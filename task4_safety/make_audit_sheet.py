from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_blind_sheet(outdir: Path, ids: list[int], policies: list[str], seed: int, all_policies: bool = False) -> pd.DataFrame:
    """Blind manual-audit sheet: prompt + response only (no AI label, no confidence), shuffled row order.

    Default: exactly one (prompt, response) per audit ID = the fixed 60-example subset. Each ID is assigned one
    policy by a seeded, class-balanced rotation so every policy is audited on ~15 examples (half safe, half unsafe).
    With `all_policies`, every ID is listed for every policy instead (4x the labelling work).
    Fill `manual_label` with one of SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS
    before looking at any judged_*.jsonl file.
    """
    gens = {}
    for name in policies:
        src = outdir / f"generated_{name}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found; run task4_safety.generate_responses first")
        gens[name] = pd.DataFrame(read_jsonl(src)).set_index("xstest_id")
    cls = gens[policies[0]]["benchmark_class"]
    rng = np.random.default_rng(seed)
    assign = []
    if all_policies:
        assign = [(i, p) for i in ids for p in policies]
    else:
        for label in ["SAFE", "UNSAFE"]:
            pool = [i for i in ids if cls.loc[i] == label]
            pool = [pool[j] for j in rng.permutation(len(pool))]
            assign += [(i, policies[k % len(policies)]) for k, i in enumerate(pool)]
    rows = [{"xstest_id": i, "policy": p, "prompt": gens[p].loc[i, "prompt"], "response": gens[p].loc[i, "response"]} for i, p in assign]
    sheet = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet["manual_label"] = ""
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--all-policies", action="store_true", help="list every audit ID for every policy (default: 60 rows, one policy per ID)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists():
        print(f"{sheet_path} already exists; not overwriting (it may contain your manual labels).")
    else:
        sheet = build_blind_sheet(outdir, ids, args.policies, int(cfg["seed"]), args.all_policies)
        sheet.to_csv(sheet_path, index=False)
        print(f"Wrote blind audit sheet with {len(sheet)} (prompt, response) rows: {sheet_path}")
    print("Label without viewing AI labels; evaluate_safety reads manual_label from manual_audit_sheet.csv.")


if __name__ == "__main__":
    main()
