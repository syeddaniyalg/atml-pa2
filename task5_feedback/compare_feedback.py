from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def _load(path):
    return load_json(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = _load(out_dir / "metrics_gsm.json")
    svamp = _load(out_dir / "metrics_transfer.json")
    diag = _load(out_dir / "diagnostic_summary.json")
    missing = [n for n, v in [("metrics_gsm.json", gsm), ("metrics_transfer.json", svamp), ("diagnostic_summary.json", diag)] if v is None]
    if missing:
        raise FileNotFoundError(f"Run the corresponding task5 scripts first; missing: {missing} in {out_dir}")

    rows = []
    for policy in gsm["policies"]:
        g, t = gsm["policies"][policy], svamp["policies"].get(policy)
        row = {"policy": policy,
               "gsm_exact_acc": g["exact_accuracy"], "gsm_format": g["format_compliance"], "gsm_len_mean": g["response_tokens"]["mean"],
               "gsm_len_std": g["response_tokens"]["std"]}
        pw_g = gsm["pairwise"].get(f"{policy}_vs_sft")
        if pw_g:
            row.update({"gsm_win_rate_vs_sft": pw_g["win_rate"], "gsm_verifier_judge_agreement": pw_g["verifier_judge_agreement_all"],
                        "gsm_judge_follows_verifier_on_discordant": pw_g["judge_follows_verifier_on_discordant"]})
        if t:
            row.update({"svamp_exact_acc": t["exact_accuracy"], "svamp_format": t["format_compliance"],
                        "svamp_len_mean": t["response_tokens"]["mean"],
                        "acc_drop_gsm_to_svamp": g["exact_accuracy"] - t["exact_accuracy"],
                        "format_drop_gsm_to_svamp": g["format_compliance"] - t["format_compliance"],
                        "len_change_gsm_to_svamp": t["response_tokens"]["mean"] - g["response_tokens"]["mean"]})
            pw_t = svamp["pairwise"].get(f"{policy}_vs_sft")
            if pw_t:
                row.update({"svamp_win_rate_vs_sft": pw_t["win_rate"],
                            "win_rate_drop_gsm_to_svamp": (pw_g["win_rate"] - pw_t["win_rate"]) if pw_g else float("nan"),
                            "svamp_verifier_judge_agreement": pw_t["verifier_judge_agreement_all"]})
            row["svamp_failure_types"] = t["failure_types"]
        row["gsm_failure_types"] = g["failure_types"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_policies.csv", index=False)

    diag_rows = []
    for mech, m in diag["mechanisms"].items():
        for pair, r in m["by_pair"].items():
            diag_rows.append({"mechanism": mech, "pair": pair, **r})
    ddf = pd.DataFrame(diag_rows)
    ddf.to_csv(out_dir / "comparison_diagnostics.csv", index=False)

    mech_summary = {mech: {k: v for k, v in m.items() if k != "by_pair"} for mech, m in diag["mechanisms"].items()}
    final = {
        "in_and_out_of_domain": rows,
        "diagnostics_by_mechanism": mech_summary,
        "verifier_judge_disagreement_by_pair": diag.get("verifier_judge_disagreement_by_pair"),
        "inference_cost_note": "RLVR reward = one regex per response (no model call); RLAIF group reward with K responses "
                               "needs K(K-1)/2 judge calls per prompt (6 for K=4). Judge calls actually made here are cached in "
                               "results/task5_feedback/judge_cache.json.",
    }
    save_json(out_dir / "feedback_comparison.json", final)

    pd.set_option("display.width", 200)
    print(df.drop(columns=[c for c in df.columns if c.endswith("failure_types")]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(ddf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for mech, m in mech_summary.items():
        print(f"[{mech}] S_reason={m['S_reason']:.3f} S_outcome={m['S_outcome']:.3f} filler_preferred={m['filler_preferred_rate']:.3f}")
    print("Saved:", out_dir / "feedback_comparison.json")


if __name__ == "__main__":
    main()
