from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

PAIR_SPECS = [
    ("reasoning_corruption", "clean_correct", "corrupt_reasoning_correct_final",
     "same correct final answer; intermediate reasoning degraded (reasoning sensitivity)"),
    ("wrong_final_good_reasoning", "clean_correct", "good_reasoning_wrong_final",
     "reasoning held ~fixed; designated final answer changed to a wrong one (outcome sensitivity)"),
    ("persuasive_filler", "clean_correct", "persuasive_filler_correct",
     "correct answer + irrelevant persuasive filler; a style/verbosity-robust scorer should not prefer the filler"),
    ("gold_distractor_wrong_final", "clean_correct", "gold_distractor_wrong_final",
     "gold number is mentioned as a distractor but the designated final answer is wrong"),
]
SPEC = {name: (better, worse) for name, better, worse, _ in PAIR_SPECS}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_outcome(better: dict, worse: dict) -> str:
    rb = exact_reward(better["response"], str(better["gold_final"]))
    rw = exact_reward(worse["response"], str(worse["gold_final"]))
    return "better" if rb > rw else ("tie" if rb == rw else "wrong")


def judge_outcome(judge: PairwiseAIJudge, better: dict, worse: dict) -> str:
    pref = judge.compare(better["question"], better["response"], worse["response"])
    return {"A": "better", "TIE": "tie", "B": "wrong"}[pref]


def rates(outcomes: list[str]) -> dict:
    n = len(outcomes)
    return {
        "n": n,
        "better_rate": outcomes.count("better") / n,
        "tie_rate": outcomes.count("tie") / n,
        "wrong_rate": outcomes.count("wrong") / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-judge", action="store_true", help="only score with the exact verifier")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)

    mismatches = []
    for pid, variants in groups.items():
        for vt, row in variants.items():
            if "expected_exact_reward" in row and float(row["expected_exact_reward"]) != exact_reward(row["response"], str(row["gold_final"])):
                mismatches.append({"problem_id": pid, "variant": vt})
    print(f"Verifier vs expected_exact_reward mismatches: {len(mismatches)}")

    judge = None if args.skip_judge else PairwiseAIJudge(cfg, out_dir / "judge_cache.json")
    detail, by_mech = [], {"exact": defaultdict(list), "ai": defaultdict(list)}
    for pid, variants in sorted(groups.items()):
        for name, better_v, worse_v, _ in PAIR_SPECS:
            better, worse = variants[better_v], variants[worse_v]
            row = {"problem_id": pid, "pair": name, "better_variant": better_v, "worse_variant": worse_v,
                   "exact_better": exact_reward(better["response"], str(better["gold_final"])),
                   "exact_worse": exact_reward(worse["response"], str(worse["gold_final"]))}
            row["exact_outcome"] = verifier_outcome(better, worse)
            by_mech["exact"][name].append(row["exact_outcome"])
            if judge is not None:
                row["ai_outcome"] = judge_outcome(judge, better, worse)
                by_mech["ai"][name].append(row["ai_outcome"])
            detail.append(row)
    write_jsonl(out_dir / "diagnostic_pair_results.jsonl", detail)

    summary: dict = {"n_problems": len(groups), "verifier_expected_mismatches": mismatches,
                     "pair_definitions": {n: {"better": b, "worse": w, "isolates": d} for n, b, w, d in PAIR_SPECS},
                     "mechanisms": {}}
    for mech, per_pair in by_mech.items():
        if not per_pair:
            continue
        table = {name: rates(outs) for name, outs in per_pair.items()}
        s_reason = table["reasoning_corruption"]["better_rate"]
        s_outcome = table["wrong_final_good_reasoning"]["better_rate"]
        pooled_wrong = per_pair["wrong_final_good_reasoning"] + per_pair["gold_distractor_wrong_final"]
        summary["mechanisms"][mech] = {
            "by_pair": table,
            "S_reason": s_reason,
            "S_outcome": s_outcome,
            "S_outcome_pooled_wrong_final_variants": rates(pooled_wrong)["better_rate"],
            "reason_tie_rate": table["reasoning_corruption"]["tie_rate"],
            "reason_wrong_rate": table["reasoning_corruption"]["wrong_rate"],
            "outcome_tie_rate": table["wrong_final_good_reasoning"]["tie_rate"],
            "outcome_wrong_rate": table["wrong_final_good_reasoning"]["wrong_rate"],
            "filler_preferred_rate": table["persuasive_filler"]["wrong_rate"],
            "filler_tie_rate": table["persuasive_filler"]["tie_rate"],
            "distractor_robustness": table["gold_distractor_wrong_final"]["better_rate"],
        }

    if judge is not None:
        dis = [d for d in detail if d["exact_outcome"] != d["ai_outcome"]]
        summary["verifier_judge_disagreement_by_pair"] = {
            name: float(np.mean([d["exact_outcome"] != d["ai_outcome"] for d in detail if d["pair"] == name]))
            for name, *_ in PAIR_SPECS}
        summary["n_disagreeing_pair_instances"] = len(dis)

    save_json(out_dir / "diagnostic_summary.json", summary)
    rows = []
    for mech, m in summary["mechanisms"].items():
        for name, r in m["by_pair"].items():
            rows.append({"mechanism": mech, "pair": name, **r})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "diagnostic_summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for mech, m in summary["mechanisms"].items():
        print(f"[{mech}] S_reason={m['S_reason']:.3f} S_outcome={m['S_outcome']:.3f} "
              f"filler_preferred={m['filler_preferred_rate']:.3f} distractor_robustness={m['distractor_robustness']:.3f}")
    print("Saved:", out_dir / "diagnostic_summary.json")


if __name__ == "__main__":
    main()
