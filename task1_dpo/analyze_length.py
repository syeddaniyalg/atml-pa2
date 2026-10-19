from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import safe_corr, word_count
from common.models import clear_gpu, load_reward_model, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def dataset_length_profile(rows: list[dict], tokenizer) -> dict:
    """Property of the DATA: how often is the preferred response the longer one?"""
    dtok, dword = [], []
    for r in rows:
        yc, yr = preference_responses(r)
        dtok.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]) - len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
        dword.append(word_count(yc) - word_count(yr))
    dtok, dword = np.asarray(dtok), np.asarray(dword)
    score = [float(r["score_chosen"]) - float(r["score_rejected"]) for r in rows if "score_chosen" in r]
    out = {
        "n": len(rows),
        "frac_chosen_longer_tokens": float((dtok > 0).mean()),
        "frac_chosen_shorter_tokens": float((dtok < 0).mean()),
        "mean_token_diff_chosen_minus_rejected": float(dtok.mean()),
        "median_token_diff": float(np.median(dtok)),
        "mean_word_diff_chosen_minus_rejected": float(dword.mean()),
    }
    if len(score) == len(rows):
        out["corr_score_gap_vs_token_diff"] = safe_corr(score, dtok)
    if rows and "length_stratum" in rows[0]:
        out["stratum_counts"] = dict(Counter(r["length_stratum"] for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--standard-adapter", help="defaults to the standard_output of the config")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing length-balanced adapter")
    ap.add_argument("--max-gen-prompts", type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    results_dir = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg["base_model"])

    standard_train = read_jsonl(cfg["paths"]["dpo_standard_train"])
    profile = {
        "standard_train": dataset_length_profile(standard_train, tokenizer),
        "length_balanced_train": dataset_length_profile(balanced, tokenizer),
        "length_stratified_eval": dataset_length_profile(stratified, tokenizer),
    }
    save_json(results_dir / "length_dataset_profile.json", profile)
    print("Dataset length profile saved (standard train: frac chosen longer = "
          f"{profile['standard_train']['frac_chosen_longer_tokens']:.3f})")

    bal_out = cfg["length_output"]
    if args.skip_train and (repo_path(bal_out) / "adapter_config.json").exists():
        print("Reusing existing length-balanced adapter")
    else:
        run_training(args.config, "length_balanced", cfg["paths"]["dpo_length_train"], bal_out)
        clear_gpu()

    std_adapter = args.standard_adapter or cfg["standard_output"]
    reward_bundle = load_reward_model(cfg)
    summary = {}
    for name, adapter in [("standard", std_adapter), ("length_balanced", bal_out)]:
        ev = evaluate_adapter(cfg, adapter, f"{name}_stratified", beta=float(cfg["beta"]), reward_bundle=reward_bundle,
                              max_gen_prompts=args.max_gen_prompts, stratified=True, tokenizer=tokenizer)
        strat = ev["length_stratified_preferences"]
        summary[name] = {
            "adapter": adapter,
            "overall_accuracy_stratified_set": strat["preference_accuracy"],
            "by_stratum": strat["by_stratum"],
            "standard_heldout_accuracy": ev["standard_heldout_preferences"]["preference_accuracy"],
            "generated_length": ev["generation"]["response_length"],
            "reward_mean": ev["generation"]["reward_mean"],
            "kl_per_token": ev["generation"]["kl_per_token"],
            "word_limit": ev["word_limit"],
        }
        print(f"[{name}] by-stratum accuracy:",
              {k: round(v["preference_accuracy"], 3) for k, v in strat["by_stratum"].items()},
              f"| gen len {ev['generation']['response_length']['mean']:.1f}"
              f" | word-limit compliance {ev['word_limit']['compliance_rate']:.2f}")
    save_json(results_dir / "length_confounding_summary.json", {"dataset_profile": profile, "models": summary})
    print("Saved:", results_dir / "length_confounding_summary.json")


if __name__ == "__main__":
    main()
