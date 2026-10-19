from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu, load_reward_model, load_tokenizer
from common.progress import bar, postfix
from common.resume import archive_file
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--restart", action="store_true", help="retrain and re-evaluate every beta (old logs and results are archived)")
    ap.add_argument("--max-gen-prompts", type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    n_short = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n_short)

    results_dir = repo_path(cfg["results_dir"])
    tokenizer = load_tokenizer(cfg["base_model"])
    reward_bundle = load_reward_model(cfg)

    rows, summaries = [], {}
    outer = bar(betas, desc="beta-study", unit="fork")
    for beta in outer:
        tag = beta_tag(beta)
        outer.set_postfix(beta=beta, stage="train")
        out_dir = f"outputs/task1_dpo/{tag}"
        train_summary = run_training(args.config, tag, None, out_dir, beta=beta, max_examples=n_short,
                                     resume=not args.restart)
        clear_gpu()
        outer.set_postfix(beta=beta, stage="eval")
        ev = evaluate_adapter(cfg, out_dir, tag, beta=beta, reward_bundle=reward_bundle,
                              max_gen_prompts=args.max_gen_prompts, tokenizer=tokenizer, resume=not args.restart)
        pref, gen = ev["standard_heldout_preferences"], ev["generation"]
        row = {
            "beta": beta,
            "train_examples": n_short,
            "heldout_dpo_loss": pref["dpo_loss"],
            "heldout_preference_accuracy": pref["preference_accuracy"],
            "heldout_mean_margin": pref["mean_margin"],
            "kl_per_token": gen["kl_per_token"],
            "kl_per_sequence": gen["kl_per_sequence"],
            "reward_mean": gen["reward_mean"],
            "reward_std": gen["reward_std"],
            "length_mean": gen["response_length"]["mean"],
            "length_std": gen["response_length"]["std"],
            "length_iqr": gen["response_length"]["iqr"],
            "frac_truncated": gen["frac_truncated"],
            "word_limit_compliance": ev["word_limit"]["compliance_rate"],
            "final_train_window_loss": train_summary["final_window_loss"],
        }
        rows.append(row)
        summaries[tag] = {"train": train_summary, "eval_file": str(results_dir / f"eval_{tag}.json")}
        outer.set_postfix(postfix(beta=beta, acc=row["heldout_preference_accuracy"], kl_tok=row["kl_per_token"],
                                  rm=row["reward_mean"], len=row["length_mean"]))
        if len(rows) == 1 and args.restart:
            archive_file(results_dir / "beta_summary.csv")
            archive_file(results_dir / "beta_summary.json")
        pd.DataFrame(rows).to_csv(results_dir / "beta_summary.csv", index=False)
        save_json(results_dir / "beta_summary.json", {"note": "short-run forks (budget differs from the one-epoch standard run)",
                                                      "rows": rows, "runs": summaries})
    outer.close()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
