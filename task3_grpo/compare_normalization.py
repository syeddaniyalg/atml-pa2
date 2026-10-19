from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import read_jsonl_safe, save_json
from common.metrics import safe_corr
from common.models import load_reward_model, load_tokenizer
from common.progress import bar, postfix
from common.resume import archive_file
from common.rl_utils import evaluate_adapter_heldout
from task3_grpo.continue_train import run_grpo

LOSS_TYPES = ["grpo", "dr_grpo"]


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return safe_corr(ra, rb)


def length_conditioned_stats(seq_stats: list[dict], max_completion_length: int) -> dict:
    """Statistics that directly test the normalisation effect, on completions that actually received gradient.

    Per-token weight: canonical GRPO divides a completion's token sum by its own length T_k (weight 1/T_k per
    token); Dr. GRPO divides by the constant L_max. The measured quantity is each completion's own gradient norm
    (before the 1/K batch average), split by median completion length.
    """
    rows = [r for r in seq_stats if not r["masked"] and r["length"] > 0]
    if not rows:
        return {"n": 0}
    lengths = np.array([r["length"] for r in rows], dtype=float)
    g = np.array([r["grad_norm"] for r in rows], dtype=float)
    absadv = np.array([abs(r["advantage"]) for r in rows], dtype=float)
    med = float(np.median(lengths))
    short, long_ = lengths <= med, lengths > med
    nz = absadv > 1e-8
    per_adv = g / np.maximum(absadv, 1e-8)
    out = {
        "n": len(rows),
        "median_length": med,
        "corr_length_gradnorm_pearson": safe_corr(lengths, g),
        "corr_length_gradnorm_spearman": spearman(lengths, g),
        "mean_gradnorm_short": float(g[short].mean()) if short.any() else float("nan"),
        "mean_gradnorm_long": float(g[long_].mean()) if long_.any() else float("nan"),
        "gradnorm_ratio_short_over_long": float(g[short].mean() / g[long_].mean()) if short.any() and long_.any() and g[long_].mean() > 0 else float("nan"),
        "corr_length_gradnorm_per_abs_adv_spearman": spearman(lengths[nz], per_adv[nz]) if nz.sum() > 2 else float("nan"),
        "mean_gradnorm_per_abs_adv_short": float(per_adv[short & nz].mean()) if (short & nz).any() else float("nan"),
        "mean_gradnorm_per_abs_adv_long": float(per_adv[long_ & nz].mean()) if (long_ & nz).any() else float("nan"),
        "per_token_weight_at_median_length": {"grpo": 1.0 / max(med, 1.0), "dr_grpo": 1.0 / float(max_completion_length)},
    }
    adv_signed = np.array([r["advantage"] for r in rows], dtype=float)
    out["mean_length_positive_adv"] = float(lengths[adv_signed > 0].mean()) if (adv_signed > 0).any() else float("nan")
    out["mean_length_negative_adv"] = float(lengths[adv_signed < 0].mean()) if (adv_signed < 0).any() else float("nan")
    return out


def length_trend(log: list[dict]) -> dict:
    lens = [h["response_length"] for h in log]
    half = max(len(lens) // 2, 1)
    return {
        "length_first_half": float(np.mean(lens[:half])) if lens else float("nan"),
        "length_second_half": float(np.mean(lens[half:])) if lens[half:] else float("nan"),
        "length_slope_per_update": float(np.polyfit(np.arange(len(lens)), lens, 1)[0]) if len(lens) > 1 else float("nan"),
        "mean_frac_truncated": float(np.mean([h["frac_truncated"] for h in log])) if log else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--restart", action="store_true", help="retrain and re-evaluate every fork (old logs and results are archived)")
    ap.add_argument("--no-grad-stats", action="store_true", help="skip per-completion gradient norms (cheaper)")
    ap.add_argument("--max-prompts", type=int, help="number of held-out prompts for fork evaluation")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    results_dir = repo_path(cfg["results_dir"])

    reward_bundle = load_reward_model(cfg)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, detail = [], {}
    outer = bar(LOSS_TYPES, desc="normalization", unit="fork")
    for lt in outer:
        tag = f"norm_{lt}"
        adapter = f"outputs/task3_grpo/{tag}"
        outer.set_postfix(loss_type=lt, stage="train")
        run_grpo(args.config, output=adapter, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=tag,
                 reward_bundle=reward_bundle, collect_grad_stats=not args.no_grad_stats, resume=not args.restart)
        outer.set_postfix(loss_type=lt, stage="eval")
        ev = evaluate_adapter_heldout(cfg, adapter, tag, results_dir, args.max_prompts, reward_bundle, tokenizer,
                                      resume=not args.restart)
        log = read_jsonl_safe(results_dir / f"{tag}_log.jsonl")
        seq_stats = read_jsonl_safe(results_dir / f"{tag}_seq_stats.jsonl")
        m = ev["metrics"]
        lc = length_conditioned_stats(seq_stats, int(cfg["max_completion_length"]))
        trend = length_trend(log)
        row = {
            "loss_type": lt,
            "heldout_reward": m["reward_mean"], "heldout_reward_std": m["reward_std"],
            "heldout_kl_per_token": m["kl_per_token"], "heldout_kl_per_sequence": m["kl_per_sequence"],
            "heldout_entropy": m["entropy_per_token"], "heldout_entropy_sampled": m["entropy_sampled_per_token"],
            "heldout_length_mean": m["response_length"]["mean"], "heldout_length_std": m["response_length"]["std"],
            "heldout_frac_truncated": m["frac_truncated"],
            "train_reward_mean": float(np.mean([h["reward_mean"] for h in log])) if log else float("nan"),
            "train_grad_norm_mean": float(np.mean([h["grad_norm"] for h in log])) if log else float("nan"),
            "train_informative_rate": 1.0 - float(np.mean([h["frac_uninformative_groups"] for h in log])) if log else float("nan"),
        }
        row.update({f"trend_{k}": v for k, v in trend.items()})
        row.update({f"lencond_{k}": v for k, v in lc.items() if not isinstance(v, dict)})
        rows.append(row)
        detail[lt] = {"length_conditioned": lc, "length_trend": trend, "eval_file": str(results_dir / f"eval_{tag}.json")}
        outer.set_postfix(postfix(loss_type=lt, reward=row["heldout_reward"], kl_tok=row["heldout_kl_per_token"],
                                  ent=row["heldout_entropy"], len=row["heldout_length_mean"],
                                  gn_ratio=lc.get("gradnorm_ratio_short_over_long")))
        if len(rows) == 1 and args.restart:
            archive_file(results_dir / "normalization_summary.csv")
        pd.DataFrame(rows).to_csv(results_dir / "normalization_summary.csv", index=False)
    outer.close()

    if args.restart:
        archive_file(results_dir / "normalization_summary.json")
    save_json(results_dir / "normalization_summary.json", {"rows": rows, "detail": detail})
    print("Saved:", results_dir / "normalization_summary.json")


if __name__ == "__main__":
    main()
