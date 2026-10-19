from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import read_jsonl_safe, save_json
from common.metrics import safe_corr
from common.models import load_reward_model, load_tokenizer
from common.rl_utils import evaluate_adapter_heldout
from task2_ppo.analyze_clipping import stability_statistics
from task2_ppo.continue_train import run_ppo


def trajectory_table(logs: dict[str, list[dict]]) -> pd.DataFrame:
    """Per-update trajectories for every KL condition, aligned on the update index."""
    rows = []
    for tag, log in logs.items():
        for rec in log:
            rows.append({
                "condition": tag, "update": rec["update"], "kl_beta": rec["kl_beta"],
                "reward": rec["reward_raw"], "kl_per_token": rec["kl_per_token"], "entropy": rec["entropy"],
                "response_length": rec["response_length"], "frac_truncated": rec["frac_truncated"],
                "policy_grad_norm": rec["policy_grad_norm"], "value_loss": rec["value_loss"],
            })
    return pd.DataFrame(rows)


def first_divergence(df: pd.DataFrame, metric: str, ref_tag: str, tag: str, rel_tol: float = 0.25) -> int | None:
    """First update at which `tag` deviates from the reference condition by > rel_tol * (reference range).

    Used for the question "which observable changes first when KL pressure is weakened". The scale is the
    reference run's own range of that metric so metrics with different units are comparable.
    """
    a = df[df.condition == ref_tag].sort_values("update")[metric].to_numpy()
    b = df[df.condition == tag].sort_values("update")[metric].to_numpy()
    n = min(len(a), len(b))
    if n == 0:
        return None
    scale = max(float(np.ptp(a[:n])), 1e-8)
    dev = np.abs(b[:n] - a[:n]) / scale
    idx = np.nonzero(dev > rel_tol)[0]
    return int(idx[0] + 1) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--reuse", action="store_true", help="reuse existing fork adapters instead of retraining")
    ap.add_argument("--max-prompts", type=int, help="number of held-out prompts for fork evaluation")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    results_dir = repo_path(cfg["results_dir"])

    reward_bundle = load_reward_model(cfg)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, logs = [], {}
    for beta in [float(b) for b in cfg["kl_values"]]:
        tag = f"kl_{beta:g}"
        adapter = f"outputs/task2_ppo/{tag}"
        if not (args.reuse and (repo_path(adapter) / "adapter_config.json").exists()):
            run_ppo(args.config, output=adapter, updates=int(cfg["fork_updates"]),
                    clip_epsilon=float(cfg["clip_epsilon"]), kl_beta=beta, run_name=tag, reward_bundle=reward_bundle)
        ev = evaluate_adapter_heldout(cfg, adapter, tag, results_dir, args.max_prompts, reward_bundle, tokenizer)
        log = read_jsonl_safe(results_dir / f"{tag}_log.jsonl")
        logs[tag] = log
        m = ev["metrics"]
        gens = read_jsonl(results_dir / f"eval_{tag}_generations.jsonl")
        row = {
            "kl_beta": beta,
            "heldout_reward": m["reward_mean"], "heldout_reward_std": m["reward_std"],
            "heldout_kl_per_token": m["kl_per_token"], "heldout_kl_per_sequence": m["kl_per_sequence"],
            "heldout_entropy": m["entropy_per_token"], "heldout_entropy_sampled": m["entropy_sampled_per_token"],
            "heldout_length_mean": m["response_length"]["mean"], "heldout_length_std": m["response_length"]["std"],
            "heldout_frac_truncated": m["frac_truncated"],
            "heldout_corr_reward_length": safe_corr([g["reward"] for g in gens], [g["n_tokens"] for g in gens]),
        }
        row.update({f"train_{k}": v for k, v in stability_statistics(log).items()})
        rows.append(row)
        print(f"[{tag}] heldout reward {row['heldout_reward']:.3f} KL/tok {row['heldout_kl_per_token']:.4f} "
              f"entropy {row['heldout_entropy']:.3f} len {row['heldout_length_mean']:.1f}")
        pd.DataFrame(rows).to_csv(results_dir / "kl_pressure_summary.csv", index=False)

    traj = trajectory_table(logs)
    traj.to_csv(results_dir / "kl_pressure_trajectories.csv", index=False)

    ref_tag = f"kl_{max(float(b) for b in cfg['kl_values']):g}"
    divergence = {}
    for tag in logs:
        if tag == ref_tag:
            continue
        divergence[tag] = {metric: first_divergence(traj, metric, ref_tag, tag)
                           for metric in ["reward", "kl_per_token", "entropy", "response_length"]}
    save_json(results_dir / "kl_pressure_summary.json", {
        "rows": rows, "reference_condition_for_divergence": ref_tag,
        "first_divergence_update": divergence,
        "note": "first_divergence_update: first update where the metric differs from the reference fork by >25% of the "
                "reference fork's own range; None = never within the fork budget.",
    })
    print(pd.DataFrame(rows)[["kl_beta", "heldout_reward", "heldout_kl_per_token", "heldout_entropy", "heldout_length_mean"]].to_string(index=False))
    print("Saved:", results_dir / "kl_pressure_summary.json")


if __name__ == "__main__":
    main()
