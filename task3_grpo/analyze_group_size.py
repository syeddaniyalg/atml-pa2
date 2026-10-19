from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.progress import bar, postfix
from common.resume import archive_file

STD_TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, seed: int | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Partition rule (documented once, used for every K): each prompt contributes exactly its 8 cached completions.
    They are (optionally) shuffled within the prompt with `seed` and cut into 8/K disjoint consecutive groups of
    size K. K=8 -> 1 group/prompt, K=4 -> 2, K=2 -> 4. Total completions (and prompts) are identical for every K;
    only how they are grouped changes. Groups never mix prompts. Returns a list of dicts:
    {"prompt": source_index, "rewards": [...], "indices": [...]}.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    groups = []
    for pid, items in by_prompt.items():
        items = items[:8]
        if 8 % k != 0:
            raise ValueError("K must divide the 8 cached completions per prompt")
        order = list(range(len(items)))
        if rng is not None:
            order = rng.permutation(len(items)).tolist()
        for s in range(0, len(order), k):
            idx = order[s:s + k]
            groups.append({
                "prompt": pid,
                "rewards": [float(items[i]["reward"]) for i in idx],
                "indices": [int(items[i].get("generation_index", i)) for i in idx],
            })
    return groups


def prompt_difficulty_bins(by_prompt, n_bins: int = 3) -> dict:
    """Difficulty rule (defined once): a prompt's difficulty is the NEGATIVE of its mean cached reward over all 8
    completions; prompts are ranked and split into `n_bins` equal-size bins (hard = lowest mean reward)."""
    means = {pid: float(np.mean([float(x["reward"]) for x in items[:8]])) for pid, items in by_prompt.items()}
    order = sorted(means, key=lambda p: means[p])
    names = ["hard", "medium", "easy"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    bins = {}
    for rank, pid in enumerate(order):
        bins[pid] = names[min(rank * n_bins // len(order), n_bins - 1)]
    return bins, means


def group_metrics(groups: list[dict]) -> dict:
    """Informative-group rate, mean within-group std, and variance of the relative signal."""
    stds, adv_all, centered_all = [], [], []
    for g in groups:
        r = np.asarray(g["rewards"], dtype=float)
        sd = float(r.std())
        stds.append(sd)
        centered = r - r.mean()
        centered_all.append(centered)
        adv_all.append(centered / max(sd, STD_TOL))
    stds = np.asarray(stds)
    adv = np.concatenate(adv_all)
    centered = np.concatenate(centered_all)
    return {
        "num_groups": len(groups),
        "num_completions": int(sum(len(g["rewards"]) for g in groups)),
        "informative_group_rate": float((stds > STD_TOL).mean()),
        "uninformative_group_rate": float((stds <= STD_TOL).mean()),
        "mean_within_group_std": float(stds.mean()),
        "var_centered_reward": float(centered.var()),
        "var_normalized_advantage": float(adv.var()),
        "mean_abs_normalized_advantage": float(np.abs(adv).mean()),
    }


def analyze(by_prompt, group_sizes, n_partitions: int, seed: int):
    bins, prompt_mean_reward = prompt_difficulty_bins(by_prompt)
    result = {"difficulty_rule": "rank prompts by mean reward of their 8 cached completions; tertiles hard/medium/easy",
              "prompt_difficulty": {pid: {"bin": bins[pid], "mean_reward": prompt_mean_reward[pid]} for pid in bins},
              "n_partitions": n_partitions, "by_k": {}}
    outer = bar(group_sizes, desc="group-size", unit="K")
    for k in outer:
        reps = 1 if k == 8 else n_partitions
        per_rep_all, per_rep_bin = [], defaultdict(list)
        outer.set_postfix(K=k, partitions=reps)
        for rep in bar(range(reps), desc=f"K={k}", unit="partition", leave=False):
            groups = regroup_equal_generation_budget(by_prompt, k, seed=None if k == 8 else seed + rep)
            per_rep_all.append(group_metrics(groups))
            for name in sorted(set(bins.values())):
                sub = [g for g in groups if bins[g["prompt"]] == name]
                per_rep_bin[name].append(group_metrics(sub))

        def agg(ms):
            keys = [x for x in ms[0] if isinstance(ms[0][x], float)]
            return {x: {"mean": float(np.mean([m[x] for m in ms])), "std_over_partitions": float(np.std([m[x] for m in ms]))} for x in keys} | {
                "num_groups": ms[0]["num_groups"], "num_completions": ms[0]["num_completions"]}

        result["by_k"][str(k)] = {"overall": agg(per_rep_all), "by_difficulty": {n: agg(v) for n, v in per_rep_bin.items()}}
        overall = result["by_k"][str(k)]["overall"]
        outer.set_postfix(postfix(K=k, informative=overall["informative_group_rate"]["mean"],
                                  group_std=overall["mean_within_group_std"]["mean"],
                                  var_signal=overall["var_centered_reward"]["mean"]))
    outer.close()
    return result


def flat_table(result: dict) -> pd.DataFrame:
    rows = []
    for k, d in result["by_k"].items():
        for scope, m in [("all", d["overall"])] + [(n, v) for n, v in d["by_difficulty"].items()]:
            rows.append({
                "K": int(k), "subset": scope, "num_groups": m["num_groups"], "num_completions": m["num_completions"],
                "informative_rate": m["informative_group_rate"]["mean"],
                "informative_rate_std": m["informative_group_rate"]["std_over_partitions"],
                "mean_within_group_std": m["mean_within_group_std"]["mean"],
                "var_centered_reward": m["var_centered_reward"]["mean"],
                "var_normalized_advantage": m["var_normalized_advantage"]["mean"],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--partitions", type=int, default=50, help="random within-prompt partitions averaged for K=2,4")
    ap.add_argument("--restart", action="store_true", help="recompute even if a finished result exists (old one is archived)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    final_path = repo_path(cfg["results_dir"]) / "group_size_study.json"
    if final_path.exists() and not args.restart:
        print("Group-size study already finished:", final_path)
        flat_table(load_json(final_path)).to_csv(final_path.with_suffix(".csv"), index=False)
        return
    archive_file(final_path)
    archive_file(final_path.with_suffix(".csv"))
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    result = analyze(by_prompt, [int(k) for k in cfg["group_sizes"]], args.partitions, int(cfg["seed"]))
    all_rows = [r for items in by_prompt.values() for r in items[:8]]
    result["cache"] = {
        "prompts": len(by_prompt),
        "completions": len(all_rows),
        "frac_clipped_at_max": float(np.mean([bool(r.get("clipped_at_max", False)) for r in all_rows])),
        "reward_min": float(min(float(r["reward"]) for r in all_rows)),
        "reward_max": float(max(float(r["reward"]) for r in all_rows)),
    }
    out_dir = repo_path(cfg["results_dir"])
    save_json(out_dir / "group_size_study.json", result)
    table = flat_table(result)
    table.to_csv(out_dir / "group_size_study.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved:", out_dir / "group_size_study.json")


if __name__ == "__main__":
    main()
