"""Build every report figure and table from the saved result files (no model code, no GPU).

    python -m scripts.make_figures                 # results/ -> results/figures, results/tables
    python -m scripts.make_figures --tasks 1 2     # subset

Missing inputs are skipped with a message, so it can be re-run as results arrive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import repo_path

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
LABEL_COLORS = {"SAFE_ANSWER": "#54A24B", "JUSTIFIED_REFUSAL": "#4C78A8", "UNSAFE_COMPLIANCE": "#E45756",
                "OVER_REFUSAL": "#F58518", "AMBIGUOUS": "#9D9D9D"}


def js(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def need(path: Path, what: str) -> bool:
    if not path.exists():
        print(f"  skip {what}: missing {path}")
        return False
    return True


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {out / (name + '.png')}")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, tables: Path, name: str):
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"{name}.csv", index=False)
    (tables / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
    print(f"  wrote {tables / (name + '.md')}")


def line_grid(rows: list[dict], metrics: list[tuple[str, str]], title: str, ncols: int = 4, x="update"):
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    xs = [r[x] for r in rows]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        ys = [r.get(key, np.nan) for r in rows]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=PALETTE[0])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(x, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    return fig


def task1(res: Path, out: Path, tables: Path):
    d = res / "task1_dpo"
    print("Task 1")
    beta = csv(d / "beta_summary.csv")
    std = js(d / "eval_standard.json")
    if beta is not None:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        for ax, (col, lab) in zip(axes, [("heldout_preference_accuracy", "held-out preference accuracy"),
                                         ("kl_per_token", "KL / token"), ("reward_mean", "RM score"),
                                         ("length_mean", "response length (tokens)")]):
            err = beta["length_std"] if col == "length_mean" else (beta["reward_std"] if col == "reward_mean" else None)
            ax.errorbar(beta["beta"], beta[col], yerr=err, marker="o", color=PALETTE[0], capsize=3, label="short forks (600 ex.)")
            if std:
                g, p = std.get("generation", {}), std["standard_heldout_preferences"]
                val = {"heldout_preference_accuracy": p["preference_accuracy"], "kl_per_token": g.get("kl_per_token"),
                       "reward_mean": g.get("reward_mean"), "length_mean": g.get("response_length", {}).get("mean")}[col]
                ax.scatter([std["beta"]], [val], marker="*", s=130, color=PALETTE[1], zorder=5, label="standard (1 epoch, different budget)")
            ax.set_xscale("log"); ax.set_xlabel("beta"); ax.set_title(lab, fontsize=9); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=6)
        save(fig, out, "task1_beta_study")

    logs = {n: jl(d / f"{n}_train_log.jsonl") for n in ["standard", "length_balanced"]}
    logs = {k: v for k, v in logs.items() if v}
    if logs:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for i, (n, lg) in enumerate(logs.items()):
            for ax, (k, lab) in zip(axes, [("loss", "train DPO loss"), ("preference_accuracy", "train preference accuracy"), ("grad_norm", "grad norm")]):
                ax.plot([r["step"] for r in lg], [r[k] for r in lg], label=n, color=PALETTE[i]); ax.set_title(lab, fontsize=9); ax.set_xlabel("optimizer step"); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        save(fig, out, "task1_training_curves")

    lc = js(d / "length_confounding_summary.json")
    if lc:
        strata = ["preferred_longer", "length_matched", "rejected_longer"]
        names = list(lc["models"])
        keys = list(next(iter(lc["models"].values()))["by_stratum"].keys())
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        w = 0.38
        for i, n in enumerate(names):
            axes[0].bar(np.arange(len(keys)) + i * w, [lc["models"][n]["by_stratum"][k]["preference_accuracy"] for k in keys], w, label=n, color=PALETTE[i])
        axes[0].set_xticks(np.arange(len(keys)) + w / 2); axes[0].set_xticklabels(keys, fontsize=7); axes[0].set_title("held-out preference accuracy by length stratum", fontsize=9); axes[0].legend(fontsize=7)
        axes[1].bar(names, [lc["models"][n]["generated_length"]["mean"] for n in names], yerr=[lc["models"][n]["generated_length"]["std"] for n in names], color=PALETTE[:2], capsize=3)
        axes[1].set_title("generated length (tokens)", fontsize=9)
        axes[2].bar(names, [lc["models"][n]["word_limit"]["compliance_rate"] for n in names], color=PALETTE[:2])
        axes[2].set_title("word-limit compliance", fontsize=9); axes[2].set_ylim(0, 1)
        save(fig, out, "task1_length_confounding")
        prof = lc["dataset_profile"]
        write_table(pd.DataFrame([{"dataset": k, "n": v["n"], "frac_chosen_longer": v["frac_chosen_longer_tokens"],
                                   "mean_token_diff": v["mean_token_diff_chosen_minus_rejected"],
                                   "corr_score_gap_vs_len_diff": v.get("corr_score_gap_vs_token_diff", np.nan)} for k, v in prof.items()]),
                    tables, "task1_dataset_length_profile")
        write_table(pd.DataFrame([{"model": n, "stratum": k, "n": v["n"], "pref_acc": v["preference_accuracy"], "mean_margin": v["mean_margin"]}
                                  for n in names for k, v in lc["models"][n]["by_stratum"].items()]), tables, "task1_length_strata")

    rows = []
    if std:
        g, p = std.get("generation", {}), std["standard_heldout_preferences"]
        rows.append({"run": "standard (1 epoch, 1500 ex.)", "beta": std["beta"], "heldout_loss": p["dpo_loss"], "pref_acc": p["preference_accuracy"],
                     "KL/tok": g.get("kl_per_token"), "KL/seq": g.get("kl_per_sequence"), "RM": g.get("reward_mean"),
                     "len_mean": g.get("response_length", {}).get("mean"), "len_std": g.get("response_length", {}).get("std"),
                     "word_limit": std.get("word_limit", {}).get("compliance_rate")})
    if beta is not None:
        for _, r in beta.iterrows():
            rows.append({"run": f"short fork (600 ex.) beta={r['beta']:g}", "beta": r["beta"], "heldout_loss": r["heldout_dpo_loss"], "pref_acc": r["heldout_preference_accuracy"],
                         "KL/tok": r["kl_per_token"], "KL/seq": r["kl_per_sequence"], "RM": r["reward_mean"], "len_mean": r["length_mean"],
                         "len_std": r["length_std"], "word_limit": r["word_limit_compliance"]})
    if rows:
        write_table(pd.DataFrame(rows), tables, "task1_summary")


def task2(res: Path, out: Path, tables: Path):
    d = res / "task2_ppo"
    print("Task 2")
    log = jl(d / "standard_log.jsonl")
    if log:
        fig = line_grid(log, [("reward_raw", "learned reward"), ("kl_per_token", "KL / token (sampled)"), ("policy_loss", "policy loss"),
                              ("value_loss", "value loss"), ("entropy", "entropy (exact)"), ("clip_fraction", "clip fraction"),
                              ("policy_grad_norm", "policy grad norm"), ("response_length", "response length")], "Standard PPO continuation (20 updates)")
        save(fig, out, "task2_standard_trajectories")
        fig = line_grid(log, [("value_explained_variance", "critic explained variance"), ("value_mse_before_update", "critic MSE before update"),
                              ("mean_value", "mean V"), ("mean_return", "mean return")], "PPO critic behaviour", ncols=4)
        save(fig, out, "task2_critic")
    s = js(d / "standard_summary.json")
    if s:
        write_table(pd.DataFrame([{"wall_clock_s": s["wall_clock_s"], "peak_vram_gb": s["peak_vram_gb"], "updates": s["hyperparameters"]["updates"],
                                   "total_generated_tokens": s.get("total_generated_tokens")}]), tables, "task2_standard_cost")

    probe = js(d / "clipping_cached_probe.json")
    forks = csv(d / "clipping_forks.csv")
    if probe or forks is not None:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
        if probe:
            for i, (eps, tr) in enumerate(probe["traces"].items()):
                xs = [t["pass"] for t in tr]
                axes[0].plot(xs, [t["by_epsilon"][eps]["clip_fraction"] for t in tr], marker="o", color=PALETTE[i], label=f"eps={eps}")
                axes[1].plot(xs, [t["by_epsilon"][eps]["affected_token_fraction"] for t in tr], marker="o", color=PALETTE[i], label=f"eps={eps}")
            axes[0].set_title("cached batch: clip fraction", fontsize=9); axes[1].set_title("cached batch: affected-token fraction", fontsize=9)
            axes[0].set_xlabel("optimizer pass on fixed batch"); axes[1].set_xlabel("optimizer pass on fixed batch"); axes[0].legend(fontsize=7)
            rows = [{"train_eps": float(e), "pass": t["pass"], "mean_abs_log_ratio": t["mean_abs_logratio"], **{f"{k}@eps={e2}": v2
                    for e2, v in t["by_epsilon"].items() for k, v2 in v.items()}} for e, tr in probe["traces"].items() for t in tr]
            write_table(pd.DataFrame(rows), tables, "task2_cached_probe")
        if forks is not None:
            axes[2].bar(forks["epsilon"].astype(str), forks["heldout_reward"], color=PALETTE[0]); axes[2].set_title("fork held-out reward", fontsize=9)
            axes[3].bar(forks["epsilon"].astype(str), forks["stability_reward_update_std"], color=PALETTE[1]); axes[3].set_title("stability: std of per-update reward", fontsize=9)
            write_table(forks, tables, "task2_clipping_forks")
        save(fig, out, "task2_clipping_study")

    kl = csv(d / "kl_pressure_summary.csv")
    traj = csv(d / "kl_pressure_trajectories.csv")
    if kl is not None:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        for ax, (c, lab) in zip(axes, [("heldout_reward", "held-out reward"), ("heldout_kl_per_token", "held-out KL / token"),
                                       ("heldout_entropy", "held-out entropy"), ("heldout_length_mean", "held-out length")]):
            ax.bar(kl["kl_beta"].astype(str), kl[c], color=PALETTE[:len(kl)]); ax.set_title(lab, fontsize=9); ax.set_xlabel("KL beta")
        save(fig, out, "task2_kl_pressure_heldout")
        write_table(kl[[c for c in kl.columns if c.startswith(("kl_beta", "heldout"))]], tables, "task2_kl_pressure")
    if traj is not None:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        for ax, (c, lab) in zip(axes, [("reward", "reward"), ("kl_per_token", "KL / token"), ("entropy", "entropy"), ("response_length", "length")]):
            for i, (tag, g) in enumerate(traj.groupby("condition")):
                ax.plot(g["update"], g[c], marker="o", ms=3, label=tag, color=PALETTE[i])
            ax.set_title(lab, fontsize=9); ax.set_xlabel("update"); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        save(fig, out, "task2_kl_pressure_trajectories")


def task3(res: Path, out: Path, tables: Path):
    d = res / "task3_grpo"
    print("Task 3")
    log = jl(d / "standard_log.jsonl")
    if log:
        fig = line_grid(log, [("reward_mean", "mean reward"), ("kl_per_token", "KL / token (sampled)"), ("group_reward_std_mean", "within-group reward std"),
                              ("frac_uninformative_groups", "uninformative-group fraction"), ("policy_loss", "policy loss"),
                              ("grad_norm", "grad norm"), ("entropy", "entropy (exact)"), ("response_length", "response length")], "Standard GRPO continuation (20 updates)")
        save(fig, out, "task3_standard_trajectories")
    s = js(d / "standard_summary.json")
    if s:
        write_table(pd.DataFrame([{"wall_clock_s": s["wall_clock_s"], "peak_vram_gb": s["peak_vram_gb"], "updates": s["hyperparameters"]["updates"],
                                   "total_generated_tokens": s.get("total_generated_tokens")}]), tables, "task3_standard_cost")

    gs = js(d / "group_size_study.json")
    if gs:
        ks = sorted(gs["by_k"], key=int)
        subsets = ["all", "hard", "medium", "easy"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        for ax, (m, lab) in zip(axes, [("informative_group_rate", "informative-group rate"), ("mean_within_group_std", "mean within-group reward std"),
                                       ("var_centered_reward", "variance of relative signal (r - group mean)")]):
            for i, sub in enumerate(subsets):
                ys, es = [], []
                for k in ks:
                    blk = gs["by_k"][k]["overall"] if sub == "all" else gs["by_k"][k]["by_difficulty"].get(sub)
                    ys.append(blk[m]["mean"] if blk else np.nan); es.append(blk[m]["std_over_partitions"] if blk else 0)
                ax.errorbar([int(k) for k in ks], ys, yerr=es, marker="o", capsize=3, label=sub, color=PALETTE[i])
            ax.set_xlabel("group size K (equal total generations)"); ax.set_xticks([int(k) for k in ks]); ax.set_title(lab, fontsize=9); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        save(fig, out, "task3_group_size")
    t = csv(d / "group_size_study.csv")
    if t is not None:
        write_table(t, tables, "task3_group_size")

    norm = js(d / "normalization_summary.json")
    if norm:
        rows = norm["rows"]
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.2))
        for ax, (c, lab) in zip(axes[:3], [("heldout_reward", "held-out reward"), ("heldout_kl_per_token", "held-out KL / token"), ("heldout_length_mean", "held-out length")]):
            ax.bar([r["loss_type"] for r in rows], [r[c] for r in rows], color=PALETTE[:2]); ax.set_title(lab, fontsize=9)
        for i, r in enumerate(rows):
            seq = [x for x in jl(d / f"norm_{r['loss_type']}_seq_stats.jsonl") if not x["masked"]]
            if seq:
                axes[3].scatter([x["length"] for x in seq], [x["grad_norm"] for x in seq], s=12, alpha=0.6, label=r["loss_type"], color=PALETTE[i])
            lg = jl(d / f"norm_{r['loss_type']}_log.jsonl")
            if lg:
                axes[4].plot([x["update"] for x in lg], [x["response_length"] for x in lg], marker="o", ms=3, label=r["loss_type"], color=PALETTE[i])
        axes[3].set_xlabel("completion length"); axes[3].set_title("per-completion grad norm vs length", fontsize=9); axes[3].legend(fontsize=7)
        axes[4].set_xlabel("update"); axes[4].set_title("train response length", fontsize=9); axes[4].legend(fontsize=7)
        save(fig, out, "task3_normalization")
        write_table(pd.DataFrame(rows), tables, "task3_normalization")


def task4(res: Path, out: Path, tables: Path):
    d = res / "task4_safety"
    print("Task 4")
    sm = js(d / "safety_summary.json")
    if sm:
        pol = list(sm["policies"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        w = 0.38
        for ax, (a, b, la, lb, title) in zip(axes, [("safe_answer_rate", "safe_over_refusal_rate", "safe answer", "over-refusal", "safe prompts"),
                                                    ("unsafe_justified_refusal_rate", "unsafe_compliance_rate", "justified refusal", "unsafe compliance", "unsafe prompts")]):
            ax.bar(np.arange(len(pol)), [sm["policies"][p][a] for p in pol], w, label=la, color=PALETTE[2])
            ax.bar(np.arange(len(pol)) + w, [sm["policies"][p][b] for p in pol], w, label=lb, color=PALETTE[3])
            ax.set_xticks(np.arange(len(pol)) + w / 2); ax.set_xticklabels(pol); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1); ax.legend(fontsize=7)
        save(fig, out, "task4_calibration")
        df = pd.DataFrame(sm["policies"]).T.reset_index().rename(columns={"index": "policy"})
        write_table(df, tables, "task4_calibration")
        audit = sm.get("manual_audit")
        if audit:
            labs = list(audit["confusion_manual_rows_by_ai_cols"])
            cm = np.array([[audit["confusion_manual_rows_by_ai_cols"][r][c] for c in labs] for r in labs])
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.imshow(cm, cmap="Blues")
            for i in range(len(labs)):
                for j in range(len(labs)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
            ax.set_xticklabels([l.replace("_", "\n") for l in labs], fontsize=6); ax.set_yticklabels([l.replace("_", "\n") for l in labs], fontsize=6)
            ax.set_xlabel("AI judge"); ax.set_ylabel("manual"); ax.set_title(f"agreement {audit['agreement']:.2f}, kappa {audit['cohen_kappa']:.2f} (n={audit['n_audited']})", fontsize=9)
            save(fig, out, "task4_audit_confusion")
            write_table(pd.DataFrame(cm, index=labs, columns=labs).reset_index().rename(columns={"index": "manual\\ai"}), tables, "task4_audit_confusion")
    cat = csv(d / "safety_by_category.csv")
    if cat is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.28 * cat["type"].nunique())))
        for ax, (cls, col, title) in zip(axes, [("SAFE", "frac_OVER_REFUSAL", "safe prompts: over-refusal rate"), ("UNSAFE", "frac_UNSAFE_COMPLIANCE", "unsafe prompts: unsafe-compliance rate")]):
            pv = cat[cat["benchmark_class"] == cls].pivot(index="type", columns="policy", values=col)
            im = ax.imshow(pv.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=7)
            ax.set_title(title, fontsize=9)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        save(fig, out, "task4_category_heatmap")
        labels = [c[2:] for c in cat.columns if c.startswith("n_") and c[2:] in LABEL_COLORS]
        pols = sorted(cat["policy"].unique())
        fig, axes = plt.subplots(1, len(pols), figsize=(3.6 * len(pols), max(4, 0.3 * cat["type"].nunique())), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], pols):
            g = cat[cat["policy"] == p].groupby("type")[[f"n_{l}" for l in labels]].sum()
            g = g.div(g.sum(1), axis=0)
            left = np.zeros(len(g))
            for l in labels:
                ax.barh(g.index, g[f"n_{l}"], left=left, color=LABEL_COLORS[l], label=l); left += g[f"n_{l}"].to_numpy()
            ax.set_title(p, fontsize=9)
        axes[0][0].legend(fontsize=5, loc="lower left")
        save(fig, out, "task4_label_distribution_by_category")


def task5(res: Path, out: Path, tables: Path):
    d = res / "task5_feedback"
    print("Task 5")
    fc = js(d / "feedback_comparison.json")
    if fc:
        df = pd.DataFrame(fc["in_and_out_of_domain"])
        keep = [c for c in df.columns if not c.endswith("failure_types")]
        write_table(df[keep], tables, "task5_policies")
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
        w = 0.38
        x = np.arange(len(df))
        axes[0].bar(x, df["gsm_exact_acc"], w, label="GSM8K", color=PALETTE[0]); axes[0].bar(x + w, df["svamp_exact_acc"], w, label="SVAMP", color=PALETTE[1])
        axes[0].set_title("exact-answer accuracy", fontsize=9); axes[0].legend(fontsize=7)
        axes[1].bar(x, df["gsm_format"], w, color=PALETTE[0]); axes[1].bar(x + w, df["svamp_format"], w, color=PALETTE[1]); axes[1].set_title("format compliance", fontsize=9)
        axes[2].bar(x, df["gsm_len_mean"], w, color=PALETTE[0]); axes[2].bar(x + w, df["svamp_len_mean"], w, color=PALETTE[1]); axes[2].set_title("response length (tokens)", fontsize=9)
        if "gsm_win_rate_vs_sft" in df:
            axes[3].bar(x, df["gsm_win_rate_vs_sft"].fillna(0.5), w, color=PALETTE[0]); axes[3].bar(x + w, df["svamp_win_rate_vs_sft"].fillna(0.5), w, color=PALETTE[1])
            axes[3].axhline(0.5, color="k", lw=0.7, ls="--"); axes[3].set_title("AI pairwise win rate vs SFT (SFT=n/a)", fontsize=9)
        for ax in axes:
            ax.set_xticks(x + w / 2); ax.set_xticklabels(df["policy"])
        save(fig, out, "task5_math_comparison")
    dg = js(d / "diagnostic_summary.json")
    if dg:
        mechs = list(dg["mechanisms"])
        fig, axes = plt.subplots(1, len(mechs), figsize=(6.5 * len(mechs), 3.4), sharey=True, squeeze=False)
        for ax, m in zip(axes[0], mechs):
            pairs = list(dg["mechanisms"][m]["by_pair"])
            left = np.zeros(len(pairs))
            for key, col in [("better_rate", PALETTE[2]), ("tie_rate", "#BBBBBB"), ("wrong_rate", PALETTE[3])]:
                vals = np.array([dg["mechanisms"][m]["by_pair"][p][key] for p in pairs])
                ax.barh(pairs, vals, left=left, color=col, label=key.replace("_rate", "")); left += vals
            ax.set_title({"exact": "exact verifier (RLVR)", "ai": "pairwise AI judge (RLAIF)"}.get(m, m), fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
        axes[0][0].legend(fontsize=7, loc="lower right")
        save(fig, out, "task5_diagnostics")
        rows = [{"mechanism": m, **{k: v for k, v in dg["mechanisms"][m].items() if k != "by_pair"}} for m in mechs]
        write_table(pd.DataFrame(rows), tables, "task5_sensitivity_scores")
        cmp = csv(d / "comparison_diagnostics.csv")
        if cmp is not None:
            write_table(cmp, tables, "task5_diagnostics_by_pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", help="figure dir (default <results>/figures)")
    ap.add_argument("--tables", help="table dir (default <results>/tables)")
    ap.add_argument("--tasks", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    res = repo_path(args.results_dir)
    out = repo_path(args.out) if args.out else res / "figures"
    tables = repo_path(args.tables) if args.tables else res / "tables"
    fns = {1: task1, 2: task2, 3: task3, 4: task4, 5: task5}
    for t in args.tasks:
        fns[t](res, out, tables)


if __name__ == "__main__":
    main()
