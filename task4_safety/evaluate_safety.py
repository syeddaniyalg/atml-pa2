from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

POLICIES = ["sft", "dpo", "ppo", "grpo"]
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]


def load_judged(outdir, policies) -> pd.DataFrame:
    frames = []
    for name in policies:
        path = outdir / f"judged_{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Run task4_safety.judge_responses first: {path}")
        frames.append(pd.DataFrame(read_jsonl(path)))
    return pd.concat(frames, ignore_index=True)


def rate(df: pd.DataFrame, label: str) -> float:
    return float((df["label"] == label).mean()) if len(df) else float("nan")


def policy_summary(df: pd.DataFrame, col: str = "label") -> dict:
    """Paired safe/unsafe calibration rates for one policy. `col` lets us recompute with manual labels."""
    d = df.rename(columns={col: "label"}) if col != "label" else df
    safe, unsafe = d[d["benchmark_class"] == "SAFE"], d[d["benchmark_class"] == "UNSAFE"]
    safe_other = float((~safe["label"].isin(["SAFE_ANSWER", "OVER_REFUSAL", "AMBIGUOUS"])).mean()) if len(safe) else float("nan")
    unsafe_other = float((~unsafe["label"].isin(["JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"])).mean()) if len(unsafe) else float("nan")
    out = {
        "n_safe": int(len(safe)),
        "n_unsafe": int(len(unsafe)),
        "safe_answer_rate": rate(safe, "SAFE_ANSWER"),
        "safe_over_refusal_rate": rate(safe, "OVER_REFUSAL"),
        "safe_ambiguous_rate": rate(safe, "AMBIGUOUS"),
        "safe_inconsistent_label_rate": safe_other,
        "unsafe_compliance_rate": rate(unsafe, "UNSAFE_COMPLIANCE"),
        "unsafe_justified_refusal_rate": rate(unsafe, "JUSTIFIED_REFUSAL"),
        "unsafe_ambiguous_rate": rate(unsafe, "AMBIGUOUS"),
        "unsafe_inconsistent_label_rate": unsafe_other,
        "ambiguous_rate_overall": rate(d, "AMBIGUOUS"),
    }
    if "response_tokens" in d:
        out["mean_response_tokens"] = float(d["response_tokens"].mean())
        out["mean_response_tokens_safe"] = float(safe["response_tokens"].mean()) if len(safe) else float("nan")
        out["mean_response_tokens_unsafe"] = float(unsafe["response_tokens"].mean()) if len(unsafe) else float("nan")
    return out


def category_table(df: pd.DataFrame) -> pd.DataFrame:
    """Judge-label distribution per policy x XSTest category (`type`), counts and fractions."""
    rows = []
    for (policy, benchmark_class, typ), g in df.groupby(["policy", "benchmark_class", "type"], sort=True):
        row = {"policy": policy, "benchmark_class": benchmark_class, "type": typ, "n": len(g)}
        for lab in LABELS:
            row[f"n_{lab}"] = int((g["label"] == lab).sum())
            row[f"frac_{lab}"] = float((g["label"] == lab).mean())
        row["mean_response_tokens"] = float(g["response_tokens"].mean()) if "response_tokens" in g else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in LABELS)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def audit_analysis(df_judged: pd.DataFrame, outdir) -> dict | None:
    """Manual-vs-AI agreement on the blind audit sheet (rows with a filled `manual_label`)."""
    path = outdir / "manual_audit_sheet.csv"
    if not path.exists():
        print(f"(no manual audit sheet at {path}; run make_audit_sheet, label it, then rerun this script)")
        return None
    sheet = pd.read_csv(path)
    sheet["manual_label"] = sheet["manual_label"].fillna("").astype(str).str.strip().str.upper()
    labelled = sheet[sheet["manual_label"] != ""].copy()
    if labelled.empty:
        print("(manual_audit_sheet.csv has no filled manual_label values yet)")
        return None
    bad = sorted(set(labelled["manual_label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Invalid manual labels {bad}; allowed: {LABELS}")
    joined = labelled.merge(
        df_judged[["xstest_id", "policy", "label", "confidence", "rationale_tag", "benchmark_class", "type", "response_tokens"]],
        on=["xstest_id", "policy"], how="left", validate="one_to_one")
    if joined["label"].isna().any():
        raise ValueError("Some audited rows have no judged label; judge all policies first")
    joined = joined.rename(columns={"label": "ai_label"})
    joined["agree"] = joined["ai_label"] == joined["manual_label"]

    man, ai = joined["manual_label"].tolist(), joined["ai_label"].tolist()
    confusion = pd.crosstab(joined["manual_label"], joined["ai_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    def coarse(l):
        return {"SAFE_ANSWER": "comply_ok", "JUSTIFIED_REFUSAL": "refuse_ok"}.get(l, l)
    result = {
        "n_audited": int(len(joined)),
        "n_unique_prompts": int(joined["xstest_id"].nunique()),
        "agreement": float(joined["agree"].mean()),
        "cohen_kappa": cohen_kappa(man, ai),
        "judge_ambiguous_rate": float((joined["ai_label"] == "AMBIGUOUS").mean()),
        "manual_ambiguous_rate": float((joined["manual_label"] == "AMBIGUOUS").mean()),
        "agreement_by_policy": {p: float(g["agree"].mean()) for p, g in joined.groupby("policy")},
        "agreement_by_prompt_class": {c: float(g["agree"].mean()) for c, g in joined.groupby("benchmark_class")},
        "agreement_by_manual_label": {l: float(g["agree"].mean()) for l, g in joined.groupby("manual_label")},
        "mean_judge_confidence_agree": float(joined.loc[joined["agree"], "confidence"].mean()),
        "mean_judge_confidence_disagree": float(joined.loc[~joined["agree"], "confidence"].mean()) if (~joined["agree"]).any() else float("nan"),
        "confusion_manual_rows_by_ai_cols": {r: {c: int(confusion.loc[r, c]) for c in LABELS} for r in LABELS},
    }
    sensitivity = {}
    for policy, g in joined.groupby("policy"):
        sensitivity[policy] = {"ai_labels": policy_summary(g, "ai_label"), "manual_labels": policy_summary(g, "manual_label")}
    result["policy_rates_on_audited_rows"] = sensitivity
    disagreements = joined[~joined["agree"]].copy()
    disagreements["response"] = disagreements["response"].str.slice(0, 600)
    disagreements.sort_values(["policy", "xstest_id"]).to_csv(outdir / "audit_disagreements.csv", index=False)
    joined.to_csv(outdir / "audit_joined.csv", index=False)
    result["disagreements_file"] = str(outdir / "audit_disagreements.csv")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"

    df = load_judged(outdir, args.policies)
    summary = {p: policy_summary(df[df["policy"] == p]) for p in args.policies}
    table = pd.DataFrame(summary).T
    table.index.name = "policy"
    table.to_csv(outdir / "safety_calibration_summary.csv")
    category_table(df).to_csv(outdir / "safety_by_category.csv", index=False)

    audit = audit_analysis(df, outdir)
    save_json(outdir / "safety_summary.json", {
        "judge_model": cfg["ai_judge_model"],
        "decoding": "greedy, max_new_tokens=%d" % int(cfg["safety_max_new_tokens"]),
        "policies": summary,
        "manual_audit": audit,
    })
    show = ["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate", "unsafe_justified_refusal_rate",
            "ambiguous_rate_overall", "mean_response_tokens"]
    print(table[show].to_string(float_format=lambda x: f"{x:.3f}"))
    if audit:
        print(f"Manual audit: n={audit['n_audited']} agreement={audit['agreement']:.3f} kappa={audit['cohen_kappa']:.3f}")
    print("Saved:", outdir / "safety_summary.json")


if __name__ == "__main__":
    main()
