from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from common.progress import bar, postfix
from common.resume import append_partial, archive_file, read_partial, remove_file
from common.rl_utils import dispersion
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def results_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def generations_path(cfg, dataset: str, policy: str) -> Path:
    return results_dir(cfg) / f"generations_{dataset}_{policy}.jsonl"


def row_id(row: dict, idx: int) -> str:
    return str(row.get("prompt_id", row.get("source_index", idx)))


def classify_outcome(rec: dict, cap: int) -> str:
    """Failure taxonomy used for the out-of-domain discussion."""
    if rec["exact"] == 1.0:
        return "correct"
    if not rec["format_ok"]:
        return "truncated_no_final" if rec["n_tokens"] >= cap else "format_failure"
    return "wrong_final"


def generate_policy_responses(cfg, tokenizer, rows, policy_name: str, batch_size: int, dataset: str, partial_path=None):
    cap = int(cfg["math_max_new_tokens"])
    model = load_frozen_policy(cfg, policy_name)
    gen_cfg = cfg.get("generation", {})
    seed = int(cfg["seed"])
    records = read_partial(partial_path)[: len(rows)] if partial_path else []
    progress = bar(total=len(rows), initial=len(records), desc=f"{dataset}:{policy_name}", unit="problem")
    for start in range(len(records), len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        set_seed(seed + start)
        gen = batch_generate(model, tokenizer, [r["messages"] for r in chunk], max_prompt_length=512,
                             max_new_tokens=cap, temperature=float(gen_cfg.get("temperature", 0.7)),
                             top_p=float(gen_cfg.get("top_p", 0.9)), do_sample=bool(gen_cfg.get("do_sample", False)))
        new = []
        for j, row in enumerate(chunk):
            text = gen["responses"][j]
            pred = extract_designated_final(text)
            rec = {
                "id": row_id(row, start + j), "policy": policy_name, "dataset": dataset,
                "question": row["question"], "gold_final": str(row["gold_final"]), "response": text,
                "pred_final": pred, "n_tokens": int(gen["response_lengths"][j]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "exact": float(exact_reward(text, str(row["gold_final"]))), "format_ok": pred is not None,
            }
            rec["failure_type"] = classify_outcome(rec, cap)
            new.append(rec)
        records.extend(new)
        if partial_path:
            append_partial(partial_path, new)
        progress.update(len(new))
        progress.set_postfix(postfix(
            acc=float(np.mean([r["exact"] for r in records])), fmt=float(np.mean([r["format_ok"] for r in records])),
            len=float(np.mean([r["n_tokens"] for r in records]))))
    progress.close()
    del model
    clear_gpu()
    return records


def policy_metrics(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "exact_accuracy": float(np.mean([r["exact"] for r in records])),
        "format_compliance": float(np.mean([r["format_ok"] for r in records])),
        "response_tokens": dispersion([r["n_tokens"] for r in records]),
        "frac_terminated_with_eos": float(np.mean([r["terminated_with_eos"] for r in records])),
        "failure_types": dict(Counter(r["failure_type"] for r in records)),
    }


def outcome_score(pref: str) -> float:
    """Pairwise judge result for candidate A (the policy under test): win 1, tie 0.5, loss 0."""
    return {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref]


def pairwise_vs(judge: PairwiseAIJudge, recs_a: list[dict], recs_b: list[dict], desc: str = "pairwise") -> tuple[dict, list[dict]]:
    """A vs B on identical problems with the fixed judge; also checks agreement with the exact verifier."""
    rows = []
    progress = bar(total=len(recs_a), desc=desc, unit="pair")
    for a, b in zip(recs_a, recs_b):
        assert a["id"] == b["id"]
        pref = judge.compare(a["question"], a["response"], b["response"])
        v = int(np.sign(a["exact"] - b["exact"]))
        j = {"A": 1, "TIE": 0, "B": -1}[pref]
        rows.append({"id": a["id"], "judge": pref, "score": outcome_score(pref), "verifier_pref": v, "judge_pref": j,
                     "exact_a": a["exact"], "exact_b": b["exact"]})
        progress.update(1)
        progress.set_postfix(postfix(
            win_rate=float(np.mean([r["score"] for r in rows])),
            agree=float(np.mean([r["verifier_pref"] == r["judge_pref"] for r in rows])),
            ties=float(np.mean([r["judge"] == "TIE" for r in rows]))))
    progress.close()
    scores = np.array([r["score"] for r in rows])
    disc = [r for r in rows if r["verifier_pref"] != 0]
    tied = [r for r in rows if r["verifier_pref"] == 0]
    summary = {
        "n": len(rows),
        "win_rate": float(scores.mean()),
        "wins": int((scores == 1).sum()), "ties": int((scores == 0.5).sum()), "losses": int((scores == 0).sum()),
        "verifier_judge_agreement_all": float(np.mean([r["verifier_pref"] == r["judge_pref"] for r in rows])),
        "n_verifier_discordant_pairs": len(disc),
        "judge_follows_verifier_on_discordant": float(np.mean([r["judge_pref"] == r["verifier_pref"] for r in disc])) if disc else float("nan"),
        "judge_ties_on_discordant": float(np.mean([r["judge_pref"] == 0 for r in disc])) if disc else float("nan"),
        "judge_opposes_verifier_on_discordant": float(np.mean([r["judge_pref"] == -r["verifier_pref"] for r in disc])) if disc else float("nan"),
        "n_verifier_tied_pairs": len(tied),
        "judge_ties_on_verifier_tied": float(np.mean([r["judge_pref"] == 0 for r in tied])) if tied else float("nan"),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--policies", nargs="*", help="subset of sft/rlvr/rlaif (default all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-examples", type=int, help="debug only: use the first N rows")
    ap.add_argument("--restart", action="store_true", help="regenerate and re-judge everything (old files are archived)")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    if args.max_examples:
        rows = rows[: args.max_examples]
    names = args.policies or list(policy_specs(cfg))
    print("Rows:", len(rows))
    print("Policies:", names)
    set_seed(int(cfg["seed"]))
    out_dir = results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(n, "sft") for n in names if n != "sft" and "sft" in names]
    metrics_path = out_dir / f"metrics_{args.dataset}.json"

    if args.restart:
        archive_file(metrics_path)
        for a, b in pairs:
            archive_file(out_dir / f"pairwise_{args.dataset}_{a}_vs_{b}.jsonl")
    elif metrics_path.exists():
        saved = load_json(metrics_path)
        wanted = {f"{a}_vs_{b}" for a, b in pairs}
        if args.skip_judge or wanted <= set(saved.get("pairwise", {})):
            print("Evaluation already finished:", metrics_path)
            return

    recs = {}
    for name in names:
        path = generations_path(cfg, args.dataset, name)
        partial = path.with_name(path.name + ".partial")
        if args.restart:
            archive_file(path)
            remove_file(partial)
        if path.exists():
            recs[name] = read_jsonl(path)
            print(f"[{name}] finished earlier: {path}")
        else:
            recs[name] = generate_policy_responses(cfg, tokenizer, rows, name, args.batch_size, args.dataset, partial_path=partial)
            write_jsonl(path, recs[name])
            remove_file(partial)
        m = policy_metrics(recs[name])
        print(f"[{name}] acc {m['exact_accuracy']:.3f} fmt {m['format_compliance']:.3f} len {m['response_tokens']['mean']:.1f}")

    metrics = {"dataset": args.dataset, "n": len(rows),
               "decoding": {**cfg.get("generation", {}), "max_new_tokens": int(cfg["math_max_new_tokens"])},
               "policies": {n: policy_metrics(recs[n]) for n in names}, "pairwise": {}}

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, out_dir / "judge_cache.json")
        for a, b in pairs:
            summ, detail = pairwise_vs(judge, recs[a], recs[b], desc=f"{args.dataset}:{a}-vs-{b}")
            metrics["pairwise"][f"{a}_vs_{b}"] = summ
            write_jsonl(out_dir / f"pairwise_{args.dataset}_{a}_vs_{b}.jsonl", detail)
            print(f"[{a} vs {b}] win rate {summ['win_rate']:.3f} (W{summ['wins']}/T{summ['ties']}/L{summ['losses']}) "
                  f"verifier-judge agreement {summ['verifier_judge_agreement_all']:.3f}")
    save_json(metrics_path, metrics)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
