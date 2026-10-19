from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.progress import bar, postfix
from common.resume import archive_file, remove_file
from common.rl_utils import dispersion, evaluate_policy_on_prompts, generate_and_measure, pick_examples
from task1_dpo.dpo import dpo_loss


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def evaluate_preferences(cfg: dict, model, tokenizer, rows: list[dict], beta: float, batch_size: int = 4, desc: str = "pref-eval"):
    device = next(model.parameters()).device
    max_len = int(cfg["max_sequence_length"])
    model.eval()
    per_example, skipped = [], 0
    pc_all, pr_all, rc_all, rr_all = [], [], [], []
    progress = bar(total=len(rows), desc=desc, unit="pair")
    for start in range(0, len(rows), batch_size):
        chunk, chosen, rejected = [], [], []
        batch_rows = rows[start:start + batch_size]
        for row in batch_rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c = encode_prompt_response(tokenizer, prompt, yc, max_len)
                r = encode_prompt_response(tokenizer, prompt, yr, max_len)
            except ValueError:
                skipped += 1
                continue
            chunk.append(row)
            chosen.append(c)
            rejected.append(r)
        if chunk:
            cb = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen).items()}
            rb = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected).items()}
            pc, _, _ = response_sequence_logprobs(model, cb)
            pr, _, _ = response_sequence_logprobs(model, rb)
            with reference_mode(model):
                rc, _, _ = response_sequence_logprobs(model, cb)
                rr, _, _ = response_sequence_logprobs(model, rb)
            pc_all.append(pc.cpu())
            pr_all.append(pr.cpu())
            rc_all.append(rc.cpu())
            rr_all.append(rr.cpu())
            margin = (pc - pr) - (rc - rr)
            for row, m, a, b in zip(chunk, margin.cpu().tolist(), pc.cpu().tolist(), pr.cpu().tolist()):
                per_example.append({
                    "prompt_id": row.get("prompt_id"),
                    "source_index": row.get("source_index"),
                    "stratum": row.get("length_stratum"),
                    "margin": m,
                    "correct": bool(m > 0),
                    "policy_chosen_logp": a,
                    "policy_rejected_logp": b,
                })
        progress.update(len(batch_rows))
        if per_example:
            progress.set_postfix(postfix(
                acc=float(np.mean([e["correct"] for e in per_example])),
                margin=float(np.mean([e["margin"] for e in per_example])), skipped=skipped))
    progress.close()
    pc, pr, rc, rr = (torch.cat(x) for x in (pc_all, pr_all, rc_all, rr_all))
    loss, _ = dpo_loss(pc, pr, rc, rr, beta)
    out = {
        "n": len(per_example),
        "skipped_prompt_too_long": skipped,
        "beta_for_loss": beta,
        "dpo_loss": float(loss.item()),
        "preference_accuracy": float(np.mean([e["correct"] for e in per_example])),
        "mean_margin": float(np.mean([e["margin"] for e in per_example])),
    }
    strata = defaultdict(list)
    for e in per_example:
        if e["stratum"] is not None:
            strata[e["stratum"]].append(e)
    if strata:
        out["by_stratum"] = {
            name: {
                "n": len(es),
                "preference_accuracy": float(np.mean([e["correct"] for e in es])),
                "mean_margin": float(np.mean([e["margin"] for e in es])),
            }
            for name, es in sorted(strata.items())
        }
    return out, per_example


def word_limit_eval(cfg: dict, model, tokenizer, max_new_tokens: int, desc: str = "word-limit") -> tuple[dict, list[dict]]:
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [r["messages"] for r in rows]
    gen_cfg = cfg.get("generation", {})
    set_seed(int(cfg["seed"]))
    recs = generate_and_measure(model, tokenizer, prompts, 256, max_new_tokens, batch_size=4,
                                do_sample=bool(gen_cfg.get("do_sample", False)),
                                temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
                                compute_kl=False, progress_desc=desc)
    comp, out = [], []
    for row, rec in zip(rows, recs):
        text = row["messages"][-1]["content"]
        c = word_limit_compliance(text, rec["response"])
        wc = word_count(rec["response"])
        if c is not None:
            comp.append(c)
        out.append({"prompt_id": row["prompt_id"], "prompt": text, "response": rec["response"], "words": wc,
                    "compliant": c, "n_tokens": rec["n_tokens"]})
    summary = {
        "n_prompts": len(rows),
        "n_parsed_limits": len(comp),
        "compliance_rate": float(np.mean(comp)) if comp else float("nan"),
        "response_words": dispersion([o["words"] for o in out]),
        "response_tokens": dispersion([o["n_tokens"] for o in out]),
    }
    return summary, out


def evaluate_adapter(
    cfg: dict,
    adapter: str | None,
    name: str,
    beta: float | None = None,
    reward_bundle=None,
    max_gen_prompts: int | None = None,
    stratified: bool = False,
    skip_generation: bool = False,
    tokenizer=None,
    resume: bool = True,
) -> dict:
    results_dir = repo_path(cfg["results_dir"])
    final_path = results_dir / f"eval_{name}.json"
    partial_path = results_dir / f"eval_{name}_generations.partial.jsonl"
    if final_path.exists():
        if resume:
            return load_json(final_path)
        archive_file(final_path)
        archive_file(results_dir / f"eval_{name}_generations.jsonl")
        remove_file(partial_path)
    elif not resume:
        remove_file(partial_path)

    set_seed(int(cfg["seed"]))
    tokenizer = tokenizer or load_tokenizer(cfg["base_model"])
    beta = float(cfg["beta"] if beta is None else beta)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    results: dict = {"name": name, "adapter": adapter, "beta": beta, "seed": int(cfg["seed"])}

    std_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    pref, per_ex = evaluate_preferences(cfg, model, tokenizer, std_rows, beta, desc=f"pref:{name}")
    results["standard_heldout_preferences"] = pref
    save_json(results_dir / f"eval_{name}_pref_examples.json", per_ex)

    if stratified:
        len_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
        spref, sper = evaluate_preferences(cfg, model, tokenizer, len_rows, beta, desc=f"strata:{name}")
        results["length_stratified_preferences"] = spref
        save_json(results_dir / f"eval_{name}_stratified_examples.json", sper)

    if not skip_generation:
        gen_rows = std_rows[: int(max_gen_prompts)] if max_gen_prompts else std_rows
        prompts = [prompt_messages_from_preference(r) for r in gen_rows]
        rm, rm_tok = reward_bundle if reward_bundle is not None else load_reward_model(cfg)
        records, metrics = evaluate_policy_on_prompts(
            cfg, model, tokenizer, rm, rm_tok, gen_rows, prompts,
            max_new_tokens=int(cfg["max_generation_tokens"]), max_prompt_length=256, batch_size=4,
            do_sample=bool(cfg.get("generation", {}).get("do_sample", False)), reward_max_length=int(cfg.get("reward_max_length", 1024)),
            cache_path=partial_path, desc=f"gen:{name}",
        )
        for rec, row in zip(records, gen_rows):
            rec["prompt"] = prompt_messages_from_preference(row)[-1]["content"][:400]
        metrics["prompt_ids"] = [r.get("prompt_id") for r in gen_rows]
        results["generation"] = metrics
        write_jsonl(results_dir / f"eval_{name}_generations.jsonl", records)
        results["examples"] = pick_examples(records)
        wl, wl_rows = word_limit_eval(cfg, model, tokenizer, int(cfg["max_generation_tokens"]), desc=f"word-limit:{name}")
        results["word_limit"] = wl
        write_jsonl(results_dir / f"eval_{name}_word_limit.jsonl", wl_rows)
        if reward_bundle is None:
            del rm
    save_json(final_path, results)
    remove_file(partial_path)
    del model
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the reported held-out DPO loss (default: config beta)")
    ap.add_argument("--max-gen-prompts", type=int, help="number of held-out prompts used for generation metrics")
    ap.add_argument("--stratified", action="store_true", help="also evaluate on the length-stratified held-out set")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--restart", action="store_true", help="redo the evaluation even if a finished result exists (old one is archived)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = evaluate_adapter(cfg, args.adapter, args.name, args.beta, None, args.max_gen_prompts,
                           args.stratified, args.skip_generation, resume=not args.restart)
    pref = res["standard_heldout_preferences"]
    print(f"[{args.name}] held-out DPO loss {pref['dpo_loss']:.4f} | pref acc {pref['preference_accuracy']:.3f}")
    if "generation" in res:
        g = res["generation"]
        print(f"[{args.name}] RM {g.get('reward_mean', float('nan')):.3f} | KL/tok {g.get('kl_per_token', float('nan')):.4f} "
              f"| len {g['response_length']['mean']:.1f}")
    print("Saved:", repo_path(cfg["results_dir"]) / f"eval_{args.name}.json")


if __name__ == "__main__":
    main()
