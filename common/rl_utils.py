from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.models import reference_mode
from common.progress import bar, postfix
from common.resume import append_partial, archive_file, read_partial, remove_file


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)


def response_logprobs(model, sequences, attention_mask, prompt_width, response_ids, with_entropy: bool = False):
    out = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
        return_dict=True,
    )
    t = response_ids.shape[1]
    logits = out.logits[:, prompt_width - 1: prompt_width - 1 + t, :].float()
    logp_all = F.log_softmax(logits, dim=-1)
    chosen = torch.gather(logp_all, -1, response_ids.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if with_entropy:
        with torch.no_grad():
            lp = logp_all.detach()
            entropy = -(lp.exp() * lp).sum(-1)
    return chosen, entropy


@torch.no_grad()
def rowwise_policy_and_reference_logprobs(model, gen: dict, with_entropy: bool = True):
    seq, attn = gen["sequences"], gen["attention_mask"]
    pw, resp = gen["prompt_width"], gen["response_ids"]
    mask = gen["response_mask"]
    was_training = model.training
    model.eval()
    pol, ref, ent = [], [], []
    for i in range(seq.shape[0]):
        n = int(mask[i].sum().item())
        if n == 0:
            z = torch.zeros(0)
            pol.append(z)
            ref.append(z)
            ent.append(z)
            continue
        s, a, r = seq[i:i + 1], attn[i:i + 1], resp[i:i + 1, :n]
        s = s[:, : pw + n]
        a = a[:, : pw + n]
        lp, en = response_logprobs(model, s, a, pw, r, with_entropy=with_entropy)
        pol.append(lp[0].cpu())
        ent.append(en[0].cpu() if en is not None else torch.zeros(n))
        with reference_mode(model):
            lr, _ = response_logprobs(model, s, a, pw, r, with_entropy=False)
        ref.append(lr[0].cpu())
    if was_training:
        model.train()
    return pol, ref, ent


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def prompt_schedule(num_rows: int, seed: int, num_updates: int, per_update: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_rows).tolist()
    need = num_updates * per_update
    while len(order) < need:
        order += rng.permutation(num_rows).tolist()
    return [order[u * per_update:(u + 1) * per_update] for u in range(num_updates)]


def score_in_batches(reward_model, reward_tokenizer, prompts, responses, max_length=1024, batch_size=4) -> list[float]:
    out: list[float] = []
    for i in range(0, len(prompts), batch_size):
        r = score_reward_pairs(
            reward_model, reward_tokenizer, prompts[i:i + batch_size], responses[i:i + batch_size], max_length=max_length
        )
        out.extend(float(x) for x in r.cpu().tolist())
    return out


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / 1024 ** 3)


def grad_norm_of(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean_or_nan(xs) -> float:
    xs = [float(x) for x in xs if x is not None and finite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def dispersion(xs) -> dict:
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan"), "iqr": float("nan")}
    q25, q75 = np.percentile(a, [25, 75])
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


@torch.no_grad()
def generate_and_measure(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int = 4,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    compute_kl: bool = True,
    with_entropy: bool = True,
    progress_desc: str | None = None,
) -> list[dict]:
    records: list[dict] = []
    starts = range(0, len(prompts), batch_size)
    progress = bar(total=len(prompts), desc=progress_desc, unit="prompt") if progress_desc else None
    for start in starts:
        chunk = prompts[start:start + batch_size]
        gen = batch_generate(
            model, tokenizer, chunk, max_prompt_length, max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        if compute_kl:
            pol, ref, ent = rowwise_policy_and_reference_logprobs(model, gen, with_entropy=with_entropy)
        for j in range(len(chunk)):
            rec = {
                "response": gen["responses"][j],
                "n_tokens": int(gen["response_lengths"][j]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
            }
            if compute_kl:
                diff = pol[j] - ref[j]
                rec["policy_logp_sum"] = float(pol[j].sum().item())
                rec["ref_logp_sum"] = float(ref[j].sum().item())
                rec["kl_sum"] = float(diff.sum().item())
                rec["entropy_sum"] = float(ent[j].sum().item())
                rec["sampled_entropy_sum"] = float(-pol[j].sum().item())
            records.append(rec)
        if progress is not None:
            progress.update(len(chunk))
            progress.set_postfix(postfix(len=float(np.mean([r["n_tokens"] for r in records]))))
    if progress is not None:
        progress.close()
    return records


def summarize_generation_records(records: list[dict]) -> dict:
    lengths = [r["n_tokens"] for r in records]
    out: dict[str, Any] = {
        "n": len(records),
        "response_length": dispersion(lengths),
        "frac_truncated": mean_or_nan([float(r["truncated"]) for r in records]),
        "frac_terminated_with_eos": mean_or_nan([float(r["terminated_with_eos"]) for r in records]),
    }
    if records and "reward" in records[0]:
        rew = [r["reward"] for r in records]
        out["reward_mean"] = float(np.mean(rew))
        out["reward_std"] = float(np.std(rew))
    if records and "kl_sum" in records[0]:
        tot = max(sum(lengths), 1)
        out["kl_per_token"] = float(sum(r["kl_sum"] for r in records) / tot)
        out["kl_per_sequence"] = float(np.mean([r["kl_sum"] for r in records]))
        out["entropy_per_token"] = float(sum(r["entropy_sum"] for r in records) / tot)
        out["entropy_sampled_per_token"] = float(sum(r["sampled_entropy_sum"] for r in records) / tot)
    return out


def pick_examples(records: list[dict], k: int = 3, text_chars: int = 700) -> dict:
    def slim(r):
        s = dict(r)
        s["response"] = s["response"][:text_chars]
        return s

    out = {}
    if records and "reward" in records[0]:
        by_r = sorted(range(len(records)), key=lambda i: records[i]["reward"])
        out["lowest_reward"] = [slim(records[i]) for i in by_r[:k]]
        out["highest_reward"] = [slim(records[i]) for i in by_r[-k:][::-1]]
        by_len = sorted(range(len(records)), key=lambda i: -records[i]["n_tokens"])[: max(2 * k, 6)]
        top = sorted(by_len, key=lambda i: -records[i]["reward"])[:k]
        out["high_reward_among_longest"] = [slim(records[i]) for i in top]
    by_len_all = sorted(range(len(records)), key=lambda i: records[i]["n_tokens"])
    out["shortest"] = [slim(records[i]) for i in by_len_all[:k]]
    out["longest"] = [slim(records[i]) for i in by_len_all[-k:][::-1]]
    return out


def evaluate_policy_on_prompts(
    cfg: dict,
    model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows: list[dict],
    prompts: list[list[dict]],
    max_new_tokens: int,
    max_prompt_length: int,
    batch_size: int = 4,
    do_sample: bool = False,
    reward_max_length: int = 1280,
    compute_kl: bool = True,
    cache_path=None,
    desc: str = "eval",
) -> tuple[list[dict], dict]:
    gen_cfg = cfg.get("generation", {})
    seed = int(cfg.get("seed", 0))
    t0 = time.perf_counter()
    records = read_partial(cache_path)[: len(prompts)] if cache_path else []
    progress = bar(total=len(prompts), initial=len(records), desc=desc, unit="prompt")
    for start in range(len(records), len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        chunk_rows = rows[start:start + batch_size]
        set_seed(seed + start)
        recs = generate_and_measure(
            model, tokenizer, chunk_prompts, max_prompt_length, max_new_tokens,
            batch_size=len(chunk_prompts), do_sample=do_sample,
            temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
            compute_kl=compute_kl,
        )
        rewards = score_in_batches(reward_model, reward_tokenizer, chunk_prompts, [r["response"] for r in recs],
                                   max_length=reward_max_length, batch_size=batch_size)
        for row, rec, rew in zip(chunk_rows, recs, rewards):
            rec["reward"] = rew
            rec["prompt_id"] = row.get("prompt_id")
            rec["source_index"] = row.get("source_index")
        records.extend(recs)
        if cache_path:
            append_partial(cache_path, recs)
        running = summarize_generation_records(records)
        progress.update(len(recs))
        progress.set_postfix(postfix(
            reward=running.get("reward_mean"), kl_tok=running.get("kl_per_token"),
            ent=running.get("entropy_per_token"), len=running["response_length"]["mean"],
            trunc=running["frac_truncated"]))
    progress.close()
    metrics = summarize_generation_records(records)
    metrics["decoding"] = {
        "do_sample": do_sample,
        "temperature": float(gen_cfg.get("temperature", 0.7)) if do_sample else None,
        "top_p": float(gen_cfg.get("top_p", 0.9)) if do_sample else None,
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": max_prompt_length,
    }
    metrics["eval_seconds"] = time.perf_counter() - t0
    return records, metrics


def evaluate_adapter_heldout(
    cfg: dict,
    adapter: str | None,
    name: str,
    results_dir,
    max_prompts: int | None = None,
    reward_bundle=None,
    tokenizer=None,
    batch_size: int = 4,
    resume: bool = True,
) -> dict:
    from common.data import prompt_messages, read_jsonl, repo_path, write_jsonl
    from common.logging_utils import load_json, save_json
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer

    out_dir = repo_path(results_dir)
    final_path = out_dir / f"eval_{name}.json"
    partial_path = out_dir / f"eval_{name}_generations.partial.jsonl"
    if final_path.exists():
        if resume:
            return load_json(final_path)
        archive_file(final_path)
        archive_file(out_dir / f"eval_{name}_generations.jsonl")
        remove_file(partial_path)
    elif not resume:
        remove_file(partial_path)

    set_seed(int(cfg["seed"]))
    tokenizer = tokenizer or load_tokenizer(cfg["base_model"])
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_prompts:
        rows = rows[: int(max_prompts)]
    prompts = [prompt_messages(r) for r in rows]
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = reward_bundle if reward_bundle is not None else load_reward_model(cfg)
    max_new = int(cfg.get("eval_max_response_length", cfg.get("max_completion_length", 512)))
    records, metrics = evaluate_policy_on_prompts(
        cfg, model, tokenizer, rm, rm_tok, rows, prompts,
        max_new_tokens=max_new, max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        batch_size=batch_size, do_sample=bool(cfg.get("generation", {}).get("do_sample", False)),
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
        cache_path=partial_path, desc=f"eval:{name}",
    )
    for rec, p in zip(records, prompts):
        rec["prompt"] = p[-1]["content"][:400]
    write_jsonl(out_dir / f"eval_{name}_generations.jsonl", records)
    result = {"name": name, "adapter": adapter, "seed": int(cfg["seed"]),
              "prompt_ids": [r.get("prompt_id") for r in rows], "metrics": metrics, "examples": pick_examples(records)}
    save_json(final_path, result)
    remove_file(partial_path)
    del model
    if reward_bundle is None:
        del rm
    clear_gpu()
    return result
