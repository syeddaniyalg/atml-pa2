from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.progress import bar, postfix
from common.resume import (
    archive_file,
    clear_checkpoint,
    completed_summary,
    load_checkpoint,
    prepare_log,
    save_checkpoint,
    save_json_atomic,
)
from common.rl_utils import (
    finite,
    grad_norm_of,
    mean_or_nan,
    peak_vram_gb,
    prompt_schedule,
    reset_peak_memory,
    response_logprobs,
    score_in_batches,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

UNINFORMATIVE_STD_TOL = 1e-6


def prepare_grpo_continuation(config_path: str, reward_bundle=None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = reward_bundle if reward_bundle is not None else load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor) -> dict:
    """Mean within-group reward std (population std) and fraction of zero-std (uninformative) groups."""
    stds = []
    for gid in torch.unique(group_ids):
        g = rewards[group_ids == gid]
        stds.append(float(g.std(unbiased=False).item()))
    stds = np.asarray(stds)
    return {
        "group_reward_std_mean": float(stds.mean()),
        "frac_uninformative_groups": float((stds <= UNINFORMATIVE_STD_TOL).mean()),
        "num_groups": int(len(stds)),
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", reward_bundle=None, collect_grad_stats: bool = False, resume: bool = True):
    """GRPO continuation from the supplied midpoint adapter.

    Per update: sample `prompts_per_update` prompts, draw K=`num_generations` completions each, score them with
    the frozen reward model, form within-prompt standardised advantages, optionally mask completions that hit the
    length cap, and take `policy_epochs` clipped-ratio steps (loss_type 'grpo' = 1/T_k, 'dr_grpo' = 1/L_max).
    With `collect_grad_stats`, the gradient norm of every completion's own loss term is also recorded (extra
    forward/backward per completion) so the normalisation effect on gradient allocation can be measured.
    Resumable: the adapter and optimizer are checkpointed after every update and the per-update log is kept.
    """
    cfg0 = load_yaml(config_path)
    out = repo_path(output or (cfg0["output"] if run_name == "standard" else f"outputs/task3_grpo/{run_name}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg0["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_log.jsonl"
    seq_path = results_dir / f"{run_name}_seq_stats.jsonl"
    summary_path = results_dir / f"{run_name}_summary.json"
    ckpt_dir = out.parent / f"{out.name}_resume"

    if resume:
        done = completed_summary(summary_path, out)
        if done is not None:
            print(f"[grpo:{run_name}] already completed, loading saved summary")
            return done
    else:
        archive_file(summary_path)
        clear_checkpoint(ckpt_dir)

    bundle = prepare_grpo_continuation(config_path, reward_bundle=reward_bundle)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    policy, optimizer, tokenizer = bundle["policy"], bundle["optimizer"], bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    rows = bundle["prompt_rows"]

    n_updates = int(cfg["updates"])
    per_update = int(cfg.get("prompts_per_update", 1))
    k = int(cfg["num_generations"])
    eps = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    epochs = int(cfg.get("policy_epochs", 1))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    max_prompt_len = int(cfg["max_prompt_length"])
    max_comp = int(cfg["max_completion_length"])
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    checkpoint_every = int(cfg.get("checkpoint_every", 1))
    gen_cfg = cfg.get("generation", {})
    seed = int(cfg["seed"])
    schedule = prompt_schedule(len(rows), seed, n_updates, per_update)
    device = next(policy.parameters()).device
    params = trainable_parameters(policy)

    meta = load_checkpoint(ckpt_dir, {"policy": policy}, {"policy": optimizer}) if resume else None
    start = int(meta["update"]) if meta else 0
    skipped = int(meta["skipped"]) if meta else 0
    elapsed_before = float(meta["elapsed_s"]) if meta else 0.0
    peak_before = float(meta["peak_vram_gb"] or 0.0) if meta else 0.0
    history = prepare_log(log_path, "update", start if meta else None)
    if collect_grad_stats:
        prepare_log(seq_path, "update", start if meta else None)
    if meta:
        print(f"[grpo:{run_name}] resuming after update {start}/{n_updates}")

    reset_peak_memory()
    elapsed = wall_timer()
    progress = bar(total=n_updates, initial=start, desc=f"grpo:{run_name}:{loss_type}", unit="update")
    for u in range(start, n_updates):
        set_seed(seed + u)
        prompts, gids, batch_rows = [], [], []
        for g, ridx in enumerate(schedule[u]):
            msgs = prompt_messages(rows[ridx])
            prompts += [msgs] * k
            gids += [g] * k
            batch_rows.append(rows[ridx])
        group_ids = torch.tensor(gids, device=device)

        gen = batch_generate(
            policy, tokenizer, prompts, max_prompt_len, max_comp,
            temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)), do_sample=True,
        )
        seq, attn, pw = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["prompt_width"]
        resp, rmask = gen["response_ids"].clone(), gen["response_mask"].to(device)
        rewards = torch.tensor(
            score_in_batches(reward_model, reward_tokenizer, prompts, gen["responses"],
                             max_length=int(cfg.get("reward_max_length", 1280)), batch_size=2),
            device=device, dtype=torch.float32)
        adv = group_relative_advantages(rewards, group_ids)
        token_mask = mask_truncated_sequences(rmask, gen["truncated"]) if mask_trunc else rmask

        with torch.no_grad():
            policy.eval()
            old_lp, ent = response_logprobs(policy, seq, attn, pw, resp, with_entropy=True)
            with reference_mode(policy):
                ref_lp, _ = response_logprobs(policy, seq, attn, pw, resp)
            kl_tok = float(masked_mean(old_lp - ref_lp, rmask).item())
            ent_mean = float(masked_mean(ent, rmask).item())
        gstats = group_reward_stats(rewards, group_ids)

        policy.train()
        if collect_grad_stats:
            for i in range(seq.shape[0]):
                n_i = int(gen["response_lengths"][i])
                if n_i == 0:
                    continue
                s_i, a_i, r_i = seq[i:i + 1, : pw + n_i], attn[i:i + 1, : pw + n_i], resp[i:i + 1, :n_i]
                lp_i, _ = response_logprobs(policy, s_i, a_i, pw, r_i)
                loss_i, _ = grpo_policy_loss(lp_i, old_lp[i:i + 1, :n_i], adv[i:i + 1], token_mask[i:i + 1, :n_i],
                                             ref_lp[i:i + 1, :n_i], eps, beta, loss_type, max_comp)
                optimizer.zero_grad(set_to_none=True)
                loss_i.backward()
                rec_i = {"update": u + 1, "completion": i, "group": int(gids[i]), "length": n_i,
                         "advantage": float(adv[i].item()), "reward": float(rewards[i].item()),
                         "truncated": bool(gen["truncated"][i]), "masked": bool(token_mask[i].sum().item() == 0),
                         "grad_norm": grad_norm_of(params), "loss_type": loss_type}
                append_jsonl(seq_path, rec_i)
            optimizer.zero_grad(set_to_none=True)

        diag_accum = {"loss": [], "policy_term": [], "sampled_kl": [], "clip_fraction": [], "ratio_mean": [], "grad_norm": []}
        for _ in range(epochs):
            new_lp, _ = response_logprobs(policy, seq, attn, pw, resp)
            loss, diag = grpo_policy_loss(new_lp, old_lp, adv, token_mask, ref_lp, eps, beta, loss_type, max_comp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
            if finite(gn):
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            diag_accum["loss"].append(float(loss.detach().item()))
            diag_accum["policy_term"].append(float(diag["policy_term"].item()))
            diag_accum["sampled_kl"].append(float(diag["sampled_kl"].item()))
            diag_accum["clip_fraction"].append(float(diag["clip_fraction"].item()))
            diag_accum["ratio_mean"].append(float(diag["ratio_mean"].item()))
            diag_accum["grad_norm"].append(gn)

        lengths = gen["response_lengths"]
        rec = {
            "update": u + 1,
            "loss_type": loss_type,
            "prompt_ids": [r.get("prompt_id") for r in batch_rows],
            "source_index": [r.get("source_index") for r in batch_rows],
            "reward_mean": float(rewards.mean().item()),
            "reward_std_overall": float(rewards.std(unbiased=False).item()),
            "group_reward_std_mean": gstats["group_reward_std_mean"],
            "frac_uninformative_groups": gstats["frac_uninformative_groups"],
            "num_groups": gstats["num_groups"],
            "kl_per_token": kl_tok,
            "sampled_kl_k3": mean_or_nan(diag_accum["sampled_kl"]),
            "entropy": ent_mean,
            "entropy_sampled": float(sample_entropy(old_lp, rmask).item()),
            "generated_tokens": int(sum(gen["response_lengths"])),
            "response_length": float(np.mean(lengths)),
            "frac_truncated": float(np.mean(gen["truncated"])),
            "frac_masked_completions": float((token_mask.sum(-1) == 0).float().mean().item()),
            "loss": mean_or_nan(diag_accum["loss"]),
            "policy_loss": mean_or_nan(diag_accum["policy_term"]),
            "clip_fraction": mean_or_nan(diag_accum["clip_fraction"]),
            "ratio_mean": mean_or_nan(diag_accum["ratio_mean"]),
            "grad_norm": mean_or_nan(diag_accum["grad_norm"]),
            "completion_lengths": [int(x) for x in lengths],
            "completion_rewards": [float(x) for x in rewards.cpu().tolist()],
            "elapsed_s": elapsed_before + elapsed(),
            "peak_vram_gb": max(peak_before, peak_vram_gb() or 0.0),
            "response_preview": gen["responses"][0][:300],
        }
        history.append(rec)
        append_jsonl(log_path, rec)
        progress.update(1)
        progress.set_postfix(postfix(
            reward=rec["reward_mean"], gstd=rec["group_reward_std_mean"], uninf=rec["frac_uninformative_groups"],
            kl=kl_tok, ent=ent_mean, len=rec["response_length"], clip=rec["clip_fraction"],
            pl=rec["policy_loss"], gn=rec["grad_norm"], mask=rec["frac_masked_completions"]))
        if (u + 1) % checkpoint_every == 0 and u + 1 < n_updates:
            save_checkpoint(ckpt_dir, {"policy": policy}, {"policy": optimizer}, {
                "update": u + 1, "skipped": skipped, "elapsed_s": elapsed_before + elapsed(),
                "peak_vram_gb": max(peak_before, peak_vram_gb() or 0.0)})
    progress.close()

    policy.save_pretrained(str(out))
    keys = ["reward_mean", "group_reward_std_mean", "frac_uninformative_groups", "kl_per_token", "entropy",
            "response_length", "policy_loss", "grad_norm", "clip_fraction", "frac_truncated"]
    half = max(len(history) // 2, 1)
    summary = {
        "run_name": run_name,
        "completed": True,
        "loss_type": loss_type,
        "adapter_path": str(out),
        "log_path": str(log_path),
        "hyperparameters": {x: cfg.get(x) for x in [
            "updates", "prompts_per_update", "policy_epochs", "learning_rate", "clip_epsilon", "kl_beta",
            "num_generations", "max_prompt_length", "max_completion_length", "mask_truncated_completions", "seed"]},
        "prompt_schedule": [[rows[i].get("prompt_id") for i in idx] for idx in schedule],
        "wall_clock_s": elapsed_before + elapsed(),
        "peak_vram_gb": max(peak_before, peak_vram_gb() or 0.0),
        "total_generated_tokens": int(sum(h["generated_tokens"] for h in history)),
        "skipped_nonfinite_steps": skipped,
        "collect_grad_stats": collect_grad_stats,
        "first_half_mean": {x: mean_or_nan([h[x] for h in history[:half]]) for x in keys},
        "second_half_mean": {x: mean_or_nan([h[x] for h in history[half:]]) for x in keys},
        "overall_mean": {x: mean_or_nan([h[x] for h in history]) for x in keys},
        "final_update": {x: v for x, v in (history[-1].items() if history else []) if x not in ("completion_lengths", "completion_rewards")},
    }
    save_json_atomic(summary_path, summary)
    clear_checkpoint(ckpt_dir)

    del policy, optimizer, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--grad-stats", action="store_true", help="also log per-completion gradient norms (extra compute)")
    ap.add_argument("--restart", action="store_true", help="ignore checkpoints and finished results; old logs are archived")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name,
             collect_grad_stats=args.grad_stats, resume=not args.restart)


if __name__ == "__main__":
    main()
