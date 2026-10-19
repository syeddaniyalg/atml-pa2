from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_utils import (
    finite,
    mean_or_nan,
    peak_vram_gb,
    prompt_schedule,
    reset_peak_memory,
    response_logprobs,
    score_in_batches,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str, reward_bundle=None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = reward_bundle if reward_bundle is not None else load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def critic_values(value_model, seq, attn, pw, t_len):
    """Per-response-token values V(s_t) = head(h_{pw-1+t}); fp16 backbone + fp32 head via autocast."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        v = token_values(value_model, seq, attn)
    return v[:, pw - 1: pw - 1 + t_len].float()


def explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.bool()
    p, t = pred[m], target[m]
    if t.numel() < 2 or float(t.var(unbiased=False)) < 1e-12:
        return float("nan")
    return float(1.0 - (t - p).var(unbiased=False) / t.var(unbiased=False))


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", reward_bundle=None):
    """PPO continuation from the supplied midpoint (policy LoRA + matched critic).

    One update = (1) sample `prompts_per_update` prompts with the shared deterministic schedule and roll out the
    current policy, (2) score with the frozen reward model (minus the missing-EOS penalty), (3) compute old and
    reference log-probs and critic values, build KL-shaped token rewards and GAE advantages/returns, (4) run
    `ppo_epochs` clipped policy + critic steps on that batch. LoRA dropout is disabled so the importance ratio is
    exactly 1 on the first epoch; all forks use the same seed, prompts and per-update RNG.
    """
    bundle = prepare_ppo_continuation(config_path, reward_bundle=reward_bundle)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or (cfg["output"] if run_name == "standard" else f"outputs/task2_ppo/{run_name}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    policy, value_model = bundle["policy"], bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    rows = bundle["prompt_rows"]

    n_updates = int(cfg["updates"])
    per_update = int(cfg.get("prompts_per_update", 1))
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma, lam = float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95))
    value_coef = float(cfg.get("value_coef", 0.5))
    eos_penalty = float(cfg.get("missing_eos_penalty", 0.0))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    max_prompt_len = int(cfg["max_prompt_length"])
    max_resp_len = int(cfg["max_response_length"])
    gen_cfg = cfg.get("generation", {})
    seed = int(cfg["seed"])
    schedule = prompt_schedule(len(rows), seed, n_updates, per_update)
    device = next(policy.parameters()).device

    reset_peak_memory()
    elapsed = wall_timer()
    history = []
    skipped = 0
    for u in range(n_updates):
        set_seed(seed + u)
        batch_rows = [rows[i] for i in schedule[u]]
        prompts = [prompt_messages(r) for r in batch_rows]

        gen = batch_generate(
            policy, tokenizer, prompts, max_prompt_len, max_resp_len,
            temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)), do_sample=True,
        )
        seq, attn, pw = gen["sequences"].clone(), gen["attention_mask"].clone(), gen["prompt_width"]
        resp, rmask = gen["response_ids"].clone(), gen["response_mask"].to(device)
        t_len = resp.shape[1]
        raw = score_in_batches(reward_model, reward_tokenizer, prompts, gen["responses"],
                               max_length=int(cfg["reward_max_length"]), batch_size=2)
        truncated = torch.tensor([float(x) for x in gen["truncated"]], device=device)
        raw_t = torch.tensor(raw, device=device, dtype=torch.float32)
        task_reward = raw_t - eos_penalty * truncated

        with torch.no_grad():
            policy.eval()
            old_lp, ent = response_logprobs(policy, seq, attn, pw, resp, with_entropy=True)
            with reference_mode(policy):
                ref_lp, _ = response_logprobs(policy, seq, attn, pw, resp)
            value_model.eval()
            vals = critic_values(value_model, seq, attn, pw, t_len) * rmask
            shaped = shaped_rewards(task_reward, old_lp, ref_lp, rmask, beta_kl)
            adv, ret = compute_gae(shaped, vals, rmask, gamma=gamma, lam=lam)
            adv_n = normalize_advantages(adv, rmask)
            kl_tok = float(masked_mean(old_lp - ref_lp, rmask).item())
            kl_seq = float(((old_lp - ref_lp) * rmask).sum(-1).mean().item())
            ent_mean = float(masked_mean(ent, rmask).item())
            ev_before = explained_variance(vals, ret, rmask)
            value_mse_before = float(value_mse_loss(vals, ret, rmask).item())

        policy.train()
        value_model.train()
        diag = {"policy_loss": [], "value_loss": [], "clip_fraction": [], "ratio_mean": [], "policy_grad_norm": [], "value_grad_norm": []}
        for _ in range(ppo_epochs):
            new_lp, _ = response_logprobs(policy, seq, attn, pw, resp)
            p_loss, ratio, clipfrac = ppo_policy_loss(new_lp, old_lp, adv_n, rmask, eps)
            popt.zero_grad(set_to_none=True)
            p_loss.backward()
            pg = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm))
            if finite(pg):
                popt.step()
            else:
                skipped += 1
            popt.zero_grad(set_to_none=True)

            v_new = critic_values(value_model, seq, attn, pw, t_len)
            v_loss = value_mse_loss(v_new, ret, rmask)
            vopt.zero_grad(set_to_none=True)
            (value_coef * v_loss).backward()
            vg = float(torch.nn.utils.clip_grad_norm_([p for g in vopt.param_groups for p in g["params"]], max_norm))
            if finite(vg):
                vopt.step()
            vopt.zero_grad(set_to_none=True)

            diag["policy_loss"].append(float(p_loss.detach().item()))
            diag["value_loss"].append(float(v_loss.detach().item()))
            diag["clip_fraction"].append(float(clipfrac.item()))
            diag["ratio_mean"].append(float(masked_mean(ratio, rmask).item()))
            diag["policy_grad_norm"].append(pg)
            diag["value_grad_norm"].append(vg)

        lengths = gen["response_lengths"]
        rec = {
            "update": u + 1,
            "prompt_ids": [r.get("prompt_id") for r in batch_rows],
            "source_index": [r.get("source_index") for r in batch_rows],
            "reward_raw": float(np.mean(raw)),
            "reward_effective": float(task_reward.mean().item()),
            "kl_per_token": kl_tok,
            "kl_per_sequence": kl_seq,
            "entropy": ent_mean,
            "entropy_sampled": float(sample_entropy(old_lp, rmask).item()),
            "generated_tokens": int(sum(gen["response_lengths"])),
            "response_length": float(np.mean(lengths)),
            "frac_truncated": float(np.mean(gen["truncated"])),
            "policy_loss": mean_or_nan(diag["policy_loss"]),
            "value_loss": mean_or_nan(diag["value_loss"]),
            "value_loss_first_epoch": diag["value_loss"][0],
            "value_mse_before_update": value_mse_before,
            "value_explained_variance": ev_before,
            "mean_value": float(masked_mean(vals, rmask).item()),
            "mean_return": float(masked_mean(ret, rmask).item()),
            "clip_fraction": mean_or_nan(diag["clip_fraction"]),
            "clip_fraction_last_epoch": diag["clip_fraction"][-1],
            "ratio_mean": mean_or_nan(diag["ratio_mean"]),
            "policy_grad_norm": mean_or_nan(diag["policy_grad_norm"]),
            "value_grad_norm": mean_or_nan(diag["value_grad_norm"]),
            "clip_epsilon": eps,
            "kl_beta": beta_kl,
            "elapsed_s": elapsed(),
            "peak_vram_gb": peak_vram_gb(),
            "response_preview": gen["responses"][0][:300],
        }
        history.append(rec)
        append_jsonl(log_path, rec)
        print(f"[ppo:{run_name}] update {u + 1}/{n_updates} reward {rec['reward_raw']:.3f} kl/tok {kl_tok:.4f} "
              f"ent {ent_mean:.3f} len {rec['response_length']:.0f} clip {rec['clip_fraction']:.3f} "
              f"vloss {rec['value_loss']:.3f}")

    policy.save_pretrained(str(out))
    keys = ["reward_raw", "reward_effective", "kl_per_token", "entropy", "response_length", "policy_loss", "value_loss",
            "clip_fraction", "policy_grad_norm", "value_grad_norm"]
    half = max(len(history) // 2, 1)
    summary = {
        "run_name": run_name,
        "adapter_path": str(out),
        "log_path": str(log_path),
        "hyperparameters": {k: cfg.get(k) for k in [
            "updates", "prompts_per_update", "ppo_epochs", "policy_learning_rate", "value_lora_learning_rate",
            "value_head_learning_rate", "clip_epsilon", "kl_beta", "gamma", "gae_lambda", "value_coef",
            "missing_eos_penalty", "max_prompt_length", "max_response_length", "value_train_mode", "seed"]},
        "prompt_schedule": [[rows[i].get("prompt_id") for i in idx] for idx in schedule],
        "wall_clock_s": elapsed(),
        "peak_vram_gb": peak_vram_gb(),
        "total_generated_tokens": int(sum(h["generated_tokens"] for h in history)),
        "skipped_nonfinite_steps": skipped,
        "first_half_mean": {k: mean_or_nan([h[k] for h in history[:half]]) for k in keys},
        "second_half_mean": {k: mean_or_nan([h[k] for h in history[half:]]) for k in keys},
        "final_update": history[-1] if history else None,
    }
    save_json(results_dir / f"{run_name}_summary.json", summary)

    del policy, value_model, popt, vopt, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
