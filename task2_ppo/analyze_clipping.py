from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import load_json, read_jsonl_safe, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.progress import bar, postfix
from common.resume import archive_file, remove_file, save_json_atomic
from common.rl_utils import evaluate_adapter_heldout, finite, response_logprobs
from task2_ppo.continue_train import run_ppo
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _prompt_lookup(cfg):
    lookup = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            lookup.setdefault(str(r["source_index"]), r)
    return lookup


def reconstruct_batch(cfg, tokenizer, cached, device):
    """Rebuild token ids for the fixed cached batch and the PPO targets (shaped rewards, GAE, advantages).

    The cache stores response *text* and per-token old/ref log-probs/values, not token ids, so the response is
    re-tokenised (plus EOS when the cached rollout terminated). The number of valid tokens per row is the cached
    log-prob length; any tokenisation mismatch is recorded in `length_mismatch`.
    """
    lookup = _prompt_lookup(cfg)
    items = []
    for row in cached:
        prow = lookup[str(row["source_index"])]
        prompt_ids = tokenizer.apply_chat_template(prompt_messages(prow), tokenize=True, add_generation_prompt=True)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        prompt_ids = list(prompt_ids)[: int(cfg["max_prompt_length"])]
        resp_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos", False):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        n_cached = int(row["old_logprobs"].numel())
        n = min(n_cached, len(resp_ids))
        items.append({
            "prompt_ids": prompt_ids,
            "resp_ids": resp_ids[:n],
            "n": n,
            "length_mismatch": int(len(resp_ids) - n_cached),
            "old": row["old_logprobs"].float()[:n],
            "ref": row["ref_logprobs"].float()[:n],
            "values": row["values"].float()[:n] if "values" in row else torch.zeros(n),
            "task_reward": float(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0))),
            "source_index": row["source_index"],
        })

    t_max = max(it["n"] for it in items)

    def pad(key):
        out = torch.zeros(len(items), t_max)
        for i, it in enumerate(items):
            out[i, : it["n"]] = it[key]
        return out

    mask = torch.zeros(len(items), t_max)
    for i, it in enumerate(items):
        mask[i, : it["n"]] = 1.0
    old, ref, values = pad("old").to(device), pad("ref").to(device), pad("values").to(device)
    mask = mask.to(device)
    task_reward = torch.tensor([it["task_reward"] for it in items], device=device)
    shaped = shaped_rewards(task_reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, ret = compute_gae(shaped, values, mask, gamma=float(cfg.get("gamma", 1.0)), lam=float(cfg.get("gae_lambda", 0.95)))
    adv = normalize_advantages(adv, mask)
    return {"items": items, "old": old, "ref": ref, "values": values, "mask": mask, "adv": adv, "returns": ret}


def clipping_stats(new_lp, old, adv, mask, eps_values):
    """Clipped-surrogate value, clip fraction and affected-token fraction for each epsilon at the current policy.

    clip_fraction     : valid tokens whose ratio lies outside [1-eps, 1+eps] (released definition, before clipping).
    affected_fraction : the subset of those tokens whose clipped branch is actually active in the min(), i.e. whose
                        gradient is zeroed: (ratio > 1+eps and A > 0) or (ratio < 1-eps and A < 0).
    """
    out = {}
    ratio = torch.exp(new_lp - old)
    for eps in eps_values:
        loss, _, clip_frac = ppo_policy_loss(new_lp, old, adv, mask, eps)
        affected = (((ratio > 1 + eps) & (adv > 0)) | ((ratio < 1 - eps) & (adv < 0))).float()
        out[f"{eps:g}"] = {
            "surrogate_objective": float(-loss.item()),
            "clip_fraction": float(clip_frac.item()),
            "affected_token_fraction": float(masked_mean(affected, mask).item()),
        }
    return out


def probe_epsilon(cfg, policy, init_state, batch, tokenizer, train_eps, probe_steps, probe_lr, eps_values):
    """Fixed-batch optimisation trace for one training epsilon, starting from the exact midpoint weights."""
    params = trainable_parameters(policy)
    with torch.no_grad():
        for p, s in zip(params, init_state):
            p.copy_(s)
    opt = AdamW(params, lr=probe_lr)
    device = next(policy.parameters()).device
    items, old, adv, mask = batch["items"], batch["old"], batch["adv"], batch["mask"]
    total_tokens = mask.sum().clamp_min(1.0)
    trace = []
    progress = bar(range(probe_steps + 1), desc=f"clip-probe eps={train_eps:g}", unit="pass", leave=False)
    for step in progress:
        new_lp = torch.zeros_like(old)
        opt.zero_grad(set_to_none=True)
        for i, it in enumerate(items):
            ids = torch.tensor([it["prompt_ids"] + it["resp_ids"]], device=device)
            attn = torch.ones_like(ids)
            pw = len(it["prompt_ids"])
            resp = ids[:, pw:]
            with torch.set_grad_enabled(step < probe_steps):
                lp, _ = response_logprobs(policy, ids, attn, pw, resp)
            n = it["n"]
            new_lp[i, :n] = lp[0, :n].detach()
            if step < probe_steps:
                ratio = torch.exp(lp[0, :n] - old[i, :n])
                a = adv[i, :n]
                obj = torch.minimum(ratio * a, ratio.clamp(1 - train_eps, 1 + train_eps) * a)
                (-(obj.sum()) / total_tokens).backward()
        stats = clipping_stats(new_lp, old, adv, mask, eps_values)
        recon = float(masked_mean((new_lp - old).abs(), mask).item())
        rec = {"pass": step, "train_epsilon": train_eps, "mean_abs_logratio": recon,
               "mean_ratio": float(masked_mean(torch.exp(new_lp - old), mask).item()), "by_epsilon": stats}
        if step < probe_steps:
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
            rec["grad_norm"] = gn
            if finite(gn):
                opt.step()
        own = stats[f"{train_eps:g}"]
        progress.set_postfix(postfix(clip=own["clip_fraction"], affected=own["affected_token_fraction"],
                                     log_ratio=recon, gnorm=rec.get("grad_norm")))
        trace.append(rec)
    progress.close()
    return trace


def run_probe(cfg, args):
    results_dir = repo_path(cfg["results_dir"])
    partial_path = results_dir / "clipping_cached_probe.partial.json"
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    cached = load_cached_rollouts(cfg["cached_rollouts"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = reconstruct_batch(cfg, tokenizer, cached, device)
    eps_values = [float(e) for e in cfg["clip_values"]]
    probe_lr = float(args.probe_lr or cfg.get("clip_probe_lr", cfg["policy_learning_rate"]))
    steps = int(args.probe_steps)
    settings = {"probe_steps": steps, "probe_lr": probe_lr}
    traces = {}
    if args.restart:
        archive_file(partial_path)
    elif partial_path.exists():
        saved = load_json(partial_path)
        if saved.get("settings") == settings:
            traces = saved["traces"]
    todo = [e for e in eps_values if f"{e:g}" not in traces]
    if todo:
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        policy.eval()
        init_state = [p.detach().clone() for p in trainable_parameters(policy)]
        outer = bar(todo, desc="clip-probe", unit="eps", initial=0)
        for eps in outer:
            outer.set_postfix(eps=eps)
            trace = probe_epsilon(cfg, policy, init_state, batch, tokenizer, eps, steps, probe_lr, eps_values)
            traces[f"{eps:g}"] = trace
            save_json_atomic(partial_path, {"settings": settings, "traces": traces})
            last = trace[-1]["by_epsilon"][f"{eps:g}"]
            outer.write(f"[clip-probe eps={eps:g}] pass0 |log-ratio| {trace[0]['mean_abs_logratio']:.2e} -> "
                        f"last pass clip {last['clip_fraction']:.4f} affected {last['affected_token_fraction']:.4f}")
        outer.close()
        del policy
        clear_gpu()
    result = {
        "n_rollouts": len(cached),
        "valid_tokens": int(batch["mask"].sum().item()),
        "probe_steps": steps,
        "probe_lr": probe_lr,
        "kl_beta": float(cfg["kl_beta"]),
        "source_index": [it["source_index"] for it in batch["items"]],
        "tokenisation_length_mismatch": [it["length_mismatch"] for it in batch["items"]],
        "traces": {f"{e:g}": traces[f"{e:g}"] for e in eps_values},
    }
    remove_file(partial_path)
    return result


def stability_statistics(log: list[dict]) -> dict:
    """Stability statistics of a fork's per-update log (all computed over the same number of updates).

    reward_update_std      : std of the per-update mean raw reward
    reward_abs_change_mean : mean |reward_{u+1} - reward_u|
    kl_final               : sampled KL/token at the last update
    grad_norm_max/mean     : policy gradient norm before clipping (max / mean over updates)
    clip_fraction_mean     : mean importance-ratio clip fraction across updates
    value_loss_mean        : mean critic loss
    """
    r = np.array([x["reward_raw"] for x in log], dtype=float)
    return {
        "updates": len(log),
        "reward_first": float(r[0]) if len(r) else float("nan"),
        "reward_last": float(r[-1]) if len(r) else float("nan"),
        "reward_mean": float(r.mean()) if len(r) else float("nan"),
        "reward_update_std": float(r.std()) if len(r) else float("nan"),
        "reward_abs_change_mean": float(np.abs(np.diff(r)).mean()) if len(r) > 1 else float("nan"),
        "kl_final": float(log[-1]["kl_per_token"]) if log else float("nan"),
        "grad_norm_max": float(max(x["policy_grad_norm"] for x in log)) if log else float("nan"),
        "grad_norm_mean": float(np.mean([x["policy_grad_norm"] for x in log])) if log else float("nan"),
        "clip_fraction_mean": float(np.mean([x["clip_fraction"] for x in log])) if log else float("nan"),
        "value_loss_mean": float(np.mean([x["value_loss"] for x in log])) if log else float("nan"),
    }


def run_forks(cfg, args):
    results_dir = repo_path(cfg["results_dir"])
    reward_bundle = load_reward_model(cfg)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = []
    outer = bar([float(e) for e in cfg["clip_values"]], desc="clip-forks", unit="fork")
    for eps in outer:
        tag = f"clip_{eps:g}"
        adapter = f"outputs/task2_ppo/{tag}"
        outer.set_postfix(eps=eps, stage="train")
        run_ppo(args.config, output=adapter, updates=int(cfg["fork_updates"]), clip_epsilon=eps,
                kl_beta=float(cfg["kl_beta"]), run_name=tag, reward_bundle=reward_bundle, resume=not args.restart)
        outer.set_postfix(eps=eps, stage="eval")
        ev = evaluate_adapter_heldout(cfg, adapter, tag, results_dir, args.max_prompts, reward_bundle, tokenizer,
                                      resume=not args.restart)
        log = read_jsonl_safe(results_dir / f"{tag}_log.jsonl")
        m = ev["metrics"]
        row = {"epsilon": eps, "heldout_reward": m["reward_mean"], "heldout_kl_per_token": m["kl_per_token"],
               "heldout_kl_per_sequence": m["kl_per_sequence"], "heldout_entropy": m["entropy_per_token"],
               "heldout_entropy_sampled": m["entropy_sampled_per_token"],
               "heldout_length_mean": m["response_length"]["mean"], "heldout_length_std": m["response_length"]["std"]}
        row.update({f"stability_{k}": v for k, v in stability_statistics(log).items()})
        rows.append(row)
        outer.set_postfix(postfix(eps=eps, reward=row["heldout_reward"], kl_tok=row["heldout_kl_per_token"],
                                  ent=row["heldout_entropy"], len=row["heldout_length_mean"],
                                  rstd=row["stability_reward_update_std"]))
        if len(rows) == 1 and args.restart:
            archive_file(results_dir / "clipping_forks.csv")
        pd.DataFrame(rows).to_csv(results_dir / "clipping_forks.csv", index=False)
    outer.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--probe-steps", type=int, default=4, help="optimizer steps on the fixed cached batch per epsilon")
    ap.add_argument("--probe-lr", type=float, help="defaults to cfg.clip_probe_lr or the PPO policy learning rate")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--restart", action="store_true", help="redo everything (old logs and results are archived)")
    ap.add_argument("--max-prompts", type=int, help="number of held-out prompts for fork evaluation")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    results_dir = repo_path(cfg["results_dir"])

    summary = {"epsilons": cfg["clip_values"]}
    if not args.skip_probe:
        probe_path = results_dir / "clipping_cached_probe.json"
        if probe_path.exists() and not args.restart:
            print("Cached-batch probe already finished, loading", probe_path)
            summary["cached_rollout_probe"] = load_json(probe_path)
        else:
            archive_file(probe_path)
            summary["cached_rollout_probe"] = run_probe(cfg, args)
            save_json(probe_path, summary["cached_rollout_probe"])
    if not args.skip_forks:
        if args.restart:
            archive_file(results_dir / "clipping_forks.json")
        summary["forks"] = run_forks(cfg, args)
        save_json(results_dir / "clipping_forks.json", summary["forks"])
    if args.restart:
        archive_file(results_dir / "clipping_summary.json")
    save_json(results_dir / "clipping_summary.json", summary)
    print("Saved:", results_dir / "clipping_summary.json")


if __name__ == "__main__":
    main()
