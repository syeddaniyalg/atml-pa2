from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
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
from common.rl_utils import finite, mean_or_nan, peak_vram_gb, reset_peak_memory
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_long_prompts(tokenizer, rows, max_length):
    kept, dropped = [], []
    for row in rows:
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        if n >= max_length:
            dropped.append({"source_index": row.get("source_index"), "prompt_id": row.get("prompt_id"), "prompt_tokens": n})
        else:
            kept.append(row)
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_long_prompts(tokenizer, read_jsonl(path), int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "filtered_long_prompts": dropped,
    }


def _data_ids(rows):
    return [{"source_index": r.get("source_index"), "prompt_id": r.get("prompt_id")} for r in rows]


def batch_orders(num_rows: int, epochs: int, seed: int) -> list[list[int]]:
    orders = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        orders.append(torch.randperm(num_rows, generator=generator).tolist())
    return orders


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, resume: bool = True):
    cfg = load_yaml(config_path)
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    summary_path = results_dir / f"{run_name}_train_summary.json"
    ckpt_dir = output.parent / f"{output.name}_resume"

    if resume:
        done = completed_summary(summary_path, output)
        if done is not None:
            print(f"[dpo:{run_name}] already completed, loading saved summary")
            return done
    else:
        archive_file(summary_path)
        clear_checkpoint(ckpt_dir)

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    model, optimizer, tokenizer, rows = bundle["model"], bundle["optimizer"], bundle["tokenizer"], bundle["rows"]
    beta_value = bundle["beta"]

    device = next(model.parameters()).device
    batch_size = int(cfg["batch_size"])
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    checkpoint_every = int(cfg.get("checkpoint_every", 5))
    seed = int(cfg["seed"])
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    orders = batch_orders(len(rows), epochs, seed)
    micro_per_epoch = math.ceil(len(rows) / batch_size)
    micro_total = micro_per_epoch * epochs

    meta = load_checkpoint(ckpt_dir, {"policy": model}, {"optimizer": optimizer}) if resume else None
    micro = int(meta["micro"]) if meta else 0
    opt_step = int(meta["opt_step"]) if meta else 0
    skipped = int(meta["skipped"]) if meta else 0
    elapsed_before = float(meta["elapsed_s"]) if meta else 0.0
    peak_before = float(meta["peak_vram_gb"] or 0.0) if meta else 0.0
    history = prepare_log(log_path, "step", opt_step if meta else None)
    if meta:
        print(f"[dpo:{run_name}] resuming from optimizer step {opt_step} (batch {micro}/{micro_total})")

    reset_peak_memory()
    elapsed = wall_timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window: list[dict] = []
    progress = bar(total=micro_total, initial=micro, desc=f"dpo:{run_name}", unit="batch")
    last = history[-1] if history else {}

    for global_micro in range(micro, micro_total):
        epoch, k = divmod(global_micro, micro_per_epoch)
        idx = orders[epoch][k * batch_size:(k + 1) * batch_size]
        chosen, rejected = collate([rows[i] for i in idx])
        chosen = {key: v.to(device) for key, v in chosen.items()}
        rejected = {key: v.to(device) for key, v in rejected.items()}

        with torch.no_grad(), reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)
        pol_c, _, _ = response_sequence_logprobs(model, chosen)
        pol_r, _, _ = response_sequence_logprobs(model, rejected)

        loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta_value)
        (loss / accum).backward()
        window.append({
            "loss": float(loss.detach().item()),
            "preference_accuracy": float(diag["preference_accuracy"].item()),
            "logit_mean": float(diag["logit_mean"].item()),
            "policy_margin_mean": float(diag["policy_margin_mean"].item()),
            "chosen_logp": float(pol_c.detach().mean().item()),
            "rejected_logp": float(pol_r.detach().mean().item()),
        })
        progress.update(1)
        progress.set_postfix(postfix(
            step=opt_step, loss=mean_or_nan([w["loss"] for w in window]),
            acc=mean_or_nan([w["preference_accuracy"] for w in window]),
            last_loss=last.get("loss"), last_acc=last.get("preference_accuracy"), gnorm=last.get("grad_norm")))

        done_micro = global_micro + 1
        if done_micro % accum == 0 or done_micro == micro_total:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm))
            if finite(grad_norm):
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            rec = {"step": opt_step, "epoch": epoch, "micro_batches": len(window), "grad_norm": grad_norm,
                   "beta": beta_value, "elapsed_s": elapsed_before + elapsed()}
            for key in window[0]:
                rec[key] = mean_or_nan([w[key] for w in window])
            window = []
            history.append(rec)
            append_jsonl(log_path, rec)
            last = rec
            if opt_step % checkpoint_every == 0 and done_micro != micro_total:
                save_checkpoint(ckpt_dir, {"policy": model}, {"optimizer": optimizer}, {
                    "micro": done_micro, "opt_step": opt_step, "skipped": skipped,
                    "elapsed_s": elapsed_before + elapsed(),
                    "peak_vram_gb": max(peak_before, peak_vram_gb() or 0.0)})
    progress.close()

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "completed": True,
        "beta": beta_value,
        "num_examples": len(rows),
        "filtered_prompt_too_long": bundle["filtered_long_prompts"],
        "dataset": dataset_path or cfg["paths"]["dpo_standard_train"],
        "epochs": epochs,
        "optimizer_steps": opt_step,
        "skipped_nonfinite_steps": skipped,
        "effective_batch_size": batch_size * accum,
        "learning_rate": float(cfg["learning_rate"]),
        "seed": seed,
        "final_window_loss": history[-1]["loss"] if history else None,
        "final_window_preference_accuracy": history[-1]["preference_accuracy"] if history else None,
        "wall_clock_s": elapsed_before + elapsed(),
        "peak_vram_gb": max(peak_before, peak_vram_gb() or 0.0),
        "adapter_path": str(output),
        "log_path": str(log_path),
        "data_ids": _data_ids(rows),
    }
    save_json_atomic(summary_path, summary)
    clear_checkpoint(ckpt_dir)

    del model, optimizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--restart", action="store_true", help="ignore checkpoints and finished results; old logs are archived")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, resume=not args.restart)


if __name__ == "__main__":
    main()
