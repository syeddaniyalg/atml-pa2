from __future__ import annotations

import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
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
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
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
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "filtered_long_prompts": dropped,
    }


def _data_ids(rows):
    return [{"source_index": r.get("source_index"), "prompt_id": r.get("prompt_id")} for r in rows]


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta_value = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = next(model.parameters()).device
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    micro_total = len(loader) * epochs
    reset_peak_memory()
    elapsed = wall_timer()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    window: list[dict] = []
    history: list[dict] = []
    micro = 0
    opt_step = 0
    skipped = 0
    for epoch in range(epochs):
        for chosen, rejected in loader:
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

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
            micro += 1

            if micro % accum == 0 or micro == micro_total:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm))
                if finite(grad_norm):
                    optimizer.step()
                else:
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                rec = {"step": opt_step, "epoch": epoch, "micro_batches": len(window), "grad_norm": grad_norm,
                       "beta": beta_value, "elapsed_s": elapsed()}
                for k in window[0]:
                    rec[k] = mean_or_nan([w[k] for w in window])
                window = []
                history.append(rec)
                append_jsonl(log_path, rec)
                if opt_step % 5 == 0 or micro == micro_total:
                    print(f"[dpo:{run_name}] step {opt_step} loss {rec['loss']:.4f} acc {rec['preference_accuracy']:.3f} gnorm {grad_norm:.2f}")

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "beta": beta_value,
        "num_examples": len(bundle["rows"]),
        "filtered_prompt_too_long": bundle["filtered_long_prompts"],
        "dataset": dataset_path or cfg["paths"]["dpo_standard_train"],
        "epochs": epochs,
        "optimizer_steps": opt_step,
        "skipped_nonfinite_steps": skipped,
        "effective_batch_size": int(cfg["batch_size"]) * accum,
        "learning_rate": float(cfg["learning_rate"]),
        "seed": int(cfg["seed"]),
        "final_window_loss": history[-1]["loss"] if history else None,
        "final_window_preference_accuracy": history[-1]["preference_accuracy"] if history else None,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": peak_vram_gb(),
        "adapter_path": str(output),
        "data_ids": _data_ids(bundle["rows"]),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)

    del model, optimizer, loader
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
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
