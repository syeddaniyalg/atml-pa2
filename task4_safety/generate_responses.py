from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from common.progress import bar, postfix
from common.resume import append_partial, archive_file, read_partial, remove_file


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, partial_path=None):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = read_partial(partial_path)[: len(df)] if partial_path else []
    progress = bar(total=len(df), initial=len(records), desc=f"safety-gen:{policy_name}", unit="prompt")
    for start in range(len(records), len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        new = []
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            new.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
        records.extend(new)
        if partial_path:
            append_partial(partial_path, new)
        progress.update(len(new))
        progress.set_postfix(postfix(mean_tokens=sum(r["response_tokens"] for r in records) / len(records)))
    progress.close()
    del model
    return records


def output_path(cfg, policy_name: str):
    return repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy_name}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", help="subset of sft/dpo/ppo/grpo (default: all four frozen policies)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--restart", action="store_true", help="regenerate even if finished files exist (old ones are archived)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    specs = policy_specs(cfg)
    names = args.policies or list(specs)
    print("Policies:", names)
    df = load_xstest(cfg)
    print("XSTest rows:", len(df))
    set_seed(int(cfg["seed"]))

    for name in names:
        dst = output_path(cfg, name)
        partial = dst.with_name(dst.name + ".partial")
        if args.restart:
            archive_file(dst)
            remove_file(partial)
        elif dst.exists():
            print(f"[{name}] finished earlier, skipping: {dst}")
            continue
        adapter = specs[name]
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter for {name} not found: {repo_path(adapter)} (run its standard training first)")
        records = generate_for_policy(cfg, name, args.batch_size, partial_path=partial)
        assert [r["xstest_id"] for r in records] == [int(x) for x in df["xstest_id"].tolist()], "prompt order changed"
        write_jsonl(dst, records)
        remove_file(partial)
        print(f"[{name}] wrote {len(records)} responses -> {dst}; mean tokens "
              f"{sum(r['response_tokens'] for r in records) / len(records):.1f}")
        clear_gpu()


if __name__ == "__main__":
    main()
