from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rl_utils import evaluate_adapter_heldout


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, help="use only the first N held-out prompts (same N for all conditions)")
    ap.add_argument("--restart", action="store_true", help="redo the evaluation even if a finished result exists (old one is archived)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = evaluate_adapter_heldout(cfg, args.adapter, args.name, cfg["results_dir"], args.max_prompts,
                                   resume=not args.restart)
    m = res["metrics"]
    print(f"[ppo:{args.name}] reward {m['reward_mean']:.3f} | KL/tok {m['kl_per_token']:.4f} | "
          f"entropy {m['entropy_per_token']:.3f} | len {m['response_length']['mean']:.1f} | trunc {m['frac_truncated']:.3f}")


if __name__ == "__main__":
    main()
