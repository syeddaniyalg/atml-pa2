# LLM Post-Training: DPO, PPO, GRPO, Safety Calibration, and Feedback Sources

**EE-5102 / CS-6304 Advanced Topics in Machine Learning, Fall 2026 — Programming Assignment 2**
Author: Daniyal Hussain Shah

This repository contains my implementation, experiment logs, and report source for Programming Assignment 2. The assignment studies how different post-training objectives and feedback sources change a small language model (`Qwen/Qwen2.5-1.5B-Instruct`, trained with LoRA adapters), and where the measured objectives fail to match the behavior we actually want.

| Task | Topic | What is compared |
|---|---|---|
| 1 | Direct Preference Optimization (DPO) | standard run, preference-strength `beta` study, length-confounding study |
| 2 | Proximal Policy Optimization (PPO) | clipping parameter, KL pressure, critic behavior |
| 3 | Group Relative Policy Optimization (GRPO) | group size, canonical vs. Dr. GRPO length normalization |
| 4 | Safety calibration | SFT, DPO, PPO, GRPO on XSTest, with a manual audit of the AI judge |
| 5 | RLVR vs. RLAIF | exact verification vs. AI pairwise feedback on GSM8K and SVAMP |
| 6 | Synthesis | written in the report (`paper/main.tex`) |

---

## 1. Repository structure

```text
.
├── README.md
├── requirements.txt            pinned Python dependencies
├── configs/                    YAML configuration for every task (seeds, budgets, paths)
│   ├── base.yaml               models, LoRA, decoding, data paths (shared)
│   ├── dpo.yaml  ppo.yaml  grpo.yaml  feedback.yaml  assets.yaml
├── common/                     code shared by several tasks
│   ├── data.py  models.py  generation.py  metrics.py   course utilities (loading, generation, metrics)
│   ├── rl_utils.py             log-probability/entropy helpers, held-out evaluation protocol
│   ├── resume.py               checkpointing, archiving of old logs, resumable partial files
│   └── progress.py             tqdm helpers
├── task1_dpo/                  dpo.py (loss), train.py, evaluate.py, ablate_beta.py, analyze_length.py
├── task2_ppo/                  ppo.py (objective), continue_train.py, evaluate.py, analyze_clipping.py, ablate_kl.py
├── task3_grpo/                 grpo.py (objective), continue_train.py, evaluate.py,
│                               analyze_group_size.py, compare_normalization.py
├── task4_safety/               generate_responses.py, judge_responses.py, make_audit_sheet.py, evaluate_safety.py
├── task5_feedback/             rlvr.py, rlaif.py, evaluate_math.py, score_perturbations.py, compare_feedback.py
├── scripts/                    download_assets.py, validate_assets.py, check_environment.py,
│                               prepare_transfer_eval.py, make_figures.py
├── results/                    my logs, metrics, generated responses, figures, and tables (committed)
├── paper/                      LaTeX source of the report (main.tex, neurips_2026.sty, figures/)
├── data/  cached/  checkpoints/  manifests/   fixed course assets (downloaded, not committed)
└── outputs/                    my trained adapters (created by the training scripts, not committed)
```

Large files are not committed. `data/`, `cached/`, `checkpoints/`, and `outputs/` are excluded by `.gitignore`; only the small `results/` files (JSON, JSONL, CSV, figures, tables) are kept in the repository.

---

## 2. Setup

### Requirements

- Python 3.11 or newer, and a CUDA GPU. A GPU with 16 GB of memory is recommended. On a Tesla T4 the measured peak memory was about 8.3 GB for DPO, 7.3 GB for the standard PPO run, and 10.1 GB for the standard GRPO run (the reward model is loaded in 8-bit, and the judge in 4-bit, through `bitsandbytes`).
- Internet access, to download the fixed course assets and the public Hugging Face models (`Qwen/Qwen2.5-1.5B-Instruct`, the reward model `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, and the judge `Qwen/Qwen2.5-3B-Instruct`).

### Installation

```bash
git clone https://github.com/syeddaniyalg/atml-pa2.git
cd atml-pa2
python -m pip install -r requirements.txt
python -m scripts.download_assets     # fixed data, cached rollouts, and supplied checkpoints
python -m scripts.validate_assets     # should report that all assets are present and valid
python -m scripts.check_environment   # prints library versions and GPU information
```

The experiments were run on Kaggle with two Tesla T4 GPUs (16 GB each), Python 3.13, PyTorch 2.11 (CUDA 12.8), Transformers 4.57.1, PEFT 0.17.1, and TRL 0.27.2, using the versions pinned in `requirements.txt`.

All commands below are run from the repository root.

---

## 3. Reproducing the experiments

Run the tasks in this order. Task 4 needs the standard adapters produced by Tasks 1–3 (`outputs/task1_dpo/standard`, `outputs/task2_ppo/standard`, `outputs/task3_grpo/standard`). Task 5 uses the supplied RLVR and RLAIF adapters and is independent of the others.

### Task 1: DPO

```bash
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta     --config configs/dpo.yaml     # beta in {0.03, 0.10, 0.30}, 600 examples each
python -m task1_dpo.analyze_length  --config configs/dpo.yaml     # length-balanced run and length-stratified evaluation
```

### Task 2: PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate       --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml    # cached-batch probe and epsilon forks
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml    # KL coefficient forks
```

### Task 3: GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard   # 20 updates from the supplied midpoint
python -m task3_grpo.evaluate       --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml   # K in {2, 4, 8} on the cached completions
python -m task3_grpo.compare_normalization --config configs/grpo.yaml   # canonical vs. Dr. GRPO forks
```

### Task 4: safety calibration

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # SFT, DPO, PPO, GRPO; greedy decoding
python -m task4_safety.judge_responses    --config configs/feedback.yaml   # fixed AI judge (Qwen2.5-3B-Instruct)
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # blind 60-row audit sheet
```

The audit sheet is `results/task4_safety/manual_audit_sheet.csv`. I filled the `manual_label` column by hand (one of `SAFE_ANSWER`, `JUSTIFIED_REFUSAL`, `UNSAFE_COMPLIANCE`, `OVER_REFUSAL`, `AMBIGUOUS`). After labeling, run:

```bash
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

This computes the paired safe/unsafe rates, category tables, and the judge-versus-manual agreement (agreement, Cohen's kappa, confusion matrix, disagreement list). Run it before labeling to get the automatic results only; it skips the audit part until the labels exist.

### Task 5: RLVR vs. RLAIF

```bash
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset gsm        # SFT, RLVR, RLAIF on GSM8K
python -m task5_feedback.score_perturbations --config configs/feedback.yaml                      # controlled diagnostic set
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset transfer   # SVAMP transfer set
python -m task5_feedback.compare_feedback    --config configs/feedback.yaml                      # combined comparison
```

### Figures and tables

```bash
python -m scripts.make_figures               # all tasks -> results/figures and results/tables
python -m scripts.make_figures --tasks 1 4   # selected tasks only
```

This script only reads the saved result files, so it needs no GPU.

### Resuming and restarting

Every training run, evaluation, and generation step saves its progress and resumes automatically if it is interrupted; simply rerun the same command. A finished step is skipped. To redo a step from scratch, add `--restart`; the previous logs and results are moved to timestamped `*.bak` files and are never deleted. Training checkpoints are stored next to the adapter in `outputs/.../<name>_resume` and are removed when the run finishes.

### Quick checks

For a fast sanity check of a task, most scripts accept small budgets, for example `--updates 1` (PPO/GRPO training), `--max-prompts 4` (held-out evaluation), `--max-examples 4` (DPO training or Task 5), or `--max-gen-prompts 4` (DPO evaluation). Do not use these flags for the reported results.

---

## 4. Output files

All results are written under `results/`. Training and evaluation logs are JSON Lines (one record per optimizer step or update) and every summary is machine readable.

| Folder | Main files |
|---|---|
| `results/task1_dpo/` | `*_train_log.jsonl`, `*_train_summary.json`, `eval_*.json`, `eval_*_generations.jsonl`, `beta_summary.csv`, `length_confounding_summary.json`, `length_dataset_profile.json` |
| `results/task2_ppo/` | `standard_log.jsonl`, `standard_summary.json`, `eval_*.json`, `clipping_cached_probe.json`, `clipping_forks.csv`, `kl_pressure_summary.csv`, `kl_pressure_trajectories.csv` |
| `results/task3_grpo/` | `standard_log.jsonl`, `standard_summary.json`, `eval_*.json`, `group_size_study.json`, `normalization_summary.json`, `norm_*_seq_stats.jsonl` |
| `results/task4_safety/` | `generated_*.jsonl`, `judged_*.jsonl`, `manual_audit_sheet.csv`, `safety_summary.json`, `safety_calibration_summary.csv`, `safety_by_category.csv`, `audit_joined.csv`, `audit_disagreements.csv` |
| `results/task5_feedback/` | `generations_*.jsonl`, `pairwise_*.jsonl`, `metrics_gsm.json`, `metrics_transfer.json`, `diagnostic_summary.json`, `feedback_comparison.json`, `judge_cache.json` |
| `results/figures/`, `results/tables/` | PDF and PNG figures, and CSV and Markdown tables, for each task |

`safety_summary_firstpass.json` is the Task 4 summary produced before the manual labels were added.

---

## 5. Summary of findings

The detailed analysis is in the report. In short:

- **Task 1.** The standard DPO model reaches 63.1% reference-adjusted preference accuracy on 290 held-out pairs. The three `beta` forks (62.1%, 63.4%, 62.4%) and their generated lengths (about 170 tokens) are not distinguishable. The length-balanced model changes accuracy mainly on pairs where the rejected answer is longer (65.8% to 77.2%), a shorter-answer preference rather than a general quality gain.
- **Task 2.** The PPO policy barely moves from the supplied midpoint. The clipping fraction is zero during normal updates, so the three epsilon forks are identical; clipping only becomes active in the fixed-batch probe. The supplied critic is poor (negative explained variance).
- **Task 3.** Larger GRPO groups give almost no extra informative groups beyond `K = 4`. Canonical normalization gives short completions larger gradients (short/long gradient-norm ratio 2.31), while the fixed-denominator Dr. GRPO variant reverses this (0.23).
- **Task 4.** The judge labels 87–89% of the unsafe-prompt responses of all four policies as justified refusals. The manual audit shows substantial over-refusal on safe prompts (11 of the 30 audited safe prompts) that the automatic judge never labels as over-refusal. Judge-versus-manual agreement is 71.7% (kappa 0.507).
- **Task 5.** RLVR and RLAIF do not improve GSM8K accuracy over SFT (27.3% and 27.0% vs. 27.7%). The AI pairwise judge returns ties on most comparisons and ignores answer correctness in the controlled diagnostics, while the exact verifier detects wrong final answers but not corrupted reasoning.

Most conclusions are limited by small update budgets and single seeds: the final policies stay very close to their starting points.

---

## 6. Implementation notes

### Corrected defects in the starter code

The assignment states that each core objective contains one deliberate defect. I checked each against the equation in the manual and corrected it:

- **DPO** (`task1_dpo/dpo.py`): the logit was `beta * (policy_margin + ref_margin)`; it is now `beta * (policy_margin - ref_margin)`.
- **PPO** (`task2_ppo/ppo.py`): the objective used `maximum`; it is now the clipped surrogate with `minimum`.
- **GRPO** (`task3_grpo/grpo.py`): advantages were standardized over the whole batch; they are now standardized within each prompt group.

### Other fixes found while running

- DPO training prompts longer than the maximum sequence length make the supplied encoder raise an error. Such examples are filtered before training (54 of the 1,500 standard pairs) and skipped during evaluation (10 of 300); the dropped IDs are saved in the training summary.
- The supplied critic keeps its trainable scalar head in fp16, which makes AdamW produce NaN values; the trainable critic parameters are cast to fp32 and the forward pass runs under autocast.
- Tensors returned by `generate` are created in inference mode and cannot be used by autograd; they are cloned before the update step.

### Design choices that the assignment leaves open

- **Decoding.** Held-out evaluation in Tasks 1, 2, 3, and 5 uses the released `generation` config (temperature 0.7, top-p 0.9) with a fixed seed derived from `seed = 6304` and the prompt index, so that every condition sees the same random stream. Task 4 uses greedy decoding, as required. The held-out generation caps are 256 tokens for DPO, 768 for PPO, and 512 for GRPO and for the Task 5 evaluations.
- **Training.** The released LoRA dropout (0.05) is kept. Old and reference log-probabilities are computed in evaluation mode. The prompt schedule for PPO and GRPO is a fixed seeded permutation of the training pool, so all forks use the same prompts and per-update random seeds.
- **Metrics.** KL is the sampled-response estimator (policy minus the adapter-disabled reference) and is reported both per token and per sequence. Entropy is reported both exactly (from the full distribution) and with the released sampled estimator.
- **Cached-batch clipping probe.** The PPO cache stores old log-probabilities but not new ones, so the probe takes a few optimizer steps on the cached batch for each epsilon and records the clip fraction and the affected-token fraction at every pass. The default probe learning rate is the PPO learning rate.
- **Group-size study.** Each prompt contributes exactly its 8 cached completions, which are split into `8/K` random disjoint groups (averaged over 50 partitions for `K = 2, 4`). Prompt difficulty is the rank of a prompt's mean cached reward, split into tertiles.
- **Task 5 diagnostics.** The four perturbed variants are each compared against the clean response (reasoning corruption, wrong final answer, persuasive filler, gold-number distractor). `S_reason` and `S_outcome` follow the manual's definitions.
- **Manual audit.** The fixed 60 audit IDs (30 safe, 30 unsafe) come from the released helper. Each ID is assigned to one policy by a seeded, class-balanced rotation; use `--all-policies` with `make_audit_sheet` for the full four-policy version.

---

## 7. Reproducibility

- Seed 6304 everywhere (`configs/base.yaml`).
- Prompt IDs and data indices used in each run are saved in the training summaries and evaluation files.
- The fixed course assets are pinned to a Hugging Face revision in `scripts/download_assets.py`.
- Compare conditions only with the same prompts, decoding settings, and caps. The standard-run versus short-fork budgets differ (1,500 versus 600 examples for DPO; 20 versus 8 updates for PPO and GRPO), and the tables label this.
- Sampled evaluations with a different batch size change the random stream; keep the batch size fixed across the conditions you compare.

---

## 8. Building the report

The report is written in LaTeX using the NeurIPS 2026 style file included in `paper/`. The five figures in `paper/figures/` are copies of the corresponding files in `results/figures/`.

```bash
cd paper
pdflatex main.tex
pdflatex main.tex     # second pass resolves cross-references
```

Alternatively, `tectonic main.tex` also works. The compiled PDF is not stored in the repository.

---

## 9. Attribution

- The starter code, configurations, fixed data, supplied checkpoints, and cached diagnostics come from the course release (`https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining` and the Hugging Face dataset `AbDu11aHHH/ATML-PA2-assets`). The loaders, the RLVR verifier, the pairwise judge class, and the Task 4 judge prompt and parser are taken from it, with the corrections listed above.
- Models and libraries: Qwen2.5 (Alibaba Cloud), `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, Hugging Face Transformers, PEFT, TRL, and `bitsandbytes`.
- Datasets: UltraFeedback preference pairs, XSTest, GSM8K, and SVAMP, as provided in the course release.