# DIAL: Direct Iterative Adversarial Learning for Realistic Multi-Turn Dialogue Simulation

Reference implementation of the paper:

> **DIAL: Direct Iterative Adversarial Learning for Realistic Multi-Turn Dialogue Simulation**
> Ziyi Zhu, Olivier Tieleman, Caitlin A. Stamatis, Luka Smyth, Thomas D. Hull, Daniel R. Cahn, Matteo Malgaroli.
> arXiv preprint [arXiv:2512.20773](https://arxiv.org/abs/2512.20773), 2025.

DIAL is a DPO-based adversarial training framework that iteratively improves a multi-turn user simulator by playing it against a learned discriminator. Each iteration:

1. **Simulate** — the current user simulator (US) interacts with a response generator (RG) to produce dialogues.
2. **Discriminate** — a token-classification discriminator is trained to distinguish *real* (human) from *simulated* user turns.
3. **Score** — the discriminator's log-odds-delta is used as a per-turn reward.
4. **Regenerate & rank** — the lowest-reward US turns are regenerated multiple times and ranked, producing `(chosen, rejected)` preference pairs.
5. **DPO** — the US is fine-tuned on those pairs and becomes the next iteration.

This repository contains the public reproduction on **MultiWOZ 2.1** as a task-oriented benchmark, mirroring the procedure used in the paper for mental-health dialogue simulation.

## Citation

```bibtex
@article{zhu2025dial,
  title  = {DIAL: Direct Iterative Adversarial Learning for Realistic Multi-Turn Dialogue Simulation},
  author = {Zhu, Ziyi and Tieleman, Olivier and Stamatis, Caitlin A. and Smyth, Luka and Hull, Thomas D. and Cahn, Daniel R. and Malgaroli, Matteo},
  journal = {arXiv preprint arXiv:2512.20773},
  year   = {2025}
}
```

## Repository structure

```
.
├── ConvLab-3/                        # ConvLab-3 source tree (see Setup)
├── generate_sft_datasets.py          # Build US + RG SFT datasets from MultiWOZ 2.1
├── launch_sft_us.py                  # Launch SFT for the user simulator on Together AI
├── launch_sft_rg.py                  # Launch SFT for the response generator on Together AI
├── generate_discriminator_data.py    # Pair real and simulated dialogues for the discriminator
├── train_discriminator.py            # Train the token-classification discriminator (LoRA)
├── generate_preference_dataset.py    # Score, regenerate, and form (chosen, rejected) pairs
├── launch_dpo_us.py                  # Launch DPO fine-tuning of the US on Together AI
├── run_baselines.py                  # Evaluate baseline US/system combinations on the test set
├── run_custom_models.py              # Evaluate DIAL-trained US (+ RG) on the test set
├── compute_diversity.py              # Lexical diversity, distinct-n, entropy, MAUVE
└── remove_empty_cache.py             # Clean failed/empty cache entries between runs
```

## Setup

### 1. Clone with ConvLab-3

The pipeline depends on [ConvLab-3](https://github.com/ConvLab/ConvLab-3) for MultiWOZ 2.1 loading, the `LLM_US` / `LLM_RG` agent wrappers, baseline simulators (rule, TUS, GenTUS), and the MultiWOZ evaluator.

```bash
git clone <this-repo> dial
cd dial
git clone https://github.com/ConvLab/ConvLab-3.git
pip install -e ConvLab-3
```

### 2. Python dependencies

```bash
pip install \
    torch transformers datasets peft \
    bitsandbytes accelerate \
    scikit-learn numpy tqdm wandb \
    together litellm openai \
    mauve-text
```

A modern GPU (≥ 24 GB VRAM, A100/H100 recommended) is required to train the 8B discriminator with 4-bit quantization and LoRA.

### 3. API keys

The pipeline uses Together AI for SFT/DPO fine-tuning of 70B models, OpenRouter or Together for inference, and OpenAI for MAUVE embeddings.

```bash
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...        # if using openrouter/* models
export OPENAI_API_KEY=...            # for compute_diversity.py (MAUVE)
export WANDB_API_KEY=...             # optional, enables W&B logging
export HF_TOKEN=...                  # to push generated datasets to the Hub
```

## End-to-end pipeline

DIAL alternates between (a) generating data with the current US/RG and (b) training the next US. Repeat steps 3–6 for each DIAL iteration (`it1`, `it2`, …); model and dataset names in the scripts include the iteration index.

### Step 1 — SFT datasets

Build SFT datasets for both the user simulator and the response generator from the MultiWOZ 2.1 train split, and push to the Hugging Face Hub.

```bash
python generate_sft_datasets.py
# Pushes:  slingshot/multiwoz-2.1-user-sim-sft
#          slingshot/multiwoz-2.1-response-gen-sft
```

Override the target repos with `HF_REPO_US` / `HF_REPO_RG`.

### Step 2 — SFT base US and RG

Fine-tune Llama-3.1-70B-Instruct on each SFT dataset via Together AI:

```bash
python launch_sft_us.py            # user simulator
python launch_sft_rg.py            # response generator
```

The resulting checkpoint IDs are used as the starting points for DIAL.

### Step 3 — Discriminator data

Pair real MultiWOZ validation dialogues with simulated dialogues produced by the current `LLM_US` × `LLM_RG`:

```bash
# Edit LLM_US_MODEL / LLM_RG_MODEL at the top of the file to point to the
# current iteration's checkpoints, then:
python generate_discriminator_data.py
# Pushes: slingshot/multiwoz-2.1-user-disc-dial-it{N}
```

Simulated dialogues are cached under `cache/discriminator/<us>__<rg>/`.

### Step 4 — Train the discriminator

Token-classification head on Llama-3.1-8B-Instruct, 4-bit + LoRA. Labels are placed at the EOT token of every assistant (US) message: `1=real`, `0=simulated`.

```bash
# Update HF_DATASET / RUN_NAME at the top of the file.
python train_discriminator.py
```

A checkpoint is written to `output/<RUN_NAME>/checkpoint-*`.

### Step 5 — Preference dataset

For each simulated dialogue, score every US turn with the discriminator, find the two lowest-reward turns, regenerate 8 alternatives at each via the current US model, re-score, and form `(chosen, rejected)` pairs using log-odds-delta as the reward signal.

```bash
# Update DISCRIMINATOR_CHECKPOINT, HF_DATASET, HF_OUTPUT_REPO, US_MODEL.
python generate_preference_dataset.py
# Pushes: slingshot/multiwoz-2.1-user-pref-dial-it{N}
```

Per-dialogue results are cached under `cache/preference/`.

### Step 6 — DPO

Launch DPO on Together AI starting from the previous iteration's US checkpoint:

```bash
python launch_dpo_us.py \
    --filter-chosen-rejected \
    --top-margin-frac 1.0 \
    --dpo-beta 1.0
```

The newly produced model becomes the US for iteration N+1; loop back to Step 3.

## Evaluation

### Baselines

`run_baselines.py` evaluates standard US/system combinations on the MultiWOZ 2.1 test set:

| Combo               | User simulator | System            |
| :------------------ | :------------- | :---------------- |
| `rule_us_rule_sys`  | Rule           | Rule              |
| `tus_rule_sys`      | TUS            | Rule              |
| `gentus_rule_sys`   | GenTUS         | Rule              |
| `llm_us_llm_rg`     | LLM US         | LLM RG            |
| `llm_us_rule_sys`   | LLM US         | Rule pipeline NLU + Policy + NLG |

```bash
python run_baselines.py
```

Per-dialogue traces are cached under `cache/baselines/<combo>/`. Aggregated results land in `experiment_results/<combo>/{results.json,summary.txt}`.

### DIAL-trained models

Run the same evaluation harness with iteration-specific US (and optionally a custom RG):

```bash
# Edit LLM_US_MODEL / LLM_RG_MODEL / COMBO_NAME at the top of the file.
python run_custom_models.py
```

### Diversity and distributional metrics

Compute TTR, Distinct-1/2/3, word entropy, mean utterance / conversation length, and **MAUVE** (with OpenAI embeddings) between simulated user utterances and the human distribution from MultiWOZ:

```bash
python compute_diversity.py
```

MAUVE results are cached under `.mauve_cache/<combo>_buckets_<N>.json`.

### Cleaning failed runs

If a baseline / discriminator / preference run errored mid-way, drop the empty entries before re-running:

```bash
python remove_empty_cache.py --dry-run    # preview
python remove_empty_cache.py              # delete
```

## Caching

Every long-running step caches per-dialogue artifacts under `cache/`:

| Path                       | Producer                          | Contents                                  |
| :------------------------- | :-------------------------------- | :---------------------------------------- |
| `cache/baselines/<combo>/` | `run_baselines.py`, `run_custom_models.py` | Conversation + evaluator stats per dialog |
| `cache/discriminator/.../` | `generate_discriminator_data.py`  | Simulated US × RG conversations           |
| `cache/preference/.../`    | `generate_preference_dataset.py`  | Preference samples per dialogue           |
| `.mauve_cache/`            | `compute_diversity.py`            | Cached MAUVE results per combo            |

Re-running a script reuses cached results; delete the relevant subdirectory to force regeneration.

## Notes

- All scripts default to model and dataset IDs under the `slingshot/` Hugging Face namespace. Update them (or override via the constants at the top of each file) when reproducing in your own workspace.
- `LLM_US` requires the model to emit a literal `[END]` token to terminate the dialogue. The data builders enforce this for ground-truth dialogues so that the SFT distribution matches inference.
- The paper's primary application is mental-health dialogue simulation; the code in this repository targets MultiWOZ 2.1 as a public, reproducible benchmark of the same DIAL procedure.
