#!/usr/bin/env python
"""
Generate a preference dataset using discriminator log-odds-delta rewards.

Pipeline:
  1. Load a trained discriminator (PEFT token-classification checkpoint).
  2. Load paired real/simulated dialogues from HuggingFace.
  3. For each simulated conversation:
     a. Feed system + assistant messages to the discriminator.
     b. Compute log-odds-delta reward at every assistant (user-sim) EOT.
     c. Identify the top 2 lowest-reward positions.
     d. At those positions, regenerate 8 alternatives with the US model via
        litellm (same prompt format and temperature as LLM_US).
     e. Re-score all 9 candidates (original + 8 new).
     f. Chosen = highest reward, rejected = lowest reward.
  4. Upload the preference dataset to HuggingFace.

Roles in the dataset (same convention as generate_discriminator_data.py):
  system   = goal + LLM_US instructions
  user     = RG (dialogue-system / response-generator) utterances
  assistant = US (user-simulator) utterances
"""

import json
import os
import traceback
from typing import Any, Dict, List

import litellm
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

# ── Constants ──────────────────────────────────────────────────────────────

DISCRIMINATOR_CHECKPOINT = (
    "/mnt/workspace/adversarial-learning/output/user-disc-dial-it3/checkpoint-107"
)
DISCRIMINATOR_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

HF_DATASET = "slingshot/multiwoz-2.1-user-disc-dial-it3"
HF_OUTPUT_REPO = "slingshot/multiwoz-2.1-user-pref-dial-it3"

US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it3-04142b70-52244b15"
US_TEMPERATURE = 0.8
US_MAX_TOKENS = 256
NUM_REGENERATIONS = 8  # +1 original = 9 total candidates
N_DIALOGUES = 1000
CACHE_DIR = "cache"

CHAT_TEMPLATE = """
{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
""".strip()


# ── Cache helpers ────────────────────────────────────────────────────────


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _preference_cache_dir() -> str:
    return os.path.join(CACHE_DIR, "preference", _sanitize(HF_DATASET))


def _load_cached(cache_dir: str, idx: int) -> List[Dict[str, Any]] | None:
    path = os.path.join(cache_dir, f"dialog_{idx}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_to_cache(cache_dir: str, idx: int, data: List[Dict[str, Any]]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"dialog_{idx}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Tokenizer / model loading ─────────────────────────────────────────────


def prepare_tokenizer(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def load_discriminator(
    checkpoint_path: str, base_model_name: str, device: str = "cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer = prepare_tokenizer(tokenizer)

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        num_labels=2,
        id2label={0: "SIMULATED", 1: "REAL"},
        label2id={"SIMULATED": 0, "REAL": 1},
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ── Discriminator inference helpers ────────────────────────────────────────


def _find_assistant_eot_indices(
    messages: List[Dict[str, str]],
    token_ids: List[int],
    tokenizer: PreTrainedTokenizerFast,
) -> List[int]:
    eos_id = tokenizer.eos_token_id
    eos_positions = [i for i, t in enumerate(token_ids) if t == eos_id]

    assistant_eots: List[int] = []
    cursor = 0
    for msg in messages:
        single = tokenizer.apply_chat_template(
            [msg], return_dict=True, padding=False, truncation=False
        )
        n_eots = sum(1 for t in single["input_ids"] if t == eos_id)
        if msg["role"] == "assistant":
            for _ in range(n_eots):
                if cursor < len(eos_positions):
                    assistant_eots.append(eos_positions[cursor])
                cursor += 1
        else:
            cursor += n_eots
    return assistant_eots


def _get_logits_at_assistant_eots(
    model, tokenizer, disc_messages: List[Dict[str, str]], device: str
) -> List[np.ndarray]:
    tokenized = tokenizer.apply_chat_template(
        disc_messages, return_dict=True, padding=False, truncation=False
    )
    input_ids = torch.tensor([tokenized["input_ids"]], device=device)
    attn_mask = torch.tensor([tokenized["attention_mask"]], device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[0]

    eot_indices = _find_assistant_eot_indices(
        disc_messages, tokenized["input_ids"], tokenizer
    )
    return [logits[idx].float().cpu().numpy() for idx in eot_indices]


def _prob_real(logits: np.ndarray) -> float:
    exp = np.exp(logits - logits.max())
    return float((exp / exp.sum())[1])


def _log_odds_delta(prob: float, prev_prob: float) -> float:
    eps = 1e-10
    p = np.clip(prob, eps, 1 - eps)
    pp = np.clip(prev_prob, eps, 1 - eps)
    return float(np.log(p / (1 - p)) - np.log(pp / (1 - pp)))


# ── High-level scoring ─────────────────────────────────────────────────────


def score_conversation(
    model, tokenizer, full_messages: List[Dict[str, str]], device: str
) -> List[Dict[str, Any]]:
    """Return per-assistant-message rewards for a full conversation."""
    disc_msgs = [m for m in full_messages if m["role"] in ("system", "assistant")]
    logits_list = _get_logits_at_assistant_eots(model, tokenizer, disc_msgs, device)

    rewards = []
    prev = 0.5
    for lg in logits_list:
        p = _prob_real(lg)
        rewards.append(
            {
                "prob_real": p,
                "prev_prob": prev,
                "log_odds_delta": _log_odds_delta(p, prev),
            }
        )
        prev = p
    return rewards


def score_single_response(
    model,
    tokenizer,
    disc_context: List[Dict[str, str]],
    response_text: str,
    prev_prob: float,
    device: str,
) -> tuple[float, float]:
    """Score one candidate response appended to disc_context. Returns (prob, delta)."""
    msgs = disc_context + [{"role": "assistant", "content": response_text}]
    logits_list = _get_logits_at_assistant_eots(model, tokenizer, msgs, device)
    if not logits_list:
        return 0.5, 0.0
    p = _prob_real(logits_list[-1])
    return p, _log_odds_delta(p, prev_prob)


# ── Regeneration ───────────────────────────────────────────────────────────


def regenerate_at_position(
    full_messages: List[Dict[str, str]],
    asst_idx: int,
    n: int = NUM_REGENERATIONS,
) -> List[str]:
    """Call the US model to regenerate the assistant message at *asst_idx*.

    Context = full_messages[:asst_idx] (ends with a "user" / RG message).
    """
    context = full_messages[:asst_idx]
    responses: List[str] = []
    for _ in range(n):
        try:
            r = litellm.completion(
                model=US_MODEL,
                messages=context,
                temperature=US_TEMPERATURE,
                max_tokens=US_MAX_TOKENS,
            )
            responses.append(r.choices[0].message.content)
        except Exception as e:
            print(f"    litellm error: {e}")
    return responses


# ── Per-conversation pipeline ──────────────────────────────────────────────


def process_conversation(
    model, tokenizer, full_messages: List[Dict[str, str]], device: str
) -> List[Dict[str, Any]]:
    rewards = score_conversation(model, tokenizer, full_messages, device)
    if len(rewards) < 2:
        return []

    assistant_indices = [
        i for i, m in enumerate(full_messages) if m["role"] == "assistant"
    ]
    deltas = np.array([r["log_odds_delta"] for r in rewards])
    # Top 2 lowest-reward positions (ascending order, take first 2)
    lowest_two_positions = np.argsort(deltas)[:2].tolist()

    samples: List[Dict[str, Any]] = []
    for pos in sorted(set(lowest_two_positions)):
        asst_idx = assistant_indices[pos]
        original_text = full_messages[asst_idx]["content"]
        prev_prob = rewards[pos]["prev_prob"]

        new_responses = regenerate_at_position(full_messages, asst_idx)

        # Disc context for scoring: system + assistant messages before this position
        disc_context = [
            m for m in full_messages[:asst_idx] if m["role"] in ("system", "assistant")
        ]

        # Score all candidates (original first)
        candidates: List[Dict[str, Any]] = []
        for text, is_orig in [(original_text, True)] + [
            (r, False) for r in new_responses
        ]:
            prob, delta = score_single_response(
                model, tokenizer, disc_context, text, prev_prob, device
            )
            candidates.append(
                {
                    "content": text,
                    "log_odds_delta": delta,
                    "prob_real": prob,
                    "is_original": is_orig,
                }
            )

        if len(candidates) < 2:
            continue

        best = max(candidates, key=lambda c: c["log_odds_delta"])
        worst = min(candidates, key=lambda c: c["log_odds_delta"])

        samples.append(
            {
                "messages": full_messages[:asst_idx],
                "chosen": {"role": "assistant", "content": best["content"]},
                "rejected": {"role": "assistant", "content": worst["content"]},
                "chosen_reward": best["log_odds_delta"],
                "rejected_reward": worst["log_odds_delta"],
                "all_responses": [
                    {
                        "content": c["content"],
                        "log_odds_delta": c["log_odds_delta"],
                        "is_original": c["is_original"],
                    }
                    for c in candidates
                ],
                "assistant_turn_index": pos,
            }
        )

    return samples


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading discriminator from {DISCRIMINATOR_CHECKPOINT} …")
    model, tokenizer = load_discriminator(
        DISCRIMINATOR_CHECKPOINT, DISCRIMINATOR_BASE_MODEL, device
    )

    print(f"Loading dataset from {HF_DATASET} …")
    raw = load_dataset(HF_DATASET, split="train")
    print(f"  {len(raw)} rows")

    cdir = _preference_cache_dir()
    print(f"  Cache dir: {cdir}")

    all_samples: List[Dict[str, Any]] = []
    errors = 0
    cached_count = 0

    n = min(N_DIALOGUES, len(raw))
    for i in tqdm(range(n), desc="Processing"):
        row = raw[i]
        sim_msgs = row["simulated_messages"]
        dialogue_id = row.get("dialogue_id", f"dialog_{i}")

        if not any(m["role"] == "assistant" for m in sim_msgs):
            continue

        cached = _load_cached(cdir, i)
        if cached is not None:
            for s in cached:
                s["dialogue_id"] = dialogue_id
            all_samples.extend(cached)
            cached_count += 1
            continue

        try:
            pref = process_conversation(model, tokenizer, sim_msgs, device)
            for s in pref:
                s["dialogue_id"] = dialogue_id
            _save_to_cache(cdir, i, pref)
            all_samples.extend(pref)
        except Exception as e:
            print(f"  Error on {dialogue_id}: {e}")
            traceback.print_exc()
            errors += 1

    print(
        f"\n{len(all_samples)} preference samples from {n} conversations "
        f"({cached_count} from cache, {errors} errors)"
    )

    if all_samples:
        ds = Dataset.from_list(all_samples)
        print(ds)
        print(f"\nPushing to {HF_OUTPUT_REPO} …")
        ds.push_to_hub(HF_OUTPUT_REPO)
        print("Done!")
    else:
        print("No samples generated.")


if __name__ == "__main__":
    main()
