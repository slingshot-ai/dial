#!/usr/bin/env python
"""
Generate SFT training datasets for both User Simulator (US) and Response Generator (RG)
from MultiWOZ 2.1 train split, then upload each to a separate HuggingFace repo.

- US dataset: Model plays the user. System prompt = goal + US instructions.
  Messages: "user" = dialogue-system utterances, "assistant" = user utterances (training target).
- RG dataset: Model plays the assistant. System prompt = RG instructions only.
  Messages: "user" = user utterances, "assistant" = dialogue-system utterances (training target).

Train with causal LM loss masked to assistant tokens only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

from convlab.util.unified_datasets_util import load_dataset as load_convlab_dataset
from convlab.base_models.llm.user_simulator import LLM_US, LLM_RG
from datasets import Dataset

# HuggingFace repos (use env HF_REPO_US / HF_REPO_RG to override)
HF_REPO_US = os.environ.get("HF_REPO_US", "slingshot/multiwoz-2.1-user-sim-sft")
HF_REPO_RG = os.environ.get("HF_REPO_RG", "slingshot/multiwoz-2.1-response-gen-sft")

INITIAL_SYSTEM_GREETING = "Hello, how may I help you today?"


def _repair_description(description):
    """Fix descriptions corrupted by preprocess.py joining a string (not a list)
    with '. ', which inserts '. ' between every character."""
    if not description:
        return ""
    parts = description.split(". ")
    if len(parts) > 10 and sum(len(p) <= 2 for p in parts) / len(parts) > 0.8:
        return "".join(parts)
    return description


# ---------------------------------------------------------------------------
# US dataset: goal + US system prompt; "user" = system, "assistant" = user
# ---------------------------------------------------------------------------
def build_us_system_prompt(goal):
    """Replicate LLM_US.init_session system prompt."""
    description = _repair_description(goal.get("description", "") or "")
    goal_description = ".\n".join(["* " + item for item in description.split(". ")])
    return f"Goal:\n{goal_description}\n\n{LLM_US.DEFAULT_SYSTEM_INSTRUCTION}"


def process_dialogue_us(dialogue):
    """One row per dialogue. 'user' = system utterances, 'assistant' = user utterances."""
    goal = dialogue["goal"]
    turns = dialogue["turns"]
    dialogue_id = dialogue["dialogue_id"]
    domains = dialogue.get("domains", [])

    system_prompt = build_us_system_prompt(goal)
    messages = [{"role": "system", "content": system_prompt}]
    first_user = True

    for turn in turns:
        if turn["speaker"] == "user":
            if first_user:
                messages.append({"role": "user", "content": INITIAL_SYSTEM_GREETING})
                first_user = False
            messages.append({"role": "assistant", "content": turn["utterance"]})
        elif turn["speaker"] == "system":
            messages.append({"role": "user", "content": turn["utterance"]})

    if messages[-1]["role"] == "user" and len(messages) > 1:
        messages.pop()

    # Append [END] to last user (assistant) message per system prompt: "If and only if you achieve your goal, express your thanks and generate **\"[END]\"** token."
    if messages[-1]["role"] == "assistant":
        last_content = messages[-1]["content"]
        if not last_content.strip().endswith("[END]"):
            messages[-1]["content"] = last_content.rstrip() + " [END]"

    n_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    return {
        "dialogue_id": dialogue_id,
        "domains": domains,
        "n_turns": n_assistant_turns,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# RG dataset: RG system prompt only; "user" = user, "assistant" = system
# ---------------------------------------------------------------------------
def build_rg_system_prompt():
    """System prompt for the response generator (no user goal)."""
    return LLM_RG.DEFAULT_SYSTEM_INSTRUCTION


def process_dialogue_rg(dialogue):
    """One row per dialogue. 'user' = user utterances, 'assistant' = system utterances.
    Replicate exactly what RG sees during inference: no placeholder first turn.
    At inference the first message to RG is the first user utterance (greeting is hard-coded)."""
    turns = dialogue["turns"]
    dialogue_id = dialogue["dialogue_id"]
    domains = dialogue.get("domains", [])

    system_prompt = build_rg_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    if not turns:
        return None

    # During inference RG never sees a "[Start of conversation]" turn; first user message is the real first user utterance.
    # If dialogue starts with system (greeting), skip it so first turn is user -> assistant (system reply).
    idx = 0
    if turns[0]["speaker"] == "system":
        idx = 1

    while idx < len(turns):
        if turns[idx]["speaker"] == "user":
            messages.append({"role": "user", "content": turns[idx]["utterance"]})
        else:
            messages.append({"role": "assistant", "content": turns[idx]["utterance"]})
        idx += 1

    if messages[-1]["role"] == "user" and len(messages) > 1:
        messages.pop()

    n_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    return {
        "dialogue_id": dialogue_id,
        "domains": domains,
        "n_turns": n_assistant_turns,
        "messages": messages,
    }


def sanity_check_us(rows):
    """US: last message must be assistant (user-simulator utterance)."""
    for row in rows[:500]:
        msgs = row["messages"]
        assert msgs[0]["role"] == "system", f"{row['dialogue_id']}: first msg not system"
        assert msgs[-1]["role"] == "assistant", f"{row['dialogue_id']}: last msg not assistant"
        for i, m in enumerate(msgs[1:], 1):
            expected = "user" if i % 2 == 1 else "assistant"
            assert m["role"] == expected, (
                f"{row['dialogue_id']}: index {i} expected {expected}, got {m['role']}"
            )
    print(f"  US sanity check passed on {min(500, len(rows))} dialogues.")


def sanity_check_rg(rows):
    """RG: last message must be assistant (system utterance)."""
    for row in rows[:500]:
        msgs = row["messages"]
        assert msgs[0]["role"] == "system", f"{row['dialogue_id']}: first msg not system"
        assert msgs[-1]["role"] == "assistant", f"{row['dialogue_id']}: last msg not assistant"
        for i, m in enumerate(msgs[1:], 1):
            expected = "user" if i % 2 == 1 else "assistant"
            assert m["role"] == expected, (
                f"{row['dialogue_id']}: index {i} expected {expected}, got {m['role']}"
            )
    print(f"  RG sanity check passed on {min(500, len(rows))} dialogues.")


def build_and_push_us(train_dialogues):
    """Build US SFT dataset and push to HF."""
    print("\n--- Building US (user simulator) SFT dataset ---")
    all_rows = []
    skipped = 0
    for dlg in train_dialogues:
        row = process_dialogue_us(dlg)
        if row["n_turns"] == 0:
            skipped += 1
            continue
        all_rows.append(row)

    total_turns = sum(r["n_turns"] for r in all_rows)
    print(f"  {len(all_rows)} dialogues, {total_turns} assistant turns ({skipped} skipped)")
    sanity_check_us(all_rows)

    hf_dataset = Dataset.from_list(all_rows)
    print(f"  Dataset: {hf_dataset}")
    print(f"\n  Example US row (first dialogue):")
    ex = hf_dataset[0]
    for m in ex["messages"][:5]:
        content_preview = m["content"][:70] + ("..." if len(m["content"]) > 70 else "")
        print(f"    [{m['role']:>9}] {content_preview}")
    if len(ex["messages"]) > 5:
        print(f"    ... {len(ex['messages']) - 5} more messages")

    print(f"\n  Pushing to {HF_REPO_US} ...")
    hf_dataset.push_to_hub(HF_REPO_US)
    print(f"  Done: {HF_REPO_US}")


def build_and_push_rg(train_dialogues):
    """Build RG SFT dataset and push to HF."""
    print("\n--- Building RG (response generator) SFT dataset ---")
    all_rows = []
    skipped = 0
    for dlg in train_dialogues:
        row = process_dialogue_rg(dlg)
        if row is None or row["n_turns"] == 0:
            skipped += 1
            continue
        all_rows.append(row)

    total_turns = sum(r["n_turns"] for r in all_rows)
    print(f"  {len(all_rows)} dialogues, {total_turns} assistant turns ({skipped} skipped)")
    sanity_check_rg(all_rows)

    hf_dataset = Dataset.from_list(all_rows)
    print(f"  Dataset: {hf_dataset}")
    print(f"\n  Example RG row (first dialogue):")
    ex = hf_dataset[0]
    for m in ex["messages"][:5]:
        content_preview = m["content"][:70] + ("..." if len(m["content"]) > 70 else "")
        print(f"    [{m['role']:>9}] {content_preview}")
    if len(ex["messages"]) > 5:
        print(f"    ... {len(ex['messages']) - 5} more messages")

    print(f"\n  Pushing to {HF_REPO_RG} ...")
    hf_dataset.push_to_hub(HF_REPO_RG)
    print(f"  Done: {HF_REPO_RG}")


def main():
    print("Loading MultiWOZ 2.1 train ...")
    dataset = load_convlab_dataset("multiwoz21")
    train_dialogues = dataset["train"]
    print(f"  {len(train_dialogues)} train dialogues")

    build_and_push_us(train_dialogues)
    build_and_push_rg(train_dialogues)

    print("\nAll done. Repos:")
    print(f"  US: {HF_REPO_US}")
    print(f"  RG: {HF_REPO_RG}")


if __name__ == "__main__":
    main()
