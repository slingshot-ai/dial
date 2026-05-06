#!/usr/bin/env python
"""
Generate discriminator training data from MultiWOZ 2.1 validation split.

Each row contains:
  - real_messages: ground truth dialogue in chat format (same as SFT data)
  - simulated_messages: LLM_US + LLM_RG simulation with the same goal, same format

Both use the same system prompt (goal + LLM_US instructions), the same initial
greeting, and the same [system, user, assistant, ...] role convention.

Simulated conversations are cached per-dialogue under cache/discriminator/
keyed by US and RG model IDs.

Uploads to HuggingFace.
"""

import json
import logging
import os
import sys
import traceback
from copy import deepcopy

os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import litellm

litellm.suppress_debug_info = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

from convlab.base_models.llm.user_simulator import LLM_RG, LLM_US
from convlab.util.unified_datasets_util import load_dataset as load_convlab_dataset
from datasets import Dataset

# LLM_US_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-sft-4dcc3672"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it1-a8c57d7f-6bb2624d"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it2-7d06c9f1-6710a3db"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it3-04142b70-90b36413"
LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it4-625c294c"
# LLM_RG_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
LLM_RG_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-rg-sft-5c55bb5c"
N_DIALOGUES = 1000
INITIAL_SYSTEM_GREETING = "Hello, how may I help you today?"
HF_REPO = "slingshot/multiwoz-2.1-user-disc-dial-it4"
CACHE_DIR = "cache"
MAX_TURNS = 20


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _sanitize(name):
    return name.replace("/", "_").replace(" ", "_")


def _sim_cache_dir():
    key = f"us__{_sanitize(LLM_US_MODEL)}__rg__{_sanitize(LLM_RG_MODEL)}"
    return os.path.join(CACHE_DIR, "discriminator", key)


def _load_cached(cache_dir, idx):
    path = os.path.join(cache_dir, f"dialog_{idx}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_to_cache(cache_dir, idx, data):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"dialog_{idx}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Description repair
# ---------------------------------------------------------------------------
def _repair_description(description):
    parts = description.split(". ")
    if len(parts) > 10 and sum(len(p) <= 2 for p in parts) / len(parts) > 0.8:
        return "".join(parts)
    return description


def _repair_goal(goal):
    desc = goal.get("description", "") or ""
    repaired = _repair_description(desc)
    if repaired != desc:
        goal = deepcopy(goal)
        goal["description"] = repaired
    return goal


def _clean_trailing_messages(messages):
    """Ensure last message is from user (role=assistant), clean [END]/[STOP]."""
    if len(messages) > 1 and messages[-1]["role"] == "user":
        messages.pop()

    if len(messages) > 1 and messages[-1]["role"] == "assistant":
        content = messages[-1]["content"]
        content = content.replace("[END]", "").replace("[STOP]", "").strip()
        if not content:
            messages.pop()
            if len(messages) > 1 and messages[-1]["role"] == "user":
                messages.pop()
        else:
            messages[-1]["content"] = content

    return messages


def _append_end_to_last_user_simulator_message(messages):
    """Append [END] to the last user simulator (assistant) message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            messages[i]["content"] = (
                messages[i]["content"].rstrip() + " [END]"
            ).strip()
            break
    return messages


def build_system_prompt(goal):
    description = _repair_description(goal.get("description", "") or "")
    goal_description = ".\n".join(["* " + item for item in description.split(". ")])
    return f"Goal:\n{goal_description}\n\n{LLM_US.DEFAULT_SYSTEM_INSTRUCTION}"


# ---------------------------------------------------------------------------
# Real / simulated message builders
# ---------------------------------------------------------------------------
def build_real_messages(dialogue, system_prompt):
    """Build message list from ground truth dialogue turns."""
    messages = [{"role": "system", "content": system_prompt}]
    first_user = True

    for turn in dialogue["turns"]:
        if turn["speaker"] == "user":
            if first_user:
                messages.append({"role": "user", "content": INITIAL_SYSTEM_GREETING})
                first_user = False
            messages.append({"role": "assistant", "content": turn["utterance"]})
        elif turn["speaker"] == "system":
            messages.append({"role": "user", "content": turn["utterance"]})

    return _clean_trailing_messages(messages)


def run_simulation(goal, system_prompt):
    """Run LLM_US + LLM_RG simulation, return messages in the same format."""
    user_model = LLM_US("litellm", LLM_US_MODEL)
    system_model = LLM_RG("litellm", LLM_RG_MODEL)

    user_model.init_session(goal)
    system_model.init_session()

    messages = [{"role": "system", "content": system_prompt}]
    sys_msg = INITIAL_SYSTEM_GREETING
    messages.append({"role": "user", "content": sys_msg})

    for _ in range(MAX_TURNS):
        user_msg = user_model.response(sys_msg)
        messages.append({"role": "assistant", "content": user_msg})

        if user_model.is_terminated:
            break

        sys_msg = system_model.response(user_msg)
        messages.append({"role": "user", "content": sys_msg})

    return _clean_trailing_messages(messages)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading MultiWOZ 2.1 validation split ...")
    dataset = load_convlab_dataset("multiwoz21")
    val_dialogues = dataset["validation"][:N_DIALOGUES]
    print(f"  {len(val_dialogues)} validation dialogues (N_DIALOGUES={N_DIALOGUES})")
    print(f"  US model: {LLM_US_MODEL}")
    print(f"  RG model: {LLM_RG_MODEL}")

    cdir = _sim_cache_dir()
    print(f"  Cache dir: {cdir}")

    all_rows = []
    errors = 0
    cached_count = 0

    for i, dlg in enumerate(val_dialogues):
        goal = _repair_goal(dlg["goal"])
        dialogue_id = dlg["dialogue_id"]
        domains = dlg.get("domains", [])
        system_prompt = build_system_prompt(goal)

        real_msgs = build_real_messages(dlg, system_prompt)
        real_n = sum(1 for m in real_msgs if m["role"] == "assistant")
        if real_n == 0:
            continue

        # Check simulation cache
        cached_sim = _load_cached(cdir, i)
        if cached_sim is not None:
            sim_msgs = cached_sim
            cached_count += 1
        else:
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  Simulating dialog {i + 1}/{len(val_dialogues)} ({dialogue_id}) ..."
                )
            try:
                sim_msgs = run_simulation(goal, system_prompt)
                _save_to_cache(cdir, i, sim_msgs)
            except Exception as e:
                print(f"    Simulation error for {dialogue_id}: {e}")
                traceback.print_exc()
                sim_msgs = [{"role": "system", "content": system_prompt}]
                errors += 1

        sim_n = sum(1 for m in sim_msgs if m["role"] == "assistant")

        _append_end_to_last_user_simulator_message(real_msgs)
        _append_end_to_last_user_simulator_message(sim_msgs)

        all_rows.append(
            {
                "dialogue_id": dialogue_id,
                "domains": domains,
                "real_n_turns": real_n,
                "simulated_n_turns": sim_n,
                "real_messages": real_msgs,
                "simulated_messages": sim_msgs,
            }
        )

    print(f"\n  {len(all_rows)} rows ({cached_count} from cache, {errors} errors)")

    print("Building HuggingFace dataset ...")
    hf_dataset = Dataset.from_list(all_rows)
    print(hf_dataset)

    print(f"\nExample row:")
    ex = hf_dataset[0]
    print(f"  dialogue_id: {ex['dialogue_id']}")
    print(f"  domains:     {ex['domains']}")
    print(f"  real_n_turns:      {ex['real_n_turns']}")
    print(f"  simulated_n_turns: {ex['simulated_n_turns']}")
    for label, key in [("real", "real_messages"), ("simulated", "simulated_messages")]:
        msgs = ex[key]
        print(f"  {label}_messages ({len(msgs)} total):")
        for m in msgs[:4]:
            preview = m["content"][:70] + ("..." if len(m["content"]) > 70 else "")
            print(f"    [{m['role']:>9}] {preview}")
        if len(msgs) > 4:
            print(f"    ... {len(msgs) - 4} more")

    print(f"\nPushing to {HF_REPO} ...")
    hf_dataset.push_to_hub(HF_REPO)
    print("Done.")


if __name__ == "__main__":
    main()
