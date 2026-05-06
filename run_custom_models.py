#!/usr/bin/env python
"""
Run llm_us_llm_rg with separate US and RG models (same output format as run_baselines).

- US (user simulator): OpenRouter Llama 3.1 8B Instruct (same as baselines).
- RG (response generator): Together AI MultiWOZ RG SFT model (OpenAI-compatible).

Requires:
  - OPENROUTER_API_KEY for the US model
  - TOGETHERAI_API_KEY for the RG model (Together AI is OpenAI-compatible; litellm uses together_ai/ prefix)
"""

import os
import sys

# Reuse path and logging setup from run_baselines before ConvLab/litellm imports
sys.path.insert(0, sys.path[0] or ".")
import run_baselines  # noqa: E402 - sets path, litellm logging, ConvLab-3 on path

# Model identifiers (litellm format)
# US: same Llama 8B Instruct as in run_baselines, via OpenRouter
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it1-a8c57d7f-6bb2624d"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it2-7d06c9f1-6710a3db"
# LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it3-04142b70-90b36413"
LLM_US_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-us-dial-it4-625c294c"
# LLM_US_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
# RG: Together AI OpenAI-compatible endpoint; litellm uses together_ai/{model_id}
LLM_RG_MODEL = "together_ai/slingshot/Meta-Llama-3.1-70B-Instruct-Reference-multiwoz-rg-sft-5c55bb5c"

COMBO_NAME = "dial_it4_us_sft_rg"


def run_llm_us_llm_rg_custom():
    from convlab.base_models.llm.user_simulator import LLM_RG, LLM_US

    cdir = run_baselines._cache_path(COMBO_NAME, None)
    goals = run_baselines._load_test_goals()

    user_model = LLM_US("litellm", LLM_US_MODEL)
    system_model = LLM_RG("litellm", LLM_RG_MODEL)

    results = []
    for i, goal in enumerate(goals):
        cached = run_baselines._load_cached(cdir, i)
        if cached:
            results.append(cached)
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Dialog {i + 1}/{len(goals)} ...")
        try:
            user_model.init_session(goal)
            system_model.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"turn": 0, "system": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                if user_model.is_terminated:
                    conversation.append({"turn": turns, "user": user_msg})
                    break
                sys_msg = system_model.response(user_msg)
                turns += 1
                conversation.append(
                    {"turn": turns, "user": user_msg, "system": sys_msg}
                )

            completed = user_model.is_terminated and any(
                "[END]" in str(c.get("user", "")) for c in conversation
            )
            reward = user_model.get_reward()
        except Exception as e:
            print(f"    Error: {e}")
            completed, reward, conversation, turns = False, None, [], 0

        result = {
            "dialogue_id": i,
            "goal_description": goal.get("description", ""),
            "turns": turns,
            "completed": completed,
            "success": completed,
            "reward": reward,
            "conversation": conversation,
        }
        run_baselines._save_to_cache(cdir, i, result)
        results.append(result)

    return run_baselines._save(
        os.path.join(run_baselines.OUT_ROOT, COMBO_NAME),
        COMBO_NAME,
        results,
        extra_summary={
            "us_model": LLM_US_MODEL,
            "rg_model": LLM_RG_MODEL,
        },
    )


def main():
    run_baselines.set_seed(run_baselines.SEED)
    os.makedirs(run_baselines.OUT_ROOT, exist_ok=True)
    print(f"US model: {LLM_US_MODEL}")
    print(f"RG model: {LLM_RG_MODEL}")
    print()
    summary = run_llm_us_llm_rg_custom()
    print()
    print("Done. Summary:", summary)


if __name__ == "__main__":
    main()
