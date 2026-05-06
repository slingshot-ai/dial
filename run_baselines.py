#!/usr/bin/env python
"""
Evaluate user simulator + system agent combinations on MultiWOZ 2.1 test set.

Combinations:
  1. rule_us_rule_sys   -- Rule-based user simulator + Rule-based system (semantic)
  2. tus_rule_sys       -- TUS neural user sim + Rule-based system
  3. gentus_rule_sys    -- GenTUS generative neural user sim + Rule-based system
  4. llm_us_llm_rg      -- LLM user simulator + LLM response generator
  5. llm_us_rule_sys     -- LLM user simulator + Rule-based pipeline system

Set N_DIALOGUES to subsample the test set (1000 = full test set).
"""

import json
import logging
import os
import random
import sys
import traceback
from copy import deepcopy

import numpy as np

# Suppress litellm / httpx noise before any imports that pull them in
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import litellm

litellm.suppress_debug_info = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

SEED = 42
N_DIALOGUES = 1000
LLM_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
OUT_ROOT = "experiment_results"
CACHE_DIR = "cache"


def _sanitize(name):
    return name.replace("/", "_").replace(" ", "_")


def _cache_path(combo, model=None):
    parts = [CACHE_DIR, "baselines", combo]
    if model:
        parts.append(_sanitize(model))
    return os.path.join(*parts)


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fix corrupted goal descriptions (preprocess.py bug: '. '.join on a string)
# ---------------------------------------------------------------------------
def _repair_description(description):
    """Fix descriptions corrupted by '. '.join() applied to a string instead of
    a list, which inserts '. ' between every character."""
    parts = description.split(". ")
    if len(parts) > 10 and sum(len(p) <= 2 for p in parts) / len(parts) > 0.8:
        return "".join(parts)
    return description


def _repair_goal(goal):
    """Return a copy of goal with a repaired description field."""
    desc = goal.get("description", "") or ""
    repaired = _repair_description(desc)
    if repaired != desc:
        goal = deepcopy(goal)
        goal["description"] = repaired
    return goal


# ---------------------------------------------------------------------------
# Compatibility DST: bridges ConvLab-3 unified format to the old semi/book
# state format expected by RuleBasedMultiwozBot.
# ---------------------------------------------------------------------------
from convlab.dst.dst import DST
from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

# ---------------------------------------------------------------------------
# Normalize old-format capitalized system DA → ConvLab-3 lowercase format.
#
# RuleBasedMultiwozBot produces DAs like ['Inform','Attraction','Addr','…']
# but GenTUS / TUS expect              ['inform','attraction','address','…'].
# ---------------------------------------------------------------------------

# (domain_lower, display_slot) → canonical old slot  (from REF_SYS_DA)
_SYS_DISPLAY_TO_CANON = {}
for _d, _sm in REF_SYS_DA.items():
    for _disp, _canon in _sm.items():
        if _canon is not None:
            _SYS_DISPLAY_TO_CANON[(_d.lower(), _disp)] = _canon

# canonical old slot → ConvLab-3 unified slot
_CANON_TO_UNIFIED = {
    "pricerange": "price range",
    "arriveBy": "arrive by",
    "leaveAt": "leave at",
    "trainID": "train id",
    "taxi_types": "type",
    "taxi_phone": "phone",
    "Ref": "reference",
}


def _normalize_sys_da(da):
    """Lowercase domain/intent & map display-slot names to unified names."""
    if not da:
        return da
    out = []
    for intent, domain, slot, value in da:
        d = domain.lower()
        i = intent.lower()
        s = _SYS_DISPLAY_TO_CANON.get((d, slot), slot)
        s = _CANON_TO_UNIFIED.get(s, s)
        out.append([i, d, s, value])
    return out


SLOT_NEW_TO_OLD = {
    "price range": "pricerange",
    "leave at": "leaveAt",
    "arrive by": "arriveBy",
    "book day": "day",
    "book people": "people",
    "book stay": "stay",
    "book time": "time",
    "train id": "trainID",
    "entrance fee": "entrance fee",
}

_SLOT_DISPLAY = {}
for _dom, _mapping in REF_USR_DA.items():
    for _old, _disp in _mapping.items():
        _SLOT_DISPLAY[(_dom.lower(), _old)] = _disp


def _to_display_slot(domain, new_slot):
    old = SLOT_NEW_TO_OLD.get(new_slot, new_slot)
    return _SLOT_DISPLAY.get((domain.lower(), old), new_slot)


class CompatRuleDST(DST):
    """Thin DST that provides old-format state and remaps user_action slots."""

    def __init__(self):
        DST.__init__(self)
        self._init_state()

    def _init_state(self):
        from convlab.util.multiwoz.state import default_state_old

        self.state = default_state_old()
        self.state.setdefault("booked", {})

    def init_session(self):
        self._init_state()

    def _place_slot(self, dom, old_slot, value):
        bs = self.state["belief_state"].get(dom)
        if not bs or old_slot in ("none", ""):
            return
        book = bs.get("book", {})
        semi = bs.get("semi", {})
        if old_slot in book and old_slot != "booked":
            book[old_slot] = value
        elif old_slot in semi:
            semi[old_slot] = value

    def update(self, user_act=None):
        if not user_act:
            return self.state

        remapped = []
        for intent, domain, slot, value in user_act:
            dom = domain.lower()
            old_slot = SLOT_NEW_TO_OLD.get(slot, slot)
            display_slot = _to_display_slot(dom, slot)

            if intent == "inform" and dom in self.state["belief_state"]:
                self._place_slot(dom, old_slot, value)
            elif intent == "request":
                if dom not in self.state["request_state"]:
                    self.state["request_state"][dom] = {}
                if old_slot not in self.state["request_state"][dom]:
                    self.state["request_state"][dom][old_slot] = 0

            remapped.append([intent, domain, display_slot, value])

        self.state["user_action"] = remapped
        return self.state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_test_goals():
    from convlab.util.unified_datasets_util import load_dataset

    dataset = load_dataset("multiwoz21")
    goals = [_repair_goal(d["goal"]) for d in dataset["test"][:N_DIALOGUES]]
    return goals


def _save(out_dir, combo_name, results, extra_summary=None):
    os.makedirs(out_dir, exist_ok=True)

    completed_count = sum(1 for r in results if r["completed"])
    success_count = sum(1 for r in results if r.get("success", False))
    total = len(results)

    summary = {
        "combo": combo_name,
        "n_dialogues": total,
        "completion_rate": completed_count / max(total, 1),
        "success_rate": success_count / max(total, 1),
        "avg_turns": float(np.mean([r["turns"] for r in results])),
    }
    if extra_summary:
        summary.update(extra_summary)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(
            {"summary": summary, "per_dialogue": results}, f, indent=2, default=str
        )

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"\n  --- {combo_name} summary ---")
    for k, v in summary.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return summary


# =========================================================================
# 1. Rule US + Rule Sys (semantic level, with NL post-processing)
# =========================================================================
def run_rule_us_rule_sys():
    from convlab.dialog_agent import BiSession, PipelineAgent
    from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.policy.rule.multiwoz import RulePolicy

    cdir = _cache_path("rule_us_rule_sys")

    sys_agent = PipelineAgent(None, CompatRuleDST(), RulePolicy(), None, name="sys")
    user_agent = PipelineAgent(
        None, None, RulePolicy(character="usr"), None, name="user"
    )
    evaluator = MultiWozEvaluator()

    user_nlg = TemplateNLG(is_user=True)
    sys_nlg = TemplateNLG(is_user=False)

    def _da_to_nl(da, nlg):
        try:
            return nlg.generate(da)
        except Exception:
            return str(da)

    set_seed(SEED)
    results = []
    for i in range(N_DIALOGUES):
        cached = _load_cached(cdir, i)
        if cached:
            results.append(cached)
            continue

        seed_i = random.randint(1, 100000)
        random.seed(seed_i)
        np.random.seed(seed_i)

        sess = BiSession(
            sys_agent=sys_agent,
            user_agent=user_agent,
            kb_query=None,
            evaluator=evaluator,
        )
        sess.init_session()

        sys_response = []
        conversation = []
        for t in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response
            )
            conversation.append(
                {
                    "turn": t,
                    "user_da": str(user_response),
                    "user_nl": _da_to_nl(user_response, user_nlg),
                    "system_da": str(sys_response),
                    "system_nl": _da_to_nl(sys_response, sys_nlg),
                }
            )
            if session_over:
                break

        task_success = sess.evaluator.task_success()
        task_complete = sess.evaluator.complete
        stats = sess.evaluator.inform_F1()

        result = {
            "dialogue_id": i,
            "seed": seed_i,
            "turns": len(conversation),
            "completed": bool(task_complete),
            "success": bool(task_success),
            "inform_f1": stats[2],
            "book_rate": sess.evaluator.book_rate(),
            "conversation": conversation,
        }
        _save_to_cache(cdir, i, result)
        results.append(result)
        if (i + 1) % 50 == 0 or i == N_DIALOGUES - 1:
            print(
                f"  Dialog {i + 1}/{N_DIALOGUES}: turns={len(conversation)} "
                f"complete={task_complete} success={task_success}"
            )

    return _save(
        os.path.join(OUT_ROOT, "rule_us_rule_sys"), "rule_us_rule_sys", results
    )


# =========================================================================
# 2. TUS (neural user sim) + Rule Sys
# =========================================================================
def run_tus_rule_sys():
    import json as _json

    from convlab.dialog_agent import BiSession, PipelineAgent
    from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.policy.tus.unify.TUS import UserPolicy as TUSPolicy

    cdir = _cache_path("tus_rule_sys")

    config = _json.load(open("ConvLab-3/convlab/policy/tus/unify/exp/multiwoz.json"))
    user_policy = TUSPolicy(config, dial_ids_order=0)
    user_agent = PipelineAgent(None, None, user_policy, None, name="user")

    sys_agent = PipelineAgent(None, CompatRuleDST(), RulePolicy(), None, name="sys")
    evaluator = MultiWozEvaluator()

    user_nlg = TemplateNLG(is_user=True)
    sys_nlg = TemplateNLG(is_user=False)

    def _da_to_nl(da, nlg):
        try:
            return nlg.generate(da)
        except Exception:
            return str(da)

    set_seed(SEED)
    results = []
    for i in range(N_DIALOGUES):
        cached = _load_cached(cdir, i)
        if cached:
            results.append(cached)
            continue

        seed_i = random.randint(1, 100000)
        random.seed(seed_i)
        np.random.seed(seed_i)

        sess = BiSession(
            sys_agent=sys_agent,
            user_agent=user_agent,
            kb_query=None,
            evaluator=evaluator,
        )
        sess.init_session()

        sys_response = []
        conversation = []
        for t in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                _normalize_sys_da(sys_response)
            )
            conversation.append(
                {
                    "turn": t,
                    "user_da": str(user_response),
                    "user_nl": _da_to_nl(user_response, user_nlg),
                    "system_da": str(sys_response),
                    "system_nl": _da_to_nl(sys_response, sys_nlg),
                }
            )
            if session_over:
                break

        task_success = sess.evaluator.task_success()
        task_complete = sess.evaluator.complete
        stats = sess.evaluator.inform_F1()

        result = {
            "dialogue_id": i,
            "seed": seed_i,
            "turns": len(conversation),
            "completed": bool(task_complete),
            "success": bool(task_success),
            "inform_f1": stats[2],
            "book_rate": sess.evaluator.book_rate(),
            "conversation": conversation,
        }
        _save_to_cache(cdir, i, result)
        results.append(result)
        if (i + 1) % 50 == 0 or i == N_DIALOGUES - 1:
            print(
                f"  Dialog {i + 1}/{N_DIALOGUES}: turns={len(conversation)} "
                f"complete={task_complete} success={task_success}"
            )

    return _save(os.path.join(OUT_ROOT, "tus_rule_sys"), "tus_rule_sys", results)


# =========================================================================
# 3. GenTUS (generative neural user sim) + Rule Sys
# =========================================================================
def run_gentus_rule_sys():
    from convlab.dialog_agent import BiSession, PipelineAgent
    from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.policy.genTUS.stepGenTUS import UserPolicy as GenTUSPolicy
    from convlab.policy.rule.multiwoz import RulePolicy

    cdir = _cache_path("gentus_rule_sys")

    user_policy = GenTUSPolicy(mode="semantic")
    user_agent = PipelineAgent(None, None, user_policy, None, name="user")

    sys_agent = PipelineAgent(None, CompatRuleDST(), RulePolicy(), None, name="sys")
    evaluator = MultiWozEvaluator()

    user_nlg = TemplateNLG(is_user=True)
    sys_nlg = TemplateNLG(is_user=False)

    def _da_to_nl(da, nlg):
        try:
            return nlg.generate(da)
        except Exception:
            return str(da)

    set_seed(SEED)
    results = []
    for i in range(N_DIALOGUES):
        cached = _load_cached(cdir, i)
        if cached:
            results.append(cached)
            continue

        seed_i = random.randint(1, 100000)
        random.seed(seed_i)
        np.random.seed(seed_i)

        sess = BiSession(
            sys_agent=sys_agent,
            user_agent=user_agent,
            kb_query=None,
            evaluator=evaluator,
        )
        sess.init_session()

        sys_response = []
        conversation = []
        for t in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                _normalize_sys_da(sys_response)
            )
            conversation.append(
                {
                    "turn": t,
                    "user_da": str(user_response),
                    "user_nl": _da_to_nl(user_response, user_nlg),
                    "system_da": str(sys_response),
                    "system_nl": _da_to_nl(sys_response, sys_nlg),
                }
            )
            if session_over:
                break

        task_success = sess.evaluator.task_success()
        task_complete = sess.evaluator.complete
        stats = sess.evaluator.inform_F1()

        result = {
            "dialogue_id": i,
            "seed": seed_i,
            "turns": len(conversation),
            "completed": bool(task_complete),
            "success": bool(task_success),
            "inform_f1": stats[2],
            "book_rate": sess.evaluator.book_rate(),
            "conversation": conversation,
        }
        _save_to_cache(cdir, i, result)
        results.append(result)
        if (i + 1) % 50 == 0 or i == N_DIALOGUES - 1:
            print(
                f"  Dialog {i + 1}/{N_DIALOGUES}: turns={len(conversation)} "
                f"complete={task_complete} success={task_success}"
            )

    return _save(os.path.join(OUT_ROOT, "gentus_rule_sys"), "gentus_rule_sys", results)


# =========================================================================
# 4. LLM US + LLM RG (end-to-end)
# =========================================================================
def run_llm_us_llm_rg():
    from convlab.base_models.llm.user_simulator import LLM_RG, LLM_US

    cdir = _cache_path("llm_us_llm_rg", LLM_MODEL)

    goals = _load_test_goals()
    user_model = LLM_US("litellm", LLM_MODEL)
    system_model = LLM_RG("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        cached = _load_cached(cdir, i)
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
        if conversation:
            _save_to_cache(cdir, i, result)
        results.append(result)

    return _save(
        os.path.join(OUT_ROOT, "llm_us_llm_rg"),
        "llm_us_llm_rg",
        results,
        extra_summary={"model": LLM_MODEL},
    )


# =========================================================================
# 5. LLM US + Rule pipeline system (NLU->DST->Policy->NLG)
# =========================================================================
def run_llm_us_rule_sys():
    from convlab.base_models.llm.nlu import LLM_NLU
    from convlab.base_models.llm.user_simulator import LLM_US
    from convlab.dialog_agent import PipelineAgent
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.util.unified_datasets_util import load_dataset

    cdir = _cache_path("llm_us_rule_sys", LLM_MODEL)

    dataset = load_dataset("multiwoz21")
    example_dialogs = dataset["train"][:3]
    goals = _load_test_goals()

    sys_nlu = LLM_NLU("multiwoz21", "litellm", LLM_MODEL, "user", example_dialogs)
    sys_agent = PipelineAgent(
        sys_nlu, CompatRuleDST(), RulePolicy(), TemplateNLG(is_user=False), name="sys"
    )
    user_model = LLM_US("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        cached = _load_cached(cdir, i)
        if cached:
            results.append(cached)
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Dialog {i + 1}/{len(goals)} ...")
        try:
            user_model.init_session(goal)
            sys_agent.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"turn": 0, "system": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                if user_model.is_terminated:
                    conversation.append({"turn": turns, "user": user_msg})
                    break
                sys_msg = sys_agent.response(user_msg)
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
            traceback.print_exc()
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
        if conversation:
            _save_to_cache(cdir, i, result)
        results.append(result)

    return _save(
        os.path.join(OUT_ROOT, "llm_us_rule_sys"),
        "llm_us_rule_sys",
        results,
        extra_summary={"model": LLM_MODEL},
    )


# =========================================================================
# Main
# =========================================================================
COMBOS = [
    ("rule_us_rule_sys", run_rule_us_rule_sys),
    ("tus_rule_sys", run_tus_rule_sys),
    ("gentus_rule_sys", run_gentus_rule_sys),
    ("llm_us_llm_rg", run_llm_us_llm_rg),
    # ("llm_us_rule_sys", run_llm_us_rule_sys),
]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_summaries = {}

    for name, func in COMBOS:
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"{'#' * 70}")
        try:
            all_summaries[name] = func()
        except Exception:
            traceback.print_exc()
            all_summaries[name] = {"error": traceback.format_exc()[:500]}
            print(f"  FAILED: {name}")

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS  ({N_DIALOGUES} test dialogues each)")
    print(f"{'=' * 70}")
    print(f"{'Combo':<25} {'Completion':>12} {'Success':>12} {'Avg Turns':>12}")
    print("-" * 61)
    for name, s in all_summaries.items():
        if "error" in s:
            print(f"{name:<25} {'ERROR':>12}")
        else:
            print(
                f"{name:<25} {s['completion_rate']:>11.0%} "
                f"{s['success_rate']:>11.0%} {s['avg_turns']:>11.1f}"
            )


if __name__ == "__main__":
    main()
