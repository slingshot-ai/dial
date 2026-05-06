#!/usr/bin/env python
"""
Compute standard diversity metrics on user utterances from experiment results.

Metrics:
  - TTR (Type-Token Ratio): unique words / total words
  - Distinct-1: unique unigrams / total unigrams
  - Distinct-2: unique bigrams / total bigrams
  - Distinct-3: unique trigrams / total trigrams
  - Word Entropy (unigram Shannon entropy)
  - Mean Utterance Length (tokens)
  - Mean Conversation Length (user turns per dialogue)
  - MAUVE (Pillutla et al., NeurIPS 2021 Outstanding Paper; JMLR 2023): distributional
    mode-collapse metric. Embeds with a configurable OpenAI embedding model, quantizes to ~500 k-means
    clusters, computes divergence frontier R_λ = λP + (1−λ)Q and area under
    (KL(P‖R_λ), KL(Q‖R_λ)) scaled to [0,1]. KL(Q‖P) = mode dropping (simulator fails to
    cover human modes); KL(P‖Q) = hallucination (simulator generates unlike human).
    Requires: pip install mauve-text openai; OPENAI_API_KEY set.

  MAUVE results are cached by combo name and num_buckets; each cache file is named
  {combo}_buckets_{num_buckets}.json so you can delete a combo's cache selectively.
  Cache dir: MAUVE_CACHE_DIR env (default: .mauve_cache next to this script).
"""

import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

RESULTS_DIR = "experiment_results"
EMBED_BATCH_SIZE = 100
MAUVE_NUM_BUCKETS = 250
EMBEDDING_MODEL = "text-embedding-3-small"
_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
MAUVE_CACHE_DIR = os.environ.get("MAUVE_CACHE_DIR", os.path.join(_SCRIPT_DIR, ".mauve_cache"))


def _mauve_cache_filename(combo, num_buckets):
    """Safe cache filename from combo and num_buckets (e.g. dial_us_sft_rg_v1_buckets_250.json)."""
    safe = str(combo).replace(os.sep, "_").replace("\\", "_")
    return f"{safe}_buckets_{num_buckets}"


def _mauve_cache_get(cache_name):
    """Load MAUVE result from cache if present. cache_name = combo_buckets_N (no .json). Returns dict or None."""
    if not cache_name:
        return None
    path = os.path.join(MAUVE_CACHE_DIR, cache_name + ".json")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _mauve_cache_set(cache_name, result):
    """Write MAUVE result to cache. cache_name = combo_buckets_N (no .json)."""
    if not cache_name:
        return
    path = os.path.join(MAUVE_CACHE_DIR, cache_name + ".json")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f)
    except OSError:
        pass


def _embed_openai(texts, model=EMBEDDING_MODEL, show_progress=True):
    """Embed texts with OpenAI. Returns (N, dim) float32 array."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("MAUVE embedding requires: pip install openai")
    client = OpenAI()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(f"OPENAI_API_KEY must be set for MAUVE ({model})")
    embeddings = []
    it = range(0, len(texts), EMBED_BATCH_SIZE)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=f"Embedding ({model})", unit="batch")
        except ImportError:
            pass
    for i in it:
        batch = texts[i : i + EMBED_BATCH_SIZE]
        r = client.embeddings.create(model=model, input=batch)
        for e in r.data:
            embeddings.append(e.embedding)
    return np.asarray(embeddings, dtype=np.float32)


def compute_mauve_with_openai(
    p_utterances,
    q_utterances=None,
    q_features=None,
    num_buckets=MAUVE_NUM_BUCKETS,
    show_progress=True,
    use_cache=True,
    cache_combo=None,
    embedding_model=EMBEDDING_MODEL,
):
    """
    MAUVE between two utterance distributions using the configured embedding model.
    P = model/simulator, Q = human/real.
    If q_features is provided, Q is not re-embedded (embed GT once and reuse).
    Returns dict with mauve, frontier_integral, kl_pq (hallucination), kl_qp (mode dropping).
    When use_cache is True and cache_combo is set, results are cached under
    {combo}_buckets_{num_buckets}.json for selective deletion.
    """
    try:
        import mauve
    except ImportError:
        raise ImportError("MAUVE requires: pip install mauve-text")

    # If cache is enabled and we have a combo, try loading from file first (same combo + num_buckets = load directly)
    if use_cache and cache_combo is not None:
        cache_name = _mauve_cache_filename(cache_combo, num_buckets)
        cached = _mauve_cache_get(cache_name)
        if cached is not None:
            if show_progress:
                print("  MAUVE (cache hit)", flush=True)
            return cached

    if not p_utterances:
        return {
            "mauve": None,
            "frontier_integral": None,
            "kl_pq_hallucination": None,
            "kl_qp_mode_dropping": None,
        }
    if q_features is None and not q_utterances:
        return {
            "mauve": None,
            "frontier_integral": None,
            "kl_pq_hallucination": None,
            "kl_qp_mode_dropping": None,
        }

    if show_progress:
        print("  Embedding P (user simulator)...", flush=True)
    p_feats = _embed_openai(p_utterances, model=embedding_model, show_progress=show_progress)
    if q_features is not None:
        q_feats = np.asarray(q_features, dtype=np.float32)
    else:
        if show_progress:
            print("  Embedding Q (real utterances)...", flush=True)
        q_feats = _embed_openai(q_utterances, model=embedding_model, show_progress=show_progress)
    if show_progress:
        print("  Computing MAUVE (k-means + divergence frontier)...", flush=True)
    out = mauve.compute_mauve(
        p_features=p_feats,
        q_features=q_feats,
        num_buckets=num_buckets,
        verbose=False,
    )
    # Histograms from mauve (discrete distributions over clusters)
    p_hist = np.asarray(out.p_hist).ravel()
    q_hist = np.asarray(out.q_hist).ravel()
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    kl_pq = float(np.sum(p_hist * (np.log(p_hist) - np.log(q_hist))))
    kl_qp = float(np.sum(q_hist * (np.log(q_hist) - np.log(p_hist))))
    result = {
        "mauve": float(out.mauve),
        "frontier_integral": float(out.frontier_integral),
        "kl_pq_hallucination": kl_pq,
        "kl_qp_mode_dropping": kl_qp,
    }
    if use_cache and cache_combo is not None:
        _mauve_cache_set(_mauve_cache_filename(cache_combo, num_buckets), result)
    return result

def extract_user_utterances(results_path):
    with open(results_path) as f:
        data = json.load(f)

    utterances = []
    conversation_lengths = []
    for dialog in data["per_dialogue"]:
        n_user = 0
        for turn in dialog["conversation"]:
            text = turn.get("user_nl") or turn.get("user") or ""
            text = str(text).strip()
            if text and text not in ("[]", ""):
                text = text.replace("[END]", "").replace("**", "").strip()
                if text:
                    utterances.append(text)
                    n_user += 1
        conversation_lengths.append(n_user)
    return utterances, conversation_lengths


def tokenize(text):
    return text.lower().split()


def compute_distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_metrics(utterances, conversation_lengths=None):
    all_tokens = []
    lengths = []
    for utt in utterances:
        toks = tokenize(utt)
        all_tokens.extend(toks)
        lengths.append(len(toks))

    n_total = len(all_tokens)
    n_unique = len(set(all_tokens))

    out = {
        "n_utterances": len(utterances),
        "n_tokens": n_total,
        "n_unique_tokens": n_unique,
        "ttr": n_unique / max(n_total, 1),
        "distinct_1": compute_distinct_n(all_tokens, 1),
        "distinct_2": compute_distinct_n(all_tokens, 2),
        "distinct_3": compute_distinct_n(all_tokens, 3),
        "entropy": compute_entropy(all_tokens),
        "mean_utt_length": sum(lengths) / max(len(lengths), 1),
    }
    if conversation_lengths:
        out["mean_conversation_length"] = sum(conversation_lengths) / max(len(conversation_lengths), 1)
    else:
        out["mean_conversation_length"] = None
    return out


def extract_ground_truth_utterances(n_dialogues=None):
    """Extract user utterances from the MultiWOZ 2.1 test split."""
    from convlab.util.unified_datasets_util import load_dataset
    dataset = load_dataset("multiwoz21")
    test_dialogues = dataset["test"]
    if n_dialogues is not None:
        test_dialogues = test_dialogues[:n_dialogues]

    utterances = []
    conversation_lengths = []
    for dlg in test_dialogues:
        n_user = 0
        for turn in dlg["turns"]:
            if turn["speaker"] == "user":
                text = turn["utterance"].strip()
                if text:
                    utterances.append(text)
                    n_user += 1
        conversation_lengths.append(n_user)
    return utterances, conversation_lengths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute diversity metrics (and optionally MAUVE).")
    parser.add_argument(
        "--mauve",
        action="store_true",
        help=f"Compute MAUVE (user simulator vs real) using {EMBEDDING_MODEL}. Needs OPENAI_API_KEY, mauve-text, openai.",
    )
    parser.add_argument(
        "--no-mauve-cache",
        action="store_true",
        help="Disable MAUVE result caching (recompute even when inputs match a previous run).",
    )
    args = parser.parse_args()

    all_metrics = {}

    # Detect how many dialogues were used in baselines (from any results file)
    n_dialogues = None
    for entry in sorted(os.listdir(RESULTS_DIR)):
        results_path = os.path.join(RESULTS_DIR, entry, "results.json")
        if os.path.isfile(results_path):
            with open(results_path) as f:
                data = json.load(f)
            n_dialogues = data.get("summary", {}).get("n_dialogues")
            if n_dialogues:
                break

    gt_utterances, gt_conv_lengths = extract_ground_truth_utterances(n_dialogues)
    all_metrics["ground_truth"] = compute_metrics(gt_utterances, gt_conv_lengths)

    entries_to_mauve = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        results_path = os.path.join(RESULTS_DIR, entry, "results.json")
        if not os.path.isfile(results_path):
            continue
        try:
            utterances, conv_lengths = extract_user_utterances(results_path)
        except (KeyError, json.JSONDecodeError):
            print(f"  {entry}: could not parse results.json, skipping")
            continue
        if not utterances:
            print(f"  {entry}: no user utterances found, skipping")
            continue
        metrics = compute_metrics(utterances, conv_lengths)
        all_metrics[entry] = metrics
        if args.mauve and entry != "ground_truth":
            entries_to_mauve.append((entry, utterances))

    # MAUVE: user simulator (P) vs real utterances (Q); embed GT once and reuse
    if args.mauve and entries_to_mauve and gt_utterances:
        use_cache = not args.no_mauve_cache
        # Try cache first for each combo so we can skip embedding Q if all hit
        cache_results = {}
        if use_cache:
            for entry, _ in entries_to_mauve:
                cache_name = _mauve_cache_filename(entry, MAUVE_NUM_BUCKETS)
                cached = _mauve_cache_get(cache_name)
                if cached is not None:
                    cache_results[entry] = cached
        cached_combos = sorted(cache_results.keys())
        new_combos = sorted(e for e, _ in entries_to_mauve if e not in cache_results)
        print(f"  MAUVE cache: cached={cached_combos}  new={new_combos}", flush=True)
        need_embed_q = use_cache and len(cache_results) < len(entries_to_mauve)
        if not use_cache:
            need_embed_q = True

        gt_features = None
        if need_embed_q:
            print("  Embedding Q (ground-truth, once)...", flush=True)
            gt_features = _embed_openai(gt_utterances, show_progress=True)
        elif cache_results:
            print("  MAUVE: all combos from cache (skipping Q embed)", flush=True)

        try:
            from tqdm import tqdm
            _mauve_iter = tqdm(entries_to_mauve, desc="MAUVE (simulator vs real)", unit="combo")
        except ImportError:
            _mauve_iter = entries_to_mauve
        for entry, utterances in _mauve_iter:
            if hasattr(_mauve_iter, "set_postfix"):
                _mauve_iter.set_postfix(combo=entry[:12])
            try:
                if entry in cache_results:
                    mauve_out = cache_results[entry]
                else:
                    mauve_out = compute_mauve_with_openai(
                        utterances,
                        q_features=gt_features,
                        num_buckets=MAUVE_NUM_BUCKETS,
                        show_progress=False,
                        use_cache=use_cache,
                        cache_combo=entry,
                    )
                for k, v in mauve_out.items():
                    all_metrics[entry][k] = v
            except Exception as e:
                print(f"  MAUVE failed for {entry}: {e}", flush=True)
                for k in ("mauve", "frontier_integral", "kl_pq_hallucination", "kl_qp_mode_dropping"):
                    all_metrics[entry][k] = None
    elif args.mauve and not gt_utterances:
        print("  No ground-truth utterances; skipping MAUVE.", flush=True)
    elif args.mauve and not entries_to_mauve:
        print("  No simulator entries to compare; skipping MAUVE.", flush=True)

    # Lexical table
    header = f"{'Combo':<22} {'Utts':>5} {'Tokens':>7} {'TTR':>6} {'D-1':>6} {'D-2':>6} {'D-3':>6} {'Entropy':>8} {'AvgLenUttr':>9} {'AvgLenConv':>9}"
    if args.mauve:
        header += f" {'MAUVE':>7} {'KL_PQ':>8} {'KL_QP':>8}"
    print(header)
    print("-" * len(header))
    for combo, m in all_metrics.items():
        avg_conv = m["mean_conversation_length"]
        avg_conv_str = f"{avg_conv:>9.1f}" if avg_conv is not None else "      n/a"
        row = (f"{combo:<22} {m['n_utterances']:>5} {m['n_tokens']:>7} "
               f"{m['ttr']:>6.3f} {m['distinct_1']:>6.3f} {m['distinct_2']:>6.3f} "
               f"{m['distinct_3']:>6.3f} {m['entropy']:>8.3f} {m['mean_utt_length']:>9.1f} {avg_conv_str}")
        if args.mauve:
            if m.get("mauve") is not None:
                row += f" {m['mauve']:>7.3f} {m.get('kl_pq_hallucination', 0):>8.3f} {m.get('kl_qp_mode_dropping', 0):>8.3f}"
            else:
                row += "      n/a      n/a      n/a"
        print(row)

    out_path = os.path.join(RESULTS_DIR, "diversity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
