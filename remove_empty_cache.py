#!/usr/bin/env python
"""
Remove empty/errored cache entries from baseline, discriminator, and preference runs.

- Baselines (cache/baselines/): removes dialog_*.json where conversation is empty
  (failed requests often produce conversation: []).
- Discriminator (cache/discriminator/): removes dialog_*.json where messages
  contain only system or have no assistant turns.
- Preference (cache/preference/): removes dialog_*.json where the cached value is
  an empty list []. process_conversation can write [] when there are too few
  rewards or all regenerations fail; removing these lets the run retry.

Usage:
  python remove_empty_cache.py           # delete empty caches
  python remove_empty_cache.py --dry-run # only print what would be removed
"""

import argparse
import json
import os

CACHE_DIR = "cache"


def is_empty_baseline(path):
    """True if this baseline cache file is an empty/errored result."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    conv = data.get("conversation", None)
    return conv is not None and len(conv) == 0


def is_empty_discriminator(path):
    """True if this discriminator cache file has no real dialogue (system-only or no assistant)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, list):
        return False
    if len(data) <= 1:
        return True
    n_assistant = sum(1 for m in data if m.get("role") == "assistant")
    return n_assistant == 0


def is_empty_preference(path):
    """True if this preference cache file is an empty list (no samples produced)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, list) and len(data) == 0


def main():
    ap = argparse.ArgumentParser(description="Remove empty cache files from baselines, discriminator, and preference.")
    ap.add_argument("--dry-run", action="store_true", help="Only print paths that would be removed")
    args = ap.parse_args()

    removed = 0

    # Baselines: cache/baselines/<combo>/[model/]dialog_*.json
    baselines_root = os.path.join(CACHE_DIR, "baselines")
    if os.path.isdir(baselines_root):
        baseline_dirs = {}  # dirpath -> num checked
        to_remove_baseline = []
        for dirpath, _dirnames, filenames in os.walk(baselines_root):
            dialog_files = [n for n in filenames if n.startswith("dialog_") and n.endswith(".json")]
            if dialog_files:
                baseline_dirs[dirpath] = len(dialog_files)
                for name in dialog_files:
                    path = os.path.join(dirpath, name)
                    if is_empty_baseline(path):
                        to_remove_baseline.append(path)
        print("Dirs checked (baselines):")
        for d in sorted(baseline_dirs):
            print(f"  {d}  ({baseline_dirs[d]} checked)")
        for path in to_remove_baseline:
            print(path)
            if not args.dry_run:
                os.remove(path)
            removed += 1
    else:
        print("Dirs checked (baselines): (none — not a directory)")
        print(f"  {baselines_root}")

    # Discriminator: cache/discriminator/<key>/dialog_*.json
    disc_root = os.path.join(CACHE_DIR, "discriminator")
    if os.path.isdir(disc_root):
        disc_dirs = {}  # dirpath -> num checked
        to_remove_disc = []
        for dirpath, _dirnames, filenames in os.walk(disc_root):
            dialog_files = [n for n in filenames if n.startswith("dialog_") and n.endswith(".json")]
            if dialog_files:
                disc_dirs[dirpath] = len(dialog_files)
                for name in dialog_files:
                    path = os.path.join(dirpath, name)
                    if is_empty_discriminator(path):
                        to_remove_disc.append(path)
        print("\nDirs checked (discriminator):")
        for d in sorted(disc_dirs):
            print(f"  {d}  ({disc_dirs[d]} checked)")
        for path in to_remove_disc:
            print(path)
            if not args.dry_run:
                os.remove(path)
            removed += 1
    else:
        print("\nDirs checked (discriminator): (none — not a directory)")
        print(f"  {disc_root}")

    # Preference: cache/preference/<dataset>/dialog_*.json
    pref_root = os.path.join(CACHE_DIR, "preference")
    if os.path.isdir(pref_root):
        pref_dirs = {}
        to_remove_pref = []
        for dirpath, _dirnames, filenames in os.walk(pref_root):
            dialog_files = [n for n in filenames if n.startswith("dialog_") and n.endswith(".json")]
            if dialog_files:
                pref_dirs[dirpath] = len(dialog_files)
                for name in dialog_files:
                    path = os.path.join(dirpath, name)
                    if is_empty_preference(path):
                        to_remove_pref.append(path)
        print("\nDirs checked (preference):")
        for d in sorted(pref_dirs):
            print(f"  {d}  ({pref_dirs[d]} checked)")
        for path in to_remove_pref:
            print(path)
            if not args.dry_run:
                os.remove(path)
            removed += 1
    else:
        print("\nDirs checked (preference): (none — not a directory)")
        print(f"  {pref_root}")

    if args.dry_run:
        print(f"\n[dry-run] Would remove {removed} empty cache file(s).")
    else:
        print(f"\nRemoved {removed} empty cache file(s).")


if __name__ == "__main__":
    main()
