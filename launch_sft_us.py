#!/usr/bin/env python
"""
Load the User Simulator (US) SFT dataset uploaded by generate_sft_datasets.py,
upload it to Together AI, and launch an SFT fine-tuning job.

Requires: pip install together datasets
Env: TOGETHER_API_KEY (required), WANDB_API_KEY (optional)
"""

import argparse
import json
import os
import tempfile

from datasets import load_dataset
from together import Together

# Same repo as in generate_sft_datasets.py
HF_REPO_US = os.environ.get("HF_REPO_US", "slingshot/multiwoz-2.1-user-sim-sft")
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference"


def main():
    parser = argparse.ArgumentParser(description="Launch US SFT on Together AI")
    parser.add_argument(
        "--dataset",
        default=HF_REPO_US,
        help=f"HuggingFace dataset repo (default: {HF_REPO_US})",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--suffix",
        default="multiwoz-us-sft",
        help="Suffix for the fine-tuned model name (default: us-sft)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and check data only; do not upload or create job",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Set TOGETHER_API_KEY to launch jobs.")

    print(f"Loading US dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  Loaded {len(ds)} examples")

    # Together expects JSONL with one object per line: {"messages": [...]}
    # Our dataset already has a "messages" column.
    def to_together_row(example):
        return {"messages": example["messages"]}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        path = f.name
        for i in range(len(ds)):
            row = to_together_row(ds[i])
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)

    print(f"  Wrote JSONL to {path}")

    if args.dry_run:
        print("Dry run: skipping upload and job creation.")
        os.unlink(path)
        return

    client = Together(api_key=api_key)

    # Validate file format
    from together.utils import check_file

    report = check_file(path)
    print("  File check:", report.get("message", report))
    if not report.get("is_check_passed", False):
        os.unlink(path)
        raise SystemExit("File check failed. Fix data format and retry.")

    print("  Uploading to Together AI ...")
    upload_resp = client.files.upload(path, purpose="fine-tune", check=True)
    os.unlink(path)
    training_file_id = upload_resp.id
    print(f"  Training file ID: {training_file_id}")

    wandb_key = os.environ.get("WANDB_API_KEY")
    create_kw = {
        "training_file": training_file_id,
        "model": MODEL,
        "n_epochs": 1,
        "n_checkpoints": 1,
        "learning_rate": 5e-6,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "training_method": "sft",
        "train_on_inputs": False,
        "wandb_project_name": "DIAL",
        "suffix": args.suffix,
    }
    if wandb_key:
        create_kw["wandb_api_key"] = wandb_key

    print("  Creating fine-tuning job ...")
    ft_resp = client.fine_tuning.create(**create_kw)
    print(f"  Job ID: {ft_resp.id}")
    print(f"  Status: {getattr(ft_resp, 'status', 'N/A')}")
    print("  Monitor: client.fine_tuning.retrieve(id=%r)" % ft_resp.id)


if __name__ == "__main__":
    main()
