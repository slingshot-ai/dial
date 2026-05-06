#!/usr/bin/env python
"""
Train a token-classification discriminator on assistant (user-simulator) messages.

Loads paired real/simulated dialogues from HuggingFace, keeps only system +
assistant messages, and trains to predict real (1) vs simulated (0) at the
EOT token of every assistant message.
"""

import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    pad_without_fast_tokenizer_warning,
)

# ── Constants ──────────────────────────────────────────────────────────────

HF_DATASET = "slingshot/multiwoz-2.1-user-disc-dial-it3"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "output"
RUN_NAME = "user-disc-dial-it3"
WANDB_PROJECT = "DIAL"

IGNORE_LABEL = -100
TEST_SPLIT_RATIO = 0.15
SEED = 42

LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EVAL_STEPS = 25
LOGGING_STEPS = 10
LR_SCHEDULER_TYPE = "cosine"
OPTIM = "adamw_torch"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LOAD_IN_4BIT = True
USE_BF16 = True
GRADIENT_CHECKPOINTING = True
USE_REENTRANT = False
ATTN_IMPLEMENTATION = "flash_attention_2"

CHAT_TEMPLATE = """
{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_tokenizer(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def flatten_dataset(raw_dataset) -> Dataset:
    """Convert paired real/simulated rows into individual labelled samples,
    keeping only system + assistant messages."""
    samples = []
    for i in range(len(raw_dataset)):
        row = raw_dataset[i]
        for key, label in [("real_messages", 1), ("simulated_messages", 0)]:
            msgs = row[key]
            filtered = [m for m in msgs if m["role"] in ("system", "assistant")]
            if any(m["role"] == "assistant" for m in filtered):
                samples.append({"messages": filtered, "label": label})
    return Dataset.from_list(samples)


def tokenize_sample(
    sample: Dict[str, Any], tokenizer: PreTrainedTokenizerFast
) -> Dict[str, Any]:
    """Tokenize messages and place labels at the EOT token of every assistant message."""
    messages = sample["messages"]
    label = sample["label"]

    tokenized = tokenizer.apply_chat_template(
        messages, return_dict=True, padding=False, truncation=False
    )

    eos_id = tokenizer.eos_token_id
    eos_indices = [i for i, t in enumerate(tokenized["input_ids"]) if t == eos_id]

    assistant_eot_indices = []
    eot_cursor = 0
    for msg in messages:
        single = tokenizer.apply_chat_template(
            [msg], return_dict=True, padding=False, truncation=False
        )
        n_eots = sum(1 for t in single["input_ids"] if t == eos_id)
        if msg["role"] == "assistant":
            for _ in range(n_eots):
                if eot_cursor < len(eos_indices):
                    assistant_eot_indices.append(eos_indices[eot_cursor])
                eot_cursor += 1
        else:
            eot_cursor += n_eots

    token_labels = [IGNORE_LABEL] * len(tokenized["input_ids"])
    for idx in assistant_eot_indices:
        token_labels[idx] = label

    tokenized["labels"] = token_labels
    return tokenized


# ── Data collator ──────────────────────────────────────────────────────────


class DiscriminatorDataCollator(DataCollatorForTokenClassification):
    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        label_name = "label" if "label" in features[0] else "labels"
        labels = (
            [f[label_name] for f in features] if label_name in features[0] else None
        )

        no_labels = [{k: v for k, v in f.items() if k != label_name} for f in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        seq_len = batch["input_ids"].shape[1]

        def to_list(x):
            return x.tolist() if isinstance(x, torch.Tensor) else list(x)

        if self.tokenizer.padding_side == "right":
            batch[label_name] = [
                to_list(l) + [self.label_pad_token_id] * (seq_len - len(l))
                for l in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (seq_len - len(l)) + to_list(l)
                for l in labels
            ]
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.long)
        return batch


# ── Metrics ────────────────────────────────────────────────────────────────


def _nth_position_mask(label_mask: torch.Tensor, n: int, reverse: bool = False):
    cumsum = label_mask.cumsum(1)
    if reverse:
        cumsum = label_mask + label_mask.sum(1).reshape(-1, 1) - cumsum
    return (cumsum == n) & label_mask


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels_t = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
    label_mask = labels_t != IGNORE_LABEL

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != IGNORE_LABEL

    pred = np.argmax(flat_logits[mask], axis=1)
    true = flat_labels[mask]

    metrics = {
        "accuracy": accuracy_score(true, pred),
        "f1": f1_score(true, pred, average="binary"),
        "mcc": matthews_corrcoef(true, pred),
    }

    for n in [1, 2, 5, 10]:
        nth = _nth_position_mask(label_mask, n).reshape(-1).numpy()
        if nth.sum() > 0:
            metrics[f"turn_{n}_accuracy"] = accuracy_score(
                flat_labels[nth], np.argmax(flat_logits[nth], axis=1)
            )
            metrics[f"turn_{n}_f1"] = f1_score(
                flat_labels[nth],
                np.argmax(flat_logits[nth], axis=1),
                average="binary",
            )
            metrics[f"turn_{n}_count"] = int(nth.sum())

    return metrics


# ── Print example tokenization ────────────────────────────────────────────


def print_token_table(input_ids, tokens, labels):
    id_w = max(max(len(str(x)) for x in input_ids), len("input_id"))
    tok_w = max(max(len(str(x)) for x in tokens), len("token"))
    lab_w = max(max(len(str(x)) for x in labels), len("label"))

    header = (
        f"{'Idx':<5} | {'input_id':<{id_w}} | {'token':<{tok_w}} | {'label':<{lab_w}}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for i, (iid, tok, lab) in enumerate(zip(input_ids, tokens, labels)):
        lab_s = str(lab) if lab != IGNORE_LABEL else "-"
        line = f"{i:<5} | {iid:<{id_w}} | {tok:<{tok_w}} | {lab_s:<{lab_w}}"
        print(f"\033[92m{line}\033[0m" if lab != IGNORE_LABEL else line)
    print(sep)


def print_example(tokenizer, dataset):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    print(f"\n{'=' * 60}")
    lbl = "REAL" if sample["label"] == 1 else "SIMULATED"
    print(f"Example #{idx}  —  label={sample['label']} ({lbl})")
    print(f"Messages ({len(sample['messages'])} total):")
    for i, m in enumerate(sample["messages"]):
        preview = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
        print(f"  {i + 1}. [{m['role']:>9}] {preview}")

    tokenized = tokenize_sample(sample, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    labeled = [
        (i, tokens[i], labels[i])
        for i in range(len(labels))
        if labels[i] != IGNORE_LABEL
    ]
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Labeled positions (assistant EOT tokens): {len(labeled)}")

    eos_positions = [i for i, t in enumerate(input_ids) if t == tokenizer.eos_token_id]
    print(f"All EOT positions: {eos_positions}")

    for pos, tok, lab in labeled:
        print(f"  pos {pos}: token='{tok}', label={lab}")

    print_token_table(input_ids, tokens, labels)
    print(f"{'=' * 60}\n")


# ── Training ───────────────────────────────────────────────────────────────


def main():
    set_seed(SEED)

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = prepare_tokenizer(tokenizer)

    print(f"Loading dataset from {HF_DATASET} …")
    raw = load_dataset(HF_DATASET, split="train")
    print(f"  Raw rows: {len(raw)}")

    print("Flattening to individual samples (system + assistant only) …")
    flat = flatten_dataset(raw)
    print(f"  Samples: {len(flat)}")

    # Print example tokenization before training
    print_example(tokenizer, flat)

    # Train / test split
    split = flat.train_test_split(test_size=TEST_SPLIT_RATIO, seed=SEED)
    print(f"  Train: {len(split['train'])}, Test: {len(split['test'])}")

    print("Tokenizing …")
    tokenized = split.map(
        tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=os.cpu_count(),
        remove_columns=split["train"].column_names,
        desc="Tokenizing",
    )

    print("Loading model …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if LOAD_IN_4BIT else None,
        # attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
        num_labels=2,
        id2label={0: "SIMULATED", 1: "REAL"},
        label2id={"SIMULATED": 0, "REAL": 1},
        use_cache=not GRADIENT_CHECKPOINTING,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if GRADIENT_CHECKPOINTING:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, RUN_NAME),
        run_name=RUN_NAME,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIM,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        bf16=USE_BF16,
        fp16=not USE_BF16,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if WANDB_PROJECT else "none",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": USE_REENTRANT},
        dataloader_num_workers=os.cpu_count(),
        remove_unused_columns=False,
    )

    if WANDB_PROJECT:
        import wandb

        wandb.init(project=WANDB_PROJECT, name=RUN_NAME)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DiscriminatorDataCollator(
            tokenizer=tokenizer,
            label_pad_token_id=IGNORE_LABEL,
        ),
        compute_metrics=compute_metrics,
    )

    print("Starting training …")
    trainer.train()
    print("Done!")


if __name__ == "__main__":
    main()
