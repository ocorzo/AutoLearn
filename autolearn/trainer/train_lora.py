#!/usr/bin/env python3
"""Train one isolated Qwen3.5 LoRA candidate from an approved trial dataset.

This script never contacts Mem0, Chatbox, the gateway, or the production
llama-server.  Evaluation files are deliberately not accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"Training dataset is empty: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trial(trial_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lesson_path = trial_dir / "lesson.json"
    train_path = trial_dir / "train.jsonl"
    lesson = json.loads(lesson_path.read_text(encoding="utf-8"))
    rows = read_jsonl(train_path)

    dataset = lesson["dataset"]
    if len(rows) != dataset["train_example_count"]:
        raise ValueError("train.jsonl count does not match lesson.json")
    if dataset["eval_file"] == dataset["train_file"]:
        raise ValueError("Evaluation data must not be used as training data")
    if (trial_dir / dataset["eval_file"]).resolve() == train_path.resolve():
        raise ValueError("Training and evaluation paths must differ")

    reasoning_rows = sum(row.get("role_in_batch") == "reasoning_anchor" for row in rows)
    ratio = reasoning_rows / len(rows)
    if ratio < 0.75:
        raise ValueError("Reasoning anchors must make up at least 75% of the batch")
    return lesson, rows


def render_and_tokenize(
    processor: Any, rows: list[dict[str, Any]], max_length: int
) -> list[dict[str, list[int]]]:
    tokenizer = processor.tokenizer
    prepared = []
    for row in rows:
        messages = row["messages"]
        if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            raise ValueError("Each training example must contain exactly one user/assistant pair")

        prompt = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = encoded["input_ids"]
        labels = input_ids.copy()
        for index in range(min(len(prompt_ids), len(labels))):
            labels[index] = -100
        if all(label == -100 for label in labels):
            raise ValueError("The assistant response was truncated entirely")
        prepared.append({"input_ids": input_ids, "attention_mask": encoded["attention_mask"], "labels": labels})
    return prepared


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor([pad(item["input_ids"], self.pad_token_id) for item in features]),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in features]),
            "labels": torch.tensor([pad(item["labels"], -100) for item in features]),
        }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this BF16 LoRA run")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16")

    lesson, rows = validate_trial(args.trial_dir)
    train_path = args.trial_dir / lesson["dataset"]["train_file"]
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing candidate: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    examples = render_and_tokenize(processor, rows, args.max_length)

    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda")

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=17,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=CausalCollator(processor.tokenizer.pad_token_id),
    )
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    manifest = {
        "trial_id": lesson["trial_id"],
        "lesson_id": lesson["lesson_id"],
        "base_model": str(args.model_dir),
        "train_file": train_path.name,
        "train_sha256": sha256(train_path),
        "eval_file": lesson["dataset"]["eval_file"],
        "sentinel_file": lesson["sentinels"]["file"],
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "precision": "bf16",
            "gradient_checkpointing": True,
        },
        "lora": {"r": 16, "alpha": 16, "target_modules": TARGET_MODULES},
        "parameters": {"trainable": trainable_parameters, "total": total_parameters},
        "production_deployment_allowed": False,
        "next_required_step": "Run post-training lesson and sentinel evaluations without Mem0.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
