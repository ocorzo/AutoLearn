#!/usr/bin/env python3
"""Evaluate a LoRA candidate directly, with no gateway or external memory.

Without ``--adapter-dir`` the same code evaluates the base checkpoint, which
gives a baseline produced by the same engine, precision and scoring as the
candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir", type=Path, help="LoRA candidate; omit it to evaluate the base model as baseline"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", value.lower())
    text = "".join(character for character in text if unicodedata.category(character) != "Mn")
    return re.sub(r"\s+", " ", text.strip())


# Scoring modes that cannot be decided by string comparison.  They are marked
# for manual review instead of being counted as failures, so that a baseline
# judged by hand and a candidate judged by this script remain comparable.
MANUAL_SCORING = {"semantic_and_length"}


def _strip_code_fence(value: str) -> str:
    match = re.fullmatch(r"\s*```[a-zA-Z]*\s*(.*?)\s*```\s*", value, flags=re.DOTALL)
    return match.group(1) if match else value


def exact_match(response: str, expected: Any, scoring: str) -> bool | None:
    if scoring in MANUAL_SCORING:
        return None
    if scoring == "json_exact":
        try:
            return json.loads(_strip_code_fence(response)) == expected
        except json.JSONDecodeError:
            return False
    if scoring == "ordered_list_exact":
        items = [normalize(item).rstrip(".") for item in response.split(",")]
        return items == [normalize(item).rstrip(".") for item in str(expected).split(",")]
    return normalize(response).rstrip(".") == normalize(str(expected)).rstrip(".")


def contains_element(normalized_response: str, element: str) -> bool:
    """Match whole words so that short elements such as "no" do not hit "bueno"."""
    pattern = r"(?<!\w)" + re.escape(normalize(element)) + r"(?!\w)"
    return re.search(pattern, normalized_response) is not None


def score_lesson(row: dict[str, Any], response: str) -> bool:
    normalized = normalize(response)
    return all(contains_element(normalized, element) for element in row["expected_elements"])


def generate(model: Any, processor: Any, prompt: str) -> str:
    tokenizer = processor.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lesson_rows = read_jsonl(args.trial_dir / "eval.jsonl")
    sentinel_rows = read_jsonl(args.trial_dir / "sentinels_eval.jsonl")
    trial_id = lesson_rows[0]["trial_id"]

    mode = "candidate" if args.adapter_dir else "baseline"
    processor = AutoProcessor.from_pretrained(args.adapter_dir or args.model_dir)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.to("cuda")
    model.eval()

    lesson_results = []
    for row in lesson_rows:
        response = generate(model, processor, row["prompt"])
        lesson_results.append(
            {
                "trial_id": trial_id,
                "lesson_id": row["lesson_id"],
                "type": row["type"],
                "prompt": row["prompt"],
                "response": response,
                "expected_elements": row["expected_elements"],
                "matches_expected": score_lesson(row, response),
                "mem0_retrieval": False,
            }
        )

    sentinel_results = []
    for row in sentinel_rows:
        response = generate(model, processor, row["prompt"])
        sentinel_results.append(
            {
                "trial_id": trial_id,
                "sentinel_id": row["sentinel_id"],
                "response": response,
                "expected": row["expected"],
                "scoring": row["scoring"],
                "matches_expected": exact_match(response, row["expected"], row["scoring"]),
                "needs_manual_review": row["scoring"] in MANUAL_SCORING,
                "mem0_retrieval": False,
            }
        )

    write_jsonl(args.output_dir / "lesson_eval.jsonl", lesson_results)
    write_jsonl(args.output_dir / "sentinels_eval.jsonl", sentinel_results)
    summary = {
        "mode": mode,
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "lesson_correct": sum(item["matches_expected"] for item in lesson_results),
        "lesson_total": len(lesson_results),
        "sentinels_correct": sum(item["matches_expected"] is True for item in sentinel_results),
        "sentinels_failed": [item["sentinel_id"] for item in sentinel_results if item["matches_expected"] is False],
        "sentinels_needs_manual_review": [
            item["sentinel_id"] for item in sentinel_results if item["matches_expected"] is None
        ],
        "sentinels_total": len(sentinel_results),
        "mem0_retrieval": False,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
