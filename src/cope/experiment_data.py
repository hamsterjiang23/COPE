"""Reproducible GSM8K preparation and paired exploratory diagnostics (no GPU imports)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

PROMPT = (
    "Solve the problem step by step. "
    "End your response with '#### ' followed by the numerical answer."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, value):
    """Exclusive creation: finalized evidence is never silently overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prepare_examples(rows, teacher_tokenizer, student_tokenizer, config):
    """Split by normalized question before train-only length filtering."""
    unique, seen, duplicates = [], set(), []
    for index, row in enumerate(rows):
        question = row["question"].strip()
        key = " ".join(question.split())
        sample_id = hashlib.sha256(key.encode()).hexdigest()
        if key in seen:
            duplicates.append(index)
            continue
        seen.add(key)
        unique.append(
            {"id": sample_id, "source_index": index, "question": question, "answer": row["answer"]}
        )
    random.Random(config["seed"]).shuffle(unique)
    n_holdout = config["dataset"]["holdout_size"]
    holdout = unique[:n_holdout]
    candidates = unique[n_holdout:]
    training, exclusions = [], []
    limits = config["training"]
    for row in holdout + candidates:
        row["teacher_ids"] = prompt_ids(teacher_tokenizer, row["question"])
        row["prompt_ids"] = prompt_ids(student_tokenizer, row["question"])
        target = student_tokenizer.encode(row["answer"], add_special_tokens=False)
        row["target_ids"] = target + [student_tokenizer.eos_token_id]
    for row in candidates:
        reason = None
        if max(len(row["teacher_ids"]), len(row["prompt_ids"])) > limits["prompt_max_tokens"]:
            reason = "prompt_too_long"
        elif len(row["prompt_ids"]) + len(row["target_ids"]) > limits["sequence_max_tokens"]:
            reason = "complete_sequence_too_long"
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        elif len(training) < config["dataset"]["train_size"]:
            training.append(row)
    if len(training) != config["dataset"]["train_size"] or len(holdout) != n_holdout:
        raise ValueError("insufficient unique eligible examples")
    assert not ({x["id"] for x in training} & {x["id"] for x in holdout})
    # Holdout examples are never discarded/truncated based on their reference answer.
    return {
        "train": training,
        "holdout": holdout,
        "train_exclusions": exclusions,
        "duplicate_source_indices": duplicates,
    }


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def donor_indices(size, seed, epoch=0):
    if size < 2:
        raise ValueError("mismatching requires at least two different examples")
    order = epoch_order(size, seed, epoch)
    donors = [0] * size
    for index, donor in zip(order, order[1:] + order[:1], strict=True):
        donors[index] = donor
    assert all(i != donor for i, donor in enumerate(donors))
    return donors


def extract_answer(text):
    matches = re.findall(r"####\s*([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))", text)
    if not matches:
        return None
    try:
        result = Decimal(matches[-1].replace(",", ""))
        return str(result.normalize()) if result.is_finite() else None
    except InvalidOperation:
        return None


def paired_diagnostics(baseline, assisted, seed=42):
    if not baseline or len(baseline) != len(assisted):
        raise ValueError("paired non-empty records required")
    if [r["id"] for r in baseline] != [r["id"] for r in assisted]:
        raise ValueError("paired sample IDs differ")
    base = np.array([r["correct"] for r in baseline], dtype=bool)
    other = np.array([r["correct"] for r in assisted], dtype=bool)
    delta = other.astype(float) - base.astype(float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(base), size=(10000, len(base)))].mean(axis=1)
    corrected = int((~base & other).sum())
    harmed = int((base & ~other).sum())
    return {
        "n": len(base),
        "corrected": corrected,
        "harmed": harmed,
        "baseline_wrong": int((~base).sum()),
        "baseline_correct": int(base.sum()),
        "correction_rate": corrected / int((~base).sum()) if (~base).any() else None,
        "harm_rate": harmed / int(base.sum()) if base.any() else None,
        "net_accuracy_change": float(delta.mean()),
        "paired_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def summarize_records(records):
    if not records:
        return {"n": 0, "accuracy": None, "nll": None}
    token_count = sum(r["target_tokens"] for r in records)
    return {
        "n": len(records),
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "nll": sum(r["nll_sum"] for r in records) / token_count if token_count else None,
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in records])),
        "parse_failure_rate": sum(r["prediction"] is None for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "mean_online_ms_per_sample": float(np.mean([r["online_ms"] for r in records])),
        "mean_teacher_prefill_ms_per_sample": float(
            np.mean([r["teacher_prefill_ms"] for r in records])
        ),
        "mean_interface_ms_per_sample": float(np.mean([r["interface_ms"] for r in records])),
        "mean_student_generate_ms_per_sample": float(np.mean([r["generate_ms"] for r in records])),
    }
