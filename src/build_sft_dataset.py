"""Build the SFT JSONL dataset for LoRA training.

Output format matches the Nemotron-3-Nano-30B-A3B chat template (ChatML +
`<think>` tags), which is what the model was pretrained on. The training
notebook will call `tokenizer.apply_chat_template(..., enable_thinking=True)`
to wrap the user content; the assistant completion is just the reasoning
followed by `</think>\\boxed{answer}<|im_end|>`.

Reference: the winning Progress Prize submission by huikang
(https://github.com/tonghuikang/nemotron) uses this exact wrapping with
`enable_thinking=True` and the completion suffix `</think>\\boxed{...}<|im_end|>`.

Each output record schema:
    {
      "user": <user message string>,
      "completion": <reasoning></think>\\boxed{answer}<|im_end|>,
      "category": one of CATEGORIES,
      "verified_cot": True/False,
    }

CoT generation policy
---------------------
1. Run the symbolic solver. If its prediction matches the ground-truth answer,
   we use the solver's reasoning steps (verified CoT).
2. If the solver fails or disagrees, we fall back to a short generic CoT that
   names the category and still emits the correct \\boxed{answer}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.categorize import categorize
from src.solvers import solve

USER_SUFFIX = "\n\nPlease put your final answer inside `\\boxed{}`."

GENERIC_TEMPLATES = {
    "bit_manip": (
        "I examine the input/output examples to find a consistent 8-bit "
        "transformation rule (rotation, shift, XOR with constant, NOT, "
        "or composition thereof). Applying that rule to the query gives "
        "the answer."
    ),
    "physics": (
        "Each example gives a (time, distance) pair under d = 0.5 * g * t^2. "
        "I fit g by least squares, then evaluate d for the query time."
    ),
    "unit_conv": (
        "The conversion is linear: y = a*x + b. I fit a and b from the "
        "examples and apply the fit to the query measurement."
    ),
    "cipher": (
        "Each example pairs a ciphertext sentence with its plaintext. "
        "Aligning words and characters reveals a single substitution map; "
        "I apply it to the query letters."
    ),
    "roman": (
        "The 'Wonderland numeral system' is just standard Roman numerals, "
        "as the examples confirm. I convert the requested number directly."
    ),
    "algebra": (
        "The puzzle is a cryptarithm: each unique symbol maps bijectively "
        "to a digit, and the operator symbol maps to one of {add, abs_diff, "
        "mul, concat, rev_concat}. I infer the mapping from examples and "
        "apply it to the query."
    ),
}


def _approx_match(pred: str, gt: str) -> bool:
    if pred == gt:
        return True
    try:
        a, b = float(pred), float(gt)
        return abs(a - b) < 0.02 + 0.001 * abs(b)
    except Exception:
        return False


def build_record(prompt: str, gt_answer: str) -> dict:
    """Return a record matching huikang's training corpus format."""
    pred, cat, steps = solve(prompt)
    verified = pred is not None and _approx_match(pred, gt_answer)
    reasoning = " ".join(steps) if verified else GENERIC_TEMPLATES.get(
        cat, "I analyze the examples and apply the inferred rule to the query."
    )

    user = prompt + USER_SUFFIX
    # The opening "<think>\n" is added by apply_chat_template with
    # enable_thinking=True. So the completion starts with reasoning content
    # directly, then closes </think> and emits the boxed answer.
    completion = f"{reasoning}</think>\\boxed{{{gt_answer}}}<|im_end|>"

    return {
        "user": user,
        "completion": completion,
        "category": cat,
        "verified_cot": verified,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--out", default="data/sft_train.jsonl")
    args = ap.parse_args()

    n_total = 0
    n_verified = 0
    with open(args.train) as fi, open(args.out, "w") as fo:
        for row in csv.DictReader(fi):
            rec = build_record(row["prompt"], row["answer"])
            fo.write(json.dumps(rec) + "\n")
            n_total += 1
            n_verified += int(rec["verified_cot"])
    print(f"Wrote {n_total} records to {args.out}; verified CoT: {n_verified} ({n_verified/n_total*100:.1f}%).")


if __name__ == "__main__":
    main()
