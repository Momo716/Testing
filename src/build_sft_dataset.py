"""Build the SFT JSONL dataset for LoRA training.

For every training example we emit a single (system, user, assistant) record:
- system: short instruction telling the model to reason and box the answer.
- user: the original puzzle prompt.
- assistant: chain-of-thought reasoning + final \\boxed{answer}.

CoT generation policy
---------------------
1. Run the symbolic solver. If its prediction matches the ground-truth answer,
   we use the solver's reasoning steps (verified CoT).
2. If the solver fails or disagrees, we fall back to a short generic CoT that
   names the category and still emits the correct \\boxed{answer}. This still
   teaches the model the answer format and the per-category style.
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

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Solve the puzzle step by step, "
    "show your work briefly, then put the final answer inside \\boxed{...}."
)

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
        "The examples reveal a per-symbol rewrite rule (and possibly the "
        "removal of operator characters). I infer the mapping and apply it "
        "to the query expression."
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


def build_assistant_text(prompt: str, gt_answer: str) -> tuple[str, bool]:
    """Returns (assistant_text, verified)."""
    pred, cat, steps = solve(prompt)
    verified = pred is not None and _approx_match(pred, gt_answer)
    if verified:
        body = " ".join(steps)
    else:
        body = GENERIC_TEMPLATES.get(cat, "I analyze the examples and apply the inferred rule to the query.")
    return f"{body}\n\nFinal answer: \\boxed{{{gt_answer}}}", verified


def make_record(prompt: str, gt: str) -> dict:
    assistant, verified = build_assistant_text(prompt, gt)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "category": categorize(prompt),
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
            rec = make_record(row["prompt"], row["answer"])
            fo.write(json.dumps(rec) + "\n")
            n_total += 1
            n_verified += int(rec["verified_cot"])
    print(f"Wrote {n_total} records to {args.out}; verified CoT: {n_verified} ({n_verified/n_total*100:.1f}%).")


if __name__ == "__main__":
    main()
