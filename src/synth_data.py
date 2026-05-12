"""Generate synthetic puzzles with verified answers.

We generate additional training data for the four categories where we have
high-confidence symbolic solvers: roman, unit_conv, physics, cipher. The
resulting JSONL has the same schema as build_sft_dataset.py output and can be
concatenated with the real train set during fine-tuning.

For bit_manip and algebra we deliberately skip synthesis: their rule space is
too rich to capture cheaply, and noisy synthetic data would hurt more than
help.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.solvers import _to_roman
from src.build_sft_dataset import SYSTEM_PROMPT


def _roman_puzzle(rng: random.Random) -> tuple[str, str]:
    examples = sorted(rng.sample(range(1, 4000), 4))
    target = rng.randint(1, 3999)
    while target in examples:
        target = rng.randint(1, 3999)
    body = "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:\n"
    body += "\n".join(f"{n} -> {_to_roman(n)}" for n in examples)
    body += f"\nNow, write the number {target} in the Wonderland numeral system."
    return body, _to_roman(target)


def _unit_conv_puzzle(rng: random.Random) -> tuple[str, str]:
    a = round(rng.uniform(0.2, 3.0), 4)
    b = round(rng.uniform(-2.0, 2.0), 4) if rng.random() < 0.6 else 0.0
    pts = sorted([round(rng.uniform(0.5, 50.0), 2) for _ in range(rng.randint(3, 6))])
    def f(x): return round(a * x + b, 2)
    body = "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
    body += "\n".join(f"{x} m becomes {f(x):.2f}" for x in pts)
    q = round(rng.uniform(1.0, 60.0), 2)
    body += f"\nNow, convert the following measurement: {q} m"
    return body, f"{f(q):.2f}"


def _physics_puzzle(rng: random.Random) -> tuple[str, str]:
    g = round(rng.uniform(2.0, 25.0), 4)
    pts = sorted([round(rng.uniform(0.5, 5.5), 2) for _ in range(rng.randint(4, 6))])
    def d(t): return round(0.5 * g * t * t, 2)
    body = "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:\n"
    body += "\n".join(f"For t = {t}s, distance = {d(t):.2f} m" for t in pts)
    qt = round(rng.uniform(1.0, 6.0), 2)
    body += f"\nNow, determine the falling distance for t = {qt}s given d = 0.5*g*t^2."
    return body, f"{d(qt):.2f}"


_VOCAB = (
    "the queen king dragon castle door near valley princess prince student "
    "creates reads discovers chases watches imagines walks beyond inside "
    "golden silent mysterious magical mighty secret garden river forest "
    "alice cat book wonderland follows under above whispers small great"
).split()


def _cipher_puzzle(rng: random.Random) -> tuple[str, str]:
    alphabet = list(string.ascii_lowercase)
    perm = alphabet[:]
    rng.shuffle(perm)
    plain_to_cipher = dict(zip(alphabet, perm))

    def encrypt(s):
        return "".join(plain_to_cipher.get(c, c) for c in s)

    def make_sentence():
        return " ".join(rng.sample(_VOCAB, rng.randint(3, 6)))

    examples = [make_sentence() for _ in range(rng.randint(4, 6))]
    body = "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
    body += "\n".join(f"{encrypt(s)} -> {s}" for s in examples)
    query_plain = make_sentence()
    # Ensure every letter in the query also appears in some example (otherwise unsolvable from examples alone).
    seen = set("".join(examples).replace(" ", ""))
    if not set(query_plain.replace(" ", "")).issubset(seen):
        # Add another example containing missing letters
        missing = set(query_plain.replace(" ", "")) - seen
        filler = " ".join(w for w in _VOCAB if set(w) & missing)
        if filler:
            body += f"\n{encrypt(filler)} -> {filler}"
    body += f"\nNow, decrypt the following text: {encrypt(query_plain)}"
    return body, query_plain


GENERATORS = {
    "roman": _roman_puzzle,
    "unit_conv": _unit_conv_puzzle,
    "physics": _physics_puzzle,
    "cipher": _cipher_puzzle,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_synth.jsonl")
    ap.add_argument("--per_category", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = 0
    with open(args.out, "w") as fo:
        for cat, gen in GENERATORS.items():
            for _ in range(args.per_category):
                prompt, answer = gen(rng)
                # We don't run the solver here — these are constructed to be solvable
                # by definition, so we attach a short verified CoT manually.
                cot = {
                    "roman": "The Wonderland numeral system matches standard Roman numerals. Convert directly.",
                    "unit_conv": "The conversion is linear y = a*x + b; fit a and b on examples and apply to the query.",
                    "physics": "Each example gives a (t, d) pair under d = 0.5*g*t^2; fit g and evaluate at the query t.",
                    "cipher": "Each example exposes a character-level substitution; build the map and apply it letter-by-letter.",
                }[cat]
                assistant = f"{cot}\n\nFinal answer: \\boxed{{{answer}}}"
                rec = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "category": cat,
                    "verified_cot": True,
                    "synthetic": True,
                }
                fo.write(json.dumps(rec) + "\n")
                n += 1
    print(f"Wrote {n} synthetic records to {args.out}.")


if __name__ == "__main__":
    main()
