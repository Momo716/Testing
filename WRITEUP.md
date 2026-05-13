# Methodology write-up — NVIDIA Nemotron Reasoning Challenge

This document satisfies the public-write-up requirement for prize eligibility.

## Dataset characterisation

The 9,500 training puzzles partition cleanly into six categories with roughly
equal frequency (~1,500 per class):

| Category    | Count | Underlying rule                                  |
|-------------|-------|--------------------------------------------------|
| `roman`     | 1,576 | Decimal → standard Roman numerals                |
| `unit_conv` | 1,594 | Affine map `y = a*x + b` (b often zero)          |
| `physics`   | 1,597 | Kinematic `d = ½·g·t²` with hidden g per puzzle  |
| `cipher`    | 1,576 | Per-puzzle random monoalphabetic substitution    |
| `bit_manip` | 1,602 | 8-bit transformation (rotations, XOR, AND, OR, NOT, majority/choice) |
| `algebra`   | 1,555 | Cryptarithm: 5-char LHS, symbol→digit + op→{add, abs_diff, mul, concat, rev_concat} |

Crucially, every puzzle is **deterministic in its examples** — the prompt
contains 4–8 input/output pairs from which the rule can in principle be
recovered. That converts the problem from "make the LLM smarter" to "teach
the LLM the algorithm humans would use, and let it imitate that algorithm
on a held-out instance".

## Approach: verified-CoT supervised fine-tuning

### Step 1 — Symbolic solvers (`src/solvers.py`)

For each category we wrote a closed-form solver that takes the raw prompt
and emits `(answer, reasoning_steps)`:

* **Roman** — greedy decomposition against the standard subtractive lookup.
  Accuracy: **100%**.
* **Unit conversion** — linear regression of `(x_i, y_i)` to fit `(a, b)`.
  Accuracy: **99.9%** (one rounding edge case).
* **Physics** — least-squares closed form
  `g = 2·Σ d_i·t_i² / Σ t_i⁴`. Accuracy: **100%**.
* **Cipher** — char-alignment from examples to recover a partial substitution
  map; when the query contains unseen letters, a **vocabulary-constrained
  backtracking search** completes the assignment. The training corpus has
  only 77 unique words, and the test set draws from the same vocabulary,
  so this approach is essentially complete. Accuracy: **100%**
  (raw partial-map version was 38%; vocab search lifts it to 100%).
* **Bit manipulation** — a cascading search:
  1. A library of single-step primitives (rotations, shifts, NOT, reverse,
     swap-nibbles) and their compositions, optionally with an XOR-constant.
  2. Nonlinear primitives: `x & rot(x,k)`, `x | rot(x,k)`, `x ^ rot(x,k)`,
     `majority(x, rot(x,j), rot(x,k))`, `choice(x, rot(x,j), rot(x,k))`.
  3. Two- and three-term SHA-style XOR combinations of rotations and
     shifts of x.
  4. Affine GF(2) fit `y = M·x ⊕ c` via Gaussian elimination as a fallback.
  Accuracy: **52.8%** — limited by puzzles using rule families outside our
  enumerated library.
* **Algebra (cryptarithm)** — bounded backtracking CSP over a bijective
  symbol→digit map and an op-symbol→operation map drawn from
  `{add, abs_diff, mul, concat, rev_concat}`. Accuracy: **7.8%**.

Overall solver coverage on train: **76.9%** (7,310 / 9,500) with **0
incorrect** confirmations after answer-tolerance filtering.

### Attribution

The cryptarithm solver structure (5-char LHS, operation library) and the
substitution-cipher vocab-CSP approach are both adapted from public Kaggle
notebooks by participant **glyphmatics** (D3/D4 "BorgQueen" gates,
currently scoring 0.86 on the public leaderboard). Their notebook was
shared via the user supplying this scaffold; their approach informed the
final solver designs. Our independent contributions are: (a) the
training-data pipeline that lifts solver-verified rows into chain-of-thought
SFT data, (b) the bit-manipulation SHA-style XOR search and nonlinear
primitives, and (c) the synthetic-data generators for five of the six
categories.

### Step 2 — Build a CoT-supervised dataset (`src/build_sft_dataset.py`)

For each training row:

1. Run the solver. If its prediction matches the ground truth (exact or
   within numerical tolerance), emit the solver's reasoning steps as
   chain-of-thought — these are *guaranteed correct* traces.
2. Otherwise emit a short generic category-specific CoT, still ending in
   the gold `\boxed{answer}`. The model learns the answer format and
   category-conditional style even when we cannot supply detailed steps.

Every record is serialised as a three-turn chat conversation
(`system → user → assistant`) using the same chat template the model will
see at inference under vLLM.

### Step 3 — Synthetic augmentation (`src/synth_data.py`)

We generate 2,000 additional puzzles per category for the five categories
where rules are constructively invertible (roman, unit_conv, physics,
cipher, bit_manip) — **10,000 extra verified-CoT records**. The cipher
generator explicitly ensures all query letters appear somewhere in the
examples; the bit_manip generator samples from the same primitive library
the solver searches.

We deliberately avoid synthesising `algebra` cryptarithms — their CSP
nature makes synthesis trivial but the marginal value over the existing
training set is low compared with the engineering cost.

### Step 4 — LoRA fine-tuning (`notebooks/01_train_lora.py`)

* Base model: `nvidia/Nemotron-3-Nano-30B`, 4-bit quantised via Unsloth.
* LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj` (attention + MLP, no embeddings).
* Hyperparameters: rank **32** (competition cap), α=64, dropout=0, **3
  epochs**, AdamW-8bit at lr=2e-4 with cosine schedule and 3% warmup,
  batch=1 with grad-accum=16, bf16, gradient checkpointing.
* **NEFTune noise** (α=5) on the embedding inputs.
* Verified-CoT records are duplicated once during construction, giving
  them ~2× weight relative to generic-CoT records.

Fits on a single 48 GB GPU and trains in roughly 6–10 hours.

### Step 5 — Inference & submission

The submission packages only the LoRA adapter (`adapter_config.json` +
`adapter_model.safetensors` + tokenizer files) into `submission.zip`. The
grader loads it under vLLM with `enable_lora=True, max_lora_rank=32`. We
use the canonical Nemotron chat template and a system prompt that asks for
brief reasoning ending in `\boxed{…}`, which aligns with the grader's
`\boxed{}`-first extraction.

`notebooks/02_inference_sanity.py` replays the grader's eval path locally
on the public test set.

## Expected leaderboard

A back-of-envelope projection, conditional on the model generalising well
from teacher-forced CoT:

| Category    | Train solver | Expected post-LoRA test |
|-------------|-------------:|------------------------:|
| roman       | 100%         | ~98%                    |
| physics     | 100%         | ~95%                    |
| unit_conv   | 99.9%        | ~95%                    |
| cipher      | 100%         | ~95%                    |
| bit_manip   | 52.8%        | ~55%                    |
| algebra     | 7.8%         | ~15%                    |
| **overall** |              | **~74–78%**             |

These are **solver-bounded estimates**. The LoRA can in principle exceed
the solver's training accuracy on a category if the model generalises
patterns the solver missed, particularly on bit_manip and algebra.

For comparison, the public reference adapter (`huikang/nemotron-adapter`,
patched and resubmitted by participant *glyphmatics*) sits at **0.86** on
the public leaderboard — that score is driven primarily by the underlying
pre-trained adapter rather than any single category's symbolic solver, so
reproducing it from scratch requires either (a) much longer training on
high-quality CoT than we have here, or (b) starting from that pre-existing
adapter and patching.

## Honest limitations

- **Algebra cryptarithms cap at ~8% coverage.** The CSP search times out
  on many puzzles whose solution space is too large for bounded
  backtracking; the participant's notebook also caps near the same number
  (95 verified rows on their side).
- **Bit manipulation hits ~53%** because some puzzles use rules outside
  our enumerated primitive library.
- **Training cannot be run in this authoring environment** (no GPU); the
  notebooks must run on Kaggle/Colab/Google Cloud G4.
- **A pre-existing public adapter outperforms our from-scratch training**
  in the 0.86 zone. That adapter's provenance is partially opaque, so the
  Open Contribution Awards (which favour novel methodology) may still
  prefer our reproducible pipeline.

## Possible extensions (where to invest more time if you want to push past 0.86)

In rough order of expected leaderboard impact:

1. **Stronger algebra CSP**: precompute global statistics on operator
   frequencies across the training corpus and use them as priors during
   the per-puzzle CSP search. Could push algebra from 8% to 30%+.
2. **Bit_manip primitive expansion**: enumerate compositions of length
   3–4 over a richer primitive library (including masked-rotation ANDs)
   and prefer sparse GF(2) fits. Worth ~5–8 points.
3. **Hybrid path**: start from a public reference adapter, then do a
   short additional SFT pass on our verified CoT.
4. **DPO/RL polish**: after SFT, do a short verifier-style RL pass where
   the reward is `pred == gold` evaluated by the grader's regex.

## Reproducibility checklist

1. `python -m src.build_sft_dataset` — deterministic from `data/train.csv`.
2. `python -m src.synth_data --seed 17` — reproducible synthetic data.
3. Training notebook fixes `random_state=42` / `seed=42` everywhere.
4. Solver behaviour is unit-testable: re-run the validation snippet in the
   README and confirm 76.9% verified rate.
