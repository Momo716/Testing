# Methodology write-up — NVIDIA Nemotron Reasoning Challenge

This document satisfies the public-write-up requirement for prize eligibility.

## Dataset characterisation

The 9,500 training puzzles partition cleanly into six categories, each with
roughly equal frequency (~1,500 examples per class):

| Category    | Count | Underlying rule                                  |
|-------------|-------|--------------------------------------------------|
| `roman`     | 1,576 | Decimal → standard Roman numerals                |
| `unit_conv` | 1,594 | Affine map `y = a*x + b` (b often zero)          |
| `physics`   | 1,597 | Kinematic `d = ½·g·t²` with hidden g per puzzle  |
| `cipher`    | 1,576 | Per-puzzle random monoalphabetic substitution    |
| `bit_manip` | 1,602 | 8-bit transformation (often GF(2)-affine)        |
| `algebra`   | 1,555 | Symbol-level rewrite (length-varying)            |

Crucially, every puzzle is **deterministic in its examples** — the prompt
contains 4-8 input/output pairs from which the rule can in principle be
recovered. That changes the problem from "make the LLM smarter" to "teach
the LLM the algorithm humans would use, and let it imitate that algorithm
on a held-out instance".

## Approach: verified-CoT supervised fine-tuning

### Step 1 — Symbolic solvers (`src/solvers.py`)

For each category we wrote a closed-form solver that takes the raw prompt
and emits `(answer, reasoning_steps)`:

* **Roman**: greedy decomposition against the standard subtractive lookup.
  Accuracy: **100%** on train.
* **Unit conversion**: linear regression of `(x_i, y_i)` to fit `(a, b)`,
  then evaluate at the query. Accuracy: **99.9%** (one rounding edge case).
* **Physics**: least-squares fit of `g` to `d_i = ½·g·t_i²` (closed form
  `g = 2·Σ d_i·t_i² / Σ t_i⁴`), then evaluate. Accuracy: **100%**.
* **Cipher**: align cipher↔plain pairs at the character level to recover
  the substitution map, then apply. Accuracy: **38%** — bounded by example
  letter coverage (when the query contains letters not seen in any
  example, no purely-symbolic solver can succeed).
* **Bit manipulation**: cascading search. First try a compact symbolic
  library (rotations, shifts, NOT, swap-nibbles, single XOR-constant);
  fall back to a general GF(2) affine fit `y = M·x ⊕ c` solved by Gaussian
  elimination over GF(2). Accuracy: **45%** (some example sets are
  underdetermined for a fully general affine map).
* **Algebra**: per-character substitution with optional operator removal.
  Coverage on this category is poor (~0.1%) because the rewrites are
  variable-length and structurally richer than a pure substitution. We
  treat it as a generic case below.

Overall solver coverage on train: **64.2%** (6,098 / 9,500) with **0
incorrect** confirmations after answer-tolerance filtering.

### Step 2 — Build a CoT-supervised dataset (`src/build_sft_dataset.py`)

For each training row:

1. Run the solver. If its prediction matches the ground truth (exact or
   within numerical tolerance), emit the solver's reasoning steps as
   chain-of-thought — these are *guaranteed correct* traces.
2. Otherwise emit a short generic CoT keyed off the puzzle category, still
   ending in the gold `\boxed{answer}`. The model learns the answer format
   and category-conditional style even when we cannot supply detailed
   reasoning.

Every record is serialised as a three-turn chat conversation
(`system → user → assistant`) which is exactly the chat template the model
will see at inference under vLLM.

### Step 3 — Synthetic augmentation (`src/synth_data.py`)

For the four categories where the rule is fully invertible (roman,
unit_conv, physics, cipher) we generate 2,000 additional puzzles per
category with known answers (8,000 extra records). The cipher generator
explicitly ensures all query letters appear somewhere in the example
sentences, eliminating the example-coverage failure mode that limited the
original cipher solver to 38%. This expands the SFT set to **17,500
records**.

We deliberately avoid synthesising `bit_manip` and `algebra` — generated
puzzles whose rules our solvers cannot capture would just inject noise.

### Step 4 — LoRA fine-tuning (`notebooks/01_train_lora.py`)

* Base model: `nvidia/Nemotron-3-Nano-30B`, 4-bit quantised loading via
  Unsloth.
* LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj` — i.e. attention + MLP, no embeddings.
* Hyperparameters: rank **32** (competition cap), α=64, dropout=0, 2 epochs,
  AdamW-8bit at lr=2e-4 with cosine schedule and 3% warmup, batch=1 with
  grad-accum=16, bf16, gradient checkpointing.
* Verified-CoT records are duplicated once during construction, giving
  them roughly 2× weight relative to generic-CoT records.

This combination fits on a single 48 GB GPU and trains in roughly 4–6 hours
on a Blackwell-class GPU.

### Step 5 — Inference & submission

The submission packages just the LoRA adapter (`adapter_config.json` +
`adapter_model.safetensors` + tokenizer files) into `submission.zip`. The
grader loads it under vLLM with `enable_lora=True, max_lora_rank=32`. We
use the canonical Nemotron chat template and a system prompt that asks for
brief reasoning ending in `\boxed{…}`, which lines up exactly with the
grader's `\boxed{}`-first extraction.

`notebooks/02_inference_sanity.py` replays the grader path locally on the
public test set so you can spot regressions before submitting.

## Reproducibility checklist

1. `python -m src.build_sft_dataset` — deterministic from `data/train.csv`.
2. `python -m src.synth_data --seed 17` — reproducible synthetic data.
3. Training notebook fixes `random_state=42` / `seed=42` everywhere.
4. Solver behaviour is unit-testable: run `python -m src.solvers` against
   train.csv and confirm 64.2% verified rate.

## Possible extensions

* **Cipher**: layer English-frequency analysis on top of the partial map
  to fill in unseen letters; this alone could push cipher from 38% to
  ~90%+ before any LoRA effects.
* **Bit manipulation**: enumerate compositions of length 3–4 over a richer
  primitive library (e.g. include `x & rot(x, k)`, `x | rot(x, k)`,
  majority(x, rot(x,k), rot(x,j))) to lift the symbolic-coverage ceiling.
* **Algebra**: parse the LHS as a prefix expression over a small operator
  set and search for a per-operator semantics that matches the RHS — a
  short symbolic-regression style sweep would likely cover most cases.
* **RL polish**: after SFT, do a short verifier-style RL pass where the
  reward is `pred == gold` evaluated by the same regex the grader uses;
  this typically squeezes out the last 1–3 points.
