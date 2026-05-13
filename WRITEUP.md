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
| `algebra`   | 1,555 | Cryptarithm: 5-char LHS, symbol→digit + op→arithmetic |

Crucially, every puzzle is **deterministic in its examples** — the prompt
contains 4–8 input/output pairs from which the rule can in principle be
recovered. That converts the problem from "make the LLM smarter" to "teach
the LLM the algorithm humans would use, and let it imitate that algorithm
on a held-out instance".

## Approach: verified-CoT supervised fine-tuning

Our pipeline mirrors the Progress Prize winning submission by **huikang**
(public code: github.com/tonghuikang/nemotron, public Kaggle notebook:
`huikang/end-to-end-finetuning-for-lb-0-85`). The two pipelines are
methodologically aligned:

| Component             | Huikang's pipeline           | Ours                          |
|-----------------------|------------------------------|-------------------------------|
| Per-category reasoner | `reasoners/*.py`             | `src/solvers.py`              |
| Synthetic data        | `augmentation.py`            | `src/synth_data.py`           |
| Corpus builder        | `corpus.py`                  | `src/build_sft_dataset.py`    |
| Trainer               | `train_sft.py` (Tinker)      | `notebooks/01_train_lora.py` (Unsloth + TRL) |
| Chat template         | `enable_thinking=True`       | `enable_thinking=True`        |
| Completion suffix     | `</think>\boxed{...}<\|im_end\|>` | `</think>\boxed{...}<\|im_end\|>` |
| LoRA rank             | 32                           | 32                            |
| Max seq length        | 8192                         | 8192                          |

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
  Accuracy: **52.8%**.
* **Algebra (cryptarithm)** — bounded backtracking CSP over a bijective
  symbol→digit map and an op-symbol→operation map drawn from a 16-operation
  library: `{add, abs_diff, mul, concat, rev_concat, weighted_add,
  shift_concat, digit_sum, digit_prod, pairwise_sum, pairwise_prod,
  square_sum, diff_squared, xor, or, and}`. Accuracy: **8.0%**.

Overall solver coverage on train: **77.0%** (7,313 / 9,500) with **0
incorrect** confirmations after answer-tolerance filtering.

### Step 2 — Build a CoT-supervised dataset (`src/build_sft_dataset.py`)

For each training row:

1. Run the solver. If its prediction matches the ground truth (exact or
   within numerical tolerance), emit the solver's reasoning steps as
   chain-of-thought — these are *guaranteed correct* traces.
2. Otherwise emit a short generic category-specific CoT, still ending in
   the gold `\boxed{answer}`.

Each record is serialised as `{user, completion, category, verified_cot}`.
The user message is the puzzle prompt plus the directive *"Please put your
final answer inside `\boxed{}`."* — matching huikang's prompt construction.
At training time the user message is wrapped by
`tokenizer.apply_chat_template([{"role":"user", ...}],
add_generation_prompt=True, enable_thinking=True)` which produces the
Nemotron-canonical opener ending in `<think>\n`. The completion field
provides the rest: `<reasoning></think>\boxed{<answer>}<|im_end|>`.

### Step 3 — Synthetic augmentation (`src/synth_data.py`)

We generate 2,000 additional puzzles per category for the five categories
where rules are constructively invertible (roman, unit_conv, physics,
cipher, bit_manip) — **10,000 extra verified-CoT records**. The cipher
generator explicitly ensures all query letters appear somewhere in the
examples; the bit_manip generator samples from the same primitive library
the solver searches.

### Step 4 — LoRA fine-tuning (`notebooks/01_train_lora.py`)

* Base model: **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** (Kaggle
  mirror: `/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default`).
* Loaded in **4-bit** via Unsloth for memory savings.
* LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj` (attention + MLP). MoE router is **not** trained.
* Rank **32** (competition cap), α=64, dropout=0.
* **3 epochs**, AdamW-8bit at lr=2e-4 with cosine schedule and 3% warmup.
* Batch=1 × grad-accum=16, bf16, gradient checkpointing.
* **NEFTune noise** (α=5) on the embedding inputs.
* Max sequence length **8192** (matches the grader's `max_model_len`).
* Verified-CoT records are duplicated once during construction, giving
  them ~2× weight relative to generic-CoT records.

Fits on a single 48 GB GPU and trains in roughly 6–10 hours.

### Step 5 — Inference & submission

The submission packages only the LoRA adapter (`adapter_config.json` +
`adapter_model.safetensors` + tokenizer files) into `submission.zip`. The
grader loads it under vLLM with `enable_lora=True, max_lora_rank=32`. Our
inference notebook (`notebooks/02_inference_sanity.py`) replays the grader's
eval path locally using the *same* chat template the model was trained with.

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
| algebra     | 8.0%         | ~15%                    |
| **overall** |              | **~74–78%**             |

For reference, the public 0.85 notebook from huikang reaches this same
range; the 0.86 figure for `glyphmatics` comes from starting from
huikang's pre-trained adapter and patching it (rank-32 SVD compression),
which is a separate path described below.

## Three paths to a strong submission

### Path A — Train from scratch with this scaffold

* Time: ~6–10 h on a single H100/A100.
* Expected: **~74–78%** (matches huikang's public notebook).
* Pros: fully reproducible, "Best Data/Synthetic Data Method" Open
  Contribution Award qualified.
* How: follow steps 1–5 in `README.md`.

### Path B — Reuse an existing public adapter

* Time: ~30 min (no training, just SVD compression to rank-32).
* Expected: **~0.85–0.86** (matches the public reference).
* Pros: known top-tier score immediately.
* Cons: dependent on huikang/glyphmatics' adapter; less novel for prize
  consideration; verify license.

### Path C (RECOMMENDED for winning) — Hybrid

1. Start from huikang's public adapter as a warm-start (Path B).
2. Apply one extra short SFT pass on **our** verified-CoT dataset using
   `notebooks/01_train_lora.py` with `learning_rate=5e-5`,
   `num_train_epochs=1` to capture our cipher → 100% and our improved
   cryptarithm coverage without forgetting his weights.
3. Submit the resulting merged-then-compressed adapter.
* Time: ~2 h.
* Expected: **0.87+** — best realistic shot at top-3.

## Attribution

The cryptarithm solver structure (5-char LHS, operation library) and the
substitution-cipher vocab-CSP approach are adapted from a public Kaggle
notebook by participant `glyphmatics`. The training pipeline format
(chat template, `<think>` tags, completion suffix, model path) follows
`huikang`'s open-source Progress Prize submission
(github.com/tonghuikang/nemotron). Independent contributions in this
scaffold are: (a) the SHA-style XOR search and nonlinear primitives in
the bit-manipulation solver, (b) the 16-operation cryptarithm library
(beyond glyphmatics' 5-op set), and (c) the synthetic-data generators
for five of the six categories.

## Honest limitations

- **Algebra cryptarithms cap at ~8% coverage.** The CSP search times out
  on many puzzles whose solution space is too large for bounded
  backtracking, even with an expanded operation library; expanding it
  further gives diminishing returns.
- **Bit manipulation hits ~53%** because some puzzles use rules outside
  our enumerated primitive library.
- **Training cannot be run in this authoring environment** (no GPU); the
  notebooks must run on Kaggle/Colab/Google Cloud G4.
- **Path A alone is unlikely to beat 0.85.** Path C (start from huikang +
  one polish epoch on our data) is the realistic path to a winning score.

## Reproducibility checklist

1. `python -m src.build_sft_dataset` — deterministic from `data/train.csv`.
2. `python -m src.synth_data --seed 17` — reproducible synthetic data.
3. Training notebook fixes `random_state=42` / `seed=42` everywhere.
4. Solver behaviour is unit-testable: re-run the validation snippet in the
   README and confirm 77.0% verified rate.

## References

* `https://github.com/tonghuikang/nemotron` — Progress Prize submission.
* `https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85`
  — public Kaggle notebook reaching 0.85.
* `https://blog.huikang.dev/2026/05/02/nemotron-progress-prize.html`
  — Huikang's methodology blog post.
* `https://unsloth.ai/docs/models/nemotron-3` — Unsloth's Nemotron-3
  fine-tuning guide (chat template, special tokens, MoE caveats).
