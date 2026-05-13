# NVIDIA Nemotron Reasoning Challenge — solution scaffold

This repo turns the Wonderland reasoning benchmark into a **verified-CoT
supervised fine-tuning** dataset for a rank-32 LoRA adapter on
`nvidia/Nemotron-3-Nano-30B`, then trains and packages a competition
submission.

## Headline numbers (verified CoT coverage on train)

| Category    | Solver accuracy on train |
|-------------|--------------------------|
| roman       | **100.0%** (1576 / 1576) |
| physics     | **100.0%** (1597 / 1597) |
| cipher      | **100.0%** (1576 / 1576) |
| unit_conv   | **99.9%**  (1593 / 1594) |
| bit_manip   | **52.8%**  (846 / 1602)  |
| algebra     | **7.8%**   (122 / 1555)  |
| **overall** | **76.9%**  (7310 / 9500) |

That 76.9% becomes high-quality teacher-forced chain-of-thought used to
fine-tune the model. The remaining 23.1% still gets the correct boxed
answer paired with a short generic CoT (so the model learns the output
format and answer distribution even when we cannot supply detailed steps).

The **cipher solver** is at 100% via vocabulary-constrained backtracking
against the 77-word Wonderland corpus, and the **algebra solver** uses a
**cryptarithm CSP** (5-char LHS = `s0 s1 op s3 s4`, with symbol→digit and
op-symbol→{add, abs_diff, mul, concat, rev_concat}) ported from a public
Kaggle notebook by participant *glyphmatics*.

## Two paths to a strong submission

This scaffold supports two distinct strategies for producing the final
LoRA adapter. **Read both before deciding.**

### Path A — Train from scratch on our verified-CoT dataset

* Time: ~6–10 h on a single H100/A100 GPU.
* Expected leaderboard: **~74–78%** (solver-bounded).
* Pros: fully reproducible, uses only this repo + the base model, satisfies
  the "Best Data/Synthetic Data Method" Open Contribution Award criteria.
* Cons: bit_manip and algebra are capped by solver coverage at training
  time.
* How: follow steps 1–5 below.

### Path B — Start from a public pre-trained adapter (e.g. `huikang/nemotron-adapter`) and patch

* Time: ~30 min on Kaggle (no training, only SVD compression to rank-32).
* Expected leaderboard: **~0.86** (matches the public reference).
* Pros: known top-tier score immediately available.
* Cons: depends on a third-party adapter (verify license / origin); less
  competitive for the "Best Fine-tuning Method" award since you did not
  train it yourself.
* How: clone the public notebook directly. The participant
  `glyphmatics` published a complete recipe; see `WRITEUP.md` for
  attribution and reproduction details.

A **hybrid** is also viable: start from the public adapter (Path B) then
do a short additional SFT pass on our verified-CoT dataset to capture
extra wins on cipher / algebra. This is the highest-upside option but
requires care to avoid catastrophic forgetting.

## Layout

```
data/
  train.csv             # competition training set (9,500 examples)
  test.csv              # public sample test set (3 examples)
  cipher_vocab.txt      # 77-word vocab inferred from train (used by solver)
  sft_train.jsonl       # generated: CoT-augmented SFT records from train.csv
  sft_synth.jsonl       # generated: 10,000 synthetic verified-CoT records
src/
  categorize.py         # 6-way puzzle classifier
  solvers.py            # symbolic solvers for each category (incl. cryptarithm)
  build_sft_dataset.py  # turns train.csv into JSONL with verified CoT
  synth_data.py         # generates extra puzzles with known answers
notebooks/
  01_train_lora.py      # Path A: Kaggle-runnable LoRA fine-tuning
  02_inference_sanity.py# Sanity-check: vLLM evaluator mirroring the grader
WRITEUP.md              # methodology write-up (required for prize eligibility)
```

## End-to-end workflow (Path A)

### 1. Build the SFT dataset locally
```bash
python -m src.build_sft_dataset --train data/train.csv --out data/sft_train.jsonl
python -m src.synth_data --out data/sft_synth.jsonl --per_category 2000
```

### 2. Upload to Kaggle

Create a Kaggle dataset named `wonderland-sft` containing:
- `sft_train.jsonl`
- `sft_synth.jsonl`
- `test.csv` (for the sanity-check notebook)

Attach `nvidia/Nemotron-3-Nano-30B` to the notebook.

### 3. Train the LoRA adapter

Open `notebooks/01_train_lora.py` in a Kaggle notebook session with a
Blackwell/H100/A100 GPU, paste each `# %%` block into its own cell, and run
all cells. The final cell writes `/kaggle/working/submission.zip`.

### 4. Sanity-check before submitting

Run `notebooks/02_inference_sanity.py`. It loads the adapter under vLLM with
the exact eval parameters the grader uses (`max_lora_rank=32`,
`max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_num_seqs=64`,
`gpu_memory_utilization=0.85`, `max_model_len=8192`).

### 5. Submit

Upload `submission.zip` to the competition.
