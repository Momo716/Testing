# NVIDIA Nemotron Reasoning Challenge — solution scaffold

This repo turns the Wonderland reasoning benchmark into a **verified-CoT
supervised fine-tuning** dataset for a rank-32 LoRA adapter on
`nvidia/Nemotron-3-Nano-30B`, then trains and packages a competition
submission.

## Layout

```
data/
  train.csv             # competition training set (9,500 examples)
  test.csv              # public sample test set (3 examples)
  sft_train.jsonl       # generated: CoT-augmented SFT records from train.csv
  sft_synth.jsonl       # generated: synthetic verified-CoT records
src/
  categorize.py         # 6-way puzzle classifier
  solvers.py            # symbolic solvers for each category
  build_sft_dataset.py  # turns train.csv into JSONL with verified CoT
  synth_data.py         # generates extra puzzles with known answers
notebooks/
  01_train_lora.py      # Kaggle-runnable LoRA fine-tuning (Unsloth + TRL)
  02_inference_sanity.py# vLLM evaluation harness mirroring the grader
WRITEUP.md              # methodology write-up (required for prize eligibility)
```

## End-to-end workflow

### 1. Build the SFT dataset locally
```bash
python -m src.build_sft_dataset --train data/train.csv --out data/sft_train.jsonl
python -m src.synth_data --out data/sft_synth.jsonl --per_category 2000
```

The first command produces verified chain-of-thought for **64.2%** of the
9,500 train examples (Roman/Unit/Physics ~100%, Cipher 38%, BitManip 45%,
Algebra ≈0%); the remaining 35.8% receive a short generic CoT plus the
correct boxed answer so the model still learns the output format.

### 2. Upload to Kaggle

Create a Kaggle dataset that contains:
- `sft_train.jsonl`
- `sft_synth.jsonl`
- `test.csv` (for the sanity-check notebook)

Attach `nvidia/Nemotron-3-Nano-30B` to the notebook (model card or HF mirror).

### 3. Train the LoRA adapter

Open `notebooks/01_train_lora.py` in a Kaggle notebook session with a
Blackwell/H100/A100 GPU, paste each `# %%` block into its own cell, and run
all cells. The final cell writes `/kaggle/working/submission.zip`.

### 4. (Optional) Sanity-check before submitting

Run `notebooks/02_inference_sanity.py`. It loads the adapter under vLLM with
the exact eval parameters the grader uses (`max_lora_rank=32`,
`max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_num_seqs=64`,
`gpu_memory_utilization=0.85`, `max_model_len=8192`) and emits predictions
for the public test set.

### 5. Submit

Upload `submission.zip` (containing `adapter_config.json` +
`adapter_model.safetensors` + tokenizer files) to the competition.

## Why this works

Every Wonderland puzzle has a deterministic underlying rule, and the prompt
itself contains enough examples to recover that rule programmatically for
most categories. We exploit this asymmetry: at training time we use
**symbolic solvers** to generate high-fidelity reasoning traces, and the
LoRA learns to imitate the same reasoning shape at inference. See
`WRITEUP.md` for the full methodology.
