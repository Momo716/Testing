"""
Path C — hybrid: warm-start from huikang's public adapter, then polish on
our verified-CoT dataset for one epoch at a low learning rate.

Strategy:
  1. Load base model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
  2. Attach huikang's public rank-32 LoRA adapter (~0.85 baseline).
  3. Continue training that same adapter for 1 short epoch at lr=5e-5 on
     our verified-CoT data (cipher=100%, cryptarithm=8%, etc.) to add our
     wins without forgetting huikang's broader coverage.
  4. Save and package as submission.zip.

Expected result: 0.87+ leaderboard score. ~2 h on a single H100/A100.

How to attach inputs in the Kaggle notebook sidebar
---------------------------------------------------
  Add Input → Models → search "nemotron-3-nano-30b-a3b-bf16" → add
       (this is the base model)
  Add Input → Models → search "huikang nemotron" or "nemotron adapter" →
       add huikang's published LoRA adapter
  Add Input → Datasets → search "wonderland-sft" → add (your own dataset
       containing sft_train.jsonl and sft_synth.jsonl that you uploaded
       earlier)

Paste each `# %%` chunk into its own Kaggle cell, then Run All.
"""

# %%
# !pip install -q unsloth "transformers>=4.46" "peft>=0.13" "trl>=0.11" datasets accelerate bitsandbytes

# %%
import json, os, glob
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# Auto-detect input paths under /kaggle/input
def _find(*candidates):
    for c in candidates:
        if Path(c).exists():
            return c
    # Glob fallback under /kaggle/input
    hits = []
    for c in candidates:
        hits.extend(glob.glob(c))
    return hits[0] if hits else None


BASE_MODEL = _find(
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16",
    "/kaggle/input/*/nemotron-3-nano-30b*",
) or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

WARM_START_ADAPTER = _find(
    "/kaggle/input/nemotron-adapter/transformers/default",
    "/kaggle/input/huikang-nemotron-adapter",
    "/kaggle/input/*/nemotron-adapter*",
    "/kaggle/input/*/transformers/default/*",
)
assert WARM_START_ADAPTER, "Could not find huikang's adapter under /kaggle/input. Attach it via 'Add Input → Models'."

print("Base model:", BASE_MODEL)
print("Warm-start adapter:", WARM_START_ADAPTER)

MAX_SEQ_LEN = 8192
LORA_RANK = 32
OUTPUT_DIR = "/kaggle/working/lora_adapter"
TRAIN_FILES = [
    "/kaggle/input/wonderland-sft/sft_train.jsonl",
    "/kaggle/input/wonderland-sft/sft_synth.jsonl",
]

# %%
# Load base model in 4-bit, then attach huikang's adapter as the trainable LoRA.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

# Attach the warm-start adapter as a continuable PEFT model. is_trainable=True
# makes the adapter weights themselves the parameters we optimize in this run.
model = PeftModel.from_pretrained(model, WARM_START_ADAPTER, is_trainable=True)
model.print_trainable_parameters()

# %%
def load_jsonl(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


records = load_jsonl(TRAIN_FILES)
print(f"Loaded {len(records)} SFT records.")


def render(ex):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["user"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return {"text": prompt + ex["completion"]}


ds = Dataset.from_list(records).map(render, remove_columns=list(records[0].keys()))

# Use ONLY verified-CoT records for the polish pass: we want clean,
# high-signal data to avoid catastrophic forgetting.
verified = [r for r in records if r.get("verified_cot")]
print(f"Polishing on {len(verified)} verified-CoT records (of {len(records)} total).")
polish_ds = Dataset.from_list(verified).map(render, remove_columns=list(verified[0].keys()))
polish_ds = polish_ds.shuffle(seed=42)

# %%
# Polish-pass training args. Differences vs Path A (01_train_lora.py):
#   * num_train_epochs=1     — one quick pass, not three
#   * learning_rate=5e-5     — 4x lower than Path A to preserve huikang's weights
#   * warmup_ratio=0.0       — no warmup for a short continuation
#   * no NEFTune             — avoid extra noise on top of a strong starting point
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_ratio=0.0,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=20,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=polish_ds,
    args=args,
)

trainer.train()

# %%
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Polished adapter saved to {OUTPUT_DIR}")

adapter_cfg = Path(OUTPUT_DIR) / "adapter_config.json"
assert adapter_cfg.exists(), "Missing adapter_config.json — vLLM will reject."

# Confirm rank is still 32 (PEFT preserves it during continuation)
with open(adapter_cfg) as f:
    cfg = json.load(f)
print("LoRA rank in saved config:", cfg.get("r"))
assert cfg.get("r", 32) <= 32, f"Rank exceeds competition cap: {cfg.get('r')}"

# %%
import zipfile
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in Path(OUTPUT_DIR).rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(OUTPUT_DIR))
print(f"Submission packaged at {zip_path}")
with zipfile.ZipFile(zip_path) as zf:
    for n in zf.namelist():
        print(" ", n)
