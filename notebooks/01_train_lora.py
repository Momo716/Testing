"""
LoRA fine-tuning script for Nemotron-3-Nano-30B (rank=32).

Designed to run on a Kaggle notebook with a single H100/A100/Blackwell GPU
(>=48 GB VRAM recommended). Paste each `# %%` chunk into its own Kaggle cell.

Usage on Kaggle
---------------
1) Attach the `nvidia/Nemotron-3-Nano-30B` model to the notebook.
2) Attach this repo as a Kaggle dataset (so `data/sft_train.jsonl` and
   `data/sft_synth.jsonl` are available under /kaggle/input/...).
3) Run all cells. The final cell saves the LoRA adapter to
   `/kaggle/working/lora_adapter/` and packages `submission.zip`.

Library versions known to work
------------------------------
unsloth==2026.4+, transformers>=4.46, peft>=0.13, trl>=0.11,
torch>=2.4 with CUDA 12.4.
"""

# %% [markdown]
# # LoRA fine-tune Nemotron-3-Nano-30B on Wonderland reasoning puzzles

# %%
# !pip install -q unsloth "transformers>=4.46" "peft>=0.13" "trl>=0.11" datasets accelerate bitsandbytes

# %%
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "nvidia/Nemotron-3-Nano-30B"  # match the exact HF repo id used in the competition
MAX_SEQ_LEN = 4096
LORA_RANK = 32
OUTPUT_DIR = "/kaggle/working/lora_adapter"
TRAIN_FILES = [
    "/kaggle/input/wonderland-sft/sft_train.jsonl",
    "/kaggle/input/wonderland-sft/sft_synth.jsonl",
]

# %%
# Load base model in 4-bit; LoRA on attention + MLP.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Use the chat template that matches Nemotron's pretraining.
try:
    tokenizer = get_chat_template(tokenizer, chat_template="nemotron")
except Exception:
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")


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


def format_example(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


ds = Dataset.from_list(records).map(format_example, remove_columns=["messages"])

# %%
# Upweight verified-CoT records (they have correct, detailed reasoning) by
# duplicating them once. Generic-CoT records still teach the answer format.
verified_idx = [i for i, r in enumerate(records) if r.get("verified_cot")]
extra = Dataset.from_list([records[i] for i in verified_idx]).map(
    format_example, remove_columns=["messages"]
)
ds = Dataset.from_list(list(ds) + list(extra))
print(f"After upweighting verified-CoT records: {len(ds)} training examples.")
ds = ds.shuffle(seed=42)

# %%
# Training args — picked to fit on a single 48GB GPU and converge in ~6-10h.
# - 3 epochs (was 2): more pattern reinforcement.
# - NEFTune noise (alpha=5): documented free accuracy boost on instruct tuning.
# - Cosine schedule with 3% warmup.
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_ratio=0.03,
    num_train_epochs=3,
    learning_rate=2e-4,
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
    neftune_noise_alpha=5,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=args,
)

trainer.train()

# %%
# Save the LoRA adapter (NOT the merged model). vLLM expects this directory
# with adapter_config.json + adapter_model.safetensors.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Sanity-check that adapter_config.json exists
adapter_cfg = Path(OUTPUT_DIR) / "adapter_config.json"
assert adapter_cfg.exists(), "Missing adapter_config.json — vLLM will reject the submission."
print("adapter_config.json present.")

# %%
# Package submission.zip in the layout the grader expects.
import zipfile

zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in Path(OUTPUT_DIR).rglob("*"):
        if p.is_file():
            zf.write(p, arcname=p.relative_to(OUTPUT_DIR))
print(f"Submission packaged at {zip_path}")
# Print zip contents so you can verify before submitting
with zipfile.ZipFile(zip_path) as zf:
    for n in zf.namelist():
        print(" ", n)
