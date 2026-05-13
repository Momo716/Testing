"""
LoRA fine-tuning for the NVIDIA Nemotron Reasoning Challenge.

Recipe matches the Progress Prize winner (huikang) on the critical details:
  * Base model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (Kaggle mirror at
    /kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default)
  * Chat template: tokenizer.apply_chat_template([{"role":"user", ...}],
                   add_generation_prompt=True, enable_thinking=True)
  * Completion format: <reasoning></think>\\boxed{answer}<|im_end|>
  * Loss masked: prompt tokens excluded, completion tokens included.
  * Max sequence length: 8192 (same as inference grader).
  * LoRA rank: 32 (competition cap).

Designed for a Kaggle notebook with a single H100/A100/Blackwell GPU
(>=48 GB VRAM). Paste each `# %%` chunk into its own Kaggle cell.
"""

# %% [markdown]
# # LoRA fine-tune Nemotron-3-Nano-30B-A3B on Wonderland reasoning puzzles

# %%
# !pip install -q unsloth "transformers>=4.46" "peft>=0.13" "trl>=0.11" datasets accelerate bitsandbytes

# %%
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# Kaggle's pre-attached Nemotron model is at this path. If you're running
# on a different platform, point this at the HuggingFace repo id or a
# local checkpoint.
MODEL_NAME = "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default"
# Fall back to HF if the Kaggle path doesn't exist
if not Path(MODEL_NAME).exists():
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

MAX_SEQ_LEN = 8192
LORA_RANK = 32
OUTPUT_DIR = "/kaggle/working/lora_adapter"
TRAIN_FILES = [
    "/kaggle/input/wonderland-sft/sft_train.jsonl",
    "/kaggle/input/wonderland-sft/sft_synth.jsonl",
]

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

# Nemotron is an MoE (A3B); include in_proj/out_proj alongside standard targets
# so the LoRA touches the expert projections too. Do NOT target the router.
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

# %%
# Load our SFT records (user + completion + flags). We build the full text
# by calling the Nemotron chat template with enable_thinking=True, then
# concatenating the completion. The completion already ends with
# `</think>\boxed{answer}<|im_end|>`.
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
    """Apply Nemotron chat template to the user turn and append completion."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["user"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return {"text": prompt + ex["completion"]}


ds = Dataset.from_list(records).map(render, remove_columns=list(records[0].keys()))

# %%
# Upweight verified-CoT records (correct, detailed reasoning) by duplicating
# once. Generic-CoT records still teach the answer format and category style.
verified = [r for r in records if r.get("verified_cot")]
extra = Dataset.from_list(verified).map(render, remove_columns=list(verified[0].keys()))
ds = Dataset.from_list(list(ds) + list(extra))
print(f"After verified-CoT upweighting: {len(ds)} examples.")
ds = ds.shuffle(seed=42)

# %%
# Training args. Chosen to fit a single 48-80 GB GPU and converge in ~6-12 h.
# * 3 epochs of SFT.
# * NEFTune embedding noise (alpha=5) — documented free accuracy boost.
# * Cosine schedule with 3% warmup.
# * Effective batch = 1 * 16 = 16; matches Unsloth-recommended QLoRA throughput
#   on this model size. Huikang used larger batches with Tinker; this is the
#   accessible equivalent.
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
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Sanity check before packaging
adapter_cfg = Path(OUTPUT_DIR) / "adapter_config.json"
assert adapter_cfg.exists(), "Missing adapter_config.json — vLLM will reject."
print("adapter_config.json present.")

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
