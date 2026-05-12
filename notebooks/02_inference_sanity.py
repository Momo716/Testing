"""
Local inference sanity check for the trained LoRA adapter.

The competition grades submissions by loading the adapter under vLLM and
prompting the model on the hidden test set; this script reproduces that path
on the public test.csv so you can estimate score before submitting.

Run on Kaggle after 01_train_lora.py has produced /kaggle/working/lora_adapter.
"""

# %%
# !pip install -q vllm

# %%
import csv
import re
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_NAME = "nvidia/Nemotron-3-Nano-30B"
ADAPTER_DIR = "/kaggle/working/lora_adapter"
TEST_CSV = "/kaggle/input/wonderland-sft/test.csv"  # or path to the competition test set

SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=7680,
)

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Solve the puzzle step by step, "
    "show your work briefly, then put the final answer inside \\boxed{...}."
)

# %%
llm = LLM(
    model=MODEL_NAME,
    enable_lora=True,
    max_lora_rank=32,
    max_num_seqs=64,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
)
lora_req = LoRARequest("wonderland", 1, ADAPTER_DIR)

# %%
def build_chat(prompt: str) -> str:
    # Use the same chat template as training. If your tokenizer has
    # apply_chat_template, prefer that.
    return (
        "<|system|>\n" + SYSTEM_PROMPT + "\n"
        "<|user|>\n" + prompt + "\n"
        "<|assistant|>\n"
    )


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMERIC = re.compile(r"-?\d+\.?\d*")


def extract_answer(text: str) -> str:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = _NUMERIC.findall(text)
    return nums[-1] if nums else text.strip()


# %%
rows = list(csv.DictReader(open(TEST_CSV)))
prompts = [build_chat(r["prompt"]) for r in rows]

outputs = llm.generate(prompts, SAMPLING, lora_request=lora_req)
preds = []
for r, o in zip(rows, outputs):
    text = o.outputs[0].text
    ans = extract_answer(text)
    preds.append({"id": r["id"], "answer": ans})
    print(r["id"], "->", ans[:60])

Path("/kaggle/working/predictions.json").write_text(json.dumps(preds, indent=2))
print("Wrote predictions.json")
