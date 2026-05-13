"""
Local inference sanity check for the trained LoRA adapter.

Reproduces the grader's exact vLLM pipeline locally. Uses the same chat
template (enable_thinking=True) the model was trained with, then extracts
the final \\boxed{...} answer.
"""

# %%
# !pip install -q vllm

# %%
import csv
import re
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_NAME = "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default"
if not Path(MODEL_NAME).exists():
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

ADAPTER_DIR = "/kaggle/working/lora_adapter"
TEST_CSV = "/kaggle/input/wonderland-sft/test.csv"

# Grader's exact parameters
SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=7680,
)

USER_SUFFIX = "\n\nPlease put your final answer inside `\\boxed{}`."

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
def build_prompt(puzzle: str) -> str:
    """Wrap the puzzle in Nemotron's chat template with thinking enabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": puzzle + USER_SUFFIX}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
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
prompts = [build_prompt(r["prompt"]) for r in rows]

outputs = llm.generate(prompts, SAMPLING, lora_request=lora_req)
preds = []
for r, o in zip(rows, outputs):
    text = o.outputs[0].text
    ans = extract_answer(text)
    preds.append({"id": r["id"], "answer": ans})
    print(r["id"], "->", ans[:60])

Path("/kaggle/working/predictions.json").write_text(json.dumps(preds, indent=2))
print("Wrote predictions.json")
