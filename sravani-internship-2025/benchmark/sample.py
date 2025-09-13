import os
from vllm import LLM
os.environ["HF_TOKEN"] = "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"

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tokenizer="meta-llama/Llama-3-8B-Instruct",
    trust_remote_code=True,
    hf_token=os.environ["HF_TOKEN"]
)
