# scripts/dataset/dataset_filters.py
from utils.openai_utils import get_response,get_client,get_model_response_batch  # Your OpenAI utility
import pandas as pd

def is_code_generation(instruction, client=None, model="gpt-4o-mini"):
    prompt = f"""You are an expert at understanding programming tasks. Given a user instruction, decide if it is asking for code generation (i.e., writing or implementing code). Respond only with "True" if it's a code generation task, or "False" if it's about code explanation, analysis, or purpose.

Instruction: {instruction}

Is this a code generation task? Respond with only "True" or "False"."""
    reply = get_response(client, model, prompt, temperature=0)
    return "True" in reply

def is_editing_task(instruction, client=None, model="gpt-4o-mini"):
    prompt = f"""You are an expert at identifying editing instructions. Given a prompt, determine if the user is asking to edit or modify existing code. Respond with "True" if the instruction is an editing task (e.g., fix, modify, refactor), otherwise respond with "False".

Instruction: {instruction}

Is this an editing task? Respond only with "True" or "False"."""
    reply = get_response(client, model, prompt, temperature=0)
    return "True" in reply

def transform_dataset(dataset, name):
    client = get_client()
    # Default: no transformation
    if name == "Multilingual-Multimodal-NLP/McEval-Instruct":
        return [ex for ex in dataset if ex.get("language", "").lower() == "python" and is_code_generation(ex["instruction"], client)]
    
    elif name == "ajibawa-2023/Python-Code-23k-ShareGPT":
        return [ex for ex in dataset if not is_editing_task(
            next((m["value"] for m in ex.get("conversations", []) if m["from"] == "human"), ""), client)]
    
    else:
        return dataset
