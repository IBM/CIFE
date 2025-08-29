import re
import json
import time
import pandas as pd
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from utils.compute_clients_GB import create_clients
import traceback
import os
# from utils.compute_clients import create_clients
import re
load_dotenv()

def difficulty_classification_prompt(instruction):
    return f"""You are a difficulty evaluator for programming tasks.

Your job is to classify a given programming instruction into one of three categories based on its overall difficulty:

- Easy: The task involves basic programming logic, no complex algorithms, and minimal constraints.
- Medium: The task involves little complexity such as use of specific data structures, standard algorithms, or multiple conditions to satisfy.
- Hard: The task involves moderate to high complexity logic, performance optimization, advanced algorithmic thinking, or multiple non-trivial constraints.

Only use the content explicitly mentioned in the instruction. Do not assume extra difficulty unless it is clearly indicated.

Respond in the following JSON format:

{{"Reason": "<Brief explanation for the chosen difficulty>",
  "Difficulty": "<easy | medium | hard>",
  
}}

[Instruction]:
{instruction}
"""
import re

def extract_difficulty_value(text: str) -> str:
    """
    Extracts the value of "Difficulty" from a JSON-like string block.
    Returns the difficulty (e.g. "Easy", "Medium", "Hard") or None if not found.
    """
    m = re.search(r'"Difficulty"\s*:\s*"(?P<diff>easy|medium|hard)"', text)
    return m.group("diff") if m else None


def classify_instructions(input_path, output_filename):


    df = pd.read_json(input_path,lines=True)
    # df = df.tail(100).copy()
    # Create client
    client = create_clients(mode="azure")
    # client = create_clients(mode="GPT-azure")

    # Build all prompts
    prompts = [
        difficulty_classification_prompt(row["combined_instruction"])
        for _, row in df.iterrows()
    ]

        
    # Send batched prompts
    print("Sending prompts in batch...")
    outputs = client.get_model_response_batch(user_prompts=prompts, temperature=0.0, max_new_tokens=2048)
    difficulties = [extract_difficulty_value(resp) for resp in outputs]


    # Save to new column
    # df["score_combined_instruction"] = parsed_results
    df["difficulty"] = difficulties
    df["difficulty_response"] = outputs

    output_path = os.getenv("OUTPUT_PATH")+f"/{output_filename}"
    # output_path = "benchmark_dataset/"+output_filename
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"\n✅ Results saved to: {output_path}")
   

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_filename", required=True, help="Path to save output JSONL file")
    args = parser.parse_args()
    classify_instructions(args.input_path, args.output_filename)
