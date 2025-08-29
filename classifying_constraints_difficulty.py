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


def constraint_difficulty_classification(instruction, constraints):
    prompt = f""" 
    You are an assistant that classifies requirement-constraints into “easy” vs. “hard” based on their implementation complexity, performance sensitivity, and testing burden. 

• Easy constraints involve only single-step logic, simple built-in calls, or trivial checks (O(1) or O(N) over very small N), have minimal performance impact, and require almost no special testing.  
• Hard constraints include anything that goes beyond trivial checks: multi-stage algorithms, strict resource bounds (e.g. O(1) space, custom prime testing), unusual data types or math, high performance or security sensitivity, or require extensive edge-case testing.  

JSON Response Format:
{{
  "Classification": [
    {{
      "Constraint": "<constraint text> (just the text inside quotes, no escape characters)",
      "Reason": "<brief explanation>",
      "label": "hard" | "easy"
    }},
    ...
  ]
}}

Strictly adhere to valid JSON. 
[Instruction]:
{instruction}

[Constraints]:
{constraints}
    """
    return prompt



def extract_label_values(text):
    """
    Extracts all occurrences of the "label" field that are either "Easy" or "Hard"
    from a JSON-like string block. Returns a list of label strings.
    """
    # Only match labels "Easy" or "Hard"
    matches = re.findall(r'"label"\s*:\s*"(easy|hard)"', text, re.IGNORECASE)
    return matches

def classify_instructions(input_path, output_filename):


    df = pd.read_json(input_path,lines=True)
    # df = df.tail(100).copy()
    # Create client
    client = create_clients(mode="azure")
    # client = create_clients(mode="GPT-azure")

    # Build all prompts
    prompts = [
        constraint_difficulty_classification(row["combined_instruction"],row["final_constraints"])
        for _, row in df.iterrows()
    ]

        
    # Send batched prompts
    print("Sending prompts in batch...")
    outputs = client.get_model_response_batch(user_prompts=prompts, temperature=0.0, max_new_tokens=2048)
    labels = [extract_label_values(resp) for resp in outputs]


    # Save to new column
    # df["score_combined_instruction"] = parsed_results
    df["constraint_difficulty_labels"] = labels
    df["constraint_difficulty_response"] = outputs

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
