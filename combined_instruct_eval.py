import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
# from utils.compute_clients_GB import create_clients
import traceback
from utils.compute_clients import create_clients
import re
load_dotenv()

# Build the structured prompt
def build_user_prompt_v2(instruction, constraints):
    return f"""You are a verifier. Your task is to evaluate whether a given instruction includes a set of constraints.

You will be provided:
- An instruction
- A list of constraints

Your task:
- Check if each constraint is mentioned in the instruction.
- Output your reasoning in a valid JSON format like below:
- The json generated should be a valid one and she be parsed by the json.loads()

JSON Response Format:
{{
  "Evaluation": [
    {{
      "Constraint": "<constraint text> just give the constraint text inside quotes do not use any escape characters",
      "Reason": "explanation",
      "Aligns": [true|false]
    }},
    ...
  ]
}}

Strictly make sure the response adheres to JSON format and it is a valid JSON.

[Instruction]:
{instruction}

[Constraints]:
{constraints}
"""

import json
import re

import json
import re

def load_model_json_response(response):
    # Remove ```json or ``` wrapping if present
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()

    # Try parsing directly
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[Direct Load Failed] {e}")

    # Fix improperly escaped quotes (e.g., \"text\") that aren't inside real strings
    def fix_escaped_quotes(text):
        # This avoids unescaping things inside already valid JSON values
        # Only target `\"` that are wrapping the entire value
        text = re.sub(r':\s+\\"', ': "', text)
        text = re.sub(r'\\"(.*?)\\"', r'"\1"', text)
        return text

    fixed_response = fix_escaped_quotes(response)

    # Try again after fixing quotes
    try:
        return json.loads(fixed_response)
    except json.JSONDecodeError as e:
        print(f"[Fix Attempt Failed] {e}")

    # Try regex-based extraction (object or list)
    json_patterns = [
        re.compile(r"\{.*\}", re.DOTALL),
        re.compile(r"\[\{.*\}\]", re.DOTALL),
    ]

    for pattern in json_patterns:
        match = pattern.search(response)
        if match:
            json_block = fix_escaped_quotes(match.group(0))
            try:
                return json.loads(json_block)
            except json.JSONDecodeError as e:
                print(f"[Regex Match Failed] {e}")
                continue

    print("[Final Failure] Unable to parse JSON.")
    return None

import re

def extract_aligns_values(text):
    # Use regex to find all occurrences of "Aligns": true or "Aligns": false
    matches = re.findall(r'"Aligns"\s*:\s*(true|false)', text, re.IGNORECASE)
    # Convert string values to boolean
    return [m.lower() == 'true' for m in matches]


# Main function
def evaluate_constraints(input_path, output_filename):
    # Load CSV
    df = pd.read_json(input_path,lines=True)
    # df = df_main.tail(100).copy()
    
    # Create client
    # client = create_clients(mode="azure")
    client = create_clients(mode="GPT-azure")

    # Build all prompts
    prompts = [
        build_user_prompt_v2(row["combined_instruction"], row["filtered_relevant_constraints"])
        for _, row in df.iterrows()
    ]

        
    # Send batched prompts
    print("Sending prompts in batch...")
    outputs = client.get_model_response_batch(user_prompts=prompts, temperature=0.0, max_new_tokens=2048)

    # Parse FinalDecision from outputs
    parsed_results = []
    constraint_wise_presence = []
    for i, output in enumerate(outputs):
        try:
            
            # parsed = json.loads(output)
            # parsed = load_model_json_response(output)
            # parsed_results.append(parsed.get("FinalDecision", None))
            # constraint_wise_presence.append(parsed.get("Evaluation",None))
            result = extract_aligns_values(output)
            constraint_wise_presence.append(result)
            
        except Exception as e:
            print(output)
            print(f"Error parsing row {i}: {e}")
            traceback.print_exc()
            parsed_results.append(None)

    # Save to new column
    # df["score_combined_instruction"] = parsed_results
    df["constraint_wise_presence"] = constraint_wise_presence
    df["constraint_presence_response"] = outputs


    # output_path = os.getenv("OUTPUT_PATH")+f"/{output_filename}"
    output_path = "benchmark_dataset/"+output_filename
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"\n✅ Results saved to: {output_path}")
   

# CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_filename", required=True, help="Path to save output JSONL file")
    args = parser.parse_args()
    evaluate_constraints(args.input_path, args.output_filename)
