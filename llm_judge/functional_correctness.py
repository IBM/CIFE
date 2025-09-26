import os
import argparse
import pandas as pd
import dotenv
from utils.compute_clients import create_clients
import re
dotenv.load_dotenv()


def code_correctness_prompt_v2(instruction, generated_code):
    prompt = f"""You are an expert Python developer and uncompromising code reviewer.  
    Your task is to evaluate whether the provided Python code strictly and robustly follows the instruction, based only on three dimensions:

    1. Syntax Correctness:  
    - The code must parse and compile without any syntax errors.

    2. Semantic Correctness:  
    - The code must run and produce the intended effect exactly as described.  
    - Any hidden bugs or logical mistakes count as failures.

    3. Constraint Correctness:  
    - The code must satisfy *every* constraint explicitly stated in the instruction, including any edge-case, privacy, or consistency requirements.  
    - If the instruction implies or adds new requirements (e.g. key ordering, redaction of sensitive data), those must be met exactly.

    Important Review Principles:
    - **Be conservative.** Only label “Completely Correct” if you are *absolutely certain* that all dimensions are fully satisfied—no exceptions.  
    - **Downgrade on any violation.**  
    - If syntax or runtime errors exist, mark as **Wrong**.  
    - If the code runs but omits or mis-implements any single required constraint, mark as **Partially Correct** at best.  
    - **No leniency for extra features.** Any extra functionality that interferes with, overrides, or exposes data beyond the instruction is a constraint violation.

    Evaluation Labels:
    - **Completely Correct**: Perfect in syntax, semantics, and constraint adherence—no deviations.  
    - **Partially Correct**: Runs without errors but misses or mis-implements one or more explicit requirements.  
    - **Wrong**: Contains syntax/runtime errors or fails to implement the core instruction at all.

    **If you have any doubt about full compliance, choose the lower category.**

    Output Format:
    Return your final evaluation as a json without any explanation:
    {{"reason": "<Your reason for the evaluation>",
      "correctness": "Completely Correct/Partially Correct/Wrong"
    }}
    Input:
    Instruction:  
    {instruction}

    Generated Code:  
    ```python
    {generated_code}
    ```
        """    
    return prompt

def extract_correctness_values(text):

    # Use regex to find all occurrences of "correctness": "Completely Correct", "Partially Correct", or "Wrong"
    matches = re.findall(r'"correctness"\s*:\s*("(?:Completely Correct|Partially Correct|Wrong)")', text, re.IGNORECASE)
    # Extract the matched strings and return them
    return matches[0].strip('"') if matches else None
# ----------- Main Function -----------
def code_correctness(df,client,temperature=0.1):
    

    instructions = df["instruction"].tolist()
    responses = df["response"].tolist()

    print("Generating evaluation prompts...")
    prompts = [
        code_correctness_prompt_v2(instr, resp)
        for instr, resp in zip(instructions, responses)
    ]

    print("Creating LLM client...")

    print("Sending prompts to model...")
    results = client.get_model_response_batch(
        user_prompts=prompts,
        system_prompt=None,
        temperature=temperature
    )

    print("Parsing model responses...")
    correctness_levels = [extract_correctness_values(resp) for resp in results]
    df["functional_correctness"] = correctness_levels
    return df



# ----------- CLI Argument Parsing -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Code Correctness Evaluator CLI")
    parser.add_argument("--mode", choices=["GPT", "rits", "GPT-azure", "azure"], required=True, help="Model mode: GPT, rits, or GPT-azure")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSONL")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature setting")
    parser.add_argument("--output_path", type=str, required=True, help="path to save the output JSON")

    args = parser.parse_args()

    code_correctness(
        mode=args.mode,
        input_path=args.input_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        output_path=args.output_path
    )


