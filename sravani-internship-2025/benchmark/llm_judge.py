from utils.compute_clients import create_clients
from utils.prompt_utils import load_model_json_response
from utils.compute_clients import LLMClient
import json
import re
import dotenv
import os
import argparse
import pandas as pd
dotenv.load_dotenv()


def extract_response_constraint_decision(output):
    if output is None:
        return None
    output_dict = load_model_json_response(output)
    if output_dict:
        return output_dict
    return (None, output)

# Function to handle batching of function calls
def function_batch_handler(function_to_batch: callable):
    return lambda *args_lists: [function_to_batch(*args) for args in zip(*args_lists)]


# Function to validate model responses against constraints
def response_constraint_validator(model_responses: str, constraints_list: list[str], instructions: str,
                                  client: LLMClient):
    """Return a boolean which indicates whether response satisfies all constraints or not."""

    def build_user_prompt(instruction, constraints, response):
        return f"""[Instruction]:
    {instruction}

    [Constraints]:
    {constraints}

    [Response]:
    ```python
    {response}
    ```"""

    system_prompt1 = """You are a verifier. Your task is to evaluate whether a given response satisfies a set of constraints for a specific instruction.

    You will be provided:
    - An instruction
    - A list of constraints
    - A response to the instruction

    Your task:
    - Analyze the response against each constraint independently.
    - For each constraint, determine whether it is satisfied, and provide explanation.
    - Do not make assumptions beyond the constraints. Only base your judgment on what is explicitly written.

    Output your evaluation in a valid JSON format like below:
    {"Evaluation": [
        {
        "Constraint": "<constraint text>",
        "Reason": "explanation",
        "Aligns": [true|false]
        },
        ...
    ],
    "FinalDecision": {
        "Score": "<number of constraints met>/<total number of constraints>",
        "Reason": "<Yes if all constraints were satisfied; No if any were violated, with explanation>",
        "Aligns": [true|false]
    }
    }

    Do not include any text outside this JSON object.
    """

    user_prompts = [
        build_user_prompt(instruction, constraints, model_response) if model_response is not None else None
        for instruction, constraints, model_response in zip(instructions, constraints_list, model_responses)
    ]

    outputs = client.get_model_response_batch(user_prompts=user_prompts,
                                                      system_prompt=system_prompt1,
                                                      temperature=0.3,
                                                      max_new_tokens=1000)
    extract_response_constraint_decision_batch = function_batch_handler(extract_response_constraint_decision)
    is_correct_list = extract_response_constraint_decision_batch(outputs)
    return is_correct_list




# Function to extract alignment scores from the evaluation list
def extract_alignment_scores(evaluation_list):
    results = []
    for entry in evaluation_list:
        eval_items = entry.get("Evaluation", [])
        aligns_row = [1 if item.get("Aligns") else 0 for item in eval_items]
        results.append(aligns_row)
    return results


def main(mode, input_path, output_dir, temperature=0.7):
    # 1. Load input JSONL
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_json(input_path, lines=True)

    # 2. Create model client
    client = create_clients(mode=mode)

    # 3. Prepare data
    instructions = df["instruction"].tolist()
    model_responses = df["response"].tolist()
    constraints_list = df["filtered_relevant_constraints"].tolist()

    print("Sending prompts to model...")
    result_list = response_constraint_validator(
        model_responses=model_responses,
        constraints_list=constraints_list,
        instructions=instructions,
        client=client
    )

    print("Extracting alignment scores...")
    result_scores = extract_alignment_scores(result_list)
    df["Constraint_adherence"] = result_scores

    # 4. Save output
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{input_filename}_llmjudge.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    df.to_json(output_path, orient="records", lines=True,force_ascii=False)

    print(f"Results saved to: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Constraint Validator CLI")
    parser.add_argument("--mode", choices=["GPT", "rits"], required=True, help="Model mode: GPT or rits")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for LLM (default=0.7)")

    args = parser.parse_args()

    main(
        mode=args.mode,
        input_path=args.input_path,
        output_dir=args.output_dir,
        temperature=args.temperature
    )



