
from utils.compute_clients import create_clients
from utils.compute_clients import LLMClient
from utils.prompt_utils import load_model_json_response

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
    return  output

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

    # system_prompt1 = """You are a verifier. Your task is to evaluate whether a given response satisfies a set of constraints for a specific instruction.

    # You will be provided:
    # - An instruction
    # - A list of constraints
    # - A response to the instruction

    # Your task:
    # - Analyze the response against each constraint independently.
    # - For each constraint, determine whether it is satisfied, and provide explanation.
    # - Do not make assumptions beyond the constraints. Only base your judgment on what is explicitly written.

    # Output your evaluation in a valid JSON format like below:
    # {"Evaluation": [
    #     {
    #     "Constraint": "<constraint text>",
    #     "Reason": "explanation",
    #     "Aligns": [true|false]
    #     },
    #     ...
    # ]
    # }

    # Do not include any text outside this JSON object.
    # """
    system_prompt1 = """You are a verifier. Your task is to evaluate whether a given response satisfies a set of constraints for a specific instruction.

    You will be provided:
    - An instruction
    - A list of constraints
    - A response to the instruction

    Your task:
    - Analyze the response against each constraint independently.
    - For each constraint, determine whether it is strictly and completely satisfied.
    - If any part of a constraint is only partially fulfilled or ambiguous, mark it as not satisfied.
    - Provide a brief but clear explanation of your judgment for each constraint.
    - Do not make assumptions beyond the constraint. Only base your judgment on what is explicitly implemented or stated in the response.

    Output your evaluation in a valid JSON format like below:
    {
    "Evaluation": [
        {
        "Constraint": "<constraint text>",
        "Reason": "explanation",
        "Aligns": [true|false]
        },
        ...
    ]
    }

    Do not include any text outside this JSON object."""


    user_prompts = [
        build_user_prompt(instruction, constraints, model_response) if model_response is not None else None
        for instruction, constraints, model_response in zip(instructions, constraints_list, model_responses)
    ]

    outputs = client.get_model_response_batch(user_prompts=user_prompts,
                                                      system_prompt=system_prompt1,
                                                      temperature=0,
                                                      max_new_tokens=4096)
    # extract_response_constraint_decision_batch = function_batch_handler(extract_response_constraint_decision)

    # is_correct_list = extract_response_constraint_decision_batch(outputs)
    return outputs




# Function to extract alignment scores from the evaluation list
def extract_alignment_scores(evaluation_list):
    results = []
    for entry in evaluation_list:
        try:
            eval_items = entry.get("Evaluation", [])
            aligns_row = [1 if item.get("Aligns") else 0 for item in eval_items[1]]
            results.append(aligns_row)
        except Exception as e:
            print(e)
            print(entry)
            exit
    return results

def extract_aligns_values(text):

    # Use regex to find all occurrences of "Aligns": true or "Aligns": false
    matches = re.findall(r'"Aligns"\s*:\s*(true|false)', text, re.IGNORECASE)
    # Convert string values to boolean
    return [m.lower() == 'true' for m in matches]

def constraint_adherence(df,client):
    instructions = df["instruction"].tolist()
    model_responses = df["response"].tolist()
    constraints_list = df["constraints"].tolist()

    print("Sending prompts to model...")
    result_list = response_constraint_validator(
        model_responses=model_responses,
        constraints_list=constraints_list,
        instructions=instructions,
        client=client
    )
    print("Extracting alignment scores...")
    result_scores = [extract_aligns_values(result) for result in result_list]
    df["constraint_adherence"] = result_scores
    return df
   
    



