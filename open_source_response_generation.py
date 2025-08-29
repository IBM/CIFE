import argparse
import pandas as pd
import json
# from utils.compute_clients_GB import create_clients
from utils.compute_clients import create_clients
from tqdm import tqdm
import os
import dotenv
dotenv.load_dotenv()
random_state = 42

def main(df, output_dir, model_id, temperature=0.1, max_new_tokens=1024, system_prompt=None):
   
    # df = df.sample(10,random_state=random_state).copy()  # Limit to first 10 rows for testing

    user_prompts = df["combined_instruction"].tolist()
    output_path = os.path.join(output_dir, f"{model_id}_results.jsonl")
    client = create_clients(
        mode = "GPT-azure",
        model_id=model_id
    )

    print(f"Generating responses for {len(user_prompts)} prompts.")
    responses = client.get_model_response_batch(
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        user_prompts=user_prompts,
        temperature=temperature

    )

    output_data = []
    for row, response in zip(df.to_dict(orient="records"), responses):
        row["response"] = response
        output_data.append(row)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    input_file = "benchmark_dataset/benchmark_v10_v2.jsonl"
    # model_ids = ["gpt-4o-2024-08-06",]
    # model_ids = ["o1-mini-2024-09-12"]
    # model_ids = ["o3-mini-2025-01-31"]
    model_ids = ["Azure/o1-mini"]
    temperature = 1
    max_new_tokens = 2048
    # output_prefix = os.getenv("OUTPUT_PATH")  # fallback to "" if not set
    # output_dir = os.path.join(output_prefix,"opensource_response_generation_results")
    output_dir = "LLMjudge_outputs_opensource/code_correctness"
    for model_id in model_ids:
        df = pd.read_json(input_file, lines=True)
        print(f"Processing model: {model_id}")    
        main(
            df=df,
            output_dir=output_dir,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
