# File: run_model.py

import pandas as pd
import argparse
import os
from tqdm import tqdm

from model_wrappers import get_runner
from utils.prompt_utils import get_code_generation_prompt,extract_json_field
from utils.prompt_formatter import format_prompt

INPUT_FILE = "benchmark_v1.csv"
INPUT_COL = "combined_instruction"
N = 10
def main(model_name, model_type, output_dir, max_new_tokens):
    full_df = pd.read_csv(INPUT_FILE)
    df = full_df.head(N).copy()
    if INPUT_COL not in df.columns:
        raise ValueError(f"'{INPUT_COL}' not found in {INPUT_FILE}")

    runner = get_runner(model_type)(model_name, max_new_tokens=max_new_tokens)
    output_col = f"response_{model_name.replace('/', '_')}"
    outputs = []

    for inst in tqdm(df[INPUT_COL], desc=f"Evaluating {model_name}"):
        task_prompt = get_code_generation_prompt(inst)  # Step 1
        formatted = format_prompt(task_prompt, model_type, model_name)  # Step 2
        raw_response = runner.generate(formatted)  # Step 3
        print(f"Model Response: {raw_response}")
        # extracted_response = extract_json_field(raw_response, field="response")  # Step 4
        outputs.append(raw_response)  

    df[output_col] = outputs
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"benchmark_{model_name.replace('/', '_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["hf", "openai", "anthropic"], required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    main(args.model_name, args.model_type, args.output_dir, args.max_new_tokens)