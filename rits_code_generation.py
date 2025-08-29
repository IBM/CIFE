import argparse
import pandas as pd
import json
from utils.compute_clients import LLMClient
from tqdm import tqdm
import os
import dotenv
dotenv.load_dotenv()
random_state = 42

def main(input_file, output_dir, api_key, model_id, base_url, temperature=0.1, max_new_tokens=1024, system_prompt=None):
    df = pd.read_json(input_file, lines=True)
    # df = df.sample(10,random_state=random_state).copy()  # Limit to first 10 rows for testing

    user_prompts = df["combined_instruction"].tolist()
    model_name= model_id.split("/")[-1]
    output_path = os.path.join(output_dir, f"{model_name}_results.jsonl")
    client = LLMClient(
        api_key=api_key,
        model_id=model_id,
        client_type="rits",
        base_url=base_url
    )

    print(f"Generating responses for {len(user_prompts)} prompts.")
    responses = client.get_model_response_batch(
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        max_new_tokens=max_new_tokens,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to JSON with 'combined_instruction'")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--api_key",  default=os.getenv("RITS_API_KEY"), help="RITS API key")
    parser.add_argument("--model_id", required=True, help="Model ID to use (e.g., rits-model)")
    parser.add_argument("--base_url", required=True, help="Base URL for RITS API")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for generation")
    parser.add_argument("--max_new_tokens",type=float, default=2048, help="Max new tokens")

    args = parser.parse_args()
    main(
        input_csv=args.input_file,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model_id=args.model_id,
        base_url=args.base_url,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
