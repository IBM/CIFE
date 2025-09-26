import argparse
from functional_correctness import code_correctness
from utils.compute_clients import create_clients
from constraint_adherence import constraint_adherence
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Constraint Validator CLI")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to responses jsonl file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path to save the LLM Judge output JSONL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI api key"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="base_url, if any"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o",
        help="Model ID , if other than gpt-4o"

    )
    # parser.add_argument("--temperature", type=float, default=0, help="Temperature for LLM (default=0.7)")

    args = parser.parse_args()
    df = pd.read_json(args.input_path, lines=True)
    client = create_clients(api_key=args.api_key,model_id=args.model_id,base_url=args.base_url)
    df_constraint_adherence = constraint_adherence(df=df, client=client)
    df_functional_correctness = code_correctness(df=df_constraint_adherence,client=client)
    Path(args.output_path).parent.mkdir(parents=True,exist_ok=True)
    df_functional_correctness.to_json(args.output_path,orient="records",lines=True)
    






