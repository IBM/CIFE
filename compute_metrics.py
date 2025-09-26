import os
import argparse
import ast
from pathlib import Path
import pandas as pd
import json


def compute_metrics_from_column(df, column_name):
    # Parse constraint adherence column
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    csr_values = []
    ssr_values = []

    for r_list in df[column_name]:
        r_list = list(map(int, r_list))
        csr = int(all(r_list))
        ssr = sum(r_list) / len(r_list) if r_list else 0
        csr_values.append(csr)
        ssr_values.append(ssr)

    df["CSR_per_row"] = csr_values
    df["SSR_per_row"] = ssr_values

    # Compute overall metrics
    m = len(df)
    overall_csr = sum(csr_values) / m if m > 0 else 0
    overall_ssr = sum(ssr_values) / m if m > 0 else 0

    return df, overall_csr, overall_ssr


def compute_combination_metrics(df):
    metrics = {}
    total = len(df)
    counts = {}

    # Count and percentage for CSR=1 by correctness level
    for correctness in ["Completely Correct","Partially Correct"]:
        key_pct = f"{correctness.replace(' ', '_')}_CSR"
        count = len(df[(df["functional_correctness"] == correctness) & (df["CSR_per_row"] == 1)])
        pct = (count / total) if total else 0
        metrics[key_pct] = pct

    at_least_count = counts.get("Completely Correct", 0) + counts.get("Partially Correct", 0)
    at_least_pct = (at_least_count / total) if total else 0
    metrics["At_Least_Partially_Correct_CSR"] = at_least_pct
    return metrics

def main(input_path, column_name, metrics_summary_file):

    df = pd.read_json(input_path, lines=True) if input_path.endswith(".jsonl") else pd.read_csv(input_path)

    # Compute CSR and SSR
    df_with_metrics, overall_csr, overall_ssr = compute_metrics_from_column(df, column_name)

    # Compute metrics per correctness category
    combo_metrics = compute_combination_metrics(df_with_metrics)

    # Save metrics summary row
    summary_data = {
        "filename": os.path.basename(input_path),
        "Overall_CSR": overall_csr,
        "Overall_SSR": overall_ssr
    }
    summary_data.update(combo_metrics)
    
    with open(metrics_summary_file, "a", encoding="utf-8") as f:
        json.dump(summary_data,f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CSR and SSR and save summary outputs")
    parser.add_argument("--input_path", type=str, required=True, help="Path to LLM Judge output")
    parser.add_argument("--metrics_summary_filepath", type=str, required=True, help="Path to save the metrics")
    args = parser.parse_args()

    Path(args.metrics_summary_filepath).parent.mkdir(parents=True,exist_ok=True)
    column_name = "constraint_adherence"
    main(
        input_path=args.input_path,
        column_name=column_name,
        metrics_summary_file=args.metrics_summary_filepath
    )

