import os
import argparse
import ast
import pandas as pd


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

    # Only keep CSR=1 combinations
    for correctness in ["Completely Correct", "Partially Correct", "Wrong"]:
        key = f"{correctness.replace(' ', '_')}_CSR1"
        metrics[key] = len(df[(df["correctness_level"] == correctness) & (df["CSR_per_row"] == 1)])

    # Avg SSR for each correctness level
    avg_ssr = (
        df.groupby("correctness_level")["SSR_per_row"]
        .mean()
        .rename(lambda x: f"Avg_SSR_{x.replace(' ', '_')}")
        .to_dict()
    )

    metrics.update(avg_ssr)
    return metrics


def main(input_path, column_name, output_dir, metrics_summary_file):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_json(input_path, lines=True) if input_path.endswith(".jsonl") else pd.read_csv(input_path)

    # Compute CSR and SSR
    df_with_metrics, overall_csr, overall_ssr = compute_metrics_from_column(df, column_name)

    # Compute metrics per correctness category
    combo_metrics = compute_combination_metrics(df_with_metrics)

    # Save df_with_metrics as JSONL in output_dir
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    detailed_output_path = os.path.join(output_dir, f"{input_filename}_metrics.jsonl")
    df_with_metrics.to_json(detailed_output_path, orient="records", lines=True, force_ascii=False)

    # Save metrics summary row
    summary_data = {
        "filename": os.path.basename(input_path),
        "Overall_CSR": overall_csr,
        "Overall_SSR": overall_ssr
    }
    summary_data.update(combo_metrics)

    with open(metrics_summary_file, "a", encoding="utf-8") as f:
        f.write(f"{pd.Series(summary_data).to_json()}\n")

    print(f"Saved detailed row-level metrics to: {detailed_output_path}")
    print(f"Appended summary metrics to: {metrics_summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CSR and SSR and save detailed and summary outputs")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input .csv or .jsonl file")
    parser.add_argument("--column_name", type=str, required=True, help="Column with constraint adherence lists")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detailed metrics JSONL")
    parser.add_argument("--metrics_summary_file", type=str, required=True, help="Path to summary JSONL file to append")

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        column_name=args.column_name,
        output_dir=args.output_dir,
        metrics_summary_file=args.metrics_summary_file
    )
