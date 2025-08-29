import os
import argparse
import ast
import pandas as pd
from collections import defaultdict


def compute_metrics_from_column(df, column_name):
    # Parse constraint adherence column into lists of ints
    df[column_name] = df[column_name].apply(
        lambda x: list(map(int, ast.literal_eval(x))) if isinstance(x, str) else x
    )

    csr_values = []
    ssr_values = []

    for r_list in df[column_name]:
        csr = int(all(r_list))
        ssr = sum(r_list) / len(r_list) if r_list else 0
        csr_values.append(csr)
        ssr_values.append(ssr)

    df["CSR_per_row"] = csr_values
    df["SSR_per_row"] = ssr_values

    m = len(df)
    overall_csr = sum(csr_values) / m if m else 0
    overall_ssr = sum(ssr_values) / m if m else 0

    return df, overall_csr, overall_ssr


def compute_combination_metrics(df):
    metrics = {}
    total = len(df)
    counts = {}

    # Count and percentage for CSR=1 by correctness level
    for correctness in ["Completely Correct", "Partially Correct", "Wrong"]:
        key_count = f"{correctness.replace(' ', '_')}_CSR1_Count"
        key_pct = f"{correctness.replace(' ', '_')}_CSR1_Pct"
        count = len(df[(df["correctness_level"] == correctness) & (df["CSR_per_row"] == 1)])
        pct = (count / total) if total else 0
        metrics[key_count] = count
        metrics[key_pct] = pct
        counts[correctness] = count

    # At least partially correct: includes completely + partially
    at_least_count = counts.get("Completely Correct", 0) + counts.get("Partially Correct", 0)
    at_least_pct = (at_least_count / total) if total else 0
    metrics["At_Least_Partially_Correct_CSR1_Count"] = at_least_count
    metrics["At_Least_Partially_Correct_CSR1_Pct"] = at_least_pct

    # Avg SSR by correctness level
    # avg_ssr = (
    #     df.groupby("correctness_level")["SSR_per_row"]
    #     .mean()
    #     .rename(lambda x: f"Avg_SSR_{x.replace(' ', '_')}")
    #     .to_dict()
    # )
    # metrics.update(avg_ssr)
    return metrics


def compute_ssr_by_dataset(df, constraints_col, adherence_col):
    """SSR for each dataset as (total matching flags) / (total constraints)"""
    stats = {}
    for dataset, group in df.groupby("dataset"):
        total_flags = 0
        total_constraints = 0
        for _, row in group.iterrows():
            flags = row[adherence_col]
            total_flags += sum(flags)
            total_constraints += len(flags)
        stats[dataset] = (total_flags / total_constraints) if total_constraints else 0
    return stats


def compute_ssr_by_category(df, constraints_col, adherence_col):
    """
    SSR for each constraint category across all rows.
    constraints_col: column name of list-of-dicts (final_constraints)
    adherence_col: column name of list-of-int flags
    """
    stats = defaultdict(lambda: {"sum": 0, "count": 0})
    for _, row in df.iterrows():
        constraints = (
            ast.literal_eval(row[constraints_col])
            if isinstance(row[constraints_col], str)
            else row[constraints_col]
        )
        flags = row[adherence_col]
        for rec, flag in zip(constraints, flags):
            cat = rec["type"]
            stats[cat]["sum"] += flag
            stats[cat]["count"] += 1

    return {cat: v["sum"] / v["count"] for cat, v in stats.items()}


def compute_ssr_by_instruction_part(df, constraints_col, adherence_col):
    """
    Return a dict mapping each instruction_part
    ("Extracted from instruction" vs "Newly Generated")
    to its overall SSR.
    """
    part_stats = defaultdict(lambda: {"sum": 0, "count": 0})

    for _, row in df.iterrows():
        constraints = (
            ast.literal_eval(row[constraints_col])
            if isinstance(row[constraints_col], str)
            else row[constraints_col]
        )
        flags = row[adherence_col]
        for rec, flag in zip(constraints, flags):
            part = rec["instruction_part"]
            part_stats[part]["sum"] += flag
            part_stats[part]["count"] += 1

    return {part: v["sum"] / v["count"] for part, v in part_stats.items()}


def compute_csr_by_dataset(df):
    """Compute CSR=1 rate per dataset: (rows with CSR_per_row==1)/(total rows in dataset)"""
    csr_stats = {}
    for dataset, group in df.groupby("dataset"):
        count_csr1 = group["CSR_per_row"].sum()
        total_rows = len(group)
        csr_stats[dataset] = (count_csr1 / total_rows) if total_rows else 0
    # overall CSR across all rows
    csr_stats["__overall__"] = df["CSR_per_row"].mean() if len(df) else 0
    return csr_stats

# New method: correctness percentage by level overall and per dataset

def compute_correctness_level_percentages(df):
    """Compute percentage breakdown of correctness levels overall and per dataset."""
    levels = ["Completely Correct", "Partially Correct", "Wrong"]
    stats = {"overall": {}, "per_dataset": {}}

    # overall percentages
    total = len(df)
    for lvl in levels:
        stats["overall"][lvl] = (df["correctness_level"].eq(lvl).sum() / total) if total else 0

    # per-dataset percentages
    for dataset, group in df.groupby("dataset"):
        grp_total = len(group)
        stats["per_dataset"][dataset] = {}
        for lvl in levels:
            stats["per_dataset"][dataset][lvl] = (group["correctness_level"].eq(lvl).sum() / grp_total) if grp_total else 0

    return stats


def main(input_path, column_name, output_dir, metrics_summary_file):
    os.makedirs(output_dir, exist_ok=True)

    # load input
    df = (
        pd.read_json(input_path, lines=True)
        if input_path.endswith(".jsonl")
        else pd.read_csv(input_path)
    )

    # parse lists
    df["final_constraints"] = df["final_constraints"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # compute CSR/SSR per row
    df, overall_csr, overall_ssr = compute_metrics_from_column(df, column_name)

    # combination metrics
    combo_metrics = compute_combination_metrics(df)

    # SSR metrics
    ssr_by_dataset = compute_ssr_by_dataset(df, "final_constraints", column_name)
    ssr_by_category = compute_ssr_by_category(df, "final_constraints", column_name)
    ssr_by_part = compute_ssr_by_instruction_part(df, "final_constraints", column_name)

    # new CSR per dataset
    csr_by_dataset = compute_csr_by_dataset(df)

    # new correctness percentages
    correctness_pct = compute_correctness_level_percentages(df)

    # save detailed metrics
    base = os.path.splitext(os.path.basename(input_path))[0]
    detailed_out = os.path.join(output_dir, f"{base}_metrics_extended.jsonl")
    df.to_json(detailed_out, orient="records", lines=True, force_ascii=False)

    # assemble summary
    summary = {
        "filename": os.path.basename(input_path),
        "Overall_CSR": overall_csr,
        "Overall_SSR": overall_ssr,
        **combo_metrics,
        "SSR_by_Dataset": ssr_by_dataset,
        "SSR_by_Category": ssr_by_category,
        "SSR_by_Instruction_Part": ssr_by_part,
        "CSR_by_Dataset": csr_by_dataset,
        "Correctness_Level_Pct": correctness_pct
    }

    # append summary
    with open(metrics_summary_file, "a", encoding="utf-8") as f:
        f.write(f"{pd.Series(summary).to_json()}\n")

    print(f"Saved detailed metrics to: {detailed_out}")
    print(f"Appended extended summary to: {metrics_summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute extended CSR/SSR metrics"
    )
    parser.add_argument(
        "--input_path", required=True, help="Path to input .csv or .jsonl file"
    )
    parser.add_argument(
        "--column_name", required=True, help="Column with adherence lists"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save detailed metrics"
    )
    parser.add_argument(
        "--metrics_summary_file", required=True,
        help="Path to summary JSONL file to append"
    )

    args = parser.parse_args()
    main(
        input_path=args.input_path,
        column_name=args.column_name,
        output_dir=args.output_dir,
        metrics_summary_file=args.metrics_summary_file,
    )
