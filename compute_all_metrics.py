import os
import argparse
from config.model_config import model_configs
import subprocess

def get_model_file_name(model_key):
    return f"{model_key}_results_correctness.jsonl"

def get_metrics_file_name(model_key):
    return f"{model_key}_results_correctness_metrics.jsonl"

def run_metric_computation(input_path, column_name, output_dir, metrics_summary_file):
    cmd = [
        "python", "compute_metrics_v2.py",
        "--input_path", input_path,
        "--column_name", column_name,
        "--output_dir", output_dir,
        "--metrics_summary_file", metrics_summary_file
    ]
    print("[→] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main(input_folder, output_folder, column_name, metrics_summary_file):
    os.makedirs(output_folder, exist_ok=True)

    for model_key in model_configs.keys():
        # model_key = model_key.split("/")[-1]
        input_file = os.path.join(input_folder, get_model_file_name(model_key))
        expected_output_file = os.path.join(output_folder, get_metrics_file_name(model_key))

        if not os.path.exists(input_file):
            print(f"[✗] Skipping {model_key}: Input file does not exist.")
            print(input_file)
            continue

        if os.path.exists(expected_output_file):
            print(f"[✓] Skipping {model_key}: Metrics already computed.")
            continue

        print(f"[+] Computing metrics for {model_key}...")
        run_metric_computation(
            input_path=input_file,
            column_name=column_name,
            output_dir=output_folder,
            metrics_summary_file=metrics_summary_file
        )

if __name__ == "__main__":
    # Hardcoded configuration (edit as needed)
    input_folder = "LLMjudge_outputs/filtered_final_data"
    output_folder = "metrics_outputs/filtered_final_data"
    column_name = "Constraint_adherence"
    metrics_summary_file = "metrics_outputs/filtered_final_data/metrics_summary_v4.jsonl"

    main(
        input_folder=input_folder,
        output_folder=output_folder,
        column_name=column_name,
        metrics_summary_file=metrics_summary_file
    )

