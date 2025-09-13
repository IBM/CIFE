import pandas as pd
import numpy as np
from datasets import load_dataset
from utils.config_loader import load_yaml_config, load_path_registry
from scripts.dataset.extractor import extract_fields
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
default_output_path = os.path.join(
    project_root, "data/outputs/final_dataset_stats.csv"
)

def compute_stats(name, dataset, source_name):
    instr_lens, code_lens, test_count = [], [], 0
    total = 0

    for example in dataset:
        try:
            nl, code, test = extract_fields(example, source_name)
            if nl and code:
                instr_lens.append(len(nl.split()))
                code_lens.append(len(code.split()))
                total += 1
            if test:
                test_count += 1
        except Exception as e:
            print(f"Failed to extract from {name}: {e}")
    
    return {
        "Dataset": name,
        "Num Examples": total,
        "Avg Instr Length": round(np.mean(instr_lens), 2) if instr_lens else 0,
        "Avg Code Length": round(np.mean(code_lens), 2) if code_lens else 0,
        "Has Test Cases (%)": round(100 * test_count / total, 1) if total else 0
    }

def generate_dataset_stats(save_path=default_output_path):
    paths = load_path_registry()
    config = load_yaml_config(paths["datasets"])
    dataset_info = config.get("datasets", [])
    stats = []

    for ds in dataset_info:
        name = ds["name"]
        split = ds["split"]
        try:
            print(f"\nLoading {name} [{split}]")
            dataset = load_dataset(name, split=split)
            dataset_list = dataset if isinstance(dataset, list) else list(dataset)

            stat = compute_stats(name, dataset_list, name)
            stats.append(stat)

            print(f"Stats computed for {name}")

        except Exception as e:
            print(f"Error loading {name}: {e}")

    df = pd.DataFrame(stats)
    df.to_csv(save_path, index=False)
    print(f"\nDataset statistics saved at {save_path}")
    return df

if __name__ == "__main__":
    generate_dataset_stats()
