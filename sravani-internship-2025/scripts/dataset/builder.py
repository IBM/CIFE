import pandas as pd
from datasets import load_dataset
import random
from utils.config_loader import load_yaml_config, load_path_registry

from scripts.dataset.extractor import extract_fields
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
default_output_path = os.path.join(
    project_root, "data/outputs/final_instruction_following_code_examples.csv"
)


def build_dataset(
    sample_size=10,
    save_path=default_output_path,
):
    paths = load_path_registry()
    config = load_yaml_config(paths["datasets"])
    dataset_info = config.get("datasets", [])
    all_data = []

    for ds in dataset_info:
        name = ds["name"]
        split = ds["split"]
        try:
            print(f"\nLoading {name} [{split}]")
            dataset = load_dataset(name, split=split)
            dataset_list = dataset if isinstance(dataset, list) else list(dataset)
            sampled = random.sample(dataset_list, min(sample_size, len(dataset_list)))

            for example in sampled:
                nl, code, test = extract_fields(example, name)
                if nl and code:
                    all_data.append(
                        {
                            "dataset": name,
                            "instruction": nl,
                            "code": code,
                            "test": test,
                        }
                    )

            print(f"Collected examples from {name}")

        except Exception as e:
            print(f"Error loading {name}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(save_path, index=False)
    print(f"Final CSV saved at {save_path}")

    return df
