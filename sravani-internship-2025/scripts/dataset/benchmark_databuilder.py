import pandas as pd
import os
import random
from datasets import load_dataset
from utils.config_loader import load_yaml_config, load_path_registry
from scripts.dataset.extractor import extract_fields
from utils.openai_utils import get_model_response_batch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
default_output_path = os.path.join(project_root, "data/outputs/benchmark_data.csv")

TARGET_COUNTS = {
    "ajibawa-2023/Python-Code-23k-ShareGPT": 1500 * 25 // 100,  # 375
    "xlangai/DS-1000": 1500 * 15 // 100,                        # 225
    "bigcode/bigcodebench": 1500 * 25 // 100,                   # 375
    "Multilingual-Multimodal-NLP/McEval-Instruct": 1500 * 35 // 100,  # 525
}


def sharegpt_filter(instructions):
    prompts = [
        f"""Is the following instruction asking for a **code generation** task only (not editing or explanation)? Respond with True or False.\n\nInstruction: {inst}"""
        for inst in instructions
    ]
    responses = get_model_response_batch(prompts)
    return ["True" in r for r in responses]

def mceval_filter(instructions):
    prompts = [
        f"""You are an expert at understanding programming tasks. Given a user instruction, decide if it is asking for code generation (i.e., writing or implementing code). Respond only with \"True\" if it's a code generation task, or \"False\" if it's about code explanation, analysis, or purpose.\n\nInstruction: {inst}\n\nIs this a code generation task?"""
        for inst in instructions
    ]
    responses = get_model_response_batch(prompts)
    return ["True" in r for r in responses]

def sample_until_target(dataset, source_name, filter_fn, target_count, batch_size=200):
    valid_data = []
    idx = 0
    total = len(dataset)

    while len(valid_data) < target_count and idx < total:
        batch = dataset[idx: idx + batch_size]
        instructions = [extract_fields(example, source_name)[0] for example in batch]

        keep_flags = filter_fn(instructions) if filter_fn else [True] * len(batch)
        for i, keep in enumerate(keep_flags):
            if keep:
                nl, code, test = extract_fields(batch[i], source_name)
                if nl and code:
                    valid_data.append({
                        "dataset": source_name,
                        "instruction": nl,
                        "code": code,
                        "test": test,
                    })
        idx += batch_size
        print(f"Collected {len(valid_data)} / {target_count} valid samples from {source_name}")

    return valid_data[:target_count]

def build_dataset(save_path=default_output_path):
    paths = load_path_registry()
    config = load_yaml_config(paths["datasets"])
    dataset_info = config.get("datasets", [])
    all_data = []

    for ds in dataset_info:
        name = ds["name"]
        split = ds["split"]

        if name not in TARGET_COUNTS:
            print(f"Skipping {name}, not part of sampling plan.")
            continue

        target = TARGET_COUNTS[name]
        print(f"\nLoading {name} [{split}], target samples: {target}")

        try:
            dataset = load_dataset(name, split=split)
            if name == "Multilingual-Multimodal-NLP/McEval-Instruct":
                dataset = [ex for ex in dataset if ex.get("language", "").lower() == "python"]
                filter_fn = mceval_filter
            elif name == "ajibawa-2023/Python-Code-23k-ShareGPT":
                filter_fn = sharegpt_filter
            else:
                filter_fn = None

            examples = dataset if isinstance(dataset, list) else list(dataset)
            print(f"Available examples: {len(examples)}")

            filtered_samples = sample_until_target(examples, name, filter_fn, target)
            all_data.extend(filtered_samples)

        except Exception as e:
            print(f"Error loading {name}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(save_path, index=False)
    print(f"\nFinal CSV saved at {save_path} with {len(df)} rows.")
    return df

if __name__ == "__main__":
    build_dataset()
