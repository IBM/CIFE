import os
import yaml
import pandas as pd
import json
from utils.io_utils import load_csv, save_yaml
from utils.openai_utils import generate_constraint_categories
from utils.config_loader import load_yaml_config, load_path_registry
from utils.openai_utils import generate_constraint_categories, generate_tree_categories
from utils.format_utils import format_list_for_prompt


def main():
    paths = load_path_registry()
    constraint_tasks = load_yaml_config(paths["constraint_tasks"])
    defaults = constraint_tasks["constraint_category_generation"]
    prompt_config = load_yaml_config(defaults["prompt_path"])

    df = load_csv(defaults["input_path"])
    model = defaults.get("model1", "")
    # client_type = "rits"
    client_type = "openai"
    base_url = defaults.get("base_url", "")
    # constraint_v1="constraint_category_generation_v1"
    constraint_v2 = "constraint_category_generation_v3"
    prompt_template = prompt_config[constraint_v2]

    all_characteristics = sorted(
        set([item for sublist in df["Characteristics_List"] for item in eval(sublist)])
    )
    all_constraints = sorted(
        set([item for sublist in df["constraints"] for item in eval(sublist)])
    )

    # Save prompt to log
    prompt_log_path = defaults.get("prompt_log_path", "")

    os.makedirs(os.path.dirname(prompt_log_path), exist_ok=True)
    temperature = defaults["temperature"]
    categories = generate_constraint_categories(
        characteristics_list=all_characteristics,
        constraints_list=all_constraints,
        prompt_template=prompt_template,
        temperature=temperature,
        model=model,
        base_url=base_url,
        client_type=client_type,
    )
    print(categories)
    with open(prompt_log_path, "a") as log_file:
        log_file.write(f"\nModel Name: {model}")
        log_file.write("===== Prompt Sent to Model =====\n")
        log_file.write(prompt_template + "\n\n")
        log_file.write("====Generated Constraints====\n")
        log_file.write(format_list_for_prompt("Categories", categories))

    # if categories:
    #     save_yaml(categories, defaults["output_path"])
    #     print(f"Saved constraint categories to {defaults['output_path']}")
    # else:
    #     print("No categories generated.")


def generate_tree(file_path, output_path, prompt_template, model, temperature):
    df = pd.read_csv(file_path)
    categories = df["Category"].astype(str).tolist()
    examples = df["Examples"].astype(str).tolist()
    prompt_filled = prompt_template.replace(
        "{{CATEGORY_LIST}}", json.dumps(categories, indent=2)
    ).replace("{{EXAMPLE_LIST}}", json.dumps(examples, indent=2))

    output_text = generate_tree_categories(prompt_filled, model, temperature)
    with open(output_path, "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    # main()
    paths = load_path_registry()
    constraint_tasks = load_yaml_config(paths["constraint_tasks"])
    prompt_config = load_yaml_config(paths["constraint_prompts"])

    defaults = constraint_tasks["constraint_category_tree_generation"]
    input_path = defaults["input_path"]
    output_path = defaults["output_path"]
    model = defaults["model"]
    temperature = defaults["temperature"]
    prompt_template = prompt_config["constraint_categories_tree"]
    generate_tree(input_path, output_path, prompt_template, model, temperature)
