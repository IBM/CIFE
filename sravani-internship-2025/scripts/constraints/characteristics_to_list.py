import argparse
from utils.io_utils import load_csv, save_csv
from utils.openai_utils import convert_text_to_list
from utils.config_loader import load_yaml_config, load_path_registry


paths = load_path_registry()


def main(
    input_path, output_path, prompt_path, prompt_name, column_name, temperature, model
):
    prompts = load_yaml_config(paths["constraint_prompts"])
    prompt_template = prompts.get(prompt_name, "")

    df = load_csv(input_path)

    if column_name not in df.columns:
        raise ValueError(f"Missing column '{column_name}' in the dataset.")

    print(
        f" Converting '{column_name}' column to lists using prompt from {prompt_path}..."
    )
    df[column_name + "_List"] = df[column_name].apply(
        lambda x: convert_text_to_list(x, prompt_template, temperature, model)
    )

    save_csv(df, output_path)
    print(f" Saved updated dataset to: {output_path}")


if __name__ == "__main__":
    constraint_tasks = load_yaml_config(paths["constraint_tasks"])
    defaults = constraint_tasks["preprocessing"]
    parser = argparse.ArgumentParser(
        description="Convert a text column to list format using OpenAI API"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=defaults["input_path"],
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=defaults["output_path"],
        help="Path to save the processed CSV file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=defaults["prompt_path"],
        help="Path to YAML file with prompt templates",
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default=defaults["column_name"],
        help="Name of the column to convert",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=defaults["model_name"],
        help="OpenAI model name to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults["temperature"],
        help="Temperature of the model",
    )
    prompt_name = defaults["prompt_name"]

    args = parser.parse_args()
    main(
        args.input_path,
        args.output_path,
        args.prompt_path,
        prompt_name,
        args.column_name,
        args.model,
        args.temperature,
    )
