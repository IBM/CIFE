from tqdm import tqdm
import pandas as pd
from utils.config_loader import load_yaml_config, load_path_registry
from utils.openai_utils import generate_constraint_categories_row_wise

paths = load_path_registry()
constraint_tasks = load_yaml_config(paths["constraint_tasks"])
prompt_config = load_yaml_config(paths["constraint_prompts"])

defaults = constraint_tasks["constraint_mapping"]
input_path = defaults["input_path2"]
output_path = defaults["output_path2"]
model = defaults["model"]

df = pd.read_csv(input_path)

prompt_template = prompt_config["constraint_category_generation_v5"]


# Apply mapping
# df["Mapped_Characteristics"] = tqdm(
#     df.apply(
#         lambda row: map_items_to_categories(
#             row, "Characteristics_List", category_data, prompt_template
#         ),
#         axis=1,
#     )
# )
# df["Mapped_Constraints"] = tqdm(
#     df.apply(
#         lambda row: map_items_to_categories(
#             row, "constraints", category_data, prompt_template
#         ),
#         axis=1,
#     )
# )

df["categories_v2"] = tqdm(
    df.apply(
        lambda row: generate_constraint_categories_row_wise(
            row, prompt_template, model=model, client_type="openai"
        ),
        axis=1,
    )
)
df.to_csv(output_path, index=False)
print("✅ Done mapping and saved file.")
