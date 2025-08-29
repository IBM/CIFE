import os
import dotenv
from config.model_config import model_configs
from rits_code_generation import main  # Import the main function directly

dotenv.load_dotenv()

# Global settings
INPUT_FILE = "benchmark_dataset/benchmark_v9_v2.jsonl"
OUTPUT_DIR = "outputs/V9_experiments"
API_KEY = os.getenv("RITS_API_KEY")
TEMPERATURE = 0
SYSTEM_PROMPT = None  
MAX_NEW_TOKENS = 4096
# Iterate over models
for model_id, base_url in model_configs.items():
    if model_id  in ["codellama/CodeLlama-34b-Instruct-hf","codellama/CodeLlama-70b-Instruct-hf"]:
        MAX_NEW_TOKENS = 2048
    else:
        MAX_NEW_TOKENS = 4096
    model_name = model_id.split("/")[-1]
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.jsonl")

    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"[✔] Skipping {model_name}, output already exists.")
        continue

    print(f"[→] Running for model: {model_id}")
    try:
        main(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            api_key=API_KEY,
            model_id=model_id,
            base_url=base_url,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            system_prompt=SYSTEM_PROMPT
        )
    except Exception as e:
        print(f"[✘] Failed for {model_id}: {str(e)}")
