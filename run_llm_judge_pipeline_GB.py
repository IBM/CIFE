import os
import subprocess
from config.model_config import model_configs  
from pathlib import Path

# === CONFIG ===
INPUT_FOLDER = "outputs/V10_experiments"
OUTPUT_PREFIX = os.getenv("OUTPUT_PATH")  # fallback to "" if not set

CONSTRAINT_OUTPUT_DIR = os.path.join(OUTPUT_PREFIX, "LLMjudge_outputs/constraint_adherence")
CORRECTNESS_OUTPUT_DIR = os.path.join(OUTPUT_PREFIX, "LLMjudge_outputs/code_correctness")
LLM_JUDGE_SCRIPT = "llm_judge.py"
LLM_CORRECTNESS_SCRIPT = "llm_judge_correctness.py"
MODE = "azure"  # can be overridden per model if needed
# MODE = "GPT-azure" 
def run_script(script, args_dict):
    cmd = ["python", script] + [f"--{k}={v}" for k, v in args_dict.items()]
    print("[→] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for model_name in model_configs.keys():
        model_name = model_name.split("/")[-1]
        print(f"\n=== Processing model: {model_name} ===")
        base_name = f"{model_name}_results.jsonl"
        input_path = os.path.join(INPUT_FOLDER, base_name)

        # 1. Check existence of response file
        if not os.path.exists(input_path):
            print(f"[✘] Skipping {model_name}: input response file not found.")
            continue

        # 2. Constraint Adherence Phase
        constraint_output_path = os.path.join(CONSTRAINT_OUTPUT_DIR, f"{model_name}_constraint_adherence.jsonl")
        if not os.path.exists(constraint_output_path):
            print(f"[✓] Running constraint adherence for {model_name}")
            run_script(LLM_JUDGE_SCRIPT, {
                "mode": MODE,
                "input_path": input_path,
                "output_dir": CONSTRAINT_OUTPUT_DIR,
                "output_path": constraint_output_path
            })
        else:
            print(f"[✔] Constraint adherence output exists. Skipping {model_name}")

        # 3. Code Correctness Phase
        correctness_output_path = os.path.join(CORRECTNESS_OUTPUT_DIR, f"{model_name}_results_correctness.jsonl")
        if not os.path.exists(correctness_output_path):
            print(f"[✓] Running code correctness for {model_name}")
            run_script(LLM_CORRECTNESS_SCRIPT, {
                "mode": MODE,
                "input_path": constraint_output_path,
                "output_dir": CORRECTNESS_OUTPUT_DIR,
                "output_path": correctness_output_path

            })
        else:
            print(f"[✔] Code correctness output exists. Skipping {model_name}")

if __name__ == "__main__":
    main()
