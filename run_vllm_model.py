import argparse
import os
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

class VLLMClient:
    def __init__(self, model_name, num_gpus=1, download_dir=None):
        from transformers import logging
        logging.set_verbosity_error()
        self.llm = LLM(model=model_name,
                       tensor_parallel_size=num_gpus,
                       download_dir=download_dir)

    def generate_responses(self, prompts, **sampling_kwargs):
        messages = [[{"role": "system", "content": "You are a helpful assistant"},{"role": "user", "content": prompt}] for prompt in prompts]
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.chat(messages, sampling_params, use_tqdm=True)
        # Extracting the text from the outputs
        final_outputs = [out.outputs[0].text for out in outputs if out.outputs]
        with open("vllm_outputs.txt", "w") as f:
            for output in final_outputs:
                f.write(output + "\n")
        print("Outputs saved to vllm_outputs.txt")
        return final_outputs

def main(model_name, input_csv, output_dir, batch_size=8, max_new_tokens=400,
         temperature=0.7, top_k=50, top_p=0.8):

    # Create output path
    os.makedirs(output_dir, exist_ok=True)
    model_id = model_name.split("/")[-1]
    output_path = os.path.join(output_dir, f"{model_id}_results.csv")

    # Load dataset
    df1 = pd.read_csv(input_csv)
    df = df1.head(10).copy()
    # Initialize model
    vllm_client = VLLMClient(model_name=model_name)

    # Batched generation
    all_outputs = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch_prompts = df["combined_instruction"].iloc[i:i+batch_size].tolist()
        responses = vllm_client.generate_responses(
            batch_prompts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens
        )
        all_outputs.extend(responses)
        print(f"Batch {i // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size} processed.")

    # Store response
    df[model_id + "_response"] = all_outputs
    df.to_csv(output_path, index=False)
    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="HuggingFace model ID")
    parser.add_argument("--input_csv", required=True, help="Path to CSV with 'combined_instruction'")
    parser.add_argument("--output_dir", default="./outputs", help="Where to save results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.8)
    # parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    # parser.add_argument("--framework", type=str, default="pt", choices=["pt", "tf"], help="Deep learning framework to use")
    args = parser.parse_args()
    main(**vars(args))
