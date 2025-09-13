import argparse
import os
import pandas as pd
from tqdm import tqdm
import dotenv
import json
dotenv.load_dotenv()

# -------- VLLM Client --------
class VLLMClient:
    def __init__(self, model_name, num_gpus=1, download_dir=None):
        from transformers import logging
        logging.set_verbosity_error()
        from vllm import LLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            download_dir=download_dir
        )

    def generate_responses(self, prompts, **sampling_kwargs):
        from vllm import SamplingParams
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.chat(messages, sampling_params, use_tqdm=True)
        final_outputs = [out.outputs[0].text for out in outputs if out.outputs]
        with open("vllm_outputs.txt", "w") as f:
            for output in final_outputs:
                f.write("Output started...................................................\n")
                f.write(output + "\n")
                f.write("output ended.....................................................\n")
        print("Outputs saved to vllm_outputs.txt")
        return final_outputs

# -------- HuggingFace Client --------
class HuggingFaceClient:
    def __init__(self, model_name, hf_token, device="cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, logging
        logging.set_verbosity_error()
        self.torch = torch
        self.set_seed = set_seed
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        self.model.eval()
        self.set_seed(42)

    def generate_responses(self, prompts, **sampling_kwargs):
        max_new_tokens = sampling_kwargs.get("max_tokens", 1024)
        temperature = sampling_kwargs.get("temperature", 0.7)

        all_predictions = []
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]

        inputs = self.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_dict=True,
            add_generation_prompt=True
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        for j in range(len(conversations)):
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[j][input_len:]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_predictions.append(prediction)

        return all_predictions

# -------- OpenAI Client --------
class OpenAIClient:
    def __init__(self, model_name, openai_key, max_workers=100):
        from openai import OpenAI
        import time
        self.client = OpenAI(api_key=openai_key)
        self.model_name = model_name
        self.time = time
        self.max_workers = max_workers

    def get_response_v2(self, messages, max_retries=1):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                self.time.sleep(2)
        return "[Error]"

    def generate_responses(self, prompts, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        messages_list = [[
            {"role": "user", "content": prompt}
        ] for prompt in prompts]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(lambda m: self.get_response_v2(m), messages_list),
                total=len(messages_list),
                desc="OpenAI Generating"
            ))

        return results


# -------- Main Function --------
def main(model_path, input_csv, output_dir, batch_size=8, max_new_tokens=1024,
         temperature=0, framework="vllm",
         hf_token=None, openai_key=None, num_gpus=1):

    os.makedirs(output_dir, exist_ok=True)
    model_id = model_path.split("/")[-1]
    output_path = os.path.join(output_dir, f"{model_id}_{framework}_results.jsonl")

    data = pd.read_csv(input_csv)
    df = data.head(10).copy() 
    prompts = df["combined_instruction"].tolist()
    all_outputs = []

    # Select client
    if framework == "vllm":
        client = VLLMClient(model_name=model_path, num_gpus=num_gpus)
        responses = client.generate_responses(
            prompts,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        all_outputs.extend(responses)
    else:
        if framework == "huggingface":                
            client = HuggingFaceClient(model_name=model_path, hf_token=hf_token)
        elif framework == "openai":
            if not openai_key:
                raise ValueError("OpenAI API key required with --openai_key")
            client = OpenAIClient(model_name=model_path, openai_key=openai_key)
        else:
            raise ValueError("Invalid framework. Choose from: vllm, huggingface, openai")

    # Generate responses in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batching"):
            batch_prompts = prompts[i:i+batch_size]
            
            responses = client.generate_responses(
                batch_prompts,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            all_outputs.extend(responses)


    df["response"] = all_outputs
    data_to_save = df.to_dict(orient="records")

    with open(output_path, 'w') as f:
        for record in data_to_save:
            f.write(json.dumps(record) + '\n')

    print(f"Output saved to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Model path or huggingface model name") 
    parser.add_argument("--input_csv", required=True, help="CSV with 'combined_instruction'")
    parser.add_argument("--output_dir", default="./response_outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--framework", type=str, default="vllm",
                        choices=["vllm", "huggingface", "openai"])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (VLLM only)")

    args = parser.parse_args()
    main(**vars(args))
