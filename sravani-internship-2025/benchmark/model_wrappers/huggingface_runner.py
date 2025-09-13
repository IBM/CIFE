from transformers import pipeline, AutoModel, AutoTokenizer, AutoConfig
import torch

class HuggingFaceRunner:
    def __init__(self, model_name, max_new_tokens=512):
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Detect model architecture
        config = AutoConfig.from_pretrained(model_name)
        arch = config.architectures[0] if config.architectures else ""
        print(f"Detected model architecture: {arch}")

        # Choose the right pipeline
        if "T5" in arch or "Bart" in arch or "MBart" in arch:
            task = "text2text-generation"
        else:
            task = "text-generation"

        self.generator = pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

def generate(self, formatted_prompt: str):
    try:
        output = self.generator(
            formatted_prompt,
            max_new_tokens=self.max_new_tokens,  # ✅ this controls generation length
            do_sample=False,
            return_full_text=True  # Optional: include prompt in output if needed
        )
        return output[0]["generated_text"] if "generated_text" in output[0] else output[0]["generated_text"]
    except Exception as e:
        print(f"Generation error: {e}")
        return "ERROR"

