from .huggingface_runner import HuggingFaceRunner
# from .openai_runner import OpenAIRunner
# from .anthropic_runner import AnthropicRunner

def get_runner(model_type):
    if model_type == "hf":
        return HuggingFaceRunner
    # elif model_type == "openai":
    #     return OpenAIRunner
    # elif model_type == "anthropic":
    #     return AnthropicRunner
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type}")
