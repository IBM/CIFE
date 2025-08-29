import os
from openai import AzureOpenAI
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

NUM_CALLS_PER_MIN = 100

class LLMClient:

    def __init__(self, api_key, model_id, client_type="azure", base_url=None):
        if client_type == "azure":
            self.llm = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version="2024-09-01-preview"
            )
        else:
            raise ValueError(f"Unsupported client_type: {client_type}")
        self.model_id = model_id

    def get_model_response(
                self,
                messages=None,
                system_prompt=None,
                user_prompt=None,
                max_new_tokens=4096,
                temperature=0
                           
            ):
        if messages is None:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                messages = [{"role": "user", "content": user_prompt}]

        response = self.llm.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()


    def apply_chat_template(self, messages_list):
        raise NotImplementedError("apply_chat_template is not applicable in AzureOpenAI client")

    def get_model_response_batch(
        self, system_prompt=None, user_prompts=None, max_new_tokens=1024, temperature=0.1
    ):
        non_none_user_prompts = [ele for ele in user_prompts if ele is not None]
        if system_prompt:
            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                for user_prompt in non_none_user_prompts
            ]
        else:
            messages_list = [
                [{"role": "user", "content": user_prompt}]
                for user_prompt in non_none_user_prompts
            ]
        with ThreadPoolExecutor(max_workers=NUM_CALLS_PER_MIN) as executor:
            response_texts = list(
                tqdm(
                    executor.map(
                        lambda messages: self.call_api(
                            messages, max_new_tokens, temperature
                        ),
                        messages_list,
                    ),
                    total=len(messages_list),
                    desc="Processing",
                )
            )
        response_iter = iter(response_texts)
        all_response_texts = [
            next(response_iter) if ele is not None else None for ele in user_prompts
        ]
        return all_response_texts

    @sleep_and_retry
    @limits(calls=1500, period=60)
    def call_api(self, messages, max_new_tokens, temperature):
        return self.get_model_response(
            messages=messages, max_new_tokens=max_new_tokens, temperature=temperature
        )


def create_clients(mode="azure",model_id="gpt-4o-2024-08-06"):
    if mode == "azure":
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
        MODEL = model_id
        print(f"AZURE_ENDPOINT - {AZURE_ENDPOINT}")
        client = LLMClient(
            api_key=AZURE_OPENAI_API_KEY,
            model_id=MODEL,
            client_type="azure",
            base_url=AZURE_ENDPOINT,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return client
