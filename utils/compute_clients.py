from openai import OpenAI

# from transformers import AutoTokenizer
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

NUM_CALLS_PER_MIN = 200


class LLMClient:

    def __init__(self, api_key,model_id,client_type="rits", base_url=None):
        if client_type == "rits":
            llm = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"RITS_API_KEY": api_key},
            )
        elif client_type == "GPT":
            llm = OpenAI(
                api_key=api_key,
                base_url=base_url)
        self.llm = llm
        self.model_id = model_id
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_model_response(
                self,
                messages=None,
                system_prompt=None,
                user_prompt=None,
                max_new_tokens=2048,
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
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


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
        response = self.get_model_response(
            messages=messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return response


def create_clients(mode="rits",model_id=None):
    if mode == "rits":
        RITS_KEY = os.getenv("RITS_API_KEY")
        print(RITS_KEY)
        base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1"
        model_id = "microsoft/phi-4"
        client = LLMClient(api_key=RITS_KEY,
                                    model_id=model_id,
                                    client_type="rits",
                                    base_url=base_url,
                                    )
    elif mode == "GPT-azure":
        openai_key = os.getenv("IBM_OPENAI_API_KEY")
        base_url="https://ete-litellm.bx.cloud9.ibm.com"
        client = LLMClient(api_key=openai_key,
                                  model_id=model_id,
                                  base_url=base_url,
                                  client_type="GPT")
        
    elif mode == "GPT":
        openai_key = os.getenv("OPENAI_API_KEY")
        client = LLMClient(api_key=openai_key,
                                  model_id="gpt-4o-mini",
                                  client_type="GPT")

    return client

# def get_model_response(
#                 self,
#                 messages=None,
#                 system_prompt=None,
#                 user_prompt=None,
#                 max_new_tokens=2048,
#                 temperature=0,
#                 enable_thinking: bool = False,                # ← new flag
#             ):
#         if messages is None:
#             if system_prompt:
#                 messages = [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ]
#             else:
#                 messages = [{"role": "user", "content": user_prompt}]

#         response = self.llm.chat.completions.create(
#             model=self.model_id,
#             messages=messages,
#             max_tokens=max_new_tokens,
#             temperature=temperature,
#             extra_body={                            # ← here
#                 "chat_template_kwargs": {
#                     "enable_thinking": enable_thinking
#                 }
#             },
#         )

#         return response.choices[0].message.content.strip()