import openai
import os
import yaml
import json
from dotenv import load_dotenv
from openai import OpenAI
from utils.format_utils import format_list_for_prompt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
rits_key = os.getenv("RITS_API_KEY")

SYSTEM_PROMPT = """You are a helpful assistant that provides concise and accurate responses to user queries."""
def get_model_response_batch(user_prompts=None, system_prompt=SYSTEM_PROMPT):
    messages_list = [[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_prompt
            }] for user_prompt in user_prompts]
    with ThreadPoolExecutor(max_workers=1000) as executor:
            response_texts = list(tqdm(
                executor.map(lambda messages: get_response_v2(messages), messages_list),
                total=len(messages_list),
                desc="Processing"
            ))
    return response_texts

def get_response_v2(messages, max_retries=1):
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
            )
            # print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    return "[]"

# client = OpenAI(api_key=rits_key, base_url=base_url)
def get_response(client, model, prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content


def convert_text_to_list(text, prompt_template, temperature, model):
    if not isinstance(text, str) or not text.strip():
        return []
    client = OpenAI(api_key=openai_api_key)
    prompt = prompt_template + "\n\n" + text.strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        output = response.choices[0].message.content[9:-3]
        parsed = eval(output.strip(), {"__builtins__": None}, {})
        if isinstance(parsed, list):
            return parsed
        return [output.strip()]
    except Exception as e:
        print(f"Error processing input: {e}")
        return []


def get_client(provider="openai", base_url=None):
    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_key)
        return client

    elif provider == "rits":
        rits_key = os.getenv("RITS_API_KEY")
        client = OpenAI(
            api_key=rits_key,
            base_url=base_url,
            default_headers={"RITS_API_KEY": rits_key},
        )
        return client

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate_constraint_categories(
    characteristics_list,
    constraints_list,
    prompt_template,
    temperature,
    model="gpt-4o-mini",
    client_type="openai",
    base_url=None,
):
    client = get_client(client_type, base_url)
    characteristics_text = format_list_for_prompt(
        "Characteristics", characteristics_list
    )
    constraints_text = format_list_for_prompt("Constraints", constraints_list)

    prompt = prompt_template + "\n\n" + characteristics_text + "\n" + constraints_text

    try:
        content = get_response(client, model, prompt, temperature)
        if content.startswith("```yaml"):
            content = content.replace("```yaml", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        elif content.endswith("```"):
            content = content.replace("```", "").strip()
        elif content.lower().startswith("yaml"):
            content = content[4:].strip()

        print(content)
        # parsed = yaml.safe_load(content)
        return content
    except Exception as e:
        print(f" Error during OpenAI call: {e}")
        return {}


def format_categories_for_prompt(category_data: dict) -> str:
    simple = category_data.get("simple_categories", [])
    complex_ = category_data.get("complex_categories", [])
    all_categories = simple + complex_
    return "\n".join([f"- {c['name']}: {c['description']}" for c in all_categories])


def map_items_to_categories(
    row,
    item_colname,
    prompt_template,
    model="gpt-4o-mini",
    temperature=0,
    client_type="openai",
    base_url=None,
):
    client = get_client(client_type, base_url)

    item_list = (
        eval(row[item_colname])
        if isinstance(row[item_colname], str)
        else row[item_colname]
    )
    instruction = row["instruction"]
    code = row["code"]

    prompt = prompt_template.format(
        instruction=instruction,
        code=code,
        item_list="\n".join([f"- {item}" for item in item_list]),
    )

    try:

        content = get_response(client, model, prompt, temperature)
        if content.startswith("```python"):
            content = content.replace("```python", "").replace("```", "").strip()
        if content.startswith("```"):
            content = content.replace("```", "").strip()
        if content.endswith("```"):
            content = content.replace("```", "").strip()
        print(content)
        parsed = eval(content.strip(), {"__builtins__": None}, {})
        return parsed if isinstance(parsed, dict) else {}

    except Exception as e:
        print(f"Error: {e}")
        return {}


def generate_constraint_categories_row_wise(
    row,
    prompt_template,
    model="gpt-4o",
    temperature=0,
    client_type="openai",
    base_url=None,
):

    client = get_client(client_type, base_url)

    instruction = row["instruction"]
    code = row["code"]
    characteristics = row["Characteristics_List"]
    constraints = row["constraints"]

    prompt = prompt_template.format(
        instruction_block=instruction.strip(),
        code_block=code.strip(),
        characteristics_block=characteristics.strip(),
        constraints_block=constraints.strip(),
    )

    try:
        content = get_response(client, model, prompt, temperature)

        if content.startswith("```python"):
            content = content.replace("```python", "").replace("```", "").strip()
        if content.startswith("```"):
            content = content.replace("```", "").strip()
        if content.endswith("```"):
            content = content.replace("```", "").strip()

        print(content)
        return content

    except Exception as e:
        print(f"Error: {e}")
        return []


def generate_tree_categories(
    prompt,
    model="gpt-4o-mini",
    temperature=0,
    client_type="openai",
    base_url=None,
):

    client = get_client(client_type, base_url)
    content = get_response(client, model, prompt, temperature)
    return content
