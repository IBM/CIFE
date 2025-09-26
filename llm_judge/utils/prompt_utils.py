import json
import re
import os
from dotenv import load_dotenv

def load_model_json_response_old(response):
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    try:
        response_dict = json.loads(response)
        return response_dict
    except:
        pass
    try:
        json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
        match = json_pattern.search(response)
        if match:
            json_block = match.group(0)
            json_block = json_block.replace("\\'", "'")
            response_dict = json.loads(json_block)
            return response_dict
        else:
            print("Not able to extract json data")
            return None
    except:
        print("Not able to extract json data")
        return None
    
def load_model_json_response(response):
    # Remove ```json or ``` wrapping if present
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()

    # Try parsing directly
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[Direct Load Failed] {e}")

    # Fix improperly escaped quotes (e.g., \"text\") that aren't inside real strings
    def fix_escaped_quotes(text):
        # This avoids unescaping things inside already valid JSON values
        # Only target `\"` that are wrapping the entire value
        text = re.sub(r':\s+\\"', ': "', text)
        text = re.sub(r'\\"(.*?)\\"', r'"\1"', text)
        return text

    fixed_response = fix_escaped_quotes(response)

    # Try again after fixing quotes
    try:
        return json.loads(fixed_response)
    except json.JSONDecodeError as e:
        print(f"[Fix Attempt Failed] {e}")

    # Try regex-based extraction (object or list)
    json_patterns = [
        re.compile(r"\{.*\}", re.DOTALL),
        re.compile(r"\[\{.*\}\]", re.DOTALL),
    ]

    for pattern in json_patterns:
        match = pattern.search(response)
        if match:
            json_block = fix_escaped_quotes(match.group(0))
            try:
                return json.loads(json_block)
            except json.JSONDecodeError as e:
                print(f"[Regex Match Failed] {e}")
                continue

    print("[Final Failure] Unable to parse JSON.")
    return None

def extract_json(string,col_name="Constraints"):
    try:
        json_string = string.strip().replace('```json\n', '', 1).replace('\n```', '', 1)
        json_string = json_string.strip().replace('```python\n', '', 1).replace('\n```', '', 1)  # Unescape quotes
        constraint_json = json.loads(json_string)
        return constraint_json.get(col_name, [])
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} in string: {string}")
        return []
    except AttributeError as e:
        print(f"Attribute error: {e} in string: {string}")
        return []
