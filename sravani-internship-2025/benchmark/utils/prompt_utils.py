import json
import re
import os
from dotenv import load_dotenv

def load_model_json_response(response):
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
    

def extract_json(string,col_name="Constraints"):
    try:
        json_string = string.strip().replace('```json\n', '', 1).replace('\n```', '', 1)
        constraint_json = json.loads(json_string)
        return constraint_json.get(col_name, [])
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} in string: {string}")
        return []
    except AttributeError as e:
        print(f"Attribute error: {e} in string: {string}")
        return []
