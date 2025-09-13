import re


def format_list_for_prompt(name, data, indent=0):
    indent_space = "  " * indent
    formatted = f"{indent_space}{name}:\n"

    if isinstance(data, dict):
        for key, value in data.items():
            formatted += format_list_for_prompt(key, value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, dict)):
                formatted += format_list_for_prompt("-", item, indent + 1)
            else:
                formatted += f"{indent_space}  - {item}\n"
    else:
        formatted += f"{indent_space}  - {data}\n"

    return formatted


def clean_input(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    return re.sub(r"\s+", " ", text).strip()
