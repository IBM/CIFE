def extract_fields(example, source):
    nl, code, test = "", "", ""

    if source == "ajibawa-2023/Python-Code-23k-ShareGPT":
        messages = example.get("conversations", [])
        nl = next((m["value"] for m in messages if m["from"] == "human"), "")
        code = next((m["value"] for m in messages if m["from"] == "gpt"), "")

    elif source == "xlangai/DS-1000":
        nl = example.get("prompt", "")
        code = example.get("reference_code", "")
        test = example.get("code_context", "")

    elif source == "bigcode/bigcodebench":
        nl = example.get("instruct_prompt") or example.get("complete_prompt", "")
        code = example.get("canonical_solution", "")
        test = example.get("test", "")

    elif source == "Multilingual-Multimodal-NLP/McEval-Instruct":
            nl = example.get("instruction", "")
            code = example.get("output", "")
            test = example.get("tests", "")

    elif source == "nuprl/CanItEdit":
        instruction = example.get("instruction_lazy", "")
        old_contents = example.get("before", "")
        nl = f"{instruction}\n{old_contents}"
        code = example.get("after", "")

    return nl.strip(), code.strip(), test.strip()
