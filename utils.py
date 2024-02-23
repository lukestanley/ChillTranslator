import json
from typing import Any, Dict, Union
import requests

from llama_cpp import (
    json_schema_to_gbnf,
)  # Only used directly to convert the JSON schema to GBNF,

# The main interface is the HTTP server, not the library directly.


def llm_streaming(
    prompt: str, pydantic_model_class, return_pydantic_object=False
) -> Union[str, Dict[str, Any]]:
    schema = pydantic_model_class.model_json_schema()

    # Optional example field from schema, is not needed for the grammar generation
    if "example" in schema:
        del schema["example"]

    json_schema = json.dumps(schema)
    grammar = json_schema_to_gbnf(json_schema)

    payload = {
        "stream": True,
        "max_tokens": 1000,
        "grammar": grammar,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        "http://localhost:5834/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
    )
    output_text = ""
    for chunk in response.iter_lines():
        if chunk:
            chunk = chunk.decode("utf-8")
            if chunk.startswith("data: "):
                chunk = chunk.split("data: ")[1]
                if chunk.strip() == "[DONE]":
                    break
                chunk = json.loads(chunk)
                new_token = chunk.get("choices")[0].get("delta").get("content")
                if new_token:
                    output_text = output_text + new_token
                    print(new_token, sep="", end="", flush=True)

    if return_pydantic_object:
        model_object = pydantic_model_class.model_validate_json(output_text)
        return model_object
    else:
        json_output = json.loads(output_text)
        return json_output


def replace_text(template: str, replacements: dict) -> str:
    for key, value in replacements.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def query_ai_prompt(prompt, replacements, model_class):
    prompt = replace_text(prompt, replacements)
    # print('prompt')
    # print(prompt)
    return llm_streaming(prompt, model_class)
