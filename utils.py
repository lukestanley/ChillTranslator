import json
from typing import Any, Dict, Union
import requests

from llama_cpp import Llama, LlamaGrammar, json_schema_to_gbnf

# The llama_cpp Python HTTP server communicates with the AI model, similar 
# to the OpenAI API but adds a unique "grammar" parameter.
# The real OpenAI API has other ways to set the output format.
# It's possible to switch to another LLM API by changing the llm_streaming function.

URL = "http://localhost:5834/v1/chat/completions"
in_memory_llm = None
IN_MEMORY_LLM_PATH = "/fast/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# TODO: Have a good way to set the model path

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
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        URL,
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
    print('\n')

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




def calculate_overall_score(faithfulness, spiciness):
    baseline_weight = 0.8
    overall = faithfulness + (1 - baseline_weight) * spiciness * faithfulness
    return overall


def llm_stream_sans_network(
    prompt: str, pydantic_model_class, return_pydantic_object=False
) -> Union[str, Dict[str, Any]]:
    global in_memory_llm
    if in_memory_llm is None:
        in_memory_llm = Llama(model_path=IN_MEMORY_LLM_PATH)
    schema = pydantic_model_class.model_json_schema()

    # Optional example field from schema, is not needed for the grammar generation
    if "example" in schema:
        del schema["example"]

    json_schema = json.dumps(schema)
    grammar = LlamaGrammar.from_json_schema(json_schema)

    output_text = in_memory_llm(
        prompt,
        max_tokens=1000,
        temperature=0.7,
        grammar=grammar,
    )["choices"][0]["text"]

    print(output_text)

    if return_pydantic_object:
        model_object = pydantic_model_class.model_validate_json(output_text)
        return model_object
    else:
        json_output = json.loads(output_text)
        return json_output

def query_ai_prompt(prompt, replacements, model_class, in_memory=True):
    prompt = replace_text(prompt, replacements)
    if in_memory:
        return llm_stream_sans_network(prompt, model_class)
    else:
        return llm_streaming(prompt, model_class)
