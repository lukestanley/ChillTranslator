import json
from os import environ as env
from typing import Any, Dict, Union
import requests

from huggingface_hub import hf_hub_download  
from llama_cpp import Llama, LlamaGrammar, json_schema_to_gbnf

# There are two ways to use the LLM model currently used:
# 1. Use the HTTP server (USE_HTTP_SERVER=True), this is good for development
# when you want to change the logic of the translator without restarting the server.
# 2. Load the model into memory
# When using the HTTP server, it must be ran separately. See the README for instructions.
# The llama_cpp Python HTTP server communicates with the AI model, similar 
# to the OpenAI API but adds a unique "grammar" parameter.
# The real OpenAI API has other ways to set the output format.
# It's possible to switch to another LLM API by changing the llm_streaming function.

URL = "http://localhost:5834/v1/chat/completions"
in_memory_llm = None

N_GPU_LAYERS = env.get("N_GPU_LAYERS", 10)
CONTEXT_SIZE = int(env.get("CONTEXT_SIZE", 4096))
LLM_MODEL_PATH = env.get("LLM_MODEL_PATH", None)
USE_HTTP_SERVER = env.get("USE_HTTP_SERVER", "false").lower() == "true"
MAX_TOKENS = int(env.get("MAX_TOKENS", 1000))
TEMPERATURE = float(env.get("TEMPERATURE", 0.7))

if LLM_MODEL_PATH and len(LLM_MODEL_PATH) > 0:
    print(f"Using local model from {LLM_MODEL_PATH}")
else:
    print("No local LLM_MODEL_PATH environment variable set. We need a model, downloading model from HuggingFace Hub")
    LLM_MODEL_PATH =hf_hub_download(
        repo_id=env.get("REPO_ID", "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        filename=env.get("MODEL_FILE", "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"),
    )
    print(f"Model downloaded to {LLM_MODEL_PATH}")

if in_memory_llm is None and USE_HTTP_SERVER is False:
    print("Loading model into memory. If you didn't want this, set the USE_HTTP_SERVER environment variable to 'true'.")
    in_memory_llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, verbose=True)

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
        "max_tokens": MAX_TOKENS,
        "grammar": grammar,
        "temperature": TEMPERATURE,
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
    schema = pydantic_model_class.model_json_schema()

    # Optional example field from schema, is not needed for the grammar generation
    if "example" in schema:
        del schema["example"]

    json_schema = json.dumps(schema)
    grammar = LlamaGrammar.from_json_schema(json_schema)

    stream = in_memory_llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        grammar=grammar,
        stream=True
    )

    output_text = ""
    for chunk in stream:
        result = chunk["choices"][0]
        print(result["text"], end='', flush=True)
        output_text = output_text + result["text"]

    print('\n')

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
