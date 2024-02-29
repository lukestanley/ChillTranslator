import json
from os import environ as env
from typing import Any, Dict, Union

import requests
from huggingface_hub import hf_hub_download  


# There are 3 ways to use the LLM model currently used:
# 1. Use the HTTP server (USE_HTTP_SERVER=True), this is good for development
# when you want to change the logic of the translator without restarting the server.
# 2. Load the model into memory
# When using the HTTP server, it must be ran separately. See the README for instructions.
# The llama_cpp Python HTTP server communicates with the AI model, similar 
# to the OpenAI API but adds a unique "grammar" parameter.
# The real OpenAI API has other ways to set the output format.
# It's possible to switch to another LLM API by changing the llm_streaming function.
# 3. Use the RunPod API, which is a paid service with severless GPU functions.
# See serverless.md for more information.

URL = "http://localhost:5834/v1/chat/completions"
in_memory_llm = None
worker_options = ["runpod", "http", "in_memory"]

LLM_WORKER = env.get("LLM_WORKER", "runpod")
if LLM_WORKER not in worker_options:
    raise ValueError(f"Invalid worker: {LLM_WORKER}")
N_GPU_LAYERS = int(env.get("N_GPU_LAYERS", -1)) # Default to -1, use all layers if available
CONTEXT_SIZE = int(env.get("CONTEXT_SIZE", 2048))
LLM_MODEL_PATH = env.get("LLM_MODEL_PATH", None)

MAX_TOKENS = int(env.get("MAX_TOKENS", 1000))
TEMPERATURE = float(env.get("TEMPERATURE", 0.3))

performing_local_inference = (LLM_WORKER == "in_memory" or LLM_WORKER == "http")

if LLM_MODEL_PATH and len(LLM_MODEL_PATH) > 0:
    print(f"Using local model from {LLM_MODEL_PATH}")
if performing_local_inference and not LLM_MODEL_PATH:
    print("No local LLM_MODEL_PATH environment variable set. We need a model, downloading model from HuggingFace Hub")
    LLM_MODEL_PATH =hf_hub_download(
        repo_id=env.get("REPO_ID", "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        filename=env.get("MODEL_FILE", "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"),
    )
    print(f"Model downloaded to {LLM_MODEL_PATH}")
if LLM_WORKER == "http" or LLM_WORKER == "in_memory":
    from llama_cpp import Llama, LlamaGrammar, json_schema_to_gbnf

if in_memory_llm is None and LLM_WORKER == "in_memory":
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


def llm_stream_serverless(prompt,model):
    RUNPOD_ENDPOINT_ID = env.get("RUNPOD_ENDPOINT_ID")
    RUNPOD_API_KEY = env.get("RUNPOD_API_KEY")
    assert RUNPOD_ENDPOINT_ID, "RUNPOD_ENDPOINT_ID environment variable not set"
    assert RUNPOD_API_KEY, "RUNPOD_API_KEY environment variable not set"
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }
    
    schema = model.schema()
    data = {
        'input': {
            'schema': json.dumps(schema),
            'prompt': prompt
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200, f"Unexpected RunPod API status code: {response.status_code} with body: {response.text}"
    result = response.json()
    print(result)
    output = result['output'].replace("model:mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf\n", "")
    # TODO: remove replacement once new version of runpod is deployed
    return json.loads(output)

def query_ai_prompt(prompt, replacements, model_class):
    prompt = replace_text(prompt, replacements)
    if LLM_WORKER == "runpod":
        return llm_stream_serverless(prompt, model_class)
    if LLM_WORKER == "http":
        return llm_streaming(prompt, model_class)
    if LLM_WORKER == "in_memory":
        return llm_stream_sans_network(prompt, model_class)
