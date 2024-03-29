import datetime
import json
import uuid
from time import time, sleep
from os import environ as env
from typing import Any, Dict, Union
from data import log_to_jsonl
import requests
from huggingface_hub import hf_hub_download


# There are 4 ways to use a LLM model currently used:
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
# 4. Use the Mistral API, which is a paid services.

URL = "http://localhost:5834/v1/chat/completions"
in_memory_llm = None
worker_options = ["runpod", "http", "in_memory", "mistral", "anthropic"]

LLM_WORKER = env.get("LLM_WORKER", "anthropic")
if LLM_WORKER not in worker_options:
    raise ValueError(f"Invalid worker: {LLM_WORKER}")
N_GPU_LAYERS = int(env.get("N_GPU_LAYERS", -1)) # Default to -1, use all layers if available
CONTEXT_SIZE = int(env.get("CONTEXT_SIZE", 2048))
LLM_MODEL_PATH = env.get("LLM_MODEL_PATH", None)

MAX_TOKENS = int(env.get("MAX_TOKENS", 1000))
TEMPERATURE = float(env.get("TEMPERATURE", 0.3))

performing_local_inference = (LLM_WORKER == "in_memory" or (LLM_WORKER == "http" and "localhost" in URL))

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
    # TODO: After a 30 second timeout, a job ID is returned in the response instead,
    # and the client must poll the job status endpoint to get the result.
    output = result['output'].replace("model:mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf\n", "")
    # TODO: remove replacement once new version of runpod is deployed
    return json.loads(output)

# Global variables to enforce rate limiting
LAST_REQUEST_TIME = None
REQUEST_INTERVAL = 0.5  # Minimum time interval between requests in seconds

def llm_stream_mistral_api(prompt: str, pydantic_model_class=None, attempts=0) -> Union[str, Dict[str, Any]]:
    global LAST_REQUEST_TIME
    current_time = time()
    if LAST_REQUEST_TIME is not None:
        elapsed_time = current_time - LAST_REQUEST_TIME
        if elapsed_time < REQUEST_INTERVAL:
            sleep_time = REQUEST_INTERVAL - elapsed_time
            sleep(sleep_time)
            print(f"Slept for {sleep_time} seconds to enforce rate limit")
    LAST_REQUEST_TIME = time()

    MISTRAL_API_URL = env.get("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
    MISTRAL_API_KEY = env.get("MISTRAL_API_KEY", None)
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {MISTRAL_API_KEY}'
    }
    data = {
        'model': 'mistral-small-latest',
        'messages': [
            {
                'role': 'user',
                'response_format': {'type': 'json_object'},
                'content': prompt
            }
        ]
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"Unexpected Mistral API status code: {response.status_code} with body: {response.text}")
    result = response.json()
    print(result)
    output = result['choices'][0]['message']['content']
    if pydantic_model_class:
        # TODO: Use more robust error handling that works for all cases without retrying?
        # Maybe APIs that dont have grammar should be avoided?
        # Investigate grammar enforcement with open ended generations?
        try:
            parsed_result = pydantic_model_class.model_validate_json(output)
            print(parsed_result)
            # This will raise an exception if the model is invalid,
        except Exception as e:
            print(f"Error validating pydantic model: {e}")
            # Let's retry by calling ourselves again if attempts < 3
            if attempts == 0:
                # We modify the prompt to remind it to output JSON in the required format
                prompt = f"{prompt} You must output the JSON in the required format!"
            if attempts < 3:
                attempts += 1
                print(f"Retrying Mistral API call, attempt {attempts}")
                return llm_stream_mistral_api(prompt, pydantic_model_class, attempts)

    else:
        print("No pydantic model class provided, returning without class validation")
    return json.loads(output)


def send_anthropic_request(prompt: str):
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return

    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }

    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Unexpected Anthropic API status code: {response.status_code} with body: {response.text}")
        raise ValueError(f"Unexpected Anthropic API status code: {response.status_code} with body: {response.text}")
    j = response.json()
    
    text = j['content'][0]["text"]
    print(text)
    return text

def llm_anthropic_api(prompt: str, pydantic_model_class=None, attempts=0) -> Union[str, Dict[str, Any]]:
    # With no streaming or rate limits, we use the Anthropic API, we have string input and output from send_anthropic_request,
    # but we need to convert it to JSON for the pydantic model class like the other APIs.
    output = send_anthropic_request(prompt)
    if pydantic_model_class:
        try:
            parsed_result = pydantic_model_class.model_validate_json(output)
            print(parsed_result)
            # This will raise an exception if the model is invalid.
            return json.loads(output)
        except Exception as e:
            print(f"Error validating pydantic model: {e}")
            # Let's retry by calling ourselves again if attempts < 3
            if attempts == 0:
                # We modify the prompt to remind it to output JSON in the required format
                prompt = f"{prompt} You must output the JSON in the required format only, with no remarks or prefacing remarks - JUST JSON!"
            if attempts < 3:
                attempts += 1
                print(f"Retrying Anthropic API call, attempt {attempts}")
                return llm_anthropic_api(prompt, pydantic_model_class, attempts)
    else:
        print("No pydantic model class provided, returning without class validation")
        return json.loads(output)

def query_ai_prompt(prompt, replacements, model_class):
    prompt = replace_text(prompt, replacements)
    if LLM_WORKER == "anthropic":
        result = llm_anthropic_api(prompt, model_class)
    if LLM_WORKER == "mistral":
        result = llm_stream_mistral_api(prompt, model_class)
    if LLM_WORKER == "runpod":
        result = llm_stream_serverless(prompt, model_class)
    if LLM_WORKER == "http":
        result = llm_streaming(prompt, model_class)
    if LLM_WORKER == "in_memory":
        result = llm_stream_sans_network(prompt, model_class)
    
    log_entry = {
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "worker": LLM_WORKER,
        "prompt_input": prompt,
        "prompt_output": result
    }
    log_to_jsonl('prompt_inputs_and_outputs.jsonl', log_entry)

    return result

