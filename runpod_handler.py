import json
from os import environ as env
from typing import Any, Dict, Union
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field
import runpod


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from huggingface_hub import hf_hub_download
small_repo = "TheBloke/phi-2-GGUF"
small_model="phi-2.Q2_K.gguf"
big_repo = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
big_model = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
LLM_MODEL_PATH =hf_hub_download(
    repo_id=big_repo,
    filename=big_model,
)
print(f"Model downloaded to {LLM_MODEL_PATH}")



in_memory_llm = None

N_GPU_LAYERS = env.get("N_GPU_LAYERS", -1) # Default to -1, which means use all layers if available
CONTEXT_SIZE = int(env.get("CONTEXT_SIZE", 2048))
USE_HTTP_SERVER = env.get("USE_HTTP_SERVER", "false").lower() == "true"
MAX_TOKENS = int(env.get("MAX_TOKENS", 1000))
TEMPERATURE = float(env.get("TEMPERATURE", 0.3))

class Movie(BaseModel):
    title: str = Field(..., title="The title of the movie")
    year: int = Field(..., title="The year the movie was released")
    director: str = Field(..., title="The director of the movie")
    genre: str = Field(..., title="The genre of the movie")
    plot:  str = Field(..., title="Plot summary of the movie")

JSON_EXAMPLE_MOVIE = """
{ "title": "The Matrix", "year": 1999, "director": "The Wachowskis", "genre": "Science Fiction", "plot":"Prgrammer realises he lives in simulation and plays key role."
"""

if in_memory_llm is None:
    print("Loading model into memory. If you didn't want this, set the USE_HTTP_SERVER environment variable to 'true'.")
    in_memory_llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, verbose=True)

def llm_stream_sans_network(
    prompt: str, pydantic_model_class=Movie, return_pydantic_object=False
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
        return output_text


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    name = job_input.get('name', 'World')

    #return f"Hello, {name}!"
    return llm_stream_sans_network(
        f"""You need to output JSON objects describing movies.
        For example for the movie called: `The Matrix`: Output: {JSON_EXAMPLE_MOVIE}
        Instruct: Output the JSON object for the movie: `{name}` Output: """)

runpod.serverless.start({"handler": handler})
