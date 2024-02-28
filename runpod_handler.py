import runpod
from os import environ as env
import json
from pydantic import BaseModel, Field
class Movie(BaseModel):
    title: str = Field(..., title="The title of the movie")
    year: int = Field(..., title="The year the movie was released")
    director: str = Field(..., title="The director of the movie")
    genre: str = Field(..., title="The genre of the movie")
    plot:  str = Field(..., title="Plot summary of the movie")

def pydantic_model_to_json_schema(pydantic_model_class):
    schema = pydantic_model_class.model_json_schema()

    # Optional example field from schema, is not needed for the grammar generation
    if "example" in schema:
        del schema["example"]

    json_schema = json.dumps(schema)
    return json_schema
default_schema_example = """{ "title": ..., "year": ..., "director": ..., "genre": ..., "plot":...}"""
default_schema = pydantic_model_to_json_schema(Movie)
default_prompt = f"Instruct: \nOutput a JSON object in this format: {default_schema_example} for the following movie: The Matrix\nOutput:\n"
from utils import llm_stream_sans_network_simple
def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    filename=env.get("MODEL_FILE", "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")
    prompt = job_input.get('prompt', default_prompt)
    schema = job_input.get('schema', default_schema)
    print("got this input", str(job_input))
    print("prompt", prompt )
    print("schema", schema )
    output = llm_stream_sans_network_simple(prompt, schema)
    #print("got this output", str(output))
    return output
    
runpod.serverless.start({
    "handler": handler, 
    #"return_aggregate_stream": True
})
