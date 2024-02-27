#!/usr/bin/env python3
import os, json

# Define your JSON and prompt as Python dictionaries and strings
schema = {
    "properties": {
        "title": {"title": "The title of the movie", "type": "string"},
        "year": {"title": "The year the movie was released", "type": "integer"},
        "director": {"title": "The director of the movie", "type": "string"},
        "genre": {"title": "The genre of the movie", "type": "string"},
        "plot": {"title": "Plot summary of the movie", "type": "string"}
    },
    "required": ["title", "year", "director", "genre", "plot"],
    "title": "Movie",
    "type": "object"
}

movie ="Toy Story"
prompt = "Instruct: Output a JSON object in this format: { \"title\": ..., \"year\": ..., \"director\": ..., \"genre\": ..., \"plot\":...} for the following movie: "+movie+"\nOutput:\n"

# Construct the JSON input string
json_input = json.dumps({"input": {"schema": json.dumps(schema), "prompt": prompt}})
print(json_input)
# Define the command to execute your Python script with the JSON string
command = f'python3.11 runpod_handler.py --test_input \'{json_input}\''

# Execute the command
os.system(command)