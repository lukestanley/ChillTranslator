import json

def log_to_jsonl(file_path, data):
    with open(file_path, 'a') as file:
        jsonl_str = json.dumps(data) + "\n"
        file.write(jsonl_str)