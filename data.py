import requests
import json
import os
import threading

def log_to_jsonl(file_path, data):
    def _log_to_jsonl():
        # Read the URL of the Gradio app from an environment variable
        url = os.environ.get("SAVE_URL")
        if url is None:
            raise ValueError("SAVE_URL environment variable not set")

        # Serialize the data to a JSON string
        json_data = json.dumps({"file_path": file_path, "data": data})

        # Create a dictionary with the JSON data as the value of a field named "data"
        request_body = {"data": json_data}

        # Convert the request body to a JSON string
        json_data = json.dumps(request_body)

        # Make the HTTP POST request
        try:
            response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

            # Check if the request was successful
            if response.status_code == 200:
                print("Data saved successfully!")
            else:
                print("Error saving data:", response.text, response.status_code)
        except Exception as e:
            print("Unexpected error saving", e, url, json_data)

    # Create a new thread and start it
    thread = threading.Thread(target=_log_to_jsonl)
    thread.start()
