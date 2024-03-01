import gradio as gr
import json
import threading

# Function to save data to disk in a non-blocking manner
def save_data_to_disk(data):
    # Append data to a JSONL file
    with open("data.jsonl", "a") as file:
        file.write(json.dumps(data) + "\n")

# Wrapper function to make `save_data_to_disk` non-blocking
def save_data(data):
    # Start a new thread to handle the saving process
    thread = threading.Thread(target=save_data_to_disk, args=(data,))
    thread.start()
    # Return a simple confirmation message
    return "Data is being saved."

# Create a Gradio interface
interface = gr.Interface(
    fn=save_data,
    inputs=gr.JSON(label="Input JSON Data"),
    outputs="text",
    title="Data Saving Service",
    description="A simple Gradio app to save arbitrary JSON data in the background.",
)

# Run the Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8435, share=True)
