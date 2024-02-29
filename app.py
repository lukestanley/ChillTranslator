import json
from os import environ as env
from os import system as run
from subprocess import check_output

import gradio as gr


def inference_binary_check():
    # Without a GPU, we need to re-install llama-cpp-python to avoid an error.
    # We use a shell command to detect if we have an NVIDIA GPU available:
    use_gpu = True
    try:
        command = "nvidia-debugdump --list|grep Device"
        output = str(check_output(command, shell=True).decode())
        if "NVIDIA" in output and "ID" in output:
            print("NVIDIA GPU detected.")
    except Exception as e:
        print("No NVIDIA GPU detected, using CPU. GPU check result:", e)
        use_gpu = False

    if use_gpu:
        print("GPU detected, existing GPU focused llama-cpp-python should work.")
    else:
        print("Avoiding error by re-installing non-GPU llama-cpp-python build because no GPU was detected.")
        run('pip uninstall llama-cpp-python -y')
        run('pip install git+https://github.com/lukestanley/llama-cpp-python.git@expose_json_grammar_convert_function --upgrade --no-cache-dir --force-reinstall')
        print("llama-cpp-python re-installed, will now attempt to load.")


LLM_WORKER = env.get("LLM_WORKER", "runpod")

if LLM_WORKER == "http" or LLM_WORKER == "in_memory":
    inference_binary_check()

examples = [
    ["You guys are so slow, we will never ship it!"],
    ["Your idea of a balanced diet is a biscuit in each hand."]
]

description = """This is an early experimental tool aimed at helping reduce online toxicity by automatically ➡️ transforming 🌶️ spicy or toxic comments into constructive, ❤️ kinder dialogues using AI and large language models.

ChillTranslator aims to help make online interactions more healthy, with a tool to **convert** text to less toxic variations, **preserve original intent**, focusing on constructive dialogue.
The project is on GitHub:
[https://github.com/lukestanley/ChillTranslator](https://github.com/lukestanley/ChillTranslator)
The repo is the same repo for the HuggingFace Space, the serverless worker, and the logic.

Contributions are very welcome! Especially pull requests, free API credits.
Help make the internet a kinder place, one comment at a time. Your contribution could make a big difference!
"""

from chill import improvement_loop


def chill_out(text):
    print("Got this input:", text)
    result: dict = improvement_loop(text)
    print("Got this result:", result)

    formatted_output = f"""
    <div>
        <h4>Edited text:</h4>
        <p>{result['edit']}</p>
        <h4>Details:</h4>
        <ul>
            <li>Critique: {result['critique']}</li>
            <li>Faithfulness score: {result['faithfulness_score']:.0%}</li>
            <li>Spicy score: {result['spicy_score']:.0%}</li>
            <li>Overall score: {result['overall_score']:.0%}</li>
            <li>LLM requests made: {result['request_count']}</li>
        </ul>
    </div>
    """
    return formatted_output

demo = gr.Interface(
    fn=chill_out, 
    inputs=gr.Textbox(lines=2, placeholder="Enter some spicy text here..."), 
    outputs="html",
    examples=examples,
    cache_examples=True,
    description=description,
    title="❄️ ChillTranslator 🤬 ➡️ 😎💬",
    allow_flagging="never",
)

demo.launch(max_threads=1, share=True)