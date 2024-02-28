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

# Now chill can import llama-cpp-python without an error:
from chill import improvement_loop


def chill_out(text):
    print("Got this input:", text)
    return str(improvement_loop(text))

examples = [
    ["You guys are so slow, we will never ship it!"],
    ["Your idea of a balanced diet is a biscuit in each hand."]
]

description = """
# ❄️ ChillTranslator 🤬 ➡️ 😎💬

This is an early experimental tool aimed at helping reduce online toxicity by automatically ➡️ transforming 🌶️ spicy or toxic comments into constructive, ❤️ kinder dialogues using AI and large language models.

ChillTranslator aims to help make online interactions more healthy. 
It aims to:
- **Convert** text to less toxic variations
- **Preserve original intent**, focusing on constructive dialogue

The project is on GitHub:
[https://github.com/lukestanley/ChillTranslator](https://github.com/lukestanley/ChillTranslator)
The repo is the same repo for the HuggingFace Space, the serverless worker, and the logic.

## Contributing 🤝

Contributions are very welcome!
Especially:
- pull requests,
- free GPU credits
- LLM API credits / access.

ChillTranslator is released under the MIT License.

Help make the internet a kinder place, one comment at a time.
Your contribution could make a big difference!
"""

demo = gr.Interface(
    fn=chill_out, 
    inputs="text", 
    outputs="text",
    examples=examples,
    cache_examples=True,
    description=description
)

demo.launch(max_threads=1, share=True)