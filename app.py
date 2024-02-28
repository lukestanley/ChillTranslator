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


def greet(text):
    return str(improvement_loop(text))

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(max_threads=1, share=True)