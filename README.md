---
title: ChillTranslator
emoji: ❄️
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---
# ❄️ ChillTranslator 🤬 ➡️ 😎💬


This is an early experimental tool aimed at helping reduce online toxicity by automatically ➡️ transforming 🌶️ spicy or toxic comments into constructive, ❤️ kinder dialogues using AI and large language models.

![ChillTranslator demo](https://github.com/lukestanley/ChillTranslator/assets/306671/128611f4-3e8e-4c52-ba20-2ae61d727d52)


You can try out the ChillTranslator via the HuggingFace Space demo at [https://huggingface.co/spaces/lukestanley/ChillTranslator](https://huggingface.co/spaces/lukestanley/ChillTranslator).


ChillTranslator aims to help make online interactions more healthy.

Currently, it "translates" a built-in example of a spicy comment, and it can be used via the command line to improve a specific text of your choice, or it can be imported as a module.

Online toxicity can undermine the quality of discourse, causing distress 😞 and driving people away from online communities. Or worse: it can create a viral toxic loop 🌀!

<img src="https://github.com/lukestanley/ChillTranslator/assets/306671/2899f311-24ee-4ce4-ba76-d1de665aab01" width="300">

ChillTranslator hopes to mitigate toxic comments by automatically rephrasing negative comments, while maintaining the original intent and promoting positive communication 🗣️➡️💬. These rephrased texts could be suggested to the original authors as alternatives, or users could enhance their internet experience with "rose-tinted glasses" 🌹😎, automatically translating spicy comments into versions that are easier and more calming to read.
There could be all kinds of failure cases, but hey, it's a start!

Could Reddit, Twitter, Hacker News, or even YouTube comments be more calm and constructive places? I think so!

## Aims to:
- **Convert** text to less toxic variations
- **Preserve original intent**, focusing on constructive dialogue
- **Self-hostable, serverless, or APIs**: running DIY could save costs, avoid needing to sign up to APIs, and avoid the risk of toxic content causing API access to be revoked. We use llama-cpp-python with Mixtral, with a HTTP server option, and a fast "serverless" backend using RunPod currently.

## Possible future directions 🌟

**Speed:**
- Generating rephrasings in parallel.
- Show intermediate results to the user, while waiting for the final result.
- Split text into sentences e.g: with “pysbd” for parallel processing of translations.

**Speed and Quality:**
- Use Jigsaw dataset to find spicy comments, making a dataset for training a translation transformer, maybe like Google's T5 to run faster than Mixtral could.
- Try using a 'Detoxify' scoring model instead of the current "spicy" score method.
- Use natural language similarity techniques to compare possible rephrasing fidelity faster.
- Collecting a dataset of spicy comments and their rephrasings.
- Feedback loop: users could score rephrasings, or suggest their own.

**Distribution:**
- Better example showing use as Python module, HTTP API, for use from other tools, browser extensions.
- Enabling easy experimenting with online hosted LLM APIs
- Making setup on different platforms easier


## Getting started 🚀

### Try it online

You can try out ChillTranslator without any installation by visiting the HuggingFace Space demo:
```
https://huggingface.co/spaces/lukestanley/ChillTranslator
```

### Installation

1. Clone the Project Repository:
   ```
   git clone https://github.com/lukestanley/ChillTranslator.git
   cd ChillTranslator
   ```
2. It will automaticaly download [Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf?download=true) by default. The model HuggingFace repo and filename can be switched by enviroment variables, or you can point to a different local path.
3. Install dependencies, including a special fork of `llama-cpp-python`, and Nvidia GPU support if needed:
   ```
   pip install requests pydantic uvicorn starlette fastapi sse_starlette starlette_context pydantic_settings

   # If you have an Nvidia GPU, install the special fork of llama-cpp-python with CUBLAS support:
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install git+https://github.com/lukestanley/llama-cpp-python.git@expose_json_grammar_convert_function
   ```
   If you don't have an Nvidia GPU, the `CMAKE_ARGS="-DLLAMA_CUBLAS=on"` is not needed before the `pip install` command.
   
4. Start the LLM server with your chosen configuration. Example for Nvidia with `--n_gpu_layers` set to 20; different GPUs fit more or less layers. If you have no GPU, you don't need the `--n_gpu_layers` flag:
   ```
   python3 -m llama_cpp.server --model mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --port 5834 --n_ctx 4096 --use_mlock false --n_gpu_layers 20 &
   ```
These config options are likely to need tweaking. Please check out https://llama-cpp-python.readthedocs.io/en/latest/ for more info.


### Local Usage

ChillTranslator can be used locally to improve specific texts. This is how to see it in action:
```python
python3 chill.py
```

For improving a specific text of your choice, use the `-t` flag followed by your text enclosed in quotes:
```bash
python3 chill.py -t "Your text goes here"
```

Or chill can be imported as a module, with the improvement_loop function provided the text to improve.

## Contributing 🤝

Contributions are very welcome!
Especially:
- pull requests,
- free GPU credits
- LLM API credits / access.

ChillTranslator is released under the MIT License.

Help make the internet a kinder place, one comment at a time.
Your contribution could make a big difference!