# ❄️ ChillTranslator 🤬 ➡️ 😎💬


This is an early experimental tool aimed at reducing online toxicity by automatically ➡️ transforming 🌶️ spicy or toxic comments into constructive, ❤️ kinder dialogues using AI and large language models.


ChillTranslator aims to foster healthier online interactions. The potential uses of this translator are vast, and exploring its integration could prove invaluable.

Currently, it "translates" a built-in example of a spicy comment.

Online toxicity can undermine the quality of discourse, causing distress 😞 and driving people away from online communities. Or worse: it can create a viral toxic loop 🌀!

ChillTranslator hopes to mitigate toxic comments by automatically rephrasing negative comments, while maintaining the original intent and promoting positive communication 🗣️➡️💬. These rephrased texts could be suggested to the original authors as alternatives, or users could enhance their internet experience with "rose-tinted glasses" 🌹😎, automatically translating spicy comments into versions that are easier and more calming to read.

Could Reddit, Twitter, Hacker News, or even YouTube comments be more calm and constructive places? I think so!

![ChillTranslator demo](demo.gif)

## Approach ✨

- **Converts** text to less toxic variations
- **Preserves original intent**, focusing on constructive dialogue
- **Offline LLM model**: running DIY could save costs, avoid needing to sign up to APIs, and avoid the risk of toxic content causing API access to be revoked. We use llama-cpp-python's server with Mixtral.


## Possible future directions 🌟
- **Integration**: offer a Python module and HTTP API, for use from other tools, browser extensions.
- **HuggingFace / Replicate.com etc**: Running this on a fast system, perhaps on a HuggingFace Space could be good.
- **Speed** improvements.
   - Split text into sentences e.g: with “pysbd” for parallel processing of translations.
   - Use a hate speech scoring model instead of the current "spicy" score method.
   - Use a dataset of hate speech to make a dataset for training a translation transformer like Google's T5 to run faster than Mixtral could.
   - Use natural language similarity techniques to compare possible rephrasing fidelity faster.
   - Enabling easy experimenting with online hosted LLM APIs
   - Code refactoring to improve development speed!

## Getting Started 🚀

### Installation

1. Clone the Project Repository:
   ```
   git clone https://github.com/lukestanley/ChillTranslator.git
   cd ChillTranslator
   ```
2. Download a compatible and capable model like: [Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf?download=true). E.g:
   ```
   wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf?download=true -O mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf &
   ```
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
These config options may need tweaking. Please check out https://llama-cpp-python.readthedocs.io/en/latest/ for more info.


### Usage

ChillTranslator currently has an example spicy comment it works on fixing right away. This is how to see it in action:
```python
python3 chill.py
```

## Contributing 🤝

Contributions are welcome!
Especially:
- pull requests,
- free GPU credits
- LLM API credits / access.

ChillTranslator is released under the MIT License.

Help make the internet a kinder place, one comment at a time.
Your contribution could make a big difference!