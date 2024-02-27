# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# Base image sets HuggingFace cache directory to use Runpod's shared cache for efficiency:
ENV HF_HOME="/runpod-volume/.cache/huggingface/"
# Also pre-downloading models may speed up start times while 
# increasing image size, but could be worth it for some use cases.

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install runpod==1.6.0

RUN python3.11 -m pip install pytest cmake \
    scikit-build setuptools pydantic-settings \
    huggingface_hub hf_transfer \
    pydantic pydantic_settings \
    llama-cpp-python

# Install llama-cpp-python (build with cuda)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN python3.11 -m pip install git+https://github.com/lukestanley/llama-cpp-python.git@expose_json_grammar_convert_function --upgrade --no-cache-dir --force-reinstall
ADD runpod_handler.py .

ADD chill.py .
ADD utils.py .
ADD promptObjects.py .

CMD nvidia-smi; python3.11 -u /runpod_handler.py
