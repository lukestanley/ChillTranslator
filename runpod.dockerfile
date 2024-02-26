# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# Base image sets HuggingFace cache directory to use Runpod's shared cache for efficiency:
ENV HF_HOME="/runpod-volume/.cache/huggingface/"
# Also pre-downloading models may speed up start times while 
# increasing image size, but could be worth it for some use cases.

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install runpod==1.6.0

ADD runpod_handler.py .

CMD python3.11 -u /runpod_handler.py
