ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN apt-get install git -y
COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
# Install dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install pytest cmake \
    scikit-build setuptools fastapi uvicorn sse-starlette \
    pydantic-settings starlette-context gradio huggingface_hub hf_transfer
RUN python3 -m pip install requests pydantic uvicorn starlette fastapi sse_starlette starlette_context pydantic_settings


# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install git+https://github.com/lukestanley/llama-cpp-python.git@expose_json_grammar_convert_function

CMD ["python3", "app.py"]

# Credit to Radamés Ajna <radames@hf.co> for the original Dockerfile