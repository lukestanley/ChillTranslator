Fast severless GPU inference with RunPod
==============================

This partly GPT-4 generated document explains the integration of Runpod with Docker, including testing the Runpod Dockerfile with Docker Compose, building and pushing the image to Docker Hub, and how `app.py` makes use of it. I skimmed it and added stuff to it, as a note to myself and others.

# Motivation
Fast inference is useful. Usually an existing hosted provider would be good for this, but I was worried about getting blocked given that we need to translate some spicy text input, the concern is that it could get flagged, and result in accounts being blocked.
Also I needed something that could infer with JSON typed output, that matches particular schemas, and fast. So I found RunPod's "serverless" GPU, service.
It can be used by chill.py and app.py, as one of the worker options.


## Testing with Docker Compose

To test the Runpod Dockerfile, you can use Docker Compose which simplifies the process of running multi-container Docker applications. Here's how you can test it:

1. Ensure you have Docker and Docker Compose installed on your system.
2. Navigate to the directory containing the `docker-compose.yml` file.
3. Run the following command to build and start the container:
   ```
   docker-compose up --build
   ```
4. The above command will build the image as defined in `runpod.dockerfile` and start a container with the configuration specified in `docker-compose.yml`, it will automatically run a test, that matches the format expected from the llm_stream_serverless client (in utils.py), though without the network layer in play.


# Direct testing with Docker, without Docker-Compose:

Something like this worked for me:

```sudo docker run --gpus all -it -v "$(pwd)/.cache:/runpod-volume/.cache/huggingface/" lukestanley/test:translate2 bash```
Note the cache mount. This saves re-downloading the LLMs!


## Building and Pushing to Docker Hub

After testing and ensuring that everything works as expected, you can build the Docker image and push it to Docker Hub for deployment. Here are the steps:

1. Log in to Docker Hub from your command line using `docker login --username [yourusername]`.
2. Build the Docker image with a tag:
   ```
   docker build -t yourusername/yourimagename:tag -f runpod.dockerfile .
   ```
3. Once the image is built, push it to Docker Hub:
   ```
   docker push yourusername/yourimagename:tag
   ```
4. Replace `yourusername`, `yourimagename`, and `tag` with your Docker Hub username, the name you want to give to your image, and the tag respectively.

# Runpod previsioning:
You'll need an account on Runpod with credit.
You'll need a serverless GPU endpoint setting up using your Docker image setup here:
https://www.runpod.io/console/serverless
It has a Flashboot feature that seems like Firecracker with GPU support, it might be using Cloud Hypervisor under the hood, currently Firecracker has no GPU support. Fly.io also has something similar, with Cloud Hypervisor.
You'll need the secret saved somewhere securely. This will likely end up as a securely treated env var for use by app.py later.
You'll also need the endpoint ID.

## Runpod Integration in `app.py`

The `app.py` file is a Gradio interface that makes use of the Runpod integration to perform inference. It checks for the presence of a GPU and installs the appropriate version of `llama-cpp-python`. Depending on the environment variable `LLM_WORKER`, it uses either the Runpod serverless API, an HTTP server, or loads the model into memory for inference.

The `greet` function in `app.py` calls `improvement_loop` from the `chill` module, which based on an environment variable, will use the Runpod worker, that is used to process the input text and generate improved text based on the model's output.

The Gradio interface is then launched with `demo.launch()`, making the application accessible via a web interface, which can be shared publicly.

Note: Ensure that the necessary environment variables such as `LLM_WORKER`, `REPO_ID`, and `MODEL_FILE` are set correctly for the integration to work properly.