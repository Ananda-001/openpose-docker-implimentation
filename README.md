#OpenPose Docker: GPU-Based Pose Detection
This repository packages the full OpenPose toolkit (from the official CMU repo) into easily deployable Docker containers with Python API support. Models are pre-bundled for reliability and faster startup. Ideal for deep learning, image processing, and rapid prototyping on GPU-equipped systems.

Features
CUDA 11.8 / cuDNN 8 base (Ubuntu 20.04)

Complete system/build dependencies handled in container

GCC 8 and Python3 support

OpenPose and Caffe built from source

Optional Python REST API via FastAPI/uvicorn

Separate slim and API containers for efficient builds

Models pre-downloaded (hand, body25) — link available for re-fetching

Output keypoints in JSON and pose overlays in PNG

How It Works
Base Image (Dockerfile):

Builds OpenPose from source with GPU, Python API, and required libraries.

The models directory, if provided, is mounted during the build for reliability.

Slim/API Layer (Dockerfile.slim or Dockerfile.api):

Adds REST endpoint and/or minimal dependencies as needed.

App logic (body-only image script or FastAPI HTTP service) is included.

Intended for use in a mounted directory, living atop the OpenPose base.

Usage
Prerequisites
Host Requirements:

NVIDIA GPU (CUDA Toolkit 11.8+, cuDNN 8+)

Nvidia Docker runtime (instructions)

Docker installed (docker --version)

Models directory (optional, fastest/reliable if pre-downloaded)

Build and Run
bash
# Clone repository
git clone <your-repo-url> openpose-docker
cd openpose-docker

# Build base OpenPose Docker image
docker build -t pose-openpose:latest -f Dockerfile .

# Build API/slim layer (optional)
docker build -t pose-openpose-api:latest -f Dockerfile.api .
Mount models for reliability:

bash
# Where you have models pre-fetched
docker run --gpus all -v "$PWD/models:/openpose/models" -it pose-openpose:latest /bin/bash
Python Body-Only CLI
bash
python3 /openpose/app/body25_only.py /path/to/input.jpg
# Outputs appear in /outputs (bind-mount for access)
FastAPI REST API
bash
docker run --gpus all -p 8080:8080 -v "$PWD/models:/openpose/models" pose-openpose-api:latest
# POST image files to http://localhost:8080/infer
Example: API Query
Send a POST request with an image file to /infer:

Receives: Keypoints (JSON), rendered pose PNG (base64-encoded)

Example output:

json
{
  "keypoints": { "pose": [[...]] },
  "render_png_base64": "<image-data>"
}
Troubleshooting
Ensure NVIDIA Docker runtime is active (nvidia-smi inside container should show GPU).

Match CUDA/cuDNN version to host driver (check with nvidia-smi).

For model downloads: If missing, request the official links or supply in models/.

Credits
Derived from CMU OpenPose

Dockerization strategies inspired by public projects

