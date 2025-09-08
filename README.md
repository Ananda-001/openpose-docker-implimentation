# OpenPose in Docker (CUDA 11.8) with Interactive and API Workflows

A GPU-accelerated Docker setup for OpenPose built from source on Ubuntu 20.04 with CUDA 11.8/cuDNN 8, including an interactive workflow (-it) to run the CMU demo and a slim/API layer for batch processing and HTTP inference.      

## Features

- CUDA 11.8 + cuDNN 8 base image with all build dependencies for OpenPose and Caffe preinstalled.      
- OpenPose compiled from source with Python API enabled and optional hand/face support if models are present.      
- Two-stage Docker design: a base image (OpenPose build) and a slim/API layer for minimal runtime or REST service.      
- Outputs rendered pose overlays (PNG) and keypoints (JSON) suitable for downstream tasks like VITON‑HD preprocessing.      


## Prerequisites

- NVIDIA GPU with recent driver and NVIDIA Container Toolkit installed to enable --gpus all in Docker.      
- Confirm GPU access by running a test container with nvidia-smi after toolkit installation.      
- Pre-download OpenPose models into a local models/ folder (body_25 required; hand/face optional) to bind-mount into the container.      


## Build

Build the base OpenPose image from the repository root (Dockerfile uses CUDA 11.8 on Ubuntu 20.04).      

```bash
docker build -t pose-openpose:latest -f Dockerfile .
```

Build an API/slim runtime on top of the base (choose your provided slim or api Dockerfile).      

```bash
# If using the provided slim layer
docker build -t pose-openpose-slim:latest -f Dockerfile.slim .

# If using your API layer variant
docker build -t pose-openpose-api:latest -f Dockerfile.api .
```


## Quickstart A: Interactive (-it) CMU Demo Test

Use interactive mode to validate GPU access and try the CMU sample images inside the container.      

Start a shell with GPU access and mount pre-downloaded models.      

```bash
docker run --gpus all -it \
  -v "$PWD/models:/openpose/models" \
  pose-openpose:latest bash
```

Inside the container, run the demo on the built-in sample images and save outputs (PNG + JSON) to a temporary folder.      

```bash
cd /openpose
mkdir -p /tmp/op_out

# Disable on-screen display for headless speed, and ensure at least one write option is set
./build/examples/openpose/openpose.bin \
  --image_dir /openpose/examples/media \
  --write_images /tmp/op_out \
  --write_json /tmp/op_out \
  --display 0
```

Note: When --display 0 is used, OpenPose requires one of --write_json, --write_images, or --write_video to be set, otherwise no output is produced.   

Inspect results (rendered_*.png and *_keypoints.json) under /tmp/op_out and copy them out later or bind-mount an output directory in the next step.   

## Quickstart B: Interactive (-it) on Your Images with Host Output

Process a local folder of images and write both rendered overlays and JSON keypoints to a host-mounted output directory for later use in VITON‑HD.      

Prepare input and output folders on the host, then run interactively with GPU and bind-mount /data.      

```bash
mkdir -p data/input data/output
# Place JPG/PNG images into data/input
docker run --gpus all -it \
  -v "$PWD/models:/openpose/models" \
  -v "$PWD/data:/data" \
  pose-openpose:latest bash
```

Inside the container, call the OpenPose binary on the mounted input and write outputs to the mounted output.      

```bash
./build/examples/openpose/openpose.bin \
  --image_dir /data/input \
  --write_images /data/output \
  --write_json /data/output \
  --display 0
```

Optional: Include hands and/or face if models are available by adding --hand and/or --face flags.   

```bash
./build/examples/openpose/openpose.bin \
  --image_dir /data/input \
  --write_images /data/output \
  --write_json /data/output \
  --hand \
  --display 0
```

The JSON contains the 25 body keypoints with coordinates and confidence (and hands/face if enabled), which is commonly used by downstream pipelines like VITON‑HD. 

## Quickstart C: REST API (Optional)

If the API image is built, start the service and POST images to get JSON keypoints and a base64 PNG of the rendered pose overlay.      

Run the API image with GPU and models mounted, exposing port 8080.      

```bash
docker run --gpus all -p 8080:8080 \
  -v "$PWD/models:/openpose/models" \
  pose-openpose-api:latest
```

Send a request with an image file; the response includes keypoints JSON and a base64 PNG that can be saved client-side for inspection.      

```bash
curl -X POST http://localhost:8080/infer \
  -F "image=@/path/to/local/image.jpg" \
  -o response.json
```

For batch workflows, prefer the interactive approach with --image_dir and bind-mounted input/output folders to avoid base64 decoding overheads.      

## Tips for Speed, VRAM, and Headless Use

- Disable on-screen display in servers/headless scenarios with --display 0, and ensure one of the write flags is set so outputs are actually produced.   
- Control GPU memory by setting --net_resolution, where smaller values are faster and use less memory; e.g., --net_resolution "320x224" on low-VRAM systems.  
- Confirm GPUs are visible inside the container via nvidia-smi; if missing, verify the NVIDIA Container Toolkit installation and the --gpus all flag.      


## Outputs and File Structure

- Rendered overlays are saved as PNGs (e.g., rendered_*.png), while keypoints are saved as *_keypoints.json in the directory specified by --write_images and --write_json. 
- Each frame or image produces a corresponding JSON with keypoints; body uses 25 points, and hands/face add their own sets if enabled. 


## Using Outputs for VITON‑HD Preprocessing

- VITON‑HD pipelines use OpenPose outputs (25 body keypoints) alongside other components like human parsing and agnostic masks to prepare training/evaluation data. 
- Place the JSON keypoints and rendered overlays into the preprocessing stage expected by the chosen VITON‑HD implementation (or convert to the pose representation it requires) as indicated by community tutorials and project docs.      


## Example: End-to-End Local Batch

Given host folders data/input (images) and data/output (results), run a single interactive session, process, and collect outputs back on the host for subsequent VITON‑HD steps.      

```bash
mkdir -p data/input data/output
# add images to data/input

docker run --gpus all -it \
  -v "$PWD/models:/openpose/models" \
  -v "$PWD/data:/data" \
  pose-openpose:latest bash

# inside container
./build/examples/openpose/openpose.bin \
  --image_dir /data/input \
  --write_images /data/output \
  --write_json /data/output \
  --display 0
```


## Troubleshooting

- If nothing is saved with --display 0, ensure at least one of --write_json, --write_images, or --write_video is provided; otherwise, the demo will not produce artifacts.   
- For memory errors, reduce --net_resolution or ensure input images share a similar aspect ratio to avoid extreme intermediate resolutions for some images.    
- If the container can’t see the GPU, verify Docker version, NVIDIA driver, and NVIDIA Container Toolkit; test with docker run --gpus all ubuntu nvidia-smi.      


## Credits

- OpenPose by CMU Perceptual Computing Lab (flags and usage documented in official demo pages).      
- Command-line examples and explanations adapted from community guides demonstrating --write_json, --write_images, and --net_resolution usage.      
- VITON‑HD context based on the original paper and community preprocessing notes that rely on OpenPose keypoints.      
