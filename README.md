# openpose-docker-implimentation
A ready to use Docker setup for CMU OpenPose with Python API and model prefetching. This project provides a two-tiered build approach: first constructing a robust CUDA-enabled OpenPose base image, then layering Python HTTP API and body only app logic. Models are pre-downloaded. Designed for NVIDIA GPUs and Ubuntu 20.04 (CUDA 11.8)
