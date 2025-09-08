# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         build-essential \
#         software-properties-common \
#         wget \
#         git \
#         cmake \
#         unzip \
#         pkg-config \
#         libopencv-dev python3-opencv \
#         python3-dev python3-pip \
#         libatlas-base-dev libprotobuf-dev \
#         libleveldb-dev libsnappy-dev \
#         libhdf5-serial-dev protobuf-compiler \
#         libgflags-dev libgoogle-glog-dev \
#         liblmdb-dev opencl-headers \
#         ocl-icd-opencl-dev libviennacl-dev \
#         ca-certificates

# # Install GCC 8 and G++ 8 from toolchain PPA
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
#     apt-get update && \
#     apt-get install -y gcc-8 g++-8
# RUN apt-get install -y libboost-system-dev libboost-filesystem-dev libboost-thread-dev
# RUN git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose

# WORKDIR /openpose

# # Build OpenPose
# RUN mkdir build && cd build && \
#     cmake .. -DBUILD_PYTHON=ON -DDOWNLOAD_HAND=OFF -DDOWNLOAD_FACE=OFF&& \
#     make -j"$(nproc)" && \
#     make install

# # Set default entrypoint
# ENTRYPOINT ["/bin/bash"]
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      software-properties-common \
      wget \
      git \
      cmake \
      unzip \
      pkg-config \
      libopencv-dev python3-opencv \
      python3-dev python3-pip \
      libatlas-base-dev libprotobuf-dev \
      libleveldb-dev libsnappy-dev \
      libhdf5-serial-dev protobuf-compiler \
      libgflags-dev libgoogle-glog-dev \
      liblmdb-dev opencl-headers \
      ocl-icd-opencl-dev libviennacl-dev \
      ca-certificates \
      libboost-system-dev libboost-filesystem-dev libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Install GCC 8 / G++ 8 (needed by OpenPose/Caffe)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-8 g++-8 && \
    rm -rf /var/lib/apt/lists/*

# Clone OpenPose
RUN git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose

WORKDIR /openpose

RUN git submodule update --init --recursive --remote

# (Optional, but strongly recommended for Docker builds)
# Pre-download models on your host and copy them in BUILD CONTEXT for reliability:
COPY models/ /openpose/models/

# Build with Python API, skip hand/face model downloads if you already provide them
RUN mkdir -p build && \
    cd build && \
    CC=gcc-8 CXX=g++-8 \
    cmake .. -DBUILD_PYTHON=ON -DDOWNLOAD_HAND=OFF -DDOWNLOAD_FACE=OFF && \
    make -j"$(nproc)" && \
    make install



