FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install TensorRT from NVIDIA PyPI
RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com

# Install Linux dependencies
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ \
    && apt upgrade --no-install-recommends -y openssl tar

# Create and set working directory
RUN mkdir -p /usr/src/ultralytics
WORKDIR /usr/src/ultralytics

# Clone specific fork and branch of Ultralytics
RUN git clone https://github.com/BartekTao/ultralytics.git . \
    && git checkout feat/p3p4p5-test2

# Install Python packages
RUN python3 -m pip install --upgrade pip wheel
# simsimd will fail on non-AVX512 CPUs, so we exclude it by manually editing pyproject.toml or skipping editable mode
# Use manual package list instead of "-e ." to avoid simsimd build errors
RUN pip install --no-cache \
    ultralytics albumentations comet thop pycocotools onnx onnx-simplifier onnxruntime-gpu

# Set environment variables
ENV OMP_NUM_THREADS=1
