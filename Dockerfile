FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Limit pip parallel downloads
ENV PIP_NO_CACHE_DIR=1
ENV MAX_JOBS=2
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install Python 3.12 and dependencies including OpenGL for pymeshlab
RUN apt update && apt install -y --no-install-recommends \
    software-properties-common build-essential cmake \
    python3 python3-pip python3-dev python3-venv \
    git git-lfs wget curl aria2 ffmpeg ca-certificates \
    libgl1 libglib2.0-0 libegl1 libgomp1 \
    libopengl0 libglx0 \
    libeigen3-dev ninja-build && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install ComfyUI
RUN git clone --branch v0.6.0 https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui
WORKDIR /opt/comfyui

# Install PyTorch first
RUN python3 -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

# Install ComfyUI requirements
RUN python3 -m pip install -r requirements.txt --break-system-packages

# Install ComfyUI-Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
RUN python3 -m pip install -r custom_nodes/ComfyUI-Manager/requirements.txt --break-system-packages

# Copy UltraShape repo
COPY ultrashape_repo /opt/comfyui/ultrashape_repo

# Install exact versions from working config
RUN python3 -m pip install triton==3.4.0 pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 \
    tqdm==4.67.1 easydict==1.13 opencv-python-headless==4.12.0.88 trimesh==4.10.1 \
    transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 timm==1.0.22 \
    --break-system-packages

# Install flash_attn (Python 3.12 compatible)
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    https://github.com/camenduru/wheels/releases/download/trellis2/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# Install UltraShape dependencies
RUN python3 -m pip install --break-system-packages \
    einops omegaconf pytorch_lightning accelerate diffusers \
    PyYAML safetensors scikit-image typeguard wandb tensorboard \
    pymeshlab pythreejs torchdiffeq onnxruntime rembg huggingface_hub

# Install PyTorch3D
RUN python3 -m pip install --break-system-packages --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install cubvh with explicit CUDA architecture
RUN ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen && \
    git clone https://github.com/ashawkey/cubvh.git /tmp/cubvh && \
    cd /tmp/cubvh && \
    TORCH_CUDA_ARCH_LIST="8.6" python3 setup.py build_ext --inplace && \
    mkdir -p /usr/local/lib/python3.12/dist-packages/cubvh && \
    cp _cubvh*.so /usr/local/lib/python3.12/dist-packages/cubvh/ && \
    echo "from ._cubvh import *" > /usr/local/lib/python3.12/dist-packages/cubvh/__init__.py && \
    cd / && rm -rf /tmp/cubvh

# Copy custom nodes
COPY custom_nodes/ComfyUI-UltraShape /opt/comfyui/custom_nodes/ComfyUI-UltraShape

WORKDIR /opt/comfyui

EXPOSE 8188

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--output-directory", "/opt/comfyui/output"]