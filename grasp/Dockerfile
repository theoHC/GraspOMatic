FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    udev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# ============================================================
# Pin numpy FIRST
# ============================================================
RUN pip3 install --no-cache-dir "numpy==1.26.2"

# ============================================================
# PyTorch 2.1.1 with CUDA 12.1 (what cgn-pytorch requires)
# ============================================================
RUN pip3 install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# PyTorch Geometric for torch 2.1.1
# ============================================================
RUN pip3 install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN pip3 install --no-cache-dir "torch-geometric==2.4.0"

# ============================================================
# Scientific stack pinned
# ============================================================
RUN pip3 install --no-cache-dir \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "opencv-python==4.9.0.80" \
    "Pillow==10.1.0" \
    "matplotlib==3.8.2" \
    "open3d==0.18.0" \
    "trimesh==4.0.4"

# ============================================================
# SAM2 (install without deps to avoid conflicts, then add what's needed)
# ============================================================
RUN pip3 install --no-cache-dir \
    hydra-core \
    iopath \
    "huggingface_hub>=0.20.0"

RUN git clone https://github.com/facebookresearch/sam2.git /tmp/sam2 && \
    cd /tmp/sam2 && \
    SAM2_BUILD_CUDA=0 pip3 install --no-cache-dir --no-deps -e . && \
    cd /

# ============================================================
# Grounding DINO - pin transformers for PyTorch 2.1.1 compatibility
# ============================================================
RUN pip3 install --no-cache-dir \
    "transformers==4.40.0" \
    "accelerate==0.28.0"

# ============================================================
# cgn-pytorch with all its dependencies
# ============================================================
RUN pip3 install --no-cache-dir \
    "pyrealsense2" \
    "pyrender>=0.1.45,<0.2.0" \
    "meshcat==0.3.2" \
    "pyzmq" \
    "pyngrok" \
    "typeguard" \
    "importlib_resources" \
    tqdm

# Install cgn-pytorch (now deps should mostly match)
RUN pip3 install --no-cache-dir --no-deps cgn-pytorch==0.4.3

# Force numpy back in case anything changed it
RUN pip3 install --no-cache-dir "numpy==1.26.2"

COPY . /app

CMD ["python3", "final_improved.py"]