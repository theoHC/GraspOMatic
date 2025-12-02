FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric dependencies
RUN pip3 install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

RUN pip3 install --no-cache-dir torch-geometric==2.7.0

RUN pip3 install meshcat

RUN pip3 install importlib_resources

# Install other dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.26.2 \
    scipy==1.11.4 \
    open3d==0.19.0 \
    pyrealsense2 \
    scikit-learn==1.3.2 \
    matplotlib \
    tqdm \
    Pillow \
    trimesh

# Install cgn-pytorch (ignore version conflicts, we've handled deps manually)
RUN pip3 install --no-cache-dir --no-deps cgn-pytorch==0.4.3

# Copy project files
COPY . /app

# Default command
CMD ["python3", "final_improved.py"]