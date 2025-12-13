FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ---------------------------
# System packages
# ---------------------------
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# ---------------------------
# Core Python packages
# ---------------------------
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    numpy scipy opencv-python pillow \
    trimesh open3d \
    flask gunicorn \
    einops tqdm

# ---------------------------
# SAM-3D-Objects
# ---------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /opt/sam3d
WORKDIR /opt/sam3d
RUN pip3 install -e '.[dev]' || true
RUN pip3 install git+https://github.com/microsoft/MoGe.git || true

# ---------------------------
# Copy server code
# ---------------------------
WORKDIR /app
COPY sam3d/sam3d_server.py /app/

EXPOSE 8000

CMD ["python3", "sam3d_server.py"]
