# PyTorch image with Python + CUDA pre-installed
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ============================================================
# Pin numpy FIRST
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.2"

# ============================================================
# PyTorch Geometric for torch 2.1.1
# ============================================================
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN pip install --no-cache-dir "torch-geometric==2.4.0"

# ============================================================
# Scientific stack pinned (use headless opencv)
# ============================================================
RUN pip install --no-cache-dir \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "opencv-python-headless==4.9.0.80" \
    "Pillow==10.1.0" \
    "matplotlib==3.8.2" \
    "open3d==0.18.0" \
    "trimesh==4.0.4"

# ============================================================
# SAM2 (download as zip instead of git clone)
# ============================================================
RUN pip install --no-cache-dir \
    hydra-core \
    iopath \
    "huggingface_hub>=0.20.0"

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip', '/tmp/sam2.zip')" && \
    python -c "import zipfile; zipfile.ZipFile('/tmp/sam2.zip', 'r').extractall('/tmp')" && \
    cd /tmp/sam2-main && \
    SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-deps -e .

# ============================================================
# Grounding DINO
# ============================================================
RUN pip install --no-cache-dir \
    "transformers==4.40.0" \
    "accelerate==0.28.0"

# ============================================================
# cgn-pytorch dependencies
# ============================================================
RUN pip install --no-cache-dir \
    "pyrender>=0.1.45,<0.2.0" \
    "meshcat==0.3.2" \
    "pyzmq" \
    "pyngrok" \
    "typeguard" \
    "importlib_resources" \
    tqdm

# Install cgn-pytorch
RUN pip install --no-cache-dir --no-deps cgn-pytorch==0.4.3

# Force numpy back
RUN pip install --no-cache-dir "numpy==1.26.2"

COPY . /app

CMD ["python", "final_improved.py"]