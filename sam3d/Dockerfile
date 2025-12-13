# PyTorch image with Python + CUDA pre-installed
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --------------------------- 
# Core Python packages (added omegaconf)
# --------------------------- 
RUN pip install --no-cache-dir \
    numpy scipy pillow \
    trimesh open3d \
    flask gunicorn \
    einops tqdm \
    opencv-python-headless \
    omegaconf \
    hydra-core \
    iopath

# --------------------------- 
# SAM-3D-Objects (download as zip instead of git clone)
# --------------------------- 
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/facebookresearch/sam-3d-objects/archive/refs/heads/main.zip', '/tmp/sam3d.zip')" && \
    python -c "import zipfile; zipfile.ZipFile('/tmp/sam3d.zip', 'r').extractall('/opt')" && \
    mv /opt/sam-3d-objects-main /opt/sam3d && \
    rm /tmp/sam3d.zip

WORKDIR /opt/sam3d
RUN pip install -e '.[dev]' || true

# Install MoGe via zip download as well
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/microsoft/MoGe/archive/refs/heads/main.zip', '/tmp/moge.zip')" && \
    python -c "import zipfile; zipfile.ZipFile('/tmp/moge.zip', 'r').extractall('/tmp')" && \
    pip install -e /tmp/MoGe-main || true

# --------------------------- 
# Copy server code
# --------------------------- 
WORKDIR /app
COPY sam3d/sam3d_server.py /app/

# Add sam3d to Python path
ENV PYTHONPATH="/opt/sam3d:/opt/sam3d/notebook:${PYTHONPATH}"

EXPOSE 8000

CMD ["python", "sam3d_server.py"]