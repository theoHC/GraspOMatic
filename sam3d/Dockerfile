# PyTorch image with Python + CUDA pre-installed
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install git via conda (apt-get doesn't work on Lambda)
RUN conda install -y git && conda clean -ya

# --------------------------- 
# Core Python packages
# --------------------------- 
RUN pip install --no-cache-dir \
    numpy scipy pillow \
    trimesh open3d \
    flask gunicorn \
    einops tqdm \
    opencv-python-headless

# --------------------------- 
# SAM-3D-Objects
# --------------------------- 
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /opt/sam3d

WORKDIR /opt/sam3d
RUN pip install -e '.[dev]' || true
RUN pip install git+https://github.com/microsoft/MoGe.git || true

# --------------------------- 
# Copy server code
# --------------------------- 
WORKDIR /app
COPY sam3d/sam3d_server.py /app/

EXPOSE 8000

CMD ["python", "sam3d_server.py"]