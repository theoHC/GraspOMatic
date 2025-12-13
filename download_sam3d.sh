#!/bin/bash
# Run this inside the container after first build

# Login to HuggingFace (you'll need your token)
huggingface-cli login

# Download SAM 3D Objects checkpoints
huggingface-cli download \
    --repo-type model \
    --local-dir /opt/sam3d-objects/checkpoints/hf-download \
    facebook/sam-3d-objects

# Move to correct location
mv /opt/sam3d-objects/checkpoints/hf-download/checkpoints /opt/sam3d-objects/checkpoints/hf
rm -rf /opt/sam3d-objects/checkpoints/hf-download

echo "SAM 3D Objects checkpoints downloaded!"