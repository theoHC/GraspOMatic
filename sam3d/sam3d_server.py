#!/usr/bin/env python3
"""
SAM3D Server - Provides 3D reconstruction as a REST API
"""

import os
import sys
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import traceback

os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CONDA_PREFIX'] = '/usr/local/cuda'

sys.path.insert(0, '/opt/sam3d/notebook')

app = Flask(__name__)

# Global model (loaded once)
sam3d_model = None

def load_model():
    global sam3d_model
    if sam3d_model is None:
        print("Loading SAM3D model...")
        try:
            from inference import Inference
            config_path = "/checkpoints/hf/pipeline.yaml"
            if os.path.exists(config_path):
                sam3d_model = Inference(config_path, compile=False)
                print("SAM3D model loaded!")
            else:
                print(f"Config not found at {config_path}")
                print("Run: huggingface-cli download facebook/sam-3d-objects --local-dir /checkpoints")
        except Exception as e:
            print(f"Failed to load SAM3D: {e}")
            traceback.print_exc()
    return sam3d_model

@app.route('/health', methods=['GET'])
def health():
    model = load_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    """
    Reconstruct 3D from image + mask
    
    Expects JSON:
    {
        "image": base64_encoded_png,
        "mask": base64_encoded_png
    }
    
    Returns JSON:
    {
        "success": true,
        "points": [[x,y,z], ...],
        "colors": [[r,g,b], ...],
        "mesh_vertices": [[x,y,z], ...],  # if available
        "mesh_faces": [[i,j,k], ...]       # if available
    }
    """
    model = load_model()
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Decode image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Decode mask
        mask_bytes = base64.b64decode(data['mask'])
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        
        print(f"Received image: {image.size}, mask: {mask.size}")
        
        # Run SAM3D
        output = model(image, mask, seed=42)
        
        result = {'success': True}
        
        # Extract mesh if available
        if "mesh" in output and output["mesh"] is not None:
            mesh = output["mesh"]
            result['mesh_vertices'] = mesh.vertices.tolist()
            result['mesh_faces'] = mesh.faces.tolist()
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                result['mesh_colors'] = (mesh.visual.vertex_colors[:, :3] / 255.0).tolist()
            print(f"Returning mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Extract Gaussian splats if available
        elif "gs" in output:
            gs = output["gs"]
            points = gs.get_xyz().detach().cpu().numpy()
            result['points'] = points.tolist()
            try:
                colors = gs.get_features()[:, :3].detach().cpu().numpy()
                colors = np.clip(colors, 0, 1)
                result['colors'] = colors.tolist()
            except:
                result['colors'] = [[0.7, 0.7, 0.7]] * len(points)
            print(f"Returning {len(points)} Gaussian splats")
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Pre-load model
    load_model()
    
    # Run server
    app.run(host='0.0.0.0', port=8000, threaded=True)