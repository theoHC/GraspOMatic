#!/usr/bin/env python3
"""
Client to communicate with SAM3D server
"""

import os
import io
import base64
import requests
import numpy as np
from PIL import Image
import open3d as o3d

SAM3D_URL = os.environ.get('SAM3D_URL', 'http://localhost:8000')

def check_sam3d_available():
    """Check if SAM3D server is running"""
    try:
        resp = requests.get(f"{SAM3D_URL}/health", timeout=5)
        data = resp.json()
        return data.get('model_loaded', False)
    except:
        return False

def reconstruct_3d(image_rgb: np.ndarray, mask: np.ndarray):
    """
    Send image + mask to SAM3D server, get back 3D reconstruction.
    
    Args:
        image_rgb: RGB image as numpy array (H, W, 3)
        mask: Binary mask as numpy array (H, W)
    
    Returns:
        o3d_mesh: Open3D mesh (or None)
        o3d_pcd: Open3D point cloud
    """
    # Encode image as base64 PNG
    img_pil = Image.fromarray(image_rgb)
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Encode mask as base64 PNG
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8, mode='L')
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format='PNG')
    mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
    
    # Send to server
    print(f"[SAM3D Client] Sending to {SAM3D_URL}/reconstruct...")
    try:
        resp = requests.post(
            f"{SAM3D_URL}/reconstruct",
            json={'image': img_b64, 'mask': mask_b64},
            timeout=120  # 2 min timeout for reconstruction
        )
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[SAM3D Client] Request failed: {e}")
        return None, None
    
    if not data.get('success', False):
        print(f"[SAM3D Client] Reconstruction failed: {data.get('error', 'Unknown')}")
        return None, None
    
    o3d_mesh = None
    o3d_pcd = None
    
    # Build mesh if available
    if 'mesh_vertices' in data:
        vertices = np.array(data['mesh_vertices'])
        faces = np.array(data['mesh_faces'])
        
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh.compute_vertex_normals()
        
        if 'mesh_colors' in data:
            colors = np.array(data['mesh_colors'])
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.9])
        
        # Sample point cloud from mesh
        o3d_pcd = o3d_mesh.sample_points_uniformly(number_of_points=30000)
        
        print(f"[SAM3D Client] Received mesh: {len(vertices)} vertices")
    
    # Build point cloud if available
    elif 'points' in data:
        points = np.array(data['points'])
        colors = np.array(data.get('colors', [[0.7, 0.7, 0.7]] * len(points)))
        
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"[SAM3D Client] Received {len(points)} points")
    
    return o3d_mesh, o3d_pcd