#!/usr/bin/env python3
"""
Integrated Grasp Pipeline:
  1. Capture RGBD from RealSense
  2. Grounding DINO detects objects from text prompt
  3. SAM2 segments the detected objects
  4. Mask point cloud to segmented region only
  5. Contact-GraspNet generates grasps on masked point cloud
  6. Visualize results
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Add SAM2 to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_PATH = os.path.join(THIS_DIR, "sam2")
sys.path.insert(0, SAM2_PATH)

# SAM2 Imports
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Grounding DINO Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Contact-GraspNet Imports
from cgn_pytorch import from_pretrained, inference as cgn_inference


class GraspPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.dino_processor = None
        self.dino_model = None
        self.sam_predictor = None
        self.cgn_model = None
        
    def load_models(self):
        """Load all models"""
        print("=" * 50)
        print("Loading models...")
        print("=" * 50)
        
        # Grounding DINO
        print("[1/3] Loading Grounding DINO...")
        dino_id = "IDEA-Research/grounding-dino-tiny"
        self.dino_processor = AutoProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device)
        
        # SAM2
        print("[2/3] Loading SAM2...")
        sam_model = build_sam2_hf("facebook/sam2-hiera-tiny", device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam_model)
        
        # Contact-GraspNet
        print("[3/3] Loading Contact-GraspNet...")
        self.cgn_model, _, _ = from_pretrained()
        self.cgn_model.to(self.device).eval()
        
        print("=" * 50)
        print("All models loaded!")
        print("=" * 50)
    
    def detect_objects(self, image_rgb, text_prompt, box_threshold=0.35):
        """
        Use Grounding DINO to detect objects matching text prompt.
        Returns bounding boxes and labels.
        """
        pil_image = Image.fromarray(image_rgb)
        
        inputs = self.dino_processor(
            images=pil_image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            target_sizes=[pil_image.size[::-1]]
        )
        
        boxes = results[0]["boxes"].cpu().numpy()
        labels = results[0]["labels"]
        scores = results[0]["scores"].cpu().numpy()
        
        return boxes, labels, scores
    
    def segment_objects(self, image_rgb, boxes):
        """
        Use SAM2 to segment objects given bounding boxes.
        Returns combined mask of all detected objects.
        """
        if len(boxes) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        
        self.sam_predictor.set_image(image_rgb)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        
        # Squeeze if needed (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Combine all masks into one
        combined_mask = np.any(masks, axis=0) if masks.ndim == 3 else masks
        
        return combined_mask.astype(bool)
    
    def create_masked_pointcloud(self, depth_raw, color_bgr, intrinsics, depth_scale,
                                  mask, min_z=0.15, max_z=1.0):
        """
        Create point cloud from RGBD, masked to segmented region only.
        """
        depth = depth_raw.astype(np.float32) * depth_scale
        h, w = depth.shape
        
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        
        ys, xs = np.indices((h, w))
        
        # Apply depth limits
        valid = (depth > min_z) & (depth < max_z)
        
        # Apply segmentation mask
        # Resize mask if dimensions don't match
        if mask.shape != depth.shape:
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_resized = mask
        
        valid &= mask_resized
        
        xs_v = xs[valid]
        ys_v = ys[valid]
        d_v = depth[valid]
        
        x = (xs_v - cx) * d_v / fx
        y = (ys_v - cy) * d_v / fy
        z = d_v
        
        pts = np.stack((x, y, z), axis=-1).astype(np.float32)
        cols_rgb = color_bgr[valid][..., ::-1] / 255.0  # BGR → RGB
        
        return pts, cols_rgb
    
    def generate_grasps(self, points, threshold=0.5, max_points=40000):
        """
        Run Contact-GraspNet on point cloud.
        """
        if len(points) == 0:
            print("Warning: Empty point cloud, no grasps generated")
            return np.array([]), np.array([]), np.array([])
        
        # Subsample if too many points
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        
        poses, scores, widths = cgn_inference(self.cgn_model, points, threshold=threshold)
        
        # Sort by score
        if len(scores) > 0:
            order = np.argsort(-scores)
            poses, scores, widths = poses[order], scores[order], widths[order]
        
        return poses, scores, widths
    
    def visualize_results(self, color_image, mask, boxes, labels, points, colors, 
                          grasp_poses, grasp_scores, num_grasps=20):
        """
        Visualize detection, segmentation, and grasps.
        """
        # --- 2D Visualization (OpenCV) ---
        vis_image = color_image.copy()
        
        # Draw mask overlay
        if mask is not None and mask.any():
            mask_resized = mask
            if mask.shape != color_image.shape[:2]:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (color_image.shape[1], color_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            overlay = vis_image.copy()
            overlay[mask_resized] = [0, 255, 0]  # Green mask
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        # Draw bounding boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(label), (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add grasp count
        cv2.putText(vis_image, f"Grasps: {len(grasp_scores)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Detection & Segmentation", vis_image)
        
        # --- 3D Visualization (Open3D) ---
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            geometries = [pcd]
            
            # Add grasp frames
            for i in range(min(num_grasps, len(grasp_scores))):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                frame.transform(grasp_poses[i])
                geometries.append(frame)
            
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Masked Point Cloud + Grasps"
            )
    
    def print_grasps(self, poses, scores, widths, limit=10):
        """Print top grasps"""
        print(f"\n{'='*50}")
        print(f"Found {len(scores)} Grasp Candidates")
        print(f"{'='*50}\n")
        
        if len(scores) == 0:
            print("No grasps found above threshold.\n")
            return
        
        for i in range(min(limit, len(scores))):
            pos = poses[i][:3, 3]
            quat = R.from_matrix(poses[i][:3, :3]).as_quat()
            print(
                f"Grasp {i+1}: "
                f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                f"score={scores[i]:.3f} width={widths[i]:.4f}"
            )


def start_realsense():
    """Initialize RealSense camera"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
    profile = pipe.start(cfg)
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"RealSense started (depth_scale={depth_scale:.6f})")
    
    align = rs.align(rs.stream.color)
    return pipe, align, depth_scale


def capture_frames(pipe, align, warmup=10):
    """Capture aligned RGBD frames"""
    time.sleep(1.0)
    
    # Warmup frames
    for _ in range(warmup):
        frames = pipe.wait_for_frames(timeout_ms=15000)
        frames = align.process(frames)
    
    # Capture frame
    frames = pipe.wait_for_frames(timeout_ms=15000)
    frames = align.process(frames)
    
    depth = np.asanyarray(frames.get_depth_frame().get_data())
    color = np.asanyarray(frames.get_color_frame().get_data())
    intrinsics = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
    
    return depth, color, intrinsics


def main():
    parser = argparse.ArgumentParser(description="Integrated Grasp Pipeline")
    parser.add_argument("--prompt", required=True, type=str,
                       help="Object to detect (e.g., 'red cup', 'cheez-it box')")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--box-threshold", default=0.35, type=float,
                       help="DINO detection threshold")
    parser.add_argument("--grasp-threshold", default=0.5, type=float,
                       help="Contact-GraspNet confidence threshold")
    parser.add_argument("--min-depth", default=0.15, type=float,
                       help="Minimum depth in meters")
    parser.add_argument("--max-depth", default=1.0, type=float,
                       help="Maximum depth in meters")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Show visualization")
    
    args = parser.parse_args()
    
    # Add period to prompt (DINO quirk)
    text_prompt = args.prompt if args.prompt.endswith(".") else args.prompt + "."
    
    # Initialize pipeline
    pipeline = GraspPipeline(device=args.device)
    pipeline.load_models()
    
    # Start RealSense
    pipe, align, depth_scale = start_realsense()
    
    try:
        print(f"\nLooking for: '{text_prompt}'")
        print("Capturing frames...")
        
        depth_raw, color_bgr, intrinsics = capture_frames(pipe, align)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"Captured: color={color_bgr.shape}, depth={depth_raw.shape}")
        
        # Step 1: Detect objects with DINO
        print("\n[Step 1] Running Grounding DINO...")
        boxes, labels, det_scores = pipeline.detect_objects(
            color_rgb, text_prompt, args.box_threshold
        )
        print(f"  Found {len(boxes)} objects: {labels}")
        
        if len(boxes) == 0:
            print("No objects detected! Try lowering --box-threshold or different prompt.")
            return
        
        # Step 2: Segment with SAM2
        print("\n[Step 2] Running SAM2 segmentation...")
        mask = pipeline.segment_objects(color_rgb, boxes)
        print(f"  Mask covers {mask.sum()} pixels ({100*mask.mean():.1f}% of image)")
        
        # Step 3: Create masked point cloud
        print("\n[Step 3] Creating masked point cloud...")
        points, colors = pipeline.create_masked_pointcloud(
            depth_raw, color_bgr, intrinsics, depth_scale,
            mask, min_z=args.min_depth, max_z=args.max_depth
        )
        print(f"  Point cloud: {len(points)} points")
        
        if len(points) < 100:
            print("Warning: Very few points in masked region. Check depth/segmentation.")
        
        # Step 4: Generate grasps
        print("\n[Step 4] Running Contact-GraspNet...")
        grasp_poses, grasp_scores, grasp_widths = pipeline.generate_grasps(
            points, threshold=args.grasp_threshold
        )
        
        # Print results
        pipeline.print_grasps(grasp_poses, grasp_scores, grasp_widths)
        
        # Visualize
        if args.visualize:
            print("\n[Visualization] Press any key on 2D window, close 3D window to exit...")
            pipeline.visualize_results(
                color_bgr, mask, boxes, labels,
                points, colors, grasp_poses, grasp_scores
            )
            cv2.waitKey(0)
        
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("\nRealSense stopped.")


if __name__ == "__main__":
    main()