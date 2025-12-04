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
            # --- DEBUG: CHECK DEPTH VALUES BEFORE FILTERING ---
            # We look at the raw depth values inside the mask
            masked_depth_raw = depth_raw[mask]
            if len(masked_depth_raw) > 0:
                # Calculate meters using the provided scale
                depth_meters = masked_depth_raw.astype(np.float32) * depth_scale
                print(f"\n[DEPTH DEBUG] Mask Stats:")
                print(f"  - Raw Units: min={masked_depth_raw.min()}, max={masked_depth_raw.max()}")
                print(f"  - In Meters: min={depth_meters.min():.4f}m, max={depth_meters.max():.4f}m")
                print(f"  - Filter Range: {min_z}m to {max_z}m")
            else:
                print("\n[DEPTH DEBUG] Mask is empty!")

            # Normal processing continues...
            depth = depth_raw.astype(np.float32) * depth_scale
            
            h, w = depth.shape
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
            
            ys, xs = np.indices((h, w))
            
            # Apply depth limits
            valid = (depth > min_z) & (depth < max_z)
            
            # Apply segmentation mask
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
            cols_rgb = color_bgr[valid][..., ::-1] / 255.0  # BGR -> RGB
            
            # --- CLEANING PIPELINE ---
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols_rgb)
            
            if len(pts) > 10:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
                labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50, print_progress=False))
                
                if len(labels) > 0 and labels.max() >= 0:
                    counts = np.bincount(labels[labels >= 0])
                    largest_cluster_idx = np.argmax(counts)
                    valid_ind = np.where(labels == largest_cluster_idx)[0]
                    pcd = pcd.select_by_index(valid_ind)

            pts_clean = np.asarray(pcd.points)
            cols_clean = np.asarray(pcd.colors)
            
            return pts_clean, cols_clean
    
    
    def generate_grasps(self, points, threshold=0.5, max_points=20000):
            """
            Run Contact-GraspNet with Automatic Unit Scaling (mm -> m) + Type Casting Fix.
            """
            if len(points) < 50:
                print("Warning: Point cloud too sparse for grasp generation")
                return np.array([]), np.array([]), np.array([])
            
            # 1. Downsample
            if len(points) > max_points:
                idx = np.random.choice(len(points), max_points, replace=False)
                points = points[idx]

            # 2. Mean Centering
            pc_mean = np.mean(points, axis=0)
            points_centered = points - pc_mean

            # 3. Inference
            print(f"DEBUG: Running inference on {len(points_centered)} points...")
            poses, scores, widths = cgn_inference(self.cgn_model, points_centered, threshold=threshold)
            
            if scores is None or len(scores) == 0:
                print("DEBUG: cgn_inference returned no grasps.")
                return np.array([]), np.array([]), np.array([])

            # --- UNIT FIX (CRASH PATCHED) ---
            # 1. Ensure widths is Float to prevent NumPy casting errors
            widths = widths.astype(np.float64)
            
            # 2. Check for Millimeters (Median > 1.0 meter is impossible for a shoe)
            median_width = np.median(widths)
            if median_width > 1.0:
                print(f"DEBUG: Detected MM scale (Median width={median_width:.2f}). Converting to Meters...")
                widths = widths / 1000.0
                
                # Check translation magnitude
                trans_mag = np.mean(np.linalg.norm(poses[:, :3, 3], axis=1))
                if trans_mag > 1.0: 
                    print(f"DEBUG: Detected MM translation (Mag={trans_mag:.2f}). Converting...")
                    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0

            # 4. Restore Coordinates (Add the mean back)
            poses[:, :3, 3] += pc_mean

            # 5. Filtering
            final_poses = []
            final_scores = []
            final_widths = []
            
            # Sort high to low
            order = np.argsort(-scores)
            poses = poses[order]
            scores = scores[order]
            widths = widths[order]

            min_width = 0.002  # 2mm
            max_width = 0.15   # 15cm
            min_dist  = 0.02   # 2cm
            
            print(f"DEBUG: Analyzing top candidates...")

            for i in range(len(scores)):
                p = poses[i]
                s = scores[i]
                w = widths[i]
                
                # Filter 1: Width
                if w < min_width or w > max_width:
                    if i < 5: print(f"  [{i}] REJECTED Width {w:.4f}m")
                    continue
                
                # Filter 2: NMS
                is_too_close = False
                for existing_p in final_poses:
                    if np.linalg.norm(p[:3, 3] - existing_p[:3, 3]) < min_dist:
                        is_too_close = True
                        break
                
                if not is_too_close:
                    final_poses.append(p)
                    final_scores.append(s)
                    final_widths.append(w)
                
                if len(final_poses) >= 50:
                    break

            print(f"DEBUG: Found {len(final_poses)} valid grasps after filtering.")
            return np.array(final_poses), np.array(final_scores), np.array(final_widths)


    def print_grasps(self, poses, scores, widths, max_to_print=10):
        """Pretty-print top grasps"""
        n = min(len(scores), max_to_print)
        print("\n=== Top Grasps ===")
        if n == 0:
            print("No grasps found.")
            return

        for i in range(n):
            print(f"\nGrasp {i+1}:")
            print(f"  Score:  {scores[i]:.3f}")
            print(f"  Width:  {widths[i]:.4f} m")
            print(f"  Pose:\n{poses[i]}")

    
    def create_gripper(self, pos, width, color, Rmat, jaw_length=0.05, tube_radius=0.004):
        """
        Create gripper geometry based on the visual evidence:
        - Z-Axis (Blue) = Approach (Hand -> Object)
        - X-Axis (Red)  = Width (Finger Separation)
        - Y-Axis (Green)= Orthogonal (Height)
        """
        # --- Force Clamp Width ---
        # Your logs showed widths like 8553.0m. We MUST clamp this.
        w_raw = float(width)
        if w_raw > 0.15 or w_raw <= 0.001: 
            width_m = 0.08 
        else:
            width_m = w_raw

        # --- EXTRACT AXES (Based on Visual Debugging) ---
        x_axis = Rmat[:3, 0] # Red   = Width / Separation
        y_axis = Rmat[:3, 1] # Green = Orthogonal
        z_axis = Rmat[:3, 2] # Blue  = Approach (Points INTO object)

        # --- GEOMETRIC CALCULATIONS ---
        # 1. Separation is along the X-axis (Red)
        half_w = width_m / 2.0
        
        # 'pos' is the grasp center (between fingertips)
        center = pos 

        # 2. Calculate Key Points
        # Fingertips are separated along X
        p_left_tip  = center - (x_axis * half_w)
        p_right_tip = center + (x_axis * half_w)
        
        # Finger Bases are "backwards" along the negative Z-axis
        # (Since Z points INTO the object, we go -Z to find the hand base)
        p_left_base   = p_left_tip - (z_axis * jaw_length)
        p_right_base  = p_right_tip - (z_axis * jaw_length)

        # 3. Centers for Cylinders
        center_left_finger  = (p_left_tip + p_left_base) / 2.0
        center_right_finger = (p_right_tip + p_right_base) / 2.0
        center_back_bar     = (p_left_base + p_right_base) / 2.0
        
        # 4. Handle (extends further back from the bar)
        handle_len = 0.04
        p_handle_end  = center_back_bar - (z_axis * handle_len)
        center_handle = (center_back_bar + p_handle_end) / 2.0

        # --- ROTATION MATRICES ---
        # We need to rotate the default Open3D cylinder (which points along Z [0,0,1])
        # to match our Grasp Axes.

        # A. Fingers & Handle: They run along the Z-axis of the Grasp Frame.
        # Since the cylinder is ALREADY Z, we just apply Rmat directly!
        R_aligned_z = Rmat

        # B. Back Bar: It runs along the X-axis of the Grasp Frame.
        # We need to rotate Cylinder-Z to Grasp-X.
        # Rotation: +90 deg around Y-axis maps Z(0,0,1) -> X(1,0,0)
        R_z_to_x = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
        R_aligned_x = Rmat @ R_z_to_x

        # --- CREATE MESHES ---
        geometries = []
        
        def make_cyl(radius, height, R, center, c):
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
            cyl.rotate(R, center=[0,0,0])
            cyl.translate(center)
            cyl.compute_vertex_normals()
            cyl.paint_uniform_color(c)
            return cyl

        # Left Finger (Runs along Z)
        geometries.append(make_cyl(tube_radius, jaw_length, R_aligned_z, center_left_finger, color))
        
        # Right Finger (Runs along Z)
        geometries.append(make_cyl(tube_radius, jaw_length, R_aligned_z, center_right_finger, color))
        
        # Back Bar (Runs along X - Connecting the bases)
        geometries.append(make_cyl(tube_radius, width_m, R_aligned_x, center_back_bar, color))
        
        # Handle (Runs along Z)
        geometries.append(make_cyl(tube_radius, handle_len, R_aligned_z, center_handle, color))

        return geometries
    
    def visualize_results(self, color_image, mask, boxes, labels, points, colors, 
                            grasp_poses, grasp_scores, grasp_widths, num_grasps=20):
            """
            Visualize detection, segmentation, and grasps with color-coded grippers.
            Non-blocking loop to keep both OpenCV and Open3D windows responsive.
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
                overlay[mask_resized] = [0, 255, 0]
                vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Draw boxes
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, str(label), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis_image, f"Grasps: {len(grasp_scores)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Create the window initially
            cv2.imshow("Detection & Segmentation", vis_image)
            cv2.waitKey(1) 

            # --- 3D Visualization (Open3D) ---
            if len(points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # Fix Open3D upside-down orientation
                flip_x = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
                pcd.transform(flip_x)

                vis = o3d.visualization.Visualizer()
                vis.create_window("Masked Point Cloud + Grippers", width=1200, height=700)
                vis.add_geometry(pcd)

                # --- COLOR MAPPING LOGIC ---
                n_show = min(num_grasps, len(grasp_scores))
                if n_show > 0:
                    min_score = np.min(grasp_scores[:n_show])
                    max_score = np.max(grasp_scores[:n_show])
                    range_score = max_score - min_score if max_score > min_score else 1.0

                for i in range(n_show):
                    pose = grasp_poses[i]
                    score = grasp_scores[i]
                    width = grasp_widths[i]

                    # Generate Color: 0.0 (Worst)->Red, 1.0 (Best)->Green
                    val = (score - min_score) / range_score
                    gripper_color = [1.0 - val, val, 0.0]

                    geom_list = self.create_gripper(pose[:3,3], width, gripper_color, pose[:3,:3], 
                                                    jaw_length=0.05, tube_radius=0.004)
                    
                    for g in geom_list:
                        if isinstance(g, o3d.geometry.Geometry):
                            g.transform(flip_x)
                        vis.add_geometry(g)

                # Camera Setup
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                try:
                    ctr.set_zoom(0.35)
                except:
                    ctr.fit_bounds()

                # --- CUSTOM EVENT LOOP ---
                print("\n[Controls]")
                print("  - Press 'q' or 'ESC' on the OpenCV window to exit.")
                print("  - Close the Open3D window to exit.")
                
                while True:
                    # 1. Update Open3D
                    # poll_events returns False if the window is closed
                    if not vis.poll_events():
                        break
                    vis.update_renderer()

                    # 2. Update OpenCV
                    # We show the image every loop to keep the window responsive
                    cv2.imshow("Detection & Segmentation", vis_image)
                    key = cv2.waitKey(10) & 0xFF
                    
                    if key == ord('q') or key == 27: # q or ESC
                        break
                
                vis.destroy_window()

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
                points, colors, grasp_poses, grasp_scores, grasp_widths
            )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("\nRealSense stopped.")


if __name__ == "__main__":
    main()