#!/usr/bin/env python3
"""
Integrated Grasp Pipeline with SAM 3D Objects for full 3D reconstruction.
Aligns SAM3D output to real-world pose using RealSense depth.
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
import copy
import traceback

# Use SAM3D client (HTTP) if available
from sam3d_client import check_sam3d_available, reconstruct_3d

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CONDA_PREFIX"] = "/usr/local/cuda"  # Fake it (keep if needed)

# Add paths - run from ~ or /home to avoid sam2 naming conflict
sys.path.insert(0, "/app/sam2")
sys.path.insert(0, "/app/sam-3d-objects")
sys.path.insert(0, "/app/sam-3d-objects/notebook")

# SAM2 Imports
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Grounding DINO Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Contact-GraspNet Imports
from cgn_pytorch import from_pretrained, inference as cgn_inference

# Try to import local SAM 3D Objects inference as fallback if client not available
SAM3D_LOCAL_AVAILABLE = False
try:
    from inference import Inference as SAM3DInference
    SAM3D_LOCAL_AVAILABLE = True
    print("Local SAM 3D Objects library available (will be used as fallback).")
except Exception as e:
    print(f"Local SAM 3D Objects not available: {e}")


class GraspPipeline:
    def __init__(self, device="cuda", use_sam3d=False):
        self.device = device
        # keep this the single source of truth for wanting SAM3D
        self.use_sam3d = bool(use_sam3d)

        self.dino_processor = None
        self.dino_model = None
        self.sam_predictor = None
        self.cgn_model = None
        self.sam3d_local = None  # local SAM3D inference object (if used)
        self.sam3d_client_available = False

        # If the user asked to use SAM3D, check client availability
        if self.use_sam3d:
            try:
                self.sam3d_client_available = check_sam3d_available()
            except Exception as e:
                print(f"WARNING: check_sam3d_available() call failed: {e}")
                self.sam3d_client_available = False

            if self.sam3d_client_available:
                print("SAM3D HTTP client available — will use remote SAM3D server.")
            else:
                # if client not available, try local fallback
                if SAM3D_LOCAL_AVAILABLE:
                    print("SAM3D client not available; will try local SAM3D library as fallback.")
                else:
                    print("WARNING: SAM3D not available (client and local missing). Disabling SAM3D.")
                    self.use_sam3d = False

    def load_models(self):
        """Load all models"""
        print("=" * 50)
        print("Loading models...")
        print("=" * 50)

        # Grounding DINO
        print("[1/4] Loading Grounding DINO...")
        dino_id = "IDEA-Research/grounding-dino-tiny"
        self.dino_processor = AutoProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device)

        # SAM2
        print("[2/4] Loading SAM2...")
        sam_model = build_sam2_hf("facebook/sam2-hiera-tiny", device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam_model)

        # Contact-GraspNet
        print("[3/4] Loading Contact-GraspNet...")
        self.cgn_model, _, _ = from_pretrained()
        self.cgn_model.to(self.device).eval()

        # SAM 3D Objects (local) - only instantiate if using local fallback
        if self.use_sam3d and not self.sam3d_client_available:
            print("[4/4] Attempting to load local SAM 3D Objects...")
            config_path = "/app/sam-3d-objects/checkpoints/hf/pipeline.yaml"
            if os.path.exists(config_path) and SAM3D_LOCAL_AVAILABLE:
                try:
                    self.sam3d_local = SAM3DInference(config_path, compile=False)
                    print("  Local SAM 3D Objects loaded!")
                except Exception as e:
                    print(f"  WARNING: Failed to load local SAM3D: {e}")
                    traceback.print_exc()
                    self.use_sam3d = False
            else:
                print(f"  WARNING: SAM3D config not found at {config_path} or local code missing.")
                print("  Run download_sam3d.sh to get checkpoints or start the SAM3D server.")
                self.use_sam3d = False
        else:
            print("[4/4] SAM 3D Objects: skipped (remote client will be used if available).")

        print("=" * 50)
        print("All models loaded!")
        print("=" * 50)

    def detect_objects(self, image_rgb, text_prompt, box_threshold=0.35):
        """Use Grounding DINO to detect objects."""
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
        """Use SAM2 to segment objects."""
        if len(boxes) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        self.sam_predictor.set_image(image_rgb)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        combined_mask = np.any(masks, axis=0) if masks.ndim == 3 else masks
        return combined_mask.astype(bool)

    def create_masked_pointcloud(self, depth_raw, color_bgr, intrinsics, depth_scale,
                                  mask, min_z=0.15, max_z=1.0):
        """Create point cloud from RGBD, masked to segmented region."""
        depth = depth_raw.astype(np.float32) * depth_scale

        h, w = depth.shape
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        ys, xs = np.indices((h, w))

        valid = (depth > min_z) & (depth < max_z)

        if mask.shape != depth.shape:
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_resized = mask

        valid &= mask_resized

        xs_v, ys_v, d_v = xs[valid], ys[valid], depth[valid]

        x = (xs_v - cx) * d_v / fx
        y = (ys_v - cy) * d_v / fy
        z = d_v

        pts = np.stack((x, y, z), axis=-1).astype(np.float32)
        cols_rgb = color_bgr[valid][..., ::-1] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols_rgb)

        if len(pts) > 10:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50, print_progress=False))
            if len(labels) > 0 and labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                largest = np.argmax(counts)
                pcd = pcd.select_by_index(np.where(labels == largest)[0])

        return pcd

    def reconstruct_3d_sam3d(self, image_rgb, mask):
        """
        Unified SAM3D reconstruction:
          - If client available, call remote server via reconstruct_3d(image, mask).
          - Else if local SAM3D loaded, call local inference.
        Returns:
          mesh (open3d mesh or None), pcd (open3d point cloud or None)
        """
        if not self.use_sam3d:
            return None, None

        # Prefer remote client
        if self.sam3d_client_available:
            print("\n[SAM 3D] Calling SAM3D server (HTTP client)...")
            try:
                mesh, pcd = reconstruct_3d(image_rgb, mask)
                if pcd is not None or mesh is not None:
                    print("  ✓ Received reconstruction from SAM3D server.")
                    return mesh, pcd
                else:
                    print("  ✗ SAM3D server returned no reconstruction.")
            except Exception as e:
                print(f"  ✗ SAM3D client call failed: {e}")
                traceback.print_exc()

            # If client failed, try local fallback
            if not SAM3D_LOCAL_AVAILABLE:
                return None, None
            print("  Falling back to local SAM3D inference...")

        # Local SAM3D fallback
        if self.sam3d_local is not None:
            print("\n[SAM 3D] Running local SAM3D inference (fallback)...")
            try:
                image_pil = Image.fromarray(image_rgb)
                mask_uint8 = (mask.astype(np.uint8) * 255)
                mask_pil = Image.fromarray(mask_uint8)

                output = self.sam3d_local(image_pil, mask_pil, seed=42)

                if output is None:
                    print("  ✗ Local SAM3D returned None.")
                    return None, None

                # Mesh case
                if "mesh" in output and output["mesh"] is not None:
                    mesh_t = output["mesh"]
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_t.vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_t.faces)
                    o3d_mesh.compute_vertex_normals()

                    # vertex colors if available
                    try:
                        if hasattr(mesh_t.visual, "vertex_colors") and mesh_t.visual.vertex_colors is not None:
                            colors = mesh_t.visual.vertex_colors[:, :3] / 255.0
                            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                    except Exception:
                        pass

                    pcd = o3d_mesh.sample_points_uniformly(number_of_points=30000)
                    print("  ✓ Local SAM3D produced mesh and sampled PCD.")
                    return o3d_mesh, pcd

                # Gaussian splat / point-based output
                if "gs" in output:
                    gs = output["gs"]
                    try:
                        points = gs.get_xyz().detach().cpu().numpy()
                        colors = gs.get_features()[:, :3].detach().cpu().numpy()
                        colors = np.clip(colors, 0, 1)
                    except Exception:
                        points = None
                        colors = None

                    if points is not None:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        if colors is not None:
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)) * 0.7)
                        print(f"  ✓ Local SAM3D returned {len(points)} splat points.")
                        return None, pcd

                print("  ✗ Local SAM3D returned unknown output keys:", list(output.keys()))
                return None, None

            except Exception as e:
                print(f"  ✗ Local SAM3D inference failed: {e}")
                traceback.print_exc()
                return None, None

        return None, None

    def align_sam3d_to_depth(self, sam3d_pcd, depth_pcd, max_iterations=50):
        """
        Align SAM3D point cloud to RealSense depth point cloud using ICP.
        Returns transformation, aligned_sam3d_pcd, scale
        """
        if sam3d_pcd is None or depth_pcd is None or len(sam3d_pcd.points) == 0 or len(depth_pcd.points) == 0:
            print("Alignment skipped: missing or empty point clouds.")
            return np.eye(4), sam3d_pcd, 1.0

        print("\n[Alignment] Aligning SAM3D to RealSense depth...")

        sam3d_center = sam3d_pcd.get_center()
        depth_center = depth_pcd.get_center()

        sam3d_bbox = sam3d_pcd.get_axis_aligned_bounding_box()
        depth_bbox = depth_pcd.get_axis_aligned_bounding_box()

        sam3d_extent = sam3d_bbox.get_extent()
        depth_extent = depth_bbox.get_extent()

        # Avoid zero extents
        sam_mean = np.mean(sam3d_extent) if np.mean(sam3d_extent) > 1e-8 else 1.0
        depth_mean = np.mean(depth_extent) if np.mean(depth_extent) > 1e-8 else 1.0

        scale = depth_mean / sam_mean

        print(f"  SAM3D extent: {sam3d_extent}")
        print(f"  Depth extent: {depth_extent}")
        print(f"  Scale factor: {scale:.4f}")

        sam3d_working = copy.deepcopy(sam3d_pcd)
        sam3d_working.translate(-sam3d_center)
        sam3d_working.scale(scale, center=[0, 0, 0])
        sam3d_working.translate(depth_center)

        sam3d_working.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        depth_pcd_copy = copy.deepcopy(depth_pcd)
        depth_pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

        threshold = 0.02  # 2cm threshold

        reg_result = o3d.pipelines.registration.registration_icp(
            sam3d_working,
            depth_pcd_copy,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )

        print(f"  ICP fitness: {reg_result.fitness:.4f}")
        print(f"  ICP RMSE: {reg_result.inlier_rmse:.6f}")

        sam3d_working.transform(reg_result.transformation)

        # Build transformation T_total = T_icp @ T_translate @ T_scale @ T_center
        T_center = np.eye(4)
        T_center[:3, 3] = -sam3d_center

        T_scale = np.eye(4)
        T_scale[:3, :3] *= scale

        T_translate = np.eye(4)
        T_translate[:3, 3] = depth_center

        transformation = reg_result.transformation @ T_translate @ T_scale @ T_center

        return transformation, sam3d_working, scale

    def generate_grasps(self, points, camera_position=None, threshold=0.5,
                        max_points=20000, target_count=50):
        """Run Contact-GraspNet with camera-aware filtering."""
        if len(points) < 50:
            print("Warning: Point cloud too sparse")
            return np.array([]), np.array([]), np.array([])

        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]

        pc_mean = np.mean(points, axis=0)
        points_centered = points - pc_mean

        poses, scores, widths = cgn_inference(self.cgn_model, points_centered, threshold=threshold)

        if scores is None or len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        widths = widths.astype(np.float64)
        if np.median(widths) > 1.0:
            widths /= 1000.0
            poses[:, :3, 3] /= 1000.0

        poses[:, :3, 3] += pc_mean

        if camera_position is None:
            camera_position = np.array([0.0, 0.0, 0.0])

        final_poses, final_scores, final_widths = [], [], []

        order = np.argsort(-scores)
        poses, scores, widths = poses[order], scores[order], widths[order]

        for i in range(len(scores)):
            p, s, w = poses[i], scores[i], widths[i]

            if w < 0.002 or w > 0.15:
                continue

            grasp_pos = p[:3, 3]
            approach_dir = p[:3, 2]
            cam_to_grasp = grasp_pos - camera_position
            cam_to_grasp /= (np.linalg.norm(cam_to_grasp) + 1e-8)

            alignment = np.dot(approach_dir, cam_to_grasp)

            is_close = any(np.linalg.norm(p[:3, 3] - ep[:3, 3]) < 0.02 for ep in final_poses)

            if not is_close:
                final_poses.append(p)
                final_scores.append(s)
                final_widths.append(w)

            if len(final_poses) >= target_count:
                break

        if final_poses:
            order = np.argsort(-np.array(final_scores))
            final_poses = [final_poses[i] for i in order]
            final_scores = [final_scores[i] for i in order]
            final_widths = [final_widths[i] for i in order]

        return np.array(final_poses), np.array(final_scores), np.array(final_widths)

    def print_grasps(self, poses, scores, widths, max_to_print=10):
        """Pretty-print top grasps"""
        n = min(len(scores), max_to_print)
        print("\n=== Top Grasps ===")
        if n == 0:
            print("No grasps found.")
            return
        for i in range(n):
            print(f"\nGrasp {i+1}: Score={scores[i]:.3f}, Width={widths[i]:.4f}m")
            print(f"  Position: [{poses[i][0,3]:.3f}, {poses[i][1,3]:.3f}, {poses[i][2,3]:.3f}]")

    def create_gripper(self, pos, width, color, Rmat, jaw_length=0.05, tube_radius=0.004):
        """Create gripper geometry."""
        w = min(max(float(width), 0.002), 0.15)
        half_w = w / 2.0

        x_axis, z_axis = Rmat[:3, 0], Rmat[:3, 2]

        p_left_tip = pos - x_axis * half_w
        p_right_tip = pos + x_axis * half_w
        p_left_base = p_left_tip - z_axis * jaw_length
        p_right_base = p_right_tip - z_axis * jaw_length

        center_back = (p_left_base + p_right_base) / 2.0
        p_handle_end = center_back - z_axis * 0.04

        R_z = Rmat
        R_x = Rmat @ o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0])

        def cyl(r, h, R, c, col):
            m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h)
            m.rotate(R, [0,0,0])
            m.translate(c)
            m.compute_vertex_normals()
            m.paint_uniform_color(col)
            return m

        return [
            cyl(tube_radius, jaw_length, R_z, (p_left_tip + p_left_base)/2, color),
            cyl(tube_radius, jaw_length, R_z, (p_right_tip + p_right_base)/2, color),
            cyl(tube_radius, w, R_x, center_back, color),
            cyl(tube_radius, 0.04, R_z, (center_back + p_handle_end)/2, color)
        ]

    def visualize_results_with_mesh(self, color_image, mask, boxes, labels,
                                     depth_pcd, sam3d_mesh, sam3d_pcd_aligned,
                                     grasp_poses, grasp_scores, grasp_widths,
                                     num_grasps=20, show_depth=True, show_mesh=True):
        """
        Visualize results with:
        - Depth point cloud (partial, from RealSense)
        - SAM3D mesh/point cloud (complete, aligned)
        - Grasps on the full 3D model
        """
        # 2D Visualization
        vis_image = color_image.copy()

        if mask is not None and mask.any():
            if mask.shape != color_image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (color_image.shape[1], color_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = vis_image.copy()
            overlay[mask] = [0, 255, 0]
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(label), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(vis_image, f"Grasps: {len(grasp_scores)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, "SAM3D + Depth Aligned", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Detection & Segmentation", vis_image)
        cv2.waitKey(1)

        # 3D Visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window("SAM3D Mesh + Grasps (Real-World Pose)", width=1400, height=800)

        # Flip for visualization (Open3D convention)
        flip = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        # Add depth point cloud
        if show_depth and depth_pcd is not None and len(depth_pcd.points) > 0:
            depth_vis = copy.deepcopy(depth_pcd)
            depth_colors = np.asarray(depth_vis.colors)
            if len(depth_colors) > 0:
                depth_colors[:, 0] = np.clip(depth_colors[:, 0] * 1.2, 0, 1)
                depth_colors[:, 1] = depth_colors[:, 1] * 0.5
                depth_colors[:, 2] = depth_colors[:, 2] * 0.5
                depth_vis.colors = o3d.utility.Vector3dVector(depth_colors)
            depth_vis.transform(flip)
            vis.add_geometry(depth_vis)

        if show_mesh and sam3d_mesh is not None:
            mesh_vis = copy.deepcopy(sam3d_mesh)
            mesh_vis.transform(flip)
            vis.add_geometry(mesh_vis)

        if sam3d_pcd_aligned is not None and len(sam3d_pcd_aligned.points) > 0:
            sam3d_vis = copy.deepcopy(sam3d_pcd_aligned)
            sam3d_colors = np.asarray(sam3d_vis.colors)
            if len(sam3d_colors) > 0:
                sam3d_colors[:, 0] = sam3d_colors[:, 0] * 0.5
                sam3d_colors[:, 1] = sam3d_colors[:, 1] * 0.7
                sam3d_colors[:, 2] = np.clip(sam3d_colors[:, 2] * 1.3, 0, 1)
                sam3d_vis.colors = o3d.utility.Vector3dVector(sam3d_colors)
            sam3d_vis.transform(flip)
            vis.add_geometry(sam3d_vis)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord_frame.transform(flip)
        vis.add_geometry(coord_frame)

        n_show = min(num_grasps, len(grasp_scores))
        if n_show > 0:
            min_s, max_s = np.min(grasp_scores[:n_show]), np.max(grasp_scores[:n_show])
            range_s = max_s - min_s if max_s > min_s else 1.0

            for i in range(n_show):
                val = (grasp_scores[i] - min_s) / range_s
                color = [1.0 - val, val, 0.0]  # Red to Green

                for g in self.create_gripper(grasp_poses[i][:3,3], grasp_widths[i],
                                              color, grasp_poses[i][:3,:3]):
                    g.transform(flip)
                    vis.add_geometry(g)

        ctr = vis.get_view_control()
        if sam3d_pcd_aligned is not None and len(sam3d_pcd_aligned.points) > 0:
            center = sam3d_pcd_aligned.get_center()
            center_flipped = flip[:3, :3] @ center + flip[:3, 3]
            ctr.set_lookat(center_flipped)

        print("\n" + "=" * 50)
        print("[Visualization]")
        print("  - Red/Orange: Depth point cloud (RealSense)")
        print("  - Blue: SAM3D reconstructed point cloud")
        print("  - Mesh: SAM3D complete 3D model")
        print("  - Grippers: Red=Low score, Green=High score")
        print("=" * 50)
        print("\n[Controls] Press 'q' or ESC to exit, close window to exit")

        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), 27]:
                break

        vis.destroy_window()


def start_realsense():
    """Initialize RealSense with filters."""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
    profile = pipe.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"RealSense started (depth_scale={depth_scale:.6f})")

    align = rs.align(rs.stream.color)
    filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]

    return pipe, align, depth_scale, filters


def capture_frames(pipe, align, filters, warmup=10):
    """Capture filtered RGBD frames."""
    time.sleep(1.0)

    for _ in range(warmup):
        frames = pipe.wait_for_frames(timeout_ms=15000)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        for f in filters:
            depth_frame = f.process(depth_frame)

    frames = pipe.wait_for_frames(timeout_ms=15000)
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    for f in filters:
        depth_frame = f.process(depth_frame)

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(frames.get_color_frame().get_data())
    intrinsics = frames.get_color_frame().profile.as_video_stream_profile().intrinsics

    return depth, color, intrinsics


def main():
    parser = argparse.ArgumentParser(description="Grasp Pipeline with SAM 3D")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--top", default=10, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--box-threshold", default=0.35, type=float)
    parser.add_argument("--grasp-threshold", default=0.5, type=float)
    parser.add_argument("--min-depth", default=0.15, type=float)
    parser.add_argument("--max-depth", default=1.0, type=float)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--sam3d", action="store_true",
                        help="Use SAM 3D Objects for full 3D reconstruction")
    parser.add_argument("--show-depth", action="store_true", default=True,
                        help="Show depth point cloud in visualization")

    args = parser.parse_args()

    text_prompt = args.prompt if args.prompt.endswith(".") else args.prompt + "."

    pipeline = GraspPipeline(device=args.device, use_sam3d=args.sam3d)
    pipeline.load_models()

    # Start RealSense
    pipe, align, depth_scale, filters = start_realsense()

    try:
        print(f"\nLooking for: '{text_prompt}'")

        # Capture frames
        depth_raw, color_bgr, intrinsics = capture_frames(pipe, align, filters)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: Detection
        print("\n[Step 1] Grounding DINO...")
        boxes, labels, _ = pipeline.detect_objects(color_rgb, text_prompt, args.box_threshold)
        print(f"  Found {len(boxes)} objects: {labels}")

        if len(boxes) == 0:
            print("No objects detected!")
            return

        # Step 2: Segmentation
        print("\n[Step 2] SAM2 segmentation...")
        mask = pipeline.segment_objects(color_rgb, boxes)
        print(f"  Mask: {mask.sum()} pixels ({100 * mask.mean():.1f}%)")

        # Step 3: Depth point cloud
        print("\n[Step 3] Creating depth point cloud...")
        depth_pcd = pipeline.create_masked_pointcloud(
            depth_raw, color_bgr, intrinsics, depth_scale,
            mask, args.min_depth, args.max_depth
        )
        print(f"  Depth point cloud: {len(depth_pcd.points)} points")

        # Step 4: SAM-3D full reconstruction (optional)
        if args.sam3d:
            print("\n[Step 4] SAM 3D Object reconstruction...")
            object_pcd = pipeline.reconstruct_3d_objects(color_rgb, depth_raw, mask, intrinsics, depth_scale)
            print(f"  Reconstructed object point cloud: {len(object_pcd.points)} points")
        else:
            object_pcd = depth_pcd  # fallback to depth-only

        # Step 5: Grasp generation
        print("\n[Step 5] Generating grasps...")
        grasps = pipeline.generate_grasps(object_pcd, top_k=args.top, threshold=args.grasp_threshold)
        print(f"  {len(grasps)} grasp candidates generated.")

        # Step 6: Visualization
        if args.visualize:
            print("\n[Step 6] Visualizing...")
            pipeline.visualize(
                color_rgb,
                boxes=boxes,
                mask=mask,
                depth_pcd=depth_pcd if args.show_depth else None,
                object_pcd=object_pcd,
                grasps=grasps
            )
            print("Visualization complete.")

    finally:
        print("\nShutting down RealSense...")
        pipe.stop()
        print("Done.")
