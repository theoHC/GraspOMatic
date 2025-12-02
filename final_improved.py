import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch, time
from scipy.spatial.transform import Rotation as R
from cgn_pytorch import from_pretrained, inference as cgn_inference


def print_grasps(poses, scores, widths, limit=25):
    print(f"\n==== Found {len(scores)} Grasp Candidates ====\n")
    if len(scores) == 0:
        print(" No grasps above threshold.\n")
        return
    for i in range(min(limit, len(scores))):
        pos = poses[i][:3, 3]
        quat = R.from_matrix(poses[i][:3, :3]).as_quat()
        print(
            f"Grasp {i+1}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}, "
            f"qx={quat[0]:.4f}, qy={quat[1]:.4f}, qz={quat[2]:.4f}, qw={quat[3]:.4f}, "
            f"score={scores[i]:.3f}, width={widths[i]:.4f}"
        )


def start_realsense():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
    profile = pipe.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"RealSense started (depth_scale={depth_scale:.6f})")

    align = rs.align(rs.stream.color)
    return pipe, align, depth_scale


def grab_aligned_frames(pipe, align, warmup=10):
    time.sleep(1.0)
    for _ in range(warmup):
        frames = pipe.wait_for_frames(timeout_ms=15000)
        frames = align.process(frames)

    frames = pipe.wait_for_frames(timeout_ms=15000)
    frames = align.process(frames)

    depth = np.asanyarray(frames.get_depth_frame().get_data())
    color = np.asanyarray(frames.get_color_frame().get_data())
    intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics

    return depth, color, intr


def rgbd_to_pointcloud(depth_raw, color_bgr, intr, depth_scale,
                       min_z=0.15, max_z=0.60, crop_margin=0.20):

    depth = depth_raw.astype(np.float32) * depth_scale
    h, w = depth.shape

    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    ys, xs = np.indices((h, w))

    # -------- Depth cropping --------
    valid = (depth > min_z) & (depth < max_z)

    # -------- Center ROI crop (keep central region) --------
    x1, x2 = int(w * crop_margin), int(w * (1 - crop_margin))
    y1, y2 = int(h * crop_margin), int(h * (1 - crop_margin))
    valid &= (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)

    xs_v = xs[valid]
    ys_v = ys[valid]
    d_v = depth[valid]

    x = (xs_v - cx) * d_v / fx
    y = (ys_v - cy) * d_v / fy
    z = d_v

    pts = np.stack((x, y, z), axis=-1).astype(np.float32)
    cols_rgb = color_bgr[valid][..., ::-1] / 255.0  # BGR → RGB

    return pts, cols_rgb


def main():
    pipe, align, depth_scale = start_realsense()

    try:
        print("Capturing frames ...")
        depth_raw, color_bgr, intr = grab_aligned_frames(pipe, align)
        print("Frames captured:", color_bgr.shape, depth_raw.shape)

        pts, cols = rgbd_to_pointcloud(
            depth_raw, color_bgr, intr, depth_scale,
            min_z=0.15, max_z=1.0, crop_margin=0.00
        )
        print("→ Filtered point cloud:", pts.shape[0])

        # Limit to 40k pts max
        if pts.shape[0] > 40000:
            idx = np.random.choice(pts.shape[0], 40000, replace=False)
            pts, cols = pts[idx], cols[idx]

        print("Loading Contact-GraspNet...")
        model, _, _ = from_pretrained()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        print("CGN ready on", device)

        poses, scores, widths = cgn_inference(model, pts, threshold=0.9)

        if len(scores) > 0:
            order = np.argsort(-scores)
            poses, scores, widths = poses[order], scores[order], widths[order]

        print_grasps(poses, scores, widths)

        # Visualize point cloud + top grasps
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        geoms = [pcd]

        for i in range(min(40, len(scores))):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.transform(poses[i])
            geoms.append(frame)

        o3d.visualization.draw_geometries(geoms)

    finally:
        pipe.stop()
        print("RealSense stopped.")


if __name__ == "__main__":
    main()
