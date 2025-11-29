import numpy as np
import open3d as o3d
import torch
from cgn_pytorch import from_pretrained
from cgn_pytorch import inference as cgn_inference
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------
#
# ---------------------------------------------------------
def print_grasps(poses, scores, widths, limit=25):
    print(f"\n==== Found {len(scores)} Grasp Candidates ====\n")

    if len(scores) == 0:
        print("⚠ No grasps above threshold.\n")
        return

    for i in range(min(limit, len(scores))):
        T = poses[i]
        pos = T[:3, 3]

        quat = R.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w

        print(f"Grasp {i + 1}:")
        print(f"  • Position    : x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        print(
            f"  • Orientation : qx={quat[0]:.4f}, qy={quat[1]:.4f}, qz={quat[2]:.4f}, qw={quat[3]:.4f}"
        )
        print(f"  • Score       : {scores[i]:.3f}")
        print(f"  • Width (m)   : {widths[i]:.4f}")
        print("-" * 45)
    print("\n")


# ---------------------------------------------------------
# THREE-object synthetic pointcloud
#   Object 1: Cube
#   Object 2: Cylinder
#   Object 3: Sphere
# ---------------------------------------------------------
def synthetic_three_object_scene():
    pts = []
    colors = []

    # OBJECT 1 — Cube
    cube = (np.random.rand(8000, 3) - 0.5) * 0.2
    cube[:, 2] += 0.25
    cube[:, 0] += 0.00
    cube[:, 1] += 0.00
    pts.append(cube)
    colors.append(np.tile([0.8, 0.2, 0.2], (cube.shape[0], 1)))  # red cube

    # OBJECT 2 — Cylinder (left)
    theta = np.random.rand(8000) * 2 * np.pi
    r = 0.07 * np.sqrt(np.random.rand(8000))
    h = np.random.rand(8000) * 0.15

    cyl = np.zeros((8000, 3))
    cyl[:, 0] = -0.3 + r * np.cos(theta)
    cyl[:, 1] = 0.0 + r * np.sin(theta)
    cyl[:, 2] = h
    pts.append(cyl)
    colors.append(np.tile([0.2, 0.7, 0.2], (cyl.shape[0], 1)))  # green cylinder

    # OBJECT 3 — Sphere (right)
    u = np.random.rand(8000)
    v = np.random.rand(8000)
    sphere = np.zeros((8000, 3))
    sphere[:, 0] = 0.35 + 0.12 * np.cos(2 * np.pi * u) * np.sqrt(1 - (2 * v - 1) ** 2)
    sphere[:, 1] = 0.00 + 0.12 * np.sin(2 * np.pi * u) * np.sqrt(1 - (2 * v - 1) ** 2)
    sphere[:, 2] = 0.12 * (2 * v - 1)
    pts.append(sphere)
    colors.append(np.tile([0.2, 0.2, 0.8], (sphere.shape[0], 1)))  # blue sphere

    # Merge all
    pts = np.vstack(pts).astype(np.float32)
    colors = np.vstack(colors).astype(np.float32)

    return pts, colors


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------
def main():
    pts, rgb = synthetic_three_object_scene()
    print("synthetic pts:", pts.shape)

    # Load CGN
    model, _, _ = from_pretrained()
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    print("→ Running CGN inference() ... (threshold=0.9)")
    poses, scores, widths = cgn_inference(model, pts, threshold=0.6)

    print(f"🎉 Got {len(scores)} grasps ≥ 90% confidence\n")

    # Pretty-print grasps
    print_grasps(poses, scores, widths)

    # Visualize all
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [pcd]

    # Add grasp pose coordinate frames
    for i in range(min(40, poses.shape[0])):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(poses[i])
        geoms.append(frame)

    print("Opening Open3D visualization...")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
