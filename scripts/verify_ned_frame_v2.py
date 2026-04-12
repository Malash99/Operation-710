"""
Verify TartanAir pose convention exhaustively.

Tests ALL 4 combinations:
  1. T_1to2 = inv(T2) @ T1 (camera-to-world convention), no NED conv
  2. T_1to2 = inv(T2) @ T1, with NED conv
  3. T_1to2 = T2 @ inv(T1) (world-to-camera convention), no NED conv
  4. T_1to2 = T2 @ inv(T1), with NED conv

Also tests if depth is Euclidean vs z-buffer.
"""

import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
W, H = 640, 480

K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0,  0,  1]], dtype=np.float64)

T_ned2cam = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]], dtype=np.float64)
T_cam2ned = np.linalg.inv(T_ned2cam)


def load_poses(pose_file):
    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            tx, ty, tz = vals[0], vals[1], vals[2]
            qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def reproject(depth, T_rel, euclidean_depth=False):
    """Reproject points from frame 1 to frame 2.

    Args:
        depth: (H, W) depth map
        T_rel: (4, 4) relative pose cam1 -> cam2
        euclidean_depth: if True, depth is Euclidean distance; if False, z-buffer
    """
    # Sample random valid pixels
    valid_mask = depth > 0.1
    ys, xs = np.where(valid_mask)
    np.random.seed(42)
    n = min(2000, len(xs))
    idx = np.random.choice(len(xs), n, replace=False)

    pts1 = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)
    d = depth[ys[idx], xs[idx]]

    # Back-project
    K_inv = np.linalg.inv(K)
    ones = np.ones((n, 1))
    uv_h = np.hstack([pts1, ones])
    rays = (K_inv @ uv_h.T).T  # (N, 3)

    if euclidean_depth:
        # Normalize rays, scale by euclidean depth
        ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
        p3d = rays / ray_norms * d[:, None]
    else:
        # z-buffer: depth = z coordinate, so scale ray by depth
        # since rays have z=1 after K_inv, p3d_z = depth
        p3d = rays * d[:, None]

    # Transform
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    p3d_2 = (R @ p3d.T).T + t

    # Project
    proj = (K @ p3d_2.T).T
    z = proj[:, 2]
    valid = z > 0.01
    pts2 = proj[:, :2] / z[:, None]

    return pts1, pts2, valid


def test_all_combinations(poses, depth, gap=1):
    print(f"\n{'='*70}")
    print(f"Frame gap = {gap}")
    print(f"{'='*70}")

    T1 = poses[0]
    T2 = poses[gap]

    combos = {}

    # 4 pose conventions × 2 depth conventions = 8 tests
    for pose_label, T_rel in [
        ("inv(T2)@T1 (cam2world)", np.linalg.inv(T2) @ T1),
        ("T2@inv(T1) (world2cam)", T2 @ np.linalg.inv(T1)),
        ("inv(T2)@T1 + NED conv", np.linalg.inv(T_ned2cam @ T2 @ T_cam2ned) @ (T_ned2cam @ T1 @ T_cam2ned)),
        ("T2@inv(T1) + NED conv", (T_ned2cam @ T2 @ T_cam2ned) @ np.linalg.inv(T_ned2cam @ T1 @ T_cam2ned)),
    ]:
        for depth_label, euc in [("z-buffer", False), ("euclidean", True)]:
            pts1, pts2, valid = reproject(depth, T_rel, euclidean_depth=euc)
            if valid.sum() > 0:
                err = np.linalg.norm(pts2[valid] - pts1[valid], axis=1)
                mean_err = err.mean()
                med_err = np.median(err)
            else:
                mean_err = float('inf')
                med_err = float('inf')

            label = f"{pose_label}, {depth_label}"
            combos[label] = mean_err
            status = " <<<< BEST" if mean_err < 2.0 else ""
            print(f"  {label:50s} -> mean={mean_err:8.2f} px  median={med_err:8.2f} px{status}")

    # Print winner
    best = min(combos, key=combos.get)
    print(f"\n  BEST: {best} ({combos[best]:.2f} px)")


def test_with_optical_flow(poses, depth, traj_path):
    """Use optical flow as ground truth to validate."""
    flow_dir = os.path.join(traj_path, "flow")
    if not os.path.isdir(flow_dir):
        print("\nNo optical flow GT available - skipping flow test")
        return

    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith(".npy") and "mask" not in f])
    if not flow_files:
        print("\nNo optical flow files found")
        return

    flow = np.load(os.path.join(flow_dir, flow_files[0]))  # (H, W, 2)
    print(f"\n{'='*70}")
    print(f"Validation against optical flow GT (frame 0->1)")
    print(f"{'='*70}")
    print(f"  Flow shape: {flow.shape}, range: [{flow.min():.1f}, {flow.max():.1f}]")

    # Sample valid pixels
    valid_mask = (depth > 0.1) & np.isfinite(flow).all(axis=2)
    ys, xs = np.where(valid_mask)
    np.random.seed(42)
    n = min(2000, len(xs))
    idx = np.random.choice(len(xs), n, replace=False)

    pts1 = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)
    pts2_flow = pts1 + flow[ys[idx], xs[idx]]  # GT destination

    T1, T2 = poses[0], poses[1]

    for pose_label, T_rel in [
        ("inv(T2)@T1 (cam2world)", np.linalg.inv(T2) @ T1),
        ("T2@inv(T1) (world2cam)", T2 @ np.linalg.inv(T1)),
        ("inv(T2)@T1 + NED", np.linalg.inv(T_ned2cam @ T2 @ T_cam2ned) @ (T_ned2cam @ T1 @ T_cam2ned)),
        ("T2@inv(T1) + NED", (T_ned2cam @ T2 @ T_cam2ned) @ np.linalg.inv(T_ned2cam @ T1 @ T_cam2ned)),
    ]:
        for depth_label, euc in [("z-buf", False), ("eucl", True)]:
            d = depth[ys[idx], xs[idx]]
            K_inv = np.linalg.inv(K)
            ones = np.ones((n, 1))
            uv_h = np.hstack([pts1, ones])
            rays = (K_inv @ uv_h.T).T

            if euc:
                ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
                p3d = rays / ray_norms * d[:, None]
            else:
                p3d = rays * d[:, None]

            R = T_rel[:3, :3]
            t = T_rel[:3, 3]
            p3d_2 = (R @ p3d.T).T + t
            proj = (K @ p3d_2.T).T
            z = proj[:, 2]
            valid = z > 0.01
            pts2_reproj = proj[:, :2] / z[:, None]

            if valid.sum() > 10:
                err = np.linalg.norm(pts2_reproj[valid] - pts2_flow[valid], axis=1)
                mean_err = err.mean()
                med_err = np.median(err)
            else:
                mean_err = float('inf')
                med_err = float('inf')

            status = " <<<< BEST" if mean_err < 1.0 else ""
            label = f"{pose_label}, {depth_label}"
            print(f"  {label:50s} -> vs flow: mean={mean_err:8.2f}  median={med_err:8.2f}{status}")


def main():
    data_root = os.path.join("data", "tartanair")

    # Find trajectory
    traj_path = None
    for env in sorted(os.listdir(data_root)):
        env_path = os.path.join(data_root, env)
        if not os.path.isdir(env_path):
            continue
        for diff in ["Easy"]:
            diff_path = os.path.join(env_path, diff)
            if not os.path.isdir(diff_path):
                continue
            for p in sorted(os.listdir(diff_path)):
                p_path = os.path.join(diff_path, p)
                if os.path.isfile(os.path.join(p_path, "pose_left.txt")):
                    traj_path = p_path
                    break
            if traj_path:
                break
        if traj_path:
            break

    print(f"Trajectory: {traj_path}")

    poses = load_poses(os.path.join(traj_path, "pose_left.txt"))
    depth_dir = os.path.join(traj_path, "depth_left")
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
    depth = np.load(os.path.join(depth_dir, depth_files[0]))

    print(f"Poses: {len(poses)}")
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")

    # Print first pose for reference
    print(f"\nPose 0 translation: {poses[0][:3, 3]}")
    print(f"Pose 0 rotation (euler deg): {Rotation.from_matrix(poses[0][:3,:3]).as_euler('xyz', degrees=True)}")

    # Test all 8 combinations
    for gap in [1, 5]:
        test_all_combinations(poses, depth, gap)

    # Validate against optical flow
    test_with_optical_flow(poses, depth, traj_path)


if __name__ == "__main__":
    main()
