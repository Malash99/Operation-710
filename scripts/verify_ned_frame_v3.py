"""
Verify TartanAir pose convention using cross-frame depth consistency.

For the CORRECT pose convention:
  1. Back-project pixel from frame 1 using depth1 -> 3D point P
  2. Transform P to frame 2 using T_1to2 -> P'
  3. Project P' to frame 2 pixel -> p2
  4. depth2[p2] should ≈ P'.z (depth consistency)

This is immune to the "measuring displacement" mistake because
it compares DEPTH values, not pixel positions.
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


def depth_consistency_test(depth1, depth2, T_rel, label, n_points=5000):
    """Test depth consistency across frames.

    For correct T_rel:
      back-project(p1, depth1) -> P -> transform(T_rel) -> P' -> project -> p2
      depth2[p2] should ≈ P'.z

    Returns median relative depth error.
    """
    valid_mask = (depth1 > 0.5) & (depth1 < 40)  # avoid edge cases
    ys, xs = np.where(valid_mask)
    np.random.seed(42)
    n = min(n_points, len(xs))
    idx = np.random.choice(len(xs), n, replace=False)

    u1 = xs[idx].astype(np.float64)
    v1 = ys[idx].astype(np.float64)
    d1 = depth1[ys[idx], xs[idx]]

    # Back-project to 3D (z-buffer depth)
    x3d = (u1 - CX) / FX * d1
    y3d = (v1 - CY) / FY * d1
    z3d = d1
    p3d = np.stack([x3d, y3d, z3d], axis=1)  # (N, 3)

    # Transform to frame 2
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    p3d_2 = (R @ p3d.T).T + t

    # Project to frame 2
    u2 = FX * p3d_2[:, 0] / p3d_2[:, 2] + CX
    v2 = FY * p3d_2[:, 1] / p3d_2[:, 2] + CY
    z2_predicted = p3d_2[:, 2]

    # Check which reprojected pixels are in bounds
    in_bounds = (u2 >= 0) & (u2 < W - 1) & (v2 >= 0) & (v2 < H - 1) & (z2_predicted > 0.1)

    if in_bounds.sum() < 10:
        print(f"  {label:50s} -> too few valid ({in_bounds.sum()})")
        return float('inf')

    # Sample depth2 at reprojected locations (bilinear interpolation)
    u2i = u2[in_bounds].astype(int).clip(0, W - 1)
    v2i = v2[in_bounds].astype(int).clip(0, H - 1)
    z2_actual = depth2[v2i, u2i]

    z2_pred = z2_predicted[in_bounds]

    # Only compare where depth2 is valid
    valid_d2 = (z2_actual > 0.5) & (z2_actual < 40)
    if valid_d2.sum() < 10:
        print(f"  {label:50s} -> too few valid depth2 ({valid_d2.sum()})")
        return float('inf')

    z_pred = z2_pred[valid_d2]
    z_actual = z2_actual[valid_d2]

    # Relative depth error: |predicted_z - actual_z| / actual_z
    rel_err = np.abs(z_pred - z_actual) / z_actual
    mean_rel_err = rel_err.mean()
    median_rel_err = np.median(rel_err)
    pct_good = (rel_err < 0.05).mean() * 100  # % within 5% error

    status = " <<<< CORRECT" if median_rel_err < 0.02 else ""
    print(f"  {label:50s} -> rel_err: mean={mean_rel_err:.4f} median={median_rel_err:.4f}  <5%: {pct_good:.1f}%{status}")
    return median_rel_err


def main():
    data_root = os.path.join("data", "tartanair")

    # Test on multiple trajectories for robustness
    tested = 0
    for env in sorted(os.listdir(data_root)):
        env_path = os.path.join(data_root, env)
        if not os.path.isdir(env_path):
            continue
        for diff in ["Easy"]:
            diff_path = os.path.join(env_path, diff)
            if not os.path.isdir(diff_path):
                continue
            for p in sorted(os.listdir(diff_path))[:1]:  # first trajectory only
                p_path = os.path.join(diff_path, p)
                pose_file = os.path.join(p_path, "pose_left.txt")
                if not os.path.isfile(pose_file):
                    continue

                print(f"\n{'#'*70}")
                print(f"# {os.path.relpath(p_path, data_root)}")
                print(f"{'#'*70}")

                poses = load_poses(pose_file)
                depth_dir = os.path.join(p_path, "depth_left")
                depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])

                for gap in [1, 3]:
                    depth1 = np.load(os.path.join(depth_dir, depth_files[0]))
                    depth2 = np.load(os.path.join(depth_dir, depth_files[gap]))

                    T1, T2 = poses[0], poses[gap]

                    print(f"\n  Gap={gap}, t_magnitude={np.linalg.norm(T2[:3,3] - T1[:3,3]):.4f}m")

                    # Test all 4 pose conventions
                    depth_consistency_test(
                        depth1, depth2,
                        np.linalg.inv(T2) @ T1,
                        f"inv(T2)@T1 (cam2world, no NED) [CURRENT]",
                    )
                    depth_consistency_test(
                        depth1, depth2,
                        T2 @ np.linalg.inv(T1),
                        f"T2@inv(T1) (world2cam, no NED)",
                    )

                    T1c = T_ned2cam @ T1 @ T_cam2ned
                    T2c = T_ned2cam @ T2 @ T_cam2ned
                    depth_consistency_test(
                        depth1, depth2,
                        np.linalg.inv(T2c) @ T1c,
                        f"inv(T2)@T1 (cam2world, WITH NED)",
                    )
                    depth_consistency_test(
                        depth1, depth2,
                        T2c @ np.linalg.inv(T1c),
                        f"T2@inv(T1) (world2cam, WITH NED)",
                    )

                tested += 1
                if tested >= 3:
                    break
            if tested >= 3:
                break
        if tested >= 3:
            break

    print(f"\n{'='*70}")
    print("SUMMARY: The convention with lowest median relative depth error")
    print("is the correct one (should be <2% for a correct pose).")
    print("='*70")


if __name__ == "__main__":
    main()
