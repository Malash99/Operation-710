"""
Verify NED-to-camera frame conversion for TartanAir dataset.

This script proves whether the missing NED conversion is the root cause
of the pose loss plateau at ~500. It does three things:

1. Loads a TartanAir frame pair (image + depth + poses)
2. Computes relative pose WITH and WITHOUT NED-to-camera conversion
3. Reprojects 3D points from frame 1 into frame 2 using both poses
4. Measures reprojection error against optical flow ground truth

If the NED conversion fixes the issue, the "with conversion" reprojection
should have near-zero error, while "without conversion" should be large.

Also computes what the pose loss would be for each case.
"""

import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# TartanAir intrinsics
FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
W, H = 640, 480

K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0,  0,  1]], dtype=np.float64)

# NED-to-camera conversion matrix
T_ned2cam = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]], dtype=np.float64)
T_cam2ned = np.linalg.inv(T_ned2cam)


def load_poses(pose_file):
    """Load raw NED poses from pose_left.txt."""
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


def relative_pose_ned(pose1, pose2):
    """Relative pose WITHOUT NED conversion (current buggy code)."""
    return np.linalg.inv(pose2) @ pose1


def relative_pose_cam(pose1, pose2):
    """Relative pose WITH NED-to-camera conversion."""
    pose1_cam = T_ned2cam @ pose1 @ T_cam2ned
    pose2_cam = T_ned2cam @ pose2 @ T_cam2ned
    return np.linalg.inv(pose2_cam) @ pose1_cam


def reproject_points(depth, T_rel, n_points=500):
    """Back-project random pixels from frame 1, transform, project to frame 2.

    Returns:
        pts1: (N, 2) pixel coordinates in image 1
        pts2_reproj: (N, 2) reprojected pixel coordinates in image 2
        valid: (N,) boolean mask for valid points
    """
    # Sample random pixels with valid depth
    valid_mask = depth > 0.1  # skip near-zero depth
    ys, xs = np.where(valid_mask)
    if len(xs) < n_points:
        n_points = len(xs)
    idx = np.random.choice(len(xs), n_points, replace=False)

    pts1 = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)  # (N, 2)
    d = depth[ys[idx], xs[idx]]  # (N,)

    # Back-project to 3D
    K_inv = np.linalg.inv(K)
    ones = np.ones((n_points, 1))
    uv_h = np.hstack([pts1, ones])  # (N, 3)
    rays = (K_inv @ uv_h.T).T  # (N, 3)
    p3d = rays * d[:, None]  # (N, 3)

    # Transform to camera 2
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    p3d_cam2 = (R @ p3d.T).T + t  # (N, 3)

    # Project to image 2
    proj = (K @ p3d_cam2.T).T  # (N, 3)
    z = proj[:, 2]
    valid = z > 0.01
    pts2_reproj = proj[:, :2] / z[:, None]

    return pts1, pts2_reproj, valid


def compute_pose_loss(R_pred, t_pred, R_gt, t_gt, lambda_r=180, lambda_t=400):
    """Compute the paper's pose loss (Eq. 13)."""
    # Translation loss
    t_pred_n = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-8)
    loss_t = lambda_t * np.linalg.norm(t_pred_n - t_gt_n)

    # Rotation loss
    R_diff = R_pred @ R_gt.T
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    loss_r = lambda_r * angle

    angle_deg = np.degrees(angle)
    t_angle = np.degrees(np.arccos(np.clip(np.dot(t_pred_n, t_gt_n), -1, 1)))

    return loss_t + loss_r, loss_t, loss_r, angle_deg, t_angle


def main():
    # Find a TartanAir trajectory
    data_root = os.path.join("data", "tartanair")
    traj_path = None

    for env in sorted(os.listdir(data_root)):
        env_path = os.path.join(data_root, env)
        if not os.path.isdir(env_path):
            continue
        for diff in ["Easy", "Hard"]:
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

    if traj_path is None:
        print("ERROR: No TartanAir trajectory found")
        return

    print(f"Using trajectory: {traj_path}")

    # Load poses
    poses = load_poses(os.path.join(traj_path, "pose_left.txt"))
    print(f"Loaded {len(poses)} poses")

    # Load depth for frame 0
    depth_dir = os.path.join(traj_path, "depth_left")
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
    depth = np.load(os.path.join(depth_dir, depth_files[0]))

    # Load images
    img_dir = os.path.join(traj_path, "image_left")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    img1 = cv2.imread(os.path.join(img_dir, img_files[0]))
    img2 = cv2.imread(os.path.join(img_dir, img_files[1]))

    # Also load optical flow if available for ground truth comparison
    flow_dir = os.path.join(traj_path, "flow")
    has_flow = os.path.isdir(flow_dir)
    flow_gt = None
    if has_flow:
        flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith(".npy") and "mask" not in f])
        if flow_files:
            flow_gt = np.load(os.path.join(flow_dir, flow_files[0]))
            print(f"Optical flow GT loaded: {flow_gt.shape}")

    print("\n" + "="*70)
    print("TEST 1: Reprojection error (frame 0 -> frame 1)")
    print("="*70)

    # Test multiple frame gaps
    for gap in [1, 5, 10]:
        if gap >= len(poses):
            continue

        print(f"\n--- Frame gap = {gap} ---")

        T_ned = relative_pose_ned(poses[0], poses[gap])
        T_cam = relative_pose_cam(poses[0], poses[gap])

        # Print the relative translations to see the difference
        print(f"  NED t = [{T_ned[0,3]:.4f}, {T_ned[1,3]:.4f}, {T_ned[2,3]:.4f}]")
        print(f"  CAM t = [{T_cam[0,3]:.4f}, {T_cam[1,3]:.4f}, {T_cam[2,3]:.4f}]")

        # Reproject with both
        np.random.seed(42)
        pts1, pts2_ned, valid_ned = reproject_points(depth, T_ned, n_points=1000)
        np.random.seed(42)
        pts1_2, pts2_cam, valid_cam = reproject_points(depth, T_cam, n_points=1000)

        # Measure reprojection error
        if valid_ned.sum() > 0:
            err_ned = np.linalg.norm(pts2_ned[valid_ned] - pts1[valid_ned], axis=1)
            print(f"  WITHOUT NED conv: reproj error = {err_ned.mean():.2f} px (median {np.median(err_ned):.2f})")
        if valid_cam.sum() > 0:
            err_cam = np.linalg.norm(pts2_cam[valid_cam] - pts1_2[valid_cam], axis=1)
            print(f"  WITH    NED conv: reproj error = {err_cam.mean():.2f} px (median {np.median(err_cam):.2f})")

        # If we have optical flow, compare against it
        if flow_gt is not None and gap == 1:
            # Flow gives the displacement: pts2 = pts1 + flow(pts1)
            u = pts1[:, 0].astype(int).clip(0, W-1)
            v = pts1[:, 1].astype(int).clip(0, H-1)
            flow_at_pts = flow_gt[v, u]  # (N, 2)
            pts2_flow = pts1 + flow_at_pts

            # Compare reprojection against flow GT
            flow_valid = np.isfinite(flow_at_pts).all(axis=1) & valid_ned & valid_cam
            if flow_valid.sum() > 10:
                err_ned_vs_flow = np.linalg.norm(pts2_ned[flow_valid] - pts2_flow[flow_valid], axis=1)
                err_cam_vs_flow = np.linalg.norm(pts2_cam[flow_valid] - pts2_flow[flow_valid], axis=1)
                print(f"\n  vs Optical Flow GT:")
                print(f"  WITHOUT NED conv: error vs flow = {err_ned_vs_flow.mean():.2f} px (median {np.median(err_ned_vs_flow):.2f})")
                print(f"  WITH    NED conv: error vs flow = {err_cam_vs_flow.mean():.2f} px (median {np.median(err_cam_vs_flow):.2f})")

    print("\n" + "="*70)
    print("TEST 2: Pose loss values (what the training loop would compute)")
    print("="*70)

    # Simulate what happens in training:
    # The model predicts R, t in CAMERA frame (from Essential matrix decomposition)
    # The GT R, t come from the dataset
    # Without NED conversion: GT is in NED frame
    # With NED conversion: GT is in camera frame

    for gap in [1, 5, 10]:
        if gap >= len(poses):
            continue

        print(f"\n--- Frame gap = {gap} ---")

        T_ned = relative_pose_ned(poses[0], poses[gap])
        T_cam = relative_pose_cam(poses[0], poses[gap])

        R_ned, t_ned = T_ned[:3, :3], T_ned[:3, 3]
        R_cam, t_cam = T_cam[:3, :3], T_cam[:3, 3]

        # If model predicts the CORRECT camera-frame pose:
        R_pred = R_cam  # "perfect" prediction
        t_pred = t_cam

        # Loss comparing perfect prediction vs NED GT (current broken code)
        loss_ned, lt_ned, lr_ned, rdeg_ned, tdeg_ned = compute_pose_loss(R_pred, t_pred, R_ned, t_ned)
        print(f"  Perfect pred vs NED GT:  loss={loss_ned:.1f} (rot={rdeg_ned:.1f}° trans={tdeg_ned:.1f}°)")

        # Loss comparing perfect prediction vs camera GT (fixed code)
        loss_cam, lt_cam, lr_cam, rdeg_cam, tdeg_cam = compute_pose_loss(R_pred, t_pred, R_cam, t_cam)
        print(f"  Perfect pred vs CAM GT:  loss={loss_cam:.1f} (rot={rdeg_cam:.1f}° trans={tdeg_cam:.1f}°)")

    print("\n" + "="*70)
    print("TEST 3: Verify NED conversion matrix is correct")
    print("="*70)

    # The NED frame has: x=forward, y=right, z=down
    # Camera frame has: x=right, y=down, z=forward
    # So: cam_x = ned_y, cam_y = ned_z, cam_z = ned_x

    # A point at (0, 0, 5) in camera frame = 5 meters in front of camera
    # In NED frame this should be (5, 0, 0) = 5 meters forward
    p_cam = np.array([0, 0, 5, 1])
    p_ned = T_cam2ned @ p_cam
    print(f"  Camera [0,0,5] (5m forward) -> NED {p_ned[:3]}")
    print(f"  Expected: NED [5,0,0] (5m forward in NED)")
    print(f"  Match: {np.allclose(p_ned[:3], [5, 0, 0])}")

    p_cam2 = np.array([1, 0, 0, 1])  # 1m to the right in camera
    p_ned2 = T_cam2ned @ p_cam2
    print(f"  Camera [1,0,0] (1m right)   -> NED {p_ned2[:3]}")
    print(f"  Expected: NED [0,1,0] (1m right in NED)")
    print(f"  Match: {np.allclose(p_ned2[:3], [0, 1, 0])}")

    p_cam3 = np.array([0, 1, 0, 1])  # 1m down in camera
    p_ned3 = T_cam2ned @ p_cam3
    print(f"  Camera [0,1,0] (1m down)    -> NED {p_ned3[:3]}")
    print(f"  Expected: NED [0,0,1] (1m down in NED)")
    print(f"  Match: {np.allclose(p_ned3[:3], [0, 0, 1])}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)


if __name__ == "__main__":
    main()
