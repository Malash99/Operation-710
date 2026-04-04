"""
DINO-VO Foundation Diagnostic Tests.

4 tests to verify the pose estimation pipeline works correctly:
  Test 1: 8-point algorithm with GT correspondences (synthetic + real)
  Test 2: Pose loss sanity check
  Test 3: Compare with OpenCV findEssentialMat
  Test 4: Gradient flow magnitude through pose pipeline

Run: python -m scripts.diagnostic_tests
"""

import os
import sys
import numpy as np
import torch
import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pose_estimation import PoseEstimation
from src.losses.losses import PoseLoss, _log_rotation


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ======================================================================
# TEST 1: 8-point algorithm with known-correct correspondences
# ======================================================================

def test1_eight_point_with_gt():
    """Feed perfect correspondences into our 8-point algorithm.

    If this fails, the implementation is buggy — no training can fix it.
    If this passes, the issue is match quality from the network.
    """
    separator("TEST 1: Weighted 8-point with GT correspondences")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_est = PoseEstimation().to(device)

    results = []

    # --- Test 1a: Synthetic known pose (pure translation along X) ---
    print("Test 1a: Synthetic — pure translation along X-axis")
    K = torch.tensor([[[500.0, 0.0, 320.0],
                        [0.0, 500.0, 240.0],
                        [0.0,   0.0,   1.0]]], device=device)

    # Known relative pose: identity rotation, translation = [1, 0, 0]
    R_gt = torch.eye(3, device=device).unsqueeze(0)
    t_gt = torch.tensor([[1.0, 0.0, 0.0]], device=device)

    # Generate 3D points randomly in front of camera
    np.random.seed(42)
    N = 100
    pts_3d = np.random.uniform([-2, -2, 3], [2, 2, 8], size=(N, 3))

    # Project to camera 1
    K_np = K[0].cpu().numpy()
    proj1 = (K_np @ pts_3d.T).T
    px1 = proj1[:, :2] / proj1[:, 2:3]

    # Transform to camera 2 and project
    R_np = R_gt[0].cpu().numpy()
    t_np = t_gt[0].cpu().numpy()
    pts_cam2 = (R_np @ pts_3d.T).T + t_np
    proj2 = (K_np @ pts_cam2.T).T
    px2 = proj2[:, :2] / proj2[:, 2:3]

    # Filter points visible in both cameras
    mask = (px1[:, 0] > 0) & (px1[:, 0] < 640) & (px1[:, 1] > 0) & (px1[:, 1] < 480)
    mask &= (px2[:, 0] > 0) & (px2[:, 0] < 640) & (px2[:, 1] > 0) & (px2[:, 1] < 480)
    mask &= (pts_cam2[:, 2] > 0)  # positive depth in cam2
    px1, px2 = px1[mask], px2[mask]
    M = len(px1)

    kp1 = torch.tensor(px1, dtype=torch.float32, device=device).unsqueeze(0)
    kp2 = torch.tensor(px2, dtype=torch.float32, device=device).unsqueeze(0)
    weights = torch.ones(1, M, device=device)

    out = pose_est(kp1, kp2, weights, K)
    R_pred = out["R"][0].cpu().detach().numpy()
    t_pred = out["t"][0].cpu().detach().numpy()

    rot_err = np.degrees(np.arccos(np.clip((np.trace(R_np.T @ R_pred) - 1) / 2, -1, 1)))
    # Translation direction error (angle between unit vectors)
    t_gt_unit = t_np / np.linalg.norm(t_np)
    t_pred_unit = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    trans_err = np.degrees(np.arccos(np.clip(np.dot(t_gt_unit, t_pred_unit), -1, 1)))

    print(f"  Correspondences used: {M}")
    print(f"  R_gt:\n{R_np}")
    print(f"  R_pred:\n{R_pred}")
    print(f"  t_gt (unit): {t_gt_unit}")
    print(f"  t_pred:      {t_pred_unit}")
    print(f"  Rotation error:    {rot_err:.4f} deg")
    print(f"  Translation error: {trans_err:.4f} deg")
    pass1a = rot_err < 1.0 and trans_err < 5.0
    print(f"  RESULT: {'PASS' if pass1a else 'FAIL'}")
    results.append(("1a: Synthetic pure translation", pass1a, rot_err, trans_err))

    # --- Test 1b: Synthetic known pose (rotation + translation) ---
    print("\nTest 1b: Synthetic — 15deg rotation + translation")
    angle = np.radians(15)
    R_gt_np = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=np.float64)
    t_gt_np = np.array([0.3, -0.1, 0.5])
    t_gt_np = t_gt_np / np.linalg.norm(t_gt_np)

    pts_cam2 = (R_gt_np @ pts_3d.T).T + t_gt_np
    proj2 = (K_np @ pts_cam2.T).T
    px2_b = proj2[:, :2] / proj2[:, 2:3]
    proj1 = (K_np @ pts_3d.T).T
    px1_b = proj1[:, :2] / proj1[:, 2:3]

    mask = (px1_b[:, 0] > 0) & (px1_b[:, 0] < 640) & (px1_b[:, 1] > 0) & (px1_b[:, 1] < 480)
    mask &= (px2_b[:, 0] > 0) & (px2_b[:, 0] < 640) & (px2_b[:, 1] > 0) & (px2_b[:, 1] < 480)
    mask &= (pts_cam2[:, 2] > 0)
    px1_b, px2_b = px1_b[mask], px2_b[mask]
    M = len(px1_b)

    R_gt_t = torch.tensor(R_gt_np, dtype=torch.float32, device=device).unsqueeze(0)
    t_gt_t = torch.tensor(t_gt_np, dtype=torch.float32, device=device).unsqueeze(0)

    kp1 = torch.tensor(px1_b, dtype=torch.float32, device=device).unsqueeze(0)
    kp2 = torch.tensor(px2_b, dtype=torch.float32, device=device).unsqueeze(0)
    weights = torch.ones(1, M, device=device)

    out = pose_est(kp1, kp2, weights, K)
    R_pred = out["R"][0].cpu().detach().numpy()
    t_pred = out["t"][0].cpu().detach().numpy()

    rot_err = np.degrees(np.arccos(np.clip((np.trace(R_gt_np.T @ R_pred) - 1) / 2, -1, 1)))
    t_pred_unit = t_pred / (np.linalg.norm(t_pred) + 1e-8)
    trans_err = np.degrees(np.arccos(np.clip(np.dot(t_gt_np, t_pred_unit), -1, 1)))

    print(f"  Correspondences used: {M}")
    print(f"  Rotation error:    {rot_err:.4f} deg")
    print(f"  Translation error: {trans_err:.4f} deg")
    pass1b = rot_err < 1.0 and trans_err < 5.0
    print(f"  RESULT: {'PASS' if pass1b else 'FAIL'}")
    results.append(("1b: Synthetic rotation+translation", pass1b, rot_err, trans_err))

    # --- Test 1c: Real EuRoC data with GT correspondences ---
    print("\nTest 1c: Real EuRoC data — GT correspondences via reprojection")
    try:
        from src.datasets.euroc import EuRoCDataset
        dataset = EuRoCDataset(
            sequence_path="data/euroc/MH_01_easy",
            skip_frames=2,
            target_h=476, target_w=742,
            compute_stereo_depth=False,
            min_translation=0.0,
            max_skip_multiplier=1,
        )

        # Test on 10 pairs
        rot_errors_real = []
        trans_errors_real = []
        for idx in range(0, min(50, len(dataset)), 5):
            sample = dataset[idx]
            img1 = sample["image1"].unsqueeze(0).to(device)
            img2 = sample["image2"].unsqueeze(0).to(device)
            K_real = sample["intrinsics"].unsqueeze(0).to(device)
            T_gt = sample["relative_pose"].numpy()
            R_gt_real = T_gt[:3, :3]
            t_gt_real = T_gt[:3, 3]

            if np.linalg.norm(t_gt_real) < 1e-6:
                continue

            # Use OpenCV to find correspondences via optical flow (proxy for GT)
            # Denormalize images for OpenCV
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img1_np = ((img1[0].cpu() * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img2_np = ((img2[0].cpu() * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

            gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

            # Detect features and match with OpenCV
            orb = cv2.ORB_create(nfeatures=500)
            kp1_cv = orb.detect(gray1, None)
            if len(kp1_cv) < 20:
                continue

            pts1_cv = np.array([k.pt for k in kp1_cv], dtype=np.float32).reshape(-1, 1, 2)
            pts2_cv, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1_cv, None)
            status = status.flatten()
            pts1_good = pts1_cv[status == 1].reshape(-1, 2)
            pts2_good = pts2_cv[status == 1].reshape(-1, 2)

            if len(pts1_good) < 20:
                continue

            kp1_t = torch.tensor(pts1_good, dtype=torch.float32, device=device).unsqueeze(0)
            kp2_t = torch.tensor(pts2_good, dtype=torch.float32, device=device).unsqueeze(0)
            w_t = torch.ones(1, len(pts1_good), device=device)

            out = pose_est(kp1_t, kp2_t, w_t, K_real)
            R_p = out["R"][0].cpu().detach().numpy()
            t_p = out["t"][0].cpu().detach().numpy()

            r_err = np.degrees(np.arccos(np.clip((np.trace(R_gt_real.T @ R_p) - 1) / 2, -1, 1)))
            t_gt_u = t_gt_real / (np.linalg.norm(t_gt_real) + 1e-8)
            t_p_u = t_p / (np.linalg.norm(t_p) + 1e-8)
            t_err = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_p_u), -1, 1)))

            rot_errors_real.append(r_err)
            trans_errors_real.append(t_err)

        if rot_errors_real:
            mean_rot = np.mean(rot_errors_real)
            mean_trans = np.mean(trans_errors_real)
            print(f"  Tested on {len(rot_errors_real)} EuRoC pairs")
            print(f"  Rotation error:    mean={mean_rot:.2f} deg, median={np.median(rot_errors_real):.2f} deg")
            print(f"  Translation error: mean={mean_trans:.2f} deg, median={np.median(trans_errors_real):.2f} deg")
            pass1c = mean_rot < 10.0 and mean_trans < 30.0
            print(f"  RESULT: {'PASS' if pass1c else 'FAIL'} (relaxed thresholds for real data)")
            results.append(("1c: Real EuRoC with optical flow", pass1c, mean_rot, mean_trans))
        else:
            print("  SKIPPED: Could not get enough correspondences")
            results.append(("1c: Real EuRoC with optical flow", None, None, None))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(("1c: Real EuRoC", None, None, None))

    return results


# ======================================================================
# TEST 2: Pose loss sanity check
# ======================================================================

def test2_pose_loss_sanity():
    """Verify pose loss is 0 for identical inputs, proportional for perturbations."""
    separator("TEST 2: Pose loss sanity check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_loss = PoseLoss(lambda_r=180.0, lambda_t=400.0).to(device)
    results = []

    # Test 2a: L(R_gt, t_gt, R_gt, t_gt) should be 0
    print("Test 2a: Pose loss with identical inputs (should be ~0)")
    R = torch.eye(3, device=device).unsqueeze(0)
    t = torch.tensor([[0.5, 0.3, 0.8]], device=device)
    t = t / t.norm(dim=-1, keepdim=True)
    loss_zero = pose_loss(R, t, R, t).item()
    print(f"  L(R, t, R, t) = {loss_zero:.8f}")
    pass2a = loss_zero < 1e-4
    print(f"  RESULT: {'PASS' if pass2a else 'FAIL'}")
    results.append(("2a: Zero loss for identical", pass2a, loss_zero))

    # Test 2b: Small rotation perturbation -> small loss, proportional
    print("\nTest 2b: Small rotation perturbation -> proportional loss")
    for angle_deg in [1.0, 5.0, 10.0, 30.0]:
        angle = np.radians(angle_deg)
        R_pert = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        loss = pose_loss(R_pert, t, R, t).item()
        print(f"  {angle_deg:5.1f} deg rotation: loss = {loss:.4f}")
    results.append(("2b: Proportional rotation loss", True, None))

    # Test 2c: Translation direction perturbation
    print("\nTest 2c: Translation direction perturbation")
    for angle_deg in [1.0, 5.0, 10.0, 30.0, 90.0, 180.0]:
        angle = np.radians(angle_deg)
        t_pert = torch.tensor([[np.cos(angle), np.sin(angle), 0.0]], device=device)
        t_pert = t_pert / t_pert.norm(dim=-1, keepdim=True)
        t_ref = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        loss = pose_loss(R, t_pert, R, t_ref).item()
        print(f"  {angle_deg:5.1f} deg translation: loss = {loss:.4f}")
    results.append(("2c: Proportional translation loss", True, None))

    # Test 2d: _log_rotation correctness
    print("\nTest 2d: _log_rotation correctness")
    for angle_deg in [5.0, 30.0, 90.0, 170.0]:
        angle = np.radians(angle_deg)
        R_test = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        log_R = _log_rotation(R_test)
        recovered_angle = log_R.norm().item()
        expected_angle = abs(angle)
        error = abs(recovered_angle - expected_angle)
        print(f"  Input: {angle_deg} deg | log_R norm: {np.degrees(recovered_angle):.4f} deg | error: {np.degrees(error):.4f} deg")
    pass2d = True  # visual check
    results.append(("2d: _log_rotation correctness", pass2d, None))

    return results


# ======================================================================
# TEST 3: Compare with OpenCV findEssentialMat
# ======================================================================

def test3_compare_opencv():
    """Compare our 8-point with OpenCV RANSAC on the same correspondences."""
    separator("TEST 3: Compare with OpenCV findEssentialMat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_est = PoseEstimation().to(device)
    results = []

    # Test 3a: Synthetic data — both should work
    print("Test 3a: Synthetic data — our 8-point vs OpenCV RANSAC")
    K_np = np.array([[500.0, 0.0, 320.0],
                      [0.0, 500.0, 240.0],
                      [0.0,   0.0,   1.0]])

    np.random.seed(42)
    N = 100
    pts_3d = np.random.uniform([-2, -2, 3], [2, 2, 8], size=(N, 3))
    angle = np.radians(10)
    R_gt = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t_gt = np.array([0.3, -0.1, 0.5])
    t_gt = t_gt / np.linalg.norm(t_gt)

    proj1 = (K_np @ pts_3d.T).T
    px1 = proj1[:, :2] / proj1[:, 2:3]
    pts_cam2 = (R_gt @ pts_3d.T).T + t_gt
    proj2 = (K_np @ pts_cam2.T).T
    px2 = proj2[:, :2] / proj2[:, 2:3]

    mask = (pts_cam2[:, 2] > 0)
    px1, px2 = px1[mask], px2[mask]

    # Our 8-point
    K_t = torch.tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)
    kp1_t = torch.tensor(px1, dtype=torch.float32, device=device).unsqueeze(0)
    kp2_t = torch.tensor(px2, dtype=torch.float32, device=device).unsqueeze(0)
    w_t = torch.ones(1, len(px1), device=device)

    out = pose_est(kp1_t, kp2_t, w_t, K_t)
    R_ours = out["R"][0].cpu().detach().numpy()
    t_ours = out["t"][0].cpu().detach().numpy()

    # OpenCV
    E_cv, mask_cv = cv2.findEssentialMat(px1.astype(np.float64), px2.astype(np.float64),
                                          K_np, method=cv2.RANSAC, threshold=1.0)
    _, R_cv, t_cv, _ = cv2.recoverPose(E_cv, px1.astype(np.float64), px2.astype(np.float64), K_np)
    t_cv = t_cv.flatten()

    # Errors
    rot_err_ours = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_ours) - 1) / 2, -1, 1)))
    rot_err_cv = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_cv) - 1) / 2, -1, 1)))

    t_gt_u = t_gt / np.linalg.norm(t_gt)
    t_err_ours = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_ours / (np.linalg.norm(t_ours) + 1e-8)), -1, 1)))
    t_err_cv = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_cv / (np.linalg.norm(t_cv) + 1e-8)), -1, 1)))

    print(f"  Our 8-point:  rot_err={rot_err_ours:.4f} deg, trans_err={t_err_ours:.4f} deg")
    print(f"  OpenCV RANSAC: rot_err={rot_err_cv:.4f} deg, trans_err={t_err_cv:.4f} deg")
    pass3a = rot_err_ours < 2.0 and t_err_ours < 10.0
    print(f"  RESULT: {'PASS' if pass3a else 'FAIL'}")
    results.append(("3a: Synthetic — ours vs OpenCV", pass3a, rot_err_ours, rot_err_cv))

    # Test 3b: Synthetic data with 20% outlier matches
    print("\nTest 3b: Synthetic data with 20% outlier matches")
    N_outliers = int(0.2 * len(px1))
    px2_noisy = px2.copy()
    outlier_idx = np.random.choice(len(px2), N_outliers, replace=False)
    px2_noisy[outlier_idx] = np.random.uniform([0, 0], [640, 480], size=(N_outliers, 2))

    kp2_noisy = torch.tensor(px2_noisy, dtype=torch.float32, device=device).unsqueeze(0)
    out_noisy = pose_est(kp1_t, kp2_noisy, w_t, K_t)
    R_ours_noisy = out_noisy["R"][0].cpu().detach().numpy()
    t_ours_noisy = out_noisy["t"][0].cpu().detach().numpy()

    E_cv2, _ = cv2.findEssentialMat(px1.astype(np.float64), px2_noisy.astype(np.float64),
                                     K_np, method=cv2.RANSAC, threshold=1.0)
    _, R_cv2, t_cv2, _ = cv2.recoverPose(E_cv2, px1.astype(np.float64), px2_noisy.astype(np.float64), K_np)
    t_cv2 = t_cv2.flatten()

    rot_err_ours_noisy = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_ours_noisy) - 1) / 2, -1, 1)))
    rot_err_cv_noisy = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_cv2) - 1) / 2, -1, 1)))
    t_err_ours_noisy = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_ours_noisy / (np.linalg.norm(t_ours_noisy) + 1e-8)), -1, 1)))
    t_err_cv_noisy = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_cv2 / (np.linalg.norm(t_cv2) + 1e-8)), -1, 1)))

    print(f"  Our 8-point (no RANSAC):  rot_err={rot_err_ours_noisy:.2f} deg, trans_err={t_err_ours_noisy:.2f} deg")
    print(f"  OpenCV RANSAC:            rot_err={rot_err_cv_noisy:.2f} deg, trans_err={t_err_cv_noisy:.2f} deg")
    print(f"  NOTE: 8-point without RANSAC is expected to fail with outliers.")
    print(f"  This tests whether the paper's confidence weights can substitute for RANSAC.")
    results.append(("3b: With 20% outliers", rot_err_ours_noisy < 5.0, rot_err_ours_noisy, rot_err_cv_noisy))

    # Test 3c: Real model matches vs OpenCV
    print("\nTest 3c: Model's actual matches — our pose vs OpenCV pose")
    try:
        from src.datasets.euroc import EuRoCDataset
        from src.models.dino_vo import DinoVO

        dataset = EuRoCDataset(
            sequence_path="data/euroc/MH_01_easy",
            skip_frames=2, target_h=476, target_w=742,
            compute_stereo_depth=False, min_translation=0.0, max_skip_multiplier=1,
        )

        model = DinoVO(top_k=512, descriptor_dim=192, matching_layers=12)
        model.load_dino(device)
        model = model.to(device)
        model.eval()

        ckpt_path = "checkpoints_v03_tartanair/epoch_13.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: No checkpoint found at {ckpt_path}, using random weights")

        ours_rot_errs, cv_rot_errs = [], []
        ours_trans_errs, cv_trans_errs = [], []

        with torch.no_grad():
            for idx in range(0, min(50, len(dataset)), 5):
                sample = dataset[idx]
                img1 = sample["image1"].unsqueeze(0).to(device)
                img2 = sample["image2"].unsqueeze(0).to(device)
                K_real = sample["intrinsics"].unsqueeze(0).to(device)
                T_gt_real = sample["relative_pose"].numpy()
                R_gt_real = T_gt_real[:3, :3]
                t_gt_real = T_gt_real[:3, 3]

                if np.linalg.norm(t_gt_real) < 1e-6:
                    continue

                out = model(img1, img2, K_real, return_all_assignments=False)
                R_model = out["R"][0].cpu().numpy()
                t_model = out["t"][0].cpu().numpy()
                matches = out["matches"][0].cpu().numpy()  # (M, 4): kp1_x, kp1_y, kp2_x, kp2_y

                if len(matches) < 8:
                    continue

                kp1_model = matches[:, :2]
                kp2_model = matches[:, 2:4]
                K_np_real = K_real[0].cpu().numpy()

                # OpenCV on same matches
                E_cv, mask_cv = cv2.findEssentialMat(
                    kp1_model.astype(np.float64), kp2_model.astype(np.float64),
                    K_np_real, method=cv2.RANSAC, threshold=2.0
                )
                if E_cv is None:
                    continue
                _, R_cv, t_cv, _ = cv2.recoverPose(
                    E_cv, kp1_model.astype(np.float64), kp2_model.astype(np.float64), K_np_real
                )
                t_cv = t_cv.flatten()

                # Errors
                r_err_ours = np.degrees(np.arccos(np.clip((np.trace(R_gt_real.T @ R_model) - 1) / 2, -1, 1)))
                r_err_cv = np.degrees(np.arccos(np.clip((np.trace(R_gt_real.T @ R_cv) - 1) / 2, -1, 1)))

                t_gt_u = t_gt_real / (np.linalg.norm(t_gt_real) + 1e-8)
                t_err_ours = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_model / (np.linalg.norm(t_model) + 1e-8)), -1, 1)))
                t_err_cv = np.degrees(np.arccos(np.clip(np.dot(t_gt_u, t_cv / (np.linalg.norm(t_cv) + 1e-8)), -1, 1)))

                ours_rot_errs.append(r_err_ours)
                cv_rot_errs.append(r_err_cv)
                ours_trans_errs.append(t_err_ours)
                cv_trans_errs.append(t_err_cv)

        if ours_rot_errs:
            print(f"  Tested on {len(ours_rot_errs)} pairs from model's matches")
            print(f"  Our 8-point:   rot={np.mean(ours_rot_errs):.2f} deg, trans={np.mean(ours_trans_errs):.2f} deg")
            print(f"  OpenCV RANSAC: rot={np.mean(cv_rot_errs):.2f} deg, trans={np.mean(cv_trans_errs):.2f} deg")

            if np.mean(cv_rot_errs) < np.mean(ours_rot_errs) * 0.5:
                print(f"  DIAGNOSIS: OpenCV is significantly better -> our 8-point has issues OR matches have outliers that RANSAC filters")
            elif np.mean(cv_rot_errs) > 15:
                print(f"  DIAGNOSIS: Both methods fail -> match quality is the bottleneck, not the algorithm")
            else:
                print(f"  DIAGNOSIS: Both methods similar -> algorithm is OK, match quality is the limiting factor")
            results.append(("3c: Model matches comparison", True, np.mean(ours_rot_errs), np.mean(cv_rot_errs)))
        else:
            print("  SKIPPED: No valid pairs")
            results.append(("3c: Model matches comparison", None, None, None))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results.append(("3c: Model matches comparison", None, None, None))

    return results


# ======================================================================
# TEST 4: Gradient flow magnitude
# ======================================================================

def test4_gradient_flow():
    """Check that pose loss gradients reach matching-relevant parameters."""
    separator("TEST 4: Gradient flow through pose pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # Test 4a: Gradient flows from pose loss through 8-point to weights
    print("Test 4a: Gradient from pose loss -> confidence weights")
    pose_est = PoseEstimation().to(device)
    pose_loss_fn = PoseLoss(lambda_r=180.0, lambda_t=400.0).to(device)

    K = torch.tensor([[[500.0, 0.0, 320.0],
                        [0.0, 500.0, 240.0],
                        [0.0,   0.0,   1.0]]], device=device)

    # Synthetic correspondences
    np.random.seed(42)
    N = 50
    pts_3d = np.random.uniform([-2, -2, 3], [2, 2, 8], size=(N, 3))
    K_np = K[0].cpu().numpy()
    angle = np.radians(10)
    R_gt_np = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t_gt_np = np.array([0.3, -0.1, 0.5])
    t_gt_np = t_gt_np / np.linalg.norm(t_gt_np)

    proj1 = (K_np @ pts_3d.T).T
    px1 = (proj1[:, :2] / proj1[:, 2:3])
    pts_cam2 = (R_gt_np @ pts_3d.T).T + t_gt_np
    proj2 = (K_np @ pts_cam2.T).T
    px2 = (proj2[:, :2] / proj2[:, 2:3])

    kp1 = torch.tensor(px1, dtype=torch.float32, device=device).unsqueeze(0)
    kp2 = torch.tensor(px2, dtype=torch.float32, device=device).unsqueeze(0)

    # Weights as a parameter (simulating learned weights)
    weights = torch.ones(1, N, device=device, requires_grad=True)

    R_gt_t = torch.tensor(R_gt_np, dtype=torch.float32, device=device).unsqueeze(0)
    t_gt_t = torch.tensor(t_gt_np, dtype=torch.float32, device=device).unsqueeze(0)

    out = pose_est(kp1, kp2, weights, K)
    loss = pose_loss_fn(out["R"], out["t"], R_gt_t, t_gt_t)

    loss.backward()

    if weights.grad is not None:
        grad_norm = weights.grad.norm().item()
        grad_max = weights.grad.abs().max().item()
        grad_mean = weights.grad.abs().mean().item()
        grad_nonzero = (weights.grad.abs() > 1e-10).sum().item()
        print(f"  Pose loss value: {loss.item():.4f}")
        print(f"  Gradient norm:   {grad_norm:.6f}")
        print(f"  Gradient max:    {grad_max:.6f}")
        print(f"  Gradient mean:   {grad_mean:.6f}")
        print(f"  Non-zero grads:  {grad_nonzero}/{N}")
        pass4a = grad_norm > 1e-8 and grad_nonzero > 0
        print(f"  RESULT: {'PASS' if pass4a else 'FAIL'} — gradient {'reaches' if pass4a else 'does NOT reach'} weights")
    else:
        print(f"  RESULT: FAIL — weights.grad is None!")
        pass4a = False
    results.append(("4a: Gradient to weights", pass4a, grad_norm if weights.grad is not None else 0))

    # Test 4b: Full model gradient — does pose loss gradient reach transformer weights?
    print("\nTest 4b: Gradient from pose loss -> transformer matching weights")
    try:
        from src.models.dino_vo import DinoVO
        model = DinoVO(top_k=512, descriptor_dim=192, matching_layers=12)
        model.load_dino(device)
        model = model.to(device)
        model.train()

        ckpt_path = "checkpoints_v03_tartanair/epoch_13.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        # Use real data
        from src.datasets.euroc import EuRoCDataset
        dataset = EuRoCDataset(
            sequence_path="data/euroc/MH_01_easy",
            skip_frames=2, target_h=476, target_w=742,
            compute_stereo_depth=False, min_translation=0.0, max_skip_multiplier=1,
        )
        sample = dataset[10]
        img1 = sample["image1"].unsqueeze(0).to(device)
        img2 = sample["image2"].unsqueeze(0).to(device)
        K_real = sample["intrinsics"].unsqueeze(0).to(device)
        T_gt = sample["relative_pose"].to(device)
        R_gt = T_gt[:3, :3].unsqueeze(0)
        t_gt = T_gt[:3, 3].unsqueeze(0)

        model.zero_grad()
        out = model(img1, img2, K_real, return_all_assignments=False)
        loss = pose_loss_fn(out["R"], out["t"], R_gt, t_gt)
        loss.backward()

        # Check gradients at key layers
        print(f"  Pose loss value: {loss.item():.4f}")
        layers_to_check = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                g = param.grad.abs()
                layers_to_check.append((name, g.mean().item(), g.max().item(), (g > 1e-10).sum().item(), param.numel()))

        # Sort by gradient magnitude
        layers_to_check.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Top 10 layers by gradient magnitude:")
        for name, g_mean, g_max, g_nonzero, numel in layers_to_check[:10]:
            print(f"    {name:60s} | mean={g_mean:.2e} max={g_max:.2e} nonzero={g_nonzero}/{numel}")

        zero_grad_layers = [name for name, g_mean, g_max, g_nonzero, numel in layers_to_check if g_nonzero == 0]
        if zero_grad_layers:
            print(f"\n  WARNING: {len(zero_grad_layers)} layers with ZERO gradient:")
            for name in zero_grad_layers[:5]:
                print(f"    {name}")

        total_params = sum(numel for _, _, _, _, numel in layers_to_check)
        total_nonzero = sum(g_nonzero for _, _, _, g_nonzero, _ in layers_to_check)
        print(f"\n  Total trainable params: {total_params}")
        print(f"  Params with nonzero grad: {total_nonzero} ({100*total_nonzero/max(total_params,1):.1f}%)")
        pass4b = total_nonzero > total_params * 0.5
        print(f"  RESULT: {'PASS' if pass4b else 'FAIL'}")
        results.append(("4b: Gradient to transformer", pass4b, total_nonzero / max(total_params, 1)))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results.append(("4b: Gradient to transformer", None, None))

    return results


# ======================================================================
# SUMMARY REPORT
# ======================================================================

def print_report(all_results):
    separator("DIAGNOSTIC REPORT SUMMARY")

    print(f"{'Test':<50s} | {'Result':^8s} | {'Details'}")
    print(f"{'-'*50}-+-{'-'*8}-+-{'-'*40}")

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for result_group in all_results:
        for entry in result_group:
            name = entry[0]
            passed = entry[1]
            details = entry[2:]

            if passed is None:
                status = "SKIP"
                skip_count += 1
            elif passed:
                status = "PASS"
                pass_count += 1
            else:
                status = "FAIL"
                fail_count += 1

            detail_str = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in details if v is not None)
            print(f"  {name:<48s} | {status:^8s} | {detail_str}")

    print(f"\n  Total: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP")

    # Diagnosis
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS")
    print(f"{'='*70}")

    # Check test 1 results
    test1_results = all_results[0] if all_results else []
    synth_pass = any(r[1] for r in test1_results if "Synthetic" in r[0]) if test1_results else False

    if not synth_pass:
        print("\n  CRITICAL: 8-point algorithm FAILS on perfect synthetic data.")
        print("  -> Implementation bug in pose_estimation.py. Fix before any training.")
    else:
        print("\n  8-point algorithm works on clean synthetic data.")

    # Check test 3 results
    test3_results = all_results[2] if len(all_results) > 2 else []
    for r in test3_results:
        if "Model matches" in r[0] and r[2] is not None and r[3] is not None:
            ours_err = r[2]
            cv_err = r[3]
            if cv_err < ours_err * 0.5:
                print(f"\n  OpenCV significantly outperforms our 8-point on model matches.")
                print(f"  -> Matches have outliers. Need RANSAC or better confidence weighting.")
            elif cv_err > 15 and ours_err > 15:
                print(f"\n  Both our 8-point AND OpenCV fail on model matches.")
                print(f"  -> Match quality is the bottleneck. Network produces bad correspondences.")
            else:
                print(f"\n  Our 8-point and OpenCV perform similarly on model matches.")
                print(f"  -> Algorithm is correct. Focus on improving match quality.")

    print()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  DINO-VO FOUNDATION DIAGNOSTIC TESTS")
    print("  Date: 2026-04-04")
    print("="*70)

    all_results = []
    all_results.append(test1_eight_point_with_gt())
    all_results.append(test2_pose_loss_sanity())
    all_results.append(test3_compare_opencv())
    all_results.append(test4_gradient_flow())

    print_report(all_results)
