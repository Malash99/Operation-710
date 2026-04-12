"""
Diagnose why pose loss plateaus at ~230 after NED fix.

Tests:
1. What does pose_raw=230 mean? (rotation vs translation breakdown)
2. Run the actual model on real pairs, compare predicted vs GT pose
3. Check: is RANSAC + 8-point producing reasonable poses on model matches?
4. Check: what does OpenCV recoverPose get on the same matches?
5. Check: are the confidence weights learning anything useful?
"""

import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.tartanair import build_tartanair_dataset
from src.models.dino_vo import DinoVO
from src.models.pose_estimation import PoseEstimation
from src.losses.losses import _log_rotation


def pose_loss_breakdown(R_est, t_est, R_gt, t_gt, lambda_r=180, lambda_t=400):
    """Compute pose loss with separate rotation/translation components."""
    log_R_est = _log_rotation(R_est)
    log_R_gt = _log_rotation(R_gt)
    rot_err_rad = (log_R_est - log_R_gt).norm(dim=-1)
    rot_err_deg = rot_err_rad * 180 / np.pi

    t_est_n = t_est / t_est.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t_gt_n = t_gt / t_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    trans_dir_err = (t_est_n - t_gt_n).norm(dim=-1)
    # Convert to angular error
    cos_angle = (t_est_n * t_gt_n).sum(dim=-1).clamp(-1, 1)
    trans_angle_deg = torch.acos(cos_angle) * 180 / np.pi

    loss_r = lambda_r * rot_err_rad
    loss_t = lambda_t * trans_dir_err
    total = loss_r + loss_t

    return {
        "total": total.mean().item(),
        "loss_r": loss_r.mean().item(),
        "loss_t": loss_t.mean().item(),
        "rot_deg": rot_err_deg.mean().item(),
        "trans_deg": trans_angle_deg.mean().item(),
    }


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load latest checkpoint
    ckpt_dir = "checkpoints_v04_tartanair"
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
    latest = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {latest}")

    # Build model
    model = DinoVO(top_k=512, descriptor_dim=192, matching_layers=12)
    model.load_dino(device)
    model = model.to(device)

    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, step {ckpt['global_step']}")

    # Load a few samples from TartanAir
    dataset = build_tartanair_dataset(
        data_root="data/tartanair",
        skip_frames=1, target_h=476, target_w=742, max_pairs=8500,
    )

    pose_est = PoseEstimation().to(device)

    # Test on N random pairs
    N = 50
    np.random.seed(42)
    indices = np.random.choice(len(dataset), N, replace=False)

    results = {
        "our_8pt": [], "our_8pt_ransac": [], "opencv": [],
        "rot_err_our": [], "trans_err_our": [],
        "rot_err_cv": [], "trans_err_cv": [],
        "weights_std": [], "weights_range": [],
        "n_matches": [], "n_inliers_ransac": [],
    }

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img1 = sample["image1"].unsqueeze(0).to(device)
        img2 = sample["image2"].unsqueeze(0).to(device)
        K = sample["intrinsics"].unsqueeze(0).to(device)
        T_gt = sample["relative_pose"].unsqueeze(0).to(device)
        R_gt = T_gt[:, :3, :3]
        t_gt = T_gt[:, :3, 3]

        # Forward pass
        out = model(img1, img2, K)

        R_pred = out["R"]
        t_pred = out["t"]

        # 1. Raw model pose loss breakdown
        bd = pose_loss_breakdown(R_pred, t_pred, R_gt, t_gt)
        results["our_8pt"].append(bd["total"])
        results["rot_err_our"].append(bd["rot_deg"])
        results["trans_err_our"].append(bd["trans_deg"])

        # 2. Get matches and weights
        matches = out["matches"][0]  # (M, 2)
        weights = out["weights"][0]  # (M,)
        kp1 = out["kp1"][0]  # (K, 2)
        kp2 = out["kp2"][0]  # (K, 2)

        n_matches = matches.shape[0]
        results["n_matches"].append(n_matches)

        kp1_m = kp1[matches[:, 0]]  # (M, 2)
        kp2_m = kp2[matches[:, 1]]  # (M, 2)

        # Weight statistics
        w = weights.cpu().numpy()
        results["weights_std"].append(w.std())
        results["weights_range"].append(w.max() - w.min())

        # 3. OpenCV recoverPose on the same matches
        pts1_np = kp1_m.cpu().numpy().astype(np.float64)
        pts2_np = kp2_m.cpu().numpy().astype(np.float64)
        K_np = K[0].cpu().numpy().astype(np.float64)

        try:
            E_cv, mask_cv = cv2.findEssentialMat(
                pts1_np, pts2_np, K_np,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if E_cv is not None and mask_cv is not None:
                n_inliers = mask_cv.sum()
                results["n_inliers_ransac"].append(n_inliers)

                _, R_cv, t_cv, _ = cv2.recoverPose(E_cv, pts1_np, pts2_np, K_np, mask=mask_cv)
                R_cv_t = torch.from_numpy(R_cv).float().unsqueeze(0).to(device)
                t_cv_t = torch.from_numpy(t_cv.flatten()).float().unsqueeze(0).to(device)

                bd_cv = pose_loss_breakdown(R_cv_t, t_cv_t, R_gt, t_gt)
                results["opencv"].append(bd_cv["total"])
                results["rot_err_cv"].append(bd_cv["rot_deg"])
                results["trans_err_cv"].append(bd_cv["trans_deg"])

                # 4. RANSAC-filtered 8-point (our implementation)
                mask_t = torch.tensor(mask_cv.flatten(), dtype=torch.bool, device=device)
                w_filtered = weights.clone()
                w_filtered[~mask_t] = 0.0

                pose_filtered = pose_est(
                    kp1_m.unsqueeze(0).float(),
                    kp2_m.unsqueeze(0).float(),
                    w_filtered.unsqueeze(0).float(),
                    K.float(),
                )
                bd_ransac = pose_loss_breakdown(
                    pose_filtered["R"], pose_filtered["t"], R_gt, t_gt
                )
                results["our_8pt_ransac"].append(bd_ransac["total"])
            else:
                results["n_inliers_ransac"].append(0)
                results["opencv"].append(float('nan'))
                results["rot_err_cv"].append(float('nan'))
                results["trans_err_cv"].append(float('nan'))
                results["our_8pt_ransac"].append(float('nan'))
        except cv2.error:
            results["n_inliers_ransac"].append(0)
            results["opencv"].append(float('nan'))
            results["rot_err_cv"].append(float('nan'))
            results["trans_err_cv"].append(float('nan'))
            results["our_8pt_ransac"].append(float('nan'))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{N} pairs...")

    # Print results
    print("\n" + "="*70)
    print("POSE LOSS BREAKDOWN (50 random TartanAir pairs)")
    print("="*70)

    def safe_mean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return np.mean(vals) if vals else float('nan')

    def safe_median(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return np.median(vals) if vals else float('nan')

    print(f"\n--- Pose loss (total, lambda_r=180, lambda_t=400) ---")
    print(f"  Our 8-point (no RANSAC):   mean={safe_mean(results['our_8pt']):.1f}  median={safe_median(results['our_8pt']):.1f}")
    print(f"  Our 8-point + RANSAC:      mean={safe_mean(results['our_8pt_ransac']):.1f}  median={safe_median(results['our_8pt_ransac']):.1f}")
    print(f"  OpenCV recoverPose:        mean={safe_mean(results['opencv']):.1f}  median={safe_median(results['opencv']):.1f}")

    print(f"\n--- Rotation error (degrees) ---")
    print(f"  Our 8-point:    mean={safe_mean(results['rot_err_our']):.1f}°  median={safe_median(results['rot_err_our']):.1f}°")
    print(f"  OpenCV:         mean={safe_mean(results['rot_err_cv']):.1f}°  median={safe_median(results['rot_err_cv']):.1f}°")

    print(f"\n--- Translation direction error (degrees) ---")
    print(f"  Our 8-point:    mean={safe_mean(results['trans_err_our']):.1f}°  median={safe_median(results['trans_err_our']):.1f}°")
    print(f"  OpenCV:         mean={safe_mean(results['trans_err_cv']):.1f}°  median={safe_median(results['trans_err_cv']):.1f}°")

    print(f"\n--- Match statistics ---")
    print(f"  Matches per pair:       mean={safe_mean(results['n_matches']):.0f}")
    print(f"  RANSAC inliers:         mean={safe_mean(results['n_inliers_ransac']):.0f}")
    print(f"  Inlier ratio:           {safe_mean(results['n_inliers_ransac'])/safe_mean(results['n_matches'])*100:.1f}%")

    print(f"\n--- Confidence weight distribution ---")
    print(f"  Weight std:             mean={safe_mean(results['weights_std']):.4f}")
    print(f"  Weight range:           mean={safe_mean(results['weights_range']):.4f}")

    # Key question: if OpenCV gets good poses but our 8-point doesn't,
    # the problem is in our pose estimation, not the matches
    cv_good = sum(1 for v in results['opencv'] if not np.isnan(v) and v < 100)
    our_good = sum(1 for v in results['our_8pt'] if v < 100)
    print(f"\n--- Pairs with pose loss < 100 ---")
    print(f"  Our 8-point:  {our_good}/{N}")
    print(f"  OpenCV:       {cv_good}/{N}")

    if safe_mean(results['opencv']) < safe_mean(results['our_8pt']) * 0.5:
        print("\n>>> VERDICT: OpenCV significantly outperforms our 8-point.")
        print("    The matches are good enough — the problem is in our pose estimation.")
    elif safe_mean(results['opencv']) > 150:
        print("\n>>> VERDICT: Even OpenCV can't get good poses from these matches.")
        print("    The matches themselves are not geometrically accurate enough.")
    else:
        print("\n>>> VERDICT: Both methods struggle similarly.")
        print("    The problem may be in match quality AND pose estimation.")


if __name__ == "__main__":
    main()
