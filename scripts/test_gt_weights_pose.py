"""
Test: Does the 8-point algorithm work if we feed it *perfect* weights?

WHAT THIS TESTS
---------------
We take the model's predicted matches (kp1, kp2), but replace the ConfMLP's
learned confidence weights with GROUND-TRUTH binary inlier labels computed
from depth + GT pose reprojection.

For each match (kp1_i, kp2_i):
    1. Back-project kp1_i using depth1 to a 3D point in camera 1's frame.
    2. Transform it to camera 2's frame using GT relative pose.
    3. Project into image 2.
    4. If reprojected position is close to kp2_i (< threshold px) -> inlier (w=1.0)
       Otherwise -> outlier (w=0.0)

Then we feed these perfect 1.0/0.0 weights into our own 8-point + E decomposition
and compare the recovered pose to GT.

WHAT IT TELLS US
----------------
Three schemes are compared:

    (A) MODEL weights       : our ConfMLP output as-is (baseline)
    (B) GT-BINARY weights   : 1.0 for inliers, 0.0 for outliers (perfect confidence)
    (C) INLIERS ONLY        : same as (B) but with outlier matches fully removed

VERDICT LOGIC
-------------
    * If (B) and (C) give small rotation/translation error (< 5 deg)
      -> Our 8-point + Essential decomposition is CORRECT.
      -> The pose-loss plateau is purely a ConfMLP learning problem.
      -> More/better confidence supervision would eventually fix it.

    * If (B) and (C) still give large error (> 20 deg)
      -> There is a structural bug in our pose layer (8-point math,
         coord normalization, Essential decomposition, or cheirality).
      -> No amount of training data will fix the plateau.

    * If (B) helps but (C) helps much more
      -> Our 8-point is sensitive to w=0 weights vs actually removing rows.
      -> Minor implementation note but not the root cause.

Run:
    python -m scripts.test_gt_weights_pose \
        --checkpoint checkpoints_v06_tartanair/epoch_14.pth \
        --trajectory data/tartanair/carwelding/Easy/P001 \
        --n_pairs 50 --reproj_thresh 2.0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.tartanair import TartanAirTrajectory
from src.models.dino_vo import DinoVO
from src.models.pose_estimation import PoseEstimation
from src.losses.losses import _log_rotation


def pose_errors(R_est, t_est, R_gt, t_gt):
    """Return rotation (deg) and translation direction (deg) errors.

    Inputs can be (3,3)/(3,) or (B,3,3)/(B,3). We batch-ify for _log_rotation.
    """
    if R_est.dim() == 2:
        R_est = R_est.unsqueeze(0)
    if R_gt.dim() == 2:
        R_gt = R_gt.unsqueeze(0)
    if t_est.dim() == 1:
        t_est = t_est.unsqueeze(0)
    if t_gt.dim() == 1:
        t_gt = t_gt.unsqueeze(0)

    R_rel = R_est.transpose(-2, -1) @ R_gt  # (B,3,3)
    log_R = _log_rotation(R_rel)            # (B,3)
    rot_deg = log_R.norm(dim=-1) * 180.0 / np.pi

    t_est_n = t_est / t_est.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t_gt_n = t_gt / t_gt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos = (t_est_n * t_gt_n).sum(dim=-1).clamp(-1, 1)
    trans_deg = torch.acos(cos) * 180.0 / np.pi
    return rot_deg[0].item(), trans_deg[0].item()


def compute_gt_inlier_mask(kp1, kp2, depth1, K, T_gt, reproj_thresh):
    """
    Compute binary inlier mask using GT depth + GT pose reprojection.
    Returns: (M,) bool tensor
    """
    device = kp1.device
    M = kp1.shape[0]

    # Sample depth at kp1 (bilinear would be more accurate, but nearest is fine)
    px = kp1[:, 0].long().clamp(0, depth1.shape[1] - 1)
    py = kp1[:, 1].long().clamp(0, depth1.shape[0] - 1)
    d = depth1[py, px]  # (M,)

    # Back-project kp1 to 3D (camera 1 frame)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam1 = (kp1[:, 0] - cx) * d / fx
    y_cam1 = (kp1[:, 1] - cy) * d / fy
    z_cam1 = d
    X1 = torch.stack([x_cam1, y_cam1, z_cam1], dim=-1)  # (M, 3)

    # Transform to camera 2 frame
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    X2 = X1 @ R_gt.T + t_gt  # (M, 3)

    # Project to image 2
    z2 = X2[:, 2]
    u2 = fx * X2[:, 0] / z2.clamp(min=1e-6) + cx
    v2 = fy * X2[:, 1] / z2.clamp(min=1e-6) + cy

    # Reprojection error in pixels
    reproj = torch.stack([u2, v2], dim=-1)
    err = (reproj - kp2).norm(dim=-1)  # (M,)

    # Valid: positive depth at kp1, positive depth after transform, error below threshold
    valid_depth = (d > 0.1) & (d < 1000.0) & (z2 > 0.1)
    inlier = valid_depth & (err < reproj_thresh)
    return inlier, err, valid_depth


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--reproj_thresh", type=float, default=2.0,
                        help="Reprojection error threshold (pixels) for inlier.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trajectory: {args.trajectory}")
    print(f"N pairs: {args.n_pairs}, reproj_thresh: {args.reproj_thresh} px")
    print("=" * 70)

    # Build model
    model = DinoVO(top_k=512, descriptor_dim=192, matching_layers=12)
    model.load_dino(device)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')}")

    # Dataset (just one trajectory)
    dataset = TartanAirTrajectory(
        trajectory_path=args.trajectory,
        skip_frames=1, target_h=476, target_w=742, augmentation=False,
    )
    print(f"Trajectory pairs available: {len(dataset)}")

    pose_est = PoseEstimation().to(device)

    # Pick evenly-spaced indices
    idxs = np.linspace(0, len(dataset) - 1, args.n_pairs, dtype=int)

    results = {
        "A_rot": [], "A_trans": [],          # Model weights
        "B_rot": [], "B_trans": [],          # GT binary weights
        "C_rot": [], "C_trans": [],          # Inliers only
        "n_matches": [], "n_gt_inliers": [], "inlier_ratio": [],
        "weight_mean_inlier": [], "weight_mean_outlier": [],
    }

    for i, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        img1 = sample["image1"].unsqueeze(0).to(device)
        img2 = sample["image2"].unsqueeze(0).to(device)
        K = sample["intrinsics"].to(device)
        depth1 = sample["depth1"].to(device)
        T_gt = sample["relative_pose"].to(device)
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        # Forward pass
        out = model(img1, img2, K.unsqueeze(0))

        matches = out["matches"][0]
        weights = out["weights"][0]
        kp1_full = out["kp1"][0]
        kp2_full = out["kp2"][0]

        kp1_m = kp1_full[matches[:, 0]].float()
        kp2_m = kp2_full[matches[:, 1]].float()
        w_model = weights.float()

        M = kp1_m.shape[0]
        if M < 10:
            continue

        # Compute GT inlier labels
        inlier_mask, reproj_err, valid_depth = compute_gt_inlier_mask(
            kp1_m, kp2_m, depth1, K, T_gt, args.reproj_thresh
        )
        n_valid = valid_depth.sum().item()
        n_inliers = inlier_mask.sum().item()
        if n_inliers < 8:
            # Not enough inliers to run 8-point
            continue

        # ----- (A) Model weights -----
        outA = pose_est(kp1_m.unsqueeze(0), kp2_m.unsqueeze(0),
                        w_model.unsqueeze(0), K.unsqueeze(0))
        rA, tA = pose_errors(outA["R"][0], outA["t"][0], R_gt, t_gt)

        # ----- (B) GT binary weights (zeros for outliers) -----
        w_gt = inlier_mask.float()
        outB = pose_est(kp1_m.unsqueeze(0), kp2_m.unsqueeze(0),
                        w_gt.unsqueeze(0), K.unsqueeze(0))
        rB, tB = pose_errors(outB["R"][0], outB["t"][0], R_gt, t_gt)

        # ----- (C) Inliers only (drop outliers completely) -----
        kp1_in = kp1_m[inlier_mask].unsqueeze(0)
        kp2_in = kp2_m[inlier_mask].unsqueeze(0)
        w_in = torch.ones(n_inliers, device=device).unsqueeze(0)
        outC = pose_est(kp1_in, kp2_in, w_in, K.unsqueeze(0))
        rC, tC = pose_errors(outC["R"][0], outC["t"][0], R_gt, t_gt)

        # Record
        results["A_rot"].append(rA); results["A_trans"].append(tA)
        results["B_rot"].append(rB); results["B_trans"].append(tB)
        results["C_rot"].append(rC); results["C_trans"].append(tC)
        results["n_matches"].append(M)
        results["n_gt_inliers"].append(n_inliers)
        results["inlier_ratio"].append(n_inliers / M)

        # Is the ConfMLP actually giving higher weight to inliers?
        if n_inliers > 0 and (M - n_inliers) > 0:
            results["weight_mean_inlier"].append(
                w_model[inlier_mask].mean().item())
            results["weight_mean_outlier"].append(
                w_model[~inlier_mask].mean().item())

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_pairs} processed")

    # ---------- Report ----------
    def stats(lst):
        arr = np.array(lst)
        return arr.mean(), np.median(arr), arr.std()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nPairs evaluated: {len(results['A_rot'])}/{args.n_pairs}")
    print(f"Avg matches/pair:    {np.mean(results['n_matches']):.0f}")
    print(f"Avg GT inliers/pair: {np.mean(results['n_gt_inliers']):.0f}")
    print(f"Avg inlier ratio:    {np.mean(results['inlier_ratio'])*100:.1f}%")

    print("\n--- ConfMLP inlier discrimination ---")
    if results["weight_mean_inlier"]:
        wi = np.mean(results["weight_mean_inlier"])
        wo = np.mean(results["weight_mean_outlier"])
        print(f"  Mean weight on GT inliers : {wi:.4f}")
        print(f"  Mean weight on GT outliers: {wo:.4f}")
        print(f"  Ratio (should be >>1):      {wi/max(wo,1e-6):.2f}x")

    print("\n--- Rotation error (deg) ---")
    mA, mdA, sA = stats(results["A_rot"])
    mB, mdB, sB = stats(results["B_rot"])
    mC, mdC, sC = stats(results["C_rot"])
    print(f"  (A) Model weights : mean={mA:6.2f}  median={mdA:6.2f}  std={sA:6.2f}")
    print(f"  (B) GT binary w   : mean={mB:6.2f}  median={mdB:6.2f}  std={sB:6.2f}")
    print(f"  (C) Inliers only  : mean={mC:6.2f}  median={mdC:6.2f}  std={sC:6.2f}")

    print("\n--- Translation direction error (deg) ---")
    mA, mdA, sA = stats(results["A_trans"])
    mB, mdB, sB = stats(results["B_trans"])
    mC, mdC, sC = stats(results["C_trans"])
    print(f"  (A) Model weights : mean={mA:6.2f}  median={mdA:6.2f}  std={sA:6.2f}")
    print(f"  (B) GT binary w   : mean={mB:6.2f}  median={mdB:6.2f}  std={sB:6.2f}")
    print(f"  (C) Inliers only  : mean={mC:6.2f}  median={mdC:6.2f}  std={sC:6.2f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    medC_rot = np.median(results["C_rot"])
    medC_trans = np.median(results["C_trans"])

    if medC_rot < 5 and medC_trans < 5:
        print("  [+] (C) Inliers-only gives <5 deg median errors.")
        print("      -> Our 8-point + E decomposition is CORRECT.")
        print("      -> Pose plateau is a ConfMLP *learning* problem.")
        print("      -> Fix: better confidence supervision, not more data.")
    elif medC_rot > 20 or medC_trans > 20:
        print("  [!] (C) Inliers-only STILL gives >20 deg errors with")
        print("      perfect inlier-only correspondences.")
        print("      -> Structural bug in pose pipeline.")
        print("      -> Check: coord normalization, E decomposition, cheirality.")
        print("      -> More data will NOT fix this.")
    else:
        print("  [?] Middle ground: (C) is better than (A) but not great.")
        print("      Pose pipeline is partially correct — likely sensitive")
        print("      to match noise or minor numeric issues.")


if __name__ == "__main__":
    main()
