"""
DINO-VO Training Script — Phase 8.

Training schedule (paper Section IV-A):
  Epochs 1-4:  lambda_p = 0.0  (matching loss only)
  Epochs 5-14: lambda_p ramps 0.0 -> 0.9 (increment 1.5e-4 per step)

NaN stability:
  - clip_grad_norm_ limits gradient magnitude
  - torch.isfinite(total_norm) check skips steps with NaN gradients from SVD
  - Keyframe selection filters degenerate small-motion pairs

Usage:
    python -m scripts.train
    python -m scripts.train --config configs/default.yaml
    python -m scripts.train --resume checkpoints/epoch_04.pth   # resume from checkpoint
    python -m scripts.train --config configs/default.yaml --max_steps 10  # quick test
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.euroc import EuRoCDataset
from src.datasets.tartanair import build_tartanair_dataset
from src.models.dino_vo import DinoVO
from src.losses.losses import DinoVOLoss, InlierLoss


# ------------------------------------------------------------------ #
#  GT correspondence helpers                                           #
# ------------------------------------------------------------------ #

@torch.no_grad()
def build_gt_matches_batch(
    kp1: torch.Tensor,
    kp2: torch.Tensor,
    depth1: torch.Tensor,
    intrinsics: torch.Tensor,
    T_gt: torch.Tensor,
    reproj_threshold: float = 5.0,
) -> tuple:
    """Generate GT correspondences for a full batch (GPU-accelerated).

    Runs entirely on GPU — no CPU transfer or numpy. ~10x faster than CPU version.

    Args:
        kp1:          (B, K, 2) detected keypoints in image 1 (x, y).
        kp2:          (B, K, 2) detected keypoints in image 2 (x, y).
        depth1:       (B, H, W) depth map for image 1 (meters).
        intrinsics:   (B, 3, 3) camera intrinsics K.
        T_gt:         (B, 4, 4) GT relative pose from cam1 to cam2.
        reproj_threshold: max pixel error to accept a correspondence.

    Returns:
        gt_matches: (B, M, 2) int64 tensor — (idx_in_kp1, idx_in_kp2) pairs.
                    Padded with -1 for invalid entries.
        gt_mask:    (B, M) bool tensor — True for valid entries.
    """
    B, K1, _ = kp1.shape
    K2 = kp2.shape[1]
    device = kp1.device

    # Move depth to GPU if needed
    if depth1.device != device:
        depth1 = depth1.to(device)

    H, W = depth1.shape[1], depth1.shape[2]

    # --- Sample depth at keypoint locations (B, K1) ---
    u = kp1[..., 0].round().long().clamp(0, W - 1)  # (B, K1)
    v = kp1[..., 1].round().long().clamp(0, H - 1)  # (B, K1)
    # Advanced indexing to sample depth
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(u)
    depths = depth1[b_idx, v, u]  # (B, K1)

    # --- Back-project to 3D ---
    K_inv = torch.linalg.inv(intrinsics)  # (B, 3, 3)
    ones = torch.ones(B, K1, 1, device=device, dtype=kp1.dtype)
    uv_h = torch.cat([kp1, ones], dim=-1)  # (B, K1, 3)
    rays = (K_inv @ uv_h.transpose(-1, -2)).transpose(-1, -2)  # (B, K1, 3)
    p3d = rays * depths.unsqueeze(-1)  # (B, K1, 3)

    # --- Transform to camera 2 ---
    R = T_gt[:, :3, :3]  # (B, 3, 3)
    t = T_gt[:, :3, 3]   # (B, 3)
    p3d_cam2 = (R @ p3d.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(1)  # (B, K1, 3)

    # --- Project to image 2 ---
    proj = (intrinsics @ p3d_cam2.transpose(-1, -2)).transpose(-1, -2)  # (B, K1, 3)
    z = proj[..., 2:3].clamp(min=1e-6)
    proj_uv = proj[..., :2] / z  # (B, K1, 2)

    # --- Valid mask: positive depth AND in front of camera 2 ---
    valid = (depths > 0) & (p3d_cam2[..., 2] > 0)  # (B, K1)

    # --- Find nearest keypoint in image 2 (batched cdist) ---
    # proj_uv: (B, K1, 2), kp2: (B, K2, 2)
    dists = torch.cdist(proj_uv, kp2)  # (B, K1, K2) — pairwise L2 distance

    # For invalid keypoints, set distance to inf so they're never matched
    dists[~valid] = float('inf')

    # Nearest kp2 for each kp1
    min_dist, min_idx = dists.min(dim=2)  # (B, K1), (B, K1)

    # Accept matches within threshold
    accept = (min_dist < reproj_threshold) & valid  # (B, K1)

    # --- Build match tensors per batch element ---
    all_matches = []
    for b in range(B):
        acc_b = accept[b]  # (K1,)
        if acc_b.sum() == 0:
            all_matches.append(torch.zeros(0, 2, dtype=torch.long, device=device))
            continue

        kp1_idx = torch.where(acc_b)[0]  # indices into kp1
        kp2_idx = min_idx[b, acc_b]       # corresponding kp2 indices

        # Deduplicate: keep best (closest) match per kp2
        unique_kp2, inverse = torch.unique(kp2_idx, return_inverse=True)
        # For each unique kp2, find the kp1 with smallest distance
        best_per_kp2 = torch.zeros(unique_kp2.shape[0], dtype=torch.long, device=device)
        best_dist_per_kp2 = torch.full((unique_kp2.shape[0],), float('inf'), device=device)
        match_dists = min_dist[b, acc_b]
        for i in range(kp1_idx.shape[0]):
            uid = inverse[i]
            if match_dists[i] < best_dist_per_kp2[uid]:
                best_dist_per_kp2[uid] = match_dists[i]
                best_per_kp2[uid] = i

        matches_b = torch.stack([kp1_idx[best_per_kp2], unique_kp2], dim=1)  # (M, 2)
        all_matches.append(matches_b)

    # Pad to fixed length M across batch
    max_m = max(m.shape[0] for m in all_matches)
    max_m = max(max_m, 1)

    gt_matches = torch.full((B, max_m, 2), -1, dtype=torch.long, device=device)
    gt_mask = torch.zeros(B, max_m, dtype=torch.bool, device=device)

    for b in range(B):
        m = all_matches[b]
        if m.shape[0] > 0:
            gt_matches[b, :m.shape[0]] = m
            gt_mask[b, :m.shape[0]] = True

    return gt_matches, gt_mask


# ------------------------------------------------------------------ #
#  RANSAC outlier filtering (Phase 12 fix)                             #
# ------------------------------------------------------------------ #

@torch.no_grad()
def ransac_filter_pose(
    kp1_matched: torch.Tensor,
    kp2_matched: torch.Tensor,
    w_matched: torch.Tensor,
    intrinsics: torch.Tensor,
    threshold: float = 3.0,
) -> tuple:
    """Use OpenCV RANSAC to identify inlier matches, zero-weight outliers.

    This breaks the chicken-and-egg problem: the 8-point algorithm gets
    clean (inlier-only) matches, producing meaningful pose gradients that
    allow the ConfMLP to learn which matches are good.

    The RANSAC mask is applied by zeroing outlier weights (not removing them),
    so tensor shapes stay the same and gradients still flow through inlier weights.

    Also returns the RANSAC-estimated R, t as a better pose target for the
    pose loss (optional — can still use GT pose).

    Args:
        kp1_matched: (B, M, 2) matched keypoints in image 1.
        kp2_matched: (B, M, 2) matched keypoints in image 2.
        w_matched:   (B, M) per-match confidence weights.
        intrinsics:  (B, 3, 3) camera intrinsics K.
        threshold:   RANSAC inlier threshold in pixels.

    Returns:
        w_filtered:    (B, M) weights with outliers zeroed.
        inlier_counts: list of int — number of inliers per batch element.
    """
    B = kp1_matched.shape[0]
    w_filtered = w_matched.clone()
    inlier_counts = []

    for b in range(B):
        pts1 = kp1_matched[b].cpu().numpy().astype(np.float64)
        pts2 = kp2_matched[b].cpu().numpy().astype(np.float64)
        K = intrinsics[b].cpu().numpy().astype(np.float64)

        # Skip if too few non-zero-weight matches
        valid = w_matched[b] > 1e-6
        n_valid = valid.sum().item()
        if n_valid < 8:
            inlier_counts.append(0)
            continue

        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=threshold,
            )
            if mask is not None:
                mask_t = torch.tensor(mask.flatten(), dtype=torch.bool, device=w_matched.device)
                # Zero out outlier weights
                w_filtered[b] = w_matched[b] * mask_t.float()
                inlier_counts.append(mask_t.sum().item())
            else:
                inlier_counts.append(0)
        except cv2.error:
            inlier_counts.append(0)

    return w_filtered, inlier_counts


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(cfg: dict, resume: str = None, max_steps: int = None):
    """Main training loop.

    Args:
        cfg: Parsed configuration dictionary (from default.yaml).
        resume: Path to a checkpoint .pth file to resume from.
        max_steps: If set, stop after this many steps (for quick testing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)

    # --- Dataset & DataLoader ---
    print("\nLoading dataset...")
    dataset_type = cfg["data"].get("dataset", "euroc")

    if dataset_type == "tartanair":
        dataset = build_tartanair_dataset(
            data_root=cfg["data"]["data_root"],
            skip_frames=cfg["data"]["skip_frames"],
            target_h=cfg["data"]["target_h"],
            target_w=cfg["data"]["target_w"],
            max_pairs=cfg["data"].get("max_pairs", 0),
            augmentation=cfg["data"].get("augmentation", False),
        )
    else:
        dataset = EuRoCDataset(
            sequence_path=cfg["data"]["sequence_path"],
            skip_frames=cfg["data"]["skip_frames"],
            target_h=cfg["data"]["target_h"],
            target_w=cfg["data"]["target_w"],
            compute_stereo_depth=cfg["data"].get("compute_stereo_depth", True),
            min_translation=cfg["data"].get("min_translation", 0.0),
            max_skip_multiplier=cfg["data"].get("max_skip_multiplier", 5),
        )
    num_workers = cfg["training"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"  Dataset pairs: {len(dataset)}")
    print(f"  Steps per epoch: {len(loader)}")

    # --- Model ---
    print("\nBuilding model...")
    model = DinoVO(
        top_k=cfg["model"]["top_k"],
        descriptor_dim=cfg["model"]["descriptor_dim"],
        matching_layers=cfg["model"]["matching_layers"],
        coord_normalization=cfg["training"].get("coord_normalization", "K_inv"),
    )
    model.load_dino(device)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    # --- Loss ---
    loss_fn = DinoVOLoss(
        lambda_p_increment=cfg["training"]["lambda_p_increment"],
        lambda_p_max=cfg["training"]["lambda_p_max"],
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # --- Training state ---
    epochs = cfg["training"]["epochs"]
    log_interval = cfg["logging"]["log_interval"]
    ckpt_interval = cfg["logging"]["checkpoint_interval"]
    lambda_p_start = cfg["training"]["lambda_p_start_epoch"]
    reproj_thr = cfg["model"]["reproj_threshold"]
    use_ransac = cfg["training"].get("ransac_filter", False)
    ransac_thr = cfg["training"].get("ransac_threshold", 3.0)
    if use_ransac:
        print(f"  RANSAC outlier filtering ENABLED (threshold={ransac_thr}px)")

    # --- Inlier pretraining loss (confidence supervision) ---
    use_inlier = cfg["training"].get("inlier_loss", False)
    lambda_inlier = cfg["training"].get("lambda_inlier", 0.5)
    # Separate threshold for inlier BCE labels — tighter than GT match threshold.
    # 2px gives ~50% inlier split (balanced BCE); 5px gives ~70% (near-entropy, weak gradient).
    inlier_reproj_thr = cfg["training"].get("inlier_reproj_threshold", reproj_thr)
    inlier_loss_fn = None
    if use_inlier:
        inlier_loss_fn = InlierLoss(reproj_threshold=inlier_reproj_thr).to(device)
        print(f"  Inlier pretraining loss ENABLED (lambda={lambda_inlier}, reproj_thr={inlier_reproj_thr}px)")

    # --- Mixed precision (AMP) for faster training on RTX GPUs ---
    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  Mixed precision (AMP) enabled — ~1.5-2x speedup")

    history = {"step": [], "total": [], "matching": [], "pose": [], "lambda_p": [], "inlier": []}
    global_step = 0
    run_steps = 0   # steps taken in this run (for max_steps stopping)
    start_epoch = 1
    nan_count = 0

    # --- Resume from checkpoint ---
    if resume:
        print(f"\nResuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_fn.set_lambda_p(ckpt.get("lambda_p", 0.0))
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"  Resumed from epoch {ckpt['epoch']}, step {global_step}, lp={ckpt.get('lambda_p', 0.0):.4f}")

    print(f"\nStarting training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        # Set lambda_p schedule (paper Section IV-A)
        if epoch < lambda_p_start:
            loss_fn.set_lambda_p(0.0)

        epoch_losses = {"total": [], "matching": [], "pose": [], "inlier": []}
        epoch_start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=True)

        for batch in pbar:
            global_step += 1
            run_steps += 1

            # Move to device
            image1 = batch["image1"].to(device)          # (B, 3, H, W)
            image2 = batch["image2"].to(device)          # (B, 3, H, W)
            T_gt = batch["relative_pose"].to(device)     # (B, 4, 4)
            K = batch["intrinsics"].to(device)           # (B, 3, 3)

            depth1 = batch["depth1"].to(device)          # (B, H, W)

            # --- Forward pass ---
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(image1, image2, K, return_all_assignments=True)

            # --- GT correspondences (GPU-accelerated) ---
            gt_matches, gt_mask = build_gt_matches_batch(
                out["kp1"], out["kp2"],
                depth1, K, T_gt,
                reproj_threshold=reproj_thr,
            )

            # Skip step if no GT matches in this batch (no matching supervision signal)
            if gt_mask.sum() == 0:
                optimizer.zero_grad()
                continue

            # --- GT R, t ---
            R_gt = T_gt[:, :3, :3]   # (B, 3, 3)
            t_gt = T_gt[:, :3, 3]    # (B, 3)

            # --- RANSAC filtering (Phase 12): re-estimate pose with inlier-only weights ---
            R_for_loss = out["R"]
            t_for_loss = out["t"]

            if use_ransac and epoch >= lambda_p_start:
                # Get the matched keypoints from the model's internal padding
                # Re-extract using the same match indices
                from src.models.dino_vo import _pad_matches
                matches_list = out["matches"]
                weights_list = [
                    torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    for w in out["weights"]
                ]
                kp1_m, kp2_m, w_m = _pad_matches(
                    out["kp1"], out["kp2"], matches_list, weights_list
                )
                # Apply RANSAC to zero outlier weights
                w_filtered, inlier_cts = ransac_filter_pose(
                    kp1_m, kp2_m, w_m, K, threshold=ransac_thr,
                )
                # Re-run pose estimation with filtered weights (still differentiable
                # through inlier weights — RANSAC mask is detached)
                H_img, W_img = image1.shape[2], image1.shape[3]
                with torch.amp.autocast("cuda", enabled=False):
                    pose_filtered = model.pose_est(
                        kp1_m.float(), kp2_m.float(),
                        w_filtered.float(), K.float(),
                        img_h=H_img, img_w=W_img,
                    )
                R_for_loss = pose_filtered["R"]
                t_for_loss = pose_filtered["t"]

            # --- Loss (in float32 for stability) ---
            loss_dict = loss_fn(
                out["all_assignments"],
                out["all_sigma1"],
                out["all_sigma2"],
                gt_matches,
                gt_mask,
                R_for_loss,
                t_for_loss,
                R_gt,
                t_gt,
            )

            total_loss = loss_dict["total"]

            # --- Inlier pretraining loss (confidence supervision) ---
            if use_inlier:
                # Clean NaN weights for stability (same as RANSAC path)
                weights_clean = [
                    torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    for w in out["weights"]
                ]
                L_inlier = inlier_loss_fn(
                    out["kp1"], out["kp2"],
                    out["matches"], weights_clean,
                    depth1, K, T_gt,
                )
                total_loss = total_loss + lambda_inlier * L_inlier
            else:
                L_inlier = torch.tensor(0.0, device=device)

            # --- NaN guard: skip step if loss exploded ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count % 100 == 1:
                    print(f"\n  [WARNING] NaN/Inf loss at step {global_step} (total so far: {nan_count}). Skipping.")
                # NOTE: do NOT step lambda_p on NaN skips — it would ramp pose
                # weight to max while no pose learning has occurred.
                continue

            # --- Backward with AMP scaler ---
            scaler.scale(total_loss).backward()

            # Unscale before clip_grad_norm_ so thresholds work correctly
            scaler.unscale_(optimizer)

            # --- NaN gradient check: skip step entirely if SVD backward exploded ---
            # Zeroing NaN grads and stepping corrupts matching head weights.
            # Better to skip ~33% of steps cleanly than poison every step.
            grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norms = torch.stack([g.norm() for g in grad_tensors])
            if not torch.isfinite(grad_norms).all():
                nan_count += 1
                optimizer.zero_grad()
                scaler.update()   # keep scaler in sync even on skipped steps
                if nan_count % 100 == 1:
                    print(f"\n  [INFO] NaN grads — step skipped at {global_step} (total: {nan_count})")
                # NOTE: do NOT step lambda_p on NaN skips — it would ramp pose
                # weight to max while no pose learning has occurred.
                continue

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()

            # Increment lambda_p from epoch 5 onward
            if epoch >= lambda_p_start:
                loss_fn.step_lambda_p()

            # --- Logging ---
            total_l = total_loss.item()
            match_l = loss_dict["matching"].item()
            pose_l  = loss_dict["pose"].item()
            inlier_l = L_inlier.item() if use_inlier else 0.0
            lp      = loss_fn.lambda_p.item()

            epoch_losses["total"].append(total_l)
            epoch_losses["matching"].append(match_l)
            epoch_losses["pose"].append(pose_l)
            epoch_losses["inlier"].append(inlier_l)

            postfix = {
                "loss": f"{total_l:.4f}",
                "match": f"{match_l:.4f}",
                "pose": f"{pose_l:.2f}",
                "lp": f"{lp:.4f}",
                "nan": nan_count,
            }
            if use_inlier:
                postfix["inlier"] = f"{inlier_l:.4f}"
            pbar.set_postfix(postfix)

            if global_step % log_interval == 0:
                n_gt = gt_mask.sum().item()
                extra_info = ""
                if use_ransac and epoch >= lambda_p_start:
                    avg_inliers = np.mean(inlier_cts) if inlier_cts else 0
                    extra_info += f" ransac_inliers={avg_inliers:.0f}"
                if use_inlier:
                    extra_info += f" inlier_loss={inlier_l:.4f}"
                print(
                    f"  Step {global_step} | "
                    f"total={total_l:.4f} match={match_l:.4f} "
                    f"pose_raw={pose_l:.2f} lp={lp:.4f} | "
                    f"gt_matches={n_gt} nan_skipped={nan_count}{extra_info}"
                )

            history["step"].append(global_step)
            history["total"].append(total_l)
            history["matching"].append(match_l)
            history["pose"].append(pose_l)
            history["lambda_p"].append(lp)
            history["inlier"].append(inlier_l)

            if max_steps and run_steps >= max_steps:
                print(f"\nReached max_steps={max_steps}, stopping.")
                _save_checkpoint(model, optimizer, loss_fn, epoch, global_step, cfg)
                _save_loss_curve(history, cfg["logging"]["output_dir"])
                return

        # --- End of epoch ---
        elapsed = time.time() - epoch_start
        avg_total = np.mean(epoch_losses["total"])
        avg_match = np.mean(epoch_losses["matching"])
        avg_pose  = np.mean(epoch_losses["pose"])
        avg_inlier = np.mean(epoch_losses["inlier"]) if epoch_losses["inlier"] else 0.0

        inlier_str = f" avg_inlier={avg_inlier:.4f}" if use_inlier else ""
        print(
            f"\nEpoch {epoch:02d} complete in {elapsed:.0f}s | "
            f"avg_total={avg_total:.4f} avg_match={avg_match:.4f} "
            f"avg_pose={avg_pose:.4f} lp={loss_fn.lambda_p.item():.4f} "
            f"nan_skipped={nan_count}{inlier_str}"
        )

        # ConfMLP discrimination check — run on a small validation batch
        # Tells us if confidence weights are learning to separate inliers from outliers.
        # Target: ratio >= 5.0 before activating pose loss (Stage 2).
        if use_inlier:
            _log_confmlp_discrimination(model, dataset, device,
                                        inlier_reproj_thr=inlier_reproj_thr,
                                        n_samples=20)

        # Save checkpoint
        if epoch % ckpt_interval == 0:
            _save_checkpoint(model, optimizer, loss_fn, epoch, global_step, cfg)

    # --- Final ---
    _save_loss_curve(history, cfg["logging"]["output_dir"])
    print("\nTraining complete!")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _log_confmlp_discrimination(model, dataset, device, inlier_reproj_thr, n_samples=20):
    """Sample n_samples pairs and report mean weight on GT-inliers vs GT-outliers.

    This is the key convergence signal for Stage 1: ratio should grow from ~1.4x
    toward >5x as ConfMLP learns to assign high weight to geometrically correct matches.
    """
    model.eval()
    w_inlier_all, w_outlier_all = [], []
    idxs = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in idxs:
        sample = dataset[int(idx)]
        img1 = sample["image1"].unsqueeze(0).to(device)
        img2 = sample["image2"].unsqueeze(0).to(device)
        K = sample["intrinsics"].to(device)
        depth1 = sample["depth1"].to(device)
        T_gt = sample["relative_pose"].to(device)

        out = model(img1, img2, K.unsqueeze(0))
        matches = out["matches"][0]
        weights = out["weights"][0]
        if matches.shape[0] == 0:
            continue

        kp1_m = out["kp1"][0][matches[:, 0]].float()
        kp2_m = out["kp2"][0][matches[:, 1]].float()
        w = torch.nan_to_num(weights, nan=0.5)

        # GT inlier labels via depth reprojection
        px = kp1_m[:, 0].long().clamp(0, depth1.shape[1] - 1)
        py = kp1_m[:, 1].long().clamp(0, depth1.shape[0] - 1)
        d = depth1[py, px]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X1 = torch.stack([(kp1_m[:, 0] - cx) * d / fx,
                           (kp1_m[:, 1] - cy) * d / fy, d], dim=-1)
        R_gt = T_gt[:3, :3]; t_gt = T_gt[:3, 3]
        X2 = X1 @ R_gt.T + t_gt
        z2 = X2[:, 2].clamp(min=1e-6)
        u2 = fx * X2[:, 0] / z2 + cx
        v2 = fy * X2[:, 1] / z2 + cy
        err = torch.stack([u2, v2], dim=-1).sub(kp2_m).norm(dim=-1)
        inlier = (d > 0.1) & (X2[:, 2] > 0.1) & (err < inlier_reproj_thr)

        if inlier.sum() > 0:
            w_inlier_all.append(w[inlier].mean().item())
        if (~inlier).sum() > 0:
            w_outlier_all.append(w[~inlier].mean().item())

    model.train()
    if w_inlier_all and w_outlier_all:
        wi = np.mean(w_inlier_all)
        wo = np.mean(w_outlier_all)
        ratio = wi / max(wo, 1e-6)
        status = "OK" if ratio >= 5.0 else ("improving" if ratio >= 2.5 else "WEAK")
        print(f"  ConfMLP discrimination: inlier={wi:.3f}  outlier={wo:.3f}  "
              f"ratio={ratio:.2f}x  [{status}] (target: >=5.0x for Stage 2)")


def _save_checkpoint(model, optimizer, loss_fn, epoch, step, cfg):
    ckpt_dir = cfg["logging"]["checkpoint_dir"]
    path = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pth")
    torch.save({
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lambda_p": loss_fn.lambda_p.item(),
    }, path)
    print(f"  Checkpoint saved: {path}")


def _save_loss_curve(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = history["step"]

    axes[0].plot(steps, history["total"],    label="total",    linewidth=1)
    axes[0].plot(steps, history["matching"], label="matching", linewidth=1, alpha=0.8)
    axes[0].plot(steps, history["pose"],     label="pose",     linewidth=1, alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history["lambda_p"], color="orange", linewidth=1)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("lp")
    axes[1].set_title("Lambda_p Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Loss curve saved: {path}")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train DINO-VO")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint .pth file to resume from (e.g. checkpoints/epoch_04.pth)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Stop after N steps (for quick testing)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
