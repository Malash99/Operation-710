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
from src.models.dino_vo import DinoVO
from src.losses.losses import DinoVOLoss
from src.utils.stereo import generate_gt_correspondences


# ------------------------------------------------------------------ #
#  GT correspondence helpers                                           #
# ------------------------------------------------------------------ #

def build_gt_matches_batch(
    kp1: torch.Tensor,
    kp2: torch.Tensor,
    depth1: torch.Tensor,
    intrinsics: torch.Tensor,
    T_gt: torch.Tensor,
    reproj_threshold: float = 5.0,
) -> tuple:
    """Generate GT correspondences for a full batch.

    Converts detected keypoints + stereo depth + GT relative pose into
    padded gt_matches and gt_mask tensors for the matching loss (Eq. 12).

    Args:
        kp1:          (B, K, 2) detected keypoints in image 1 (x, y).
        kp2:          (B, K, 2) detected keypoints in image 2 (x, y).
        depth1:       (B, H, W) stereo depth map for image 1 (meters).
        intrinsics:   (B, 3, 3) camera intrinsics K.
        T_gt:         (B, 4, 4) GT relative pose from cam1 to cam2.
        reproj_threshold: max pixel error to accept a correspondence.

    Returns:
        gt_matches: (B, M, 2) int64 tensor — (idx_in_kp1, idx_in_kp2) pairs.
                    Padded with -1 for invalid entries.
        gt_mask:    (B, M) bool tensor — True for valid entries.
    """
    B = kp1.shape[0]
    device = kp1.device

    # Convert to numpy for stereo.generate_gt_correspondences
    kp1_np = kp1.detach().cpu().numpy()          # (B, K, 2)
    kp2_np = kp2.detach().cpu().numpy()          # (B, K, 2)
    depth1_np = depth1.detach().cpu().numpy()    # (B, H, W)
    K_np = intrinsics.detach().cpu().numpy()     # (B, 3, 3)
    T_np = T_gt.detach().cpu().numpy()           # (B, 4, 4)

    all_matches = []
    for b in range(B):
        # Sample depth at keypoint locations
        kp1_b = kp1_np[b]           # (K, 2) — (x, y)
        depth_map = depth1_np[b]    # (H, W)
        H, W = depth_map.shape

        kp1_depths = np.zeros(kp1_b.shape[0], dtype=np.float32)
        for i in range(kp1_b.shape[0]):
            u, v = int(round(kp1_b[i, 0])), int(round(kp1_b[i, 1]))
            if 0 <= u < W and 0 <= v < H:
                kp1_depths[i] = depth_map[v, u]

        gt_matches_b, _ = generate_gt_correspondences(
            kp1_b, kp2_np[b], kp1_depths,
            K_np[b], T_np[b],
            reproj_threshold=reproj_threshold,
        )
        all_matches.append(gt_matches_b)   # (M_b, 2) or (0, 2)

    # Pad to fixed length M across batch
    max_m = max(m.shape[0] for m in all_matches)
    max_m = max(max_m, 1)   # at least 1 to avoid empty tensors

    gt_matches = torch.full((B, max_m, 2), -1, dtype=torch.long, device=device)
    gt_mask = torch.zeros(B, max_m, dtype=torch.bool, device=device)

    for b in range(B):
        m = all_matches[b]
        if m.shape[0] > 0:
            gt_matches[b, :m.shape[0]] = torch.from_numpy(m).to(device)
            gt_mask[b, :m.shape[0]] = True

    return gt_matches, gt_mask


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
    dataset = EuRoCDataset(
        sequence_path=cfg["data"]["sequence_path"],
        skip_frames=cfg["data"]["skip_frames"],
        target_h=cfg["data"]["target_h"],
        target_w=cfg["data"]["target_w"],
        compute_stereo_depth=cfg["data"]["compute_stereo_depth"],
        min_translation=cfg["data"].get("min_translation", 0.0),
        max_skip_multiplier=cfg["data"].get("max_skip_multiplier", 5),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Dataset pairs: {len(dataset)}")
    print(f"  Steps per epoch: {len(loader)}")

    # --- Model ---
    print("\nBuilding model...")
    model = DinoVO(
        top_k=cfg["model"]["top_k"],
        descriptor_dim=cfg["model"]["descriptor_dim"],
        matching_layers=cfg["model"]["matching_layers"],
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

    history = {"step": [], "total": [], "matching": [], "pose": [], "lambda_p": []}
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

        epoch_losses = {"total": [], "matching": [], "pose": []}
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

            # depth1 stays on CPU for numpy GT correspondence generation
            depth1 = batch["depth1"]                     # (B, H, W) — CPU

            # --- Forward pass ---
            optimizer.zero_grad()

            out = model(image1, image2, K, return_all_assignments=True)

            # --- GT correspondences ---
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

            # --- Loss ---
            loss_dict = loss_fn(
                out["all_assignments"],
                out["all_sigma1"],
                out["all_sigma2"],
                gt_matches,
                gt_mask,
                out["R"],
                out["t"],
                R_gt,
                t_gt,
            )

            total_loss = loss_dict["total"]

            # --- NaN guard: skip step if loss exploded ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count % 10 == 1:
                    print(f"\n  [WARNING] NaN/Inf loss at step {global_step} (total so far: {nan_count}). Skipping.")
                if epoch >= lambda_p_start:
                    loss_fn.step_lambda_p()
                continue

            # --- Backward ---
            total_loss.backward()

            # Clip gradients and check for NaN in one shot.
            # clip_grad_norm_ returns total norm — if NaN/Inf, SVD produced
            # degenerate gradients. Skip the step to protect weights.
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )
            if not torch.isfinite(total_norm):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count % 10 == 1:
                    print(f"\n  [WARNING] NaN gradient at step {global_step} (total: {nan_count}). Skipping.")
                if epoch >= lambda_p_start:
                    loss_fn.step_lambda_p()
                continue

            optimizer.step()

            # Increment lambda_p from epoch 5 onward
            if epoch >= lambda_p_start:
                loss_fn.step_lambda_p()

            # --- Logging ---
            total_l = total_loss.item()
            match_l = loss_dict["matching"].item()
            pose_l  = loss_dict["pose"].item()
            lp      = loss_fn.lambda_p.item()

            epoch_losses["total"].append(total_l)
            epoch_losses["matching"].append(match_l)
            epoch_losses["pose"].append(pose_l)

            pbar.set_postfix({
                "loss": f"{total_l:.4f}",
                "match": f"{match_l:.4f}",
                "pose": f"{pose_l:.2f}",
                "lp": f"{lp:.4f}",
                "nan": nan_count,
            })

            if global_step % log_interval == 0:
                n_gt = gt_mask.sum().item()
                print(
                    f"  Step {global_step} | "
                    f"total={total_l:.4f} match={match_l:.4f} "
                    f"pose_raw={pose_l:.2f} lp={lp:.4f} | "
                    f"gt_matches={n_gt} nan_skipped={nan_count}"
                )

            history["step"].append(global_step)
            history["total"].append(total_l)
            history["matching"].append(match_l)
            history["pose"].append(pose_l)
            history["lambda_p"].append(lp)

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

        print(
            f"\nEpoch {epoch:02d} complete in {elapsed:.0f}s | "
            f"avg_total={avg_total:.4f} avg_match={avg_match:.4f} "
            f"avg_pose={avg_pose:.4f} lp={loss_fn.lambda_p.item():.4f} "
            f"nan_skipped={nan_count}"
        )

        # Save checkpoint
        if epoch % ckpt_interval == 0:
            _save_checkpoint(model, optimizer, loss_fn, epoch, global_step, cfg)

    # --- Final ---
    _save_loss_curve(history, cfg["logging"]["output_dir"])
    print("\nTraining complete!")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

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
