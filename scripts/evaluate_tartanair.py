"""
DINO-VO TartanAir Evaluation Script — Diagnostic.

Runs the same trajectory evaluation pipeline as scripts/evaluate.py but on a
single TartanAir trajectory instead of an EuRoC sequence. This is the key
diagnostic for isolating domain gap (synthetic -> real) from model bugs:

  - If TartanAir ATE is good (<1m) but EuRoC ATE is bad (>0.5m) => domain gap
  - If both are bad => model / training issue

All accumulation, scale alignment, ATE, rotation error, and plotting logic
is reused from scripts/evaluate.py to guarantee identical protocol.

Usage:
    python -m scripts.evaluate_tartanair \\
        --checkpoint checkpoints_v04_tartanair/epoch_14.pth \\
        --trajectory data/tartanair/carwelding/Easy/P001 \\
        --max_pairs 200
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.tartanair import TartanAirTrajectory
from src.models.dino_vo import DinoVO

# Reuse existing evaluation functions — identical protocol to EuRoC eval
from scripts.evaluate import (
    accumulate_trajectory,
    extract_positions,
    compute_global_scale,
    apply_scale,
    compute_ate,
    compute_rotation_errors,
    plot_trajectory,
)


@torch.no_grad()
def evaluate_tartanair(
    checkpoint_path: str,
    trajectory_path: str,
    output_dir: str = "outputs",
    max_pairs: int = None,
    target_h: int = 476,
    target_w: int = 742,
    top_k: int = 512,
    descriptor_dim: int = 192,
    matching_layers: int = 12,
    coord_normalization: str = "K_inv",
):
    """Run full trajectory evaluation on a single TartanAir trajectory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset: single TartanAir trajectory, skip_frames=1 (consecutive) ---
    print(f"\nLoading TartanAir trajectory: {trajectory_path}")
    dataset = TartanAirTrajectory(
        trajectory_path=trajectory_path,
        skip_frames=1,          # consecutive frames (matches training)
        target_h=target_h,
        target_w=target_w,
    )
    print(f"  Total pairs: {len(dataset)}")

    n_pairs = min(max_pairs, len(dataset)) if max_pairs else len(dataset)
    print(f"  Evaluating: {n_pairs} pairs")

    # --- Model ---
    print("\nLoading model...")
    model = DinoVO(
        top_k=top_k,
        descriptor_dim=descriptor_dim,
        matching_layers=matching_layers,
        coord_normalization=coord_normalization,
    )
    model.load_dino(device)
    model = model.to(device)
    model.eval()

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_step = ckpt.get("global_step", "?")
    print(f"  Checkpoint from epoch {ckpt_epoch}, step {ckpt_step}")

    # --- Inference loop ---
    print(f"\nRunning inference on {n_pairs} pairs...")
    pred_relatives = []
    gt_relatives = []
    match_counts = []

    t_start = time.time()

    for idx in tqdm(range(n_pairs), desc="Inference"):
        sample = dataset[idx]

        image1 = sample["image1"].unsqueeze(0).to(device)
        image2 = sample["image2"].unsqueeze(0).to(device)
        K = sample["intrinsics"].unsqueeze(0).to(device)
        T_gt = sample["relative_pose"].numpy()

        out = model(image1, image2, K, return_all_assignments=False)

        R_pred = out["R"][0].cpu().numpy()
        t_pred = out["t"][0].cpu().numpy()
        n_matches = out["matches"][0].shape[0]

        T_pred = np.eye(4, dtype=np.float64)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = t_pred

        pred_relatives.append(T_pred)
        gt_relatives.append(T_gt.astype(np.float64))
        match_counts.append(n_matches)

    elapsed = time.time() - t_start
    fps = n_pairs / elapsed
    print(f"  Inference complete: {elapsed:.1f}s ({fps:.1f} pairs/sec)")
    print(f"  Average matches per pair: {np.mean(match_counts):.0f}")

    # --- Accumulate trajectories (reused) ---
    print("\nAccumulating trajectories...")
    pred_traj = accumulate_trajectory(pred_relatives)
    gt_traj = accumulate_trajectory(gt_relatives)

    pred_positions = extract_positions(pred_traj)
    gt_positions = extract_positions(gt_traj)

    gt_path_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
    pred_path_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1))
    print(f"  GT path length: {gt_path_length:.3f} m")
    print(f"  Pred path length (unscaled): {pred_path_length:.3f} m")

    # --- Scale alignment (reused, identical protocol to EuRoC eval) ---
    scale = compute_global_scale(gt_positions, pred_positions)
    print(f"  Global scale factor: {scale:.4f}")

    pred_traj_scaled = apply_scale(pred_traj, scale)
    pred_positions_scaled = extract_positions(pred_traj_scaled)

    scaled_path_length = np.sum(np.linalg.norm(
        np.diff(pred_positions_scaled, axis=0), axis=1))
    print(f"  Pred path length (scaled): {scaled_path_length:.3f} m")

    # --- ATE (reused) ---
    ate = compute_ate(gt_positions, pred_positions_scaled)
    print(f"\n{'='*50}")
    print(f"  ATE Results (checkpoint: epoch {ckpt_epoch})")
    print(f"{'='*50}")
    print(f"  RMSE:   {ate['ate_rmse']:.4f} m")
    print(f"  Mean:   {ate['ate_mean']:.4f} m")
    print(f"  Median: {ate['ate_median']:.4f} m")
    print(f"  Std:    {ate['ate_std']:.4f} m")
    print(f"  Max:    {ate['ate_max']:.4f} m")

    # --- Rotation error (reused) ---
    rot_errors = compute_rotation_errors(gt_traj, pred_traj_scaled)
    print(f"\n  Rotation Error:")
    print(f"  Mean:   {np.mean(rot_errors):.2f} deg")
    print(f"  Median: {np.median(rot_errors):.2f} deg")
    print(f"  Max:    {np.max(rot_errors):.2f} deg")

    # --- Plot (reused) ---
    traj_name = os.path.basename(os.path.normpath(trajectory_path))
    ckpt_name = os.path.basename(checkpoint_path).replace(".pth", "")
    plot_path = os.path.join(output_dir, f"trajectory_tartanair_{traj_name}_{ckpt_name}.png")
    plot_trajectory(
        gt_positions, pred_positions_scaled,
        ate, rot_errors,
        plot_path,
        checkpoint_name=f"TartanAir {traj_name} - epoch {ckpt_epoch} ({n_pairs} pairs)",
    )

    # --- Save results ---
    results_path = os.path.join(output_dir, f"eval_tartanair_{traj_name}_{ckpt_name}.txt")
    with open(results_path, "w") as f:
        f.write(f"DINO-VO TartanAir Evaluation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {ckpt_epoch}\n")
        f.write(f"Trajectory: {trajectory_path}\n")
        f.write(f"Pairs evaluated: {n_pairs}\n")
        f.write(f"Skip frames: 1 (consecutive)\n")
        f.write(f"Inference time: {elapsed:.1f}s ({fps:.1f} pairs/sec)\n")
        f.write(f"Avg matches per pair: {np.mean(match_counts):.0f}\n\n")
        f.write(f"Scale factor: {scale:.4f}\n")
        f.write(f"GT path length: {gt_path_length:.3f} m\n")
        f.write(f"Pred path length (unscaled): {pred_path_length:.3f} m\n")
        f.write(f"Pred path length (scaled): {scaled_path_length:.3f} m\n\n")
        f.write(f"ATE RMSE:   {ate['ate_rmse']:.4f} m\n")
        f.write(f"ATE Mean:   {ate['ate_mean']:.4f} m\n")
        f.write(f"ATE Median: {ate['ate_median']:.4f} m\n")
        f.write(f"ATE Std:    {ate['ate_std']:.4f} m\n")
        f.write(f"ATE Max:    {ate['ate_max']:.4f} m\n\n")
        f.write(f"Rotation Mean:   {np.mean(rot_errors):.2f} deg\n")
        f.write(f"Rotation Median: {np.median(rot_errors):.2f} deg\n")
        f.write(f"Rotation Max:    {np.max(rot_errors):.2f} deg\n")
    print(f"  Results saved: {results_path}")

    return ate, rot_errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DINO-VO on a single TartanAir trajectory"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pth file"
    )
    parser.add_argument(
        "--trajectory", required=True,
        help="Path to a TartanAir trajectory (e.g., data/tartanair/carwelding/Easy/P001)"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directory to save plots and metrics"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Evaluate only first N pairs (for quick testing)"
    )
    parser.add_argument(
        "--target_h", type=int, default=476,
        help="Target image height"
    )
    parser.add_argument(
        "--target_w", type=int, default=742,
        help="Target image width"
    )
    parser.add_argument(
        "--coord_norm", default="K_inv",
        choices=["K_inv", "image"],
        help="Coordinate normalization mode (K_inv or image)"
    )
    args = parser.parse_args()

    evaluate_tartanair(
        checkpoint_path=args.checkpoint,
        trajectory_path=args.trajectory,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        target_h=args.target_h,
        target_w=args.target_w,
        coord_normalization=args.coord_norm,
    )


if __name__ == "__main__":
    main()
