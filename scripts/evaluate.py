"""
DINO-VO Evaluation Script — Phase 9.

Evaluation protocol (paper Section IV):
  1. Run inference on sequential frame pairs (no shuffling, batch=1)
  2. Accumulate relative poses into a global trajectory
  3. Compute single global scale factor: s = sum(||t_gt||) / sum(||t_pred||)
     (follows TartanVO [7] — one scalar for the entire sequence)
  4. Apply scale to predicted translations
  5. Compute ATE = RMSE of aligned position differences
  6. Plot predicted vs GT trajectory (3D + 2D top-down)

Scale ambiguity:
  Monocular VO can only recover translation direction, not magnitude.
  The paper resolves this with a single global scale factor computed from
  ground truth, NOT per-frame scaling or Umeyama alignment. This is the
  standard evaluation protocol for monocular VO (TartanVO, ORB-SLAM, etc.).

Usage:
    python -m scripts.evaluate --checkpoint checkpoints/epoch_04.pth
    python -m scripts.evaluate --checkpoint checkpoints/epoch_04.pth --sequence data/euroc/MH_01_easy
    python -m scripts.evaluate --checkpoint checkpoints/epoch_04.pth --max_pairs 100
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.euroc import EuRoCDataset
from src.models.dino_vo import DinoVO


# ------------------------------------------------------------------ #
#  Evaluation dataset — sequential, no filtering, no stereo depth     #
# ------------------------------------------------------------------ #

def load_eval_dataset(sequence_path: str, skip_frames: int = 2,
                      target_h: int = 476, target_w: int = 742):
    """Load EuRoC dataset for evaluation (sequential, no keyframe filtering).

    Unlike training, evaluation uses ALL sequential pairs to build a
    complete trajectory. No min_translation filtering, no stereo depth.
    """
    dataset = EuRoCDataset(
        sequence_path=sequence_path,
        skip_frames=skip_frames,
        target_h=target_h,
        target_w=target_w,
        compute_stereo_depth=False,   # not needed for inference
        min_translation=0.0,          # no filtering — use all pairs
        max_skip_multiplier=1,        # fixed skip, no dynamic skipping
    )
    return dataset


# ------------------------------------------------------------------ #
#  Trajectory accumulation                                             #
# ------------------------------------------------------------------ #

def accumulate_trajectory(relative_poses: list) -> np.ndarray:
    """Chain relative poses into a global trajectory.

    T_global[0] = I (identity — origin at first camera)
    T_global[k] = T_global[k-1] @ T_rel[k-1]

    where T_rel[k] transforms points from camera k to camera k+1,
    so T_global[k] is the pose of camera k in the world (first camera) frame.

    Args:
        relative_poses: List of N (4, 4) relative pose matrices.

    Returns:
        trajectory: (N+1, 4, 4) array of global poses.
    """
    N = len(relative_poses)
    trajectory = np.zeros((N + 1, 4, 4), dtype=np.float64)
    trajectory[0] = np.eye(4)

    for k in range(N):
        # T_rel transforms cam_k points to cam_{k+1} frame
        # Global pose of cam_{k+1} = Global pose of cam_k @ inv(T_rel)
        # Because T_rel = inv(T_WC_{k+1}) @ T_WC_k
        # => T_WC_{k+1} = T_WC_k @ inv(T_rel)
        T_rel = relative_poses[k]
        T_rel_inv = np.linalg.inv(T_rel)
        trajectory[k + 1] = trajectory[k] @ T_rel_inv

    return trajectory


def extract_positions(trajectory: np.ndarray) -> np.ndarray:
    """Extract translation vectors from trajectory.

    Args:
        trajectory: (N, 4, 4) array of poses.

    Returns:
        positions: (N, 3) array of xyz positions.
    """
    return trajectory[:, :3, 3]


# ------------------------------------------------------------------ #
#  Scale alignment                                                     #
# ------------------------------------------------------------------ #

def compute_global_scale(gt_positions: np.ndarray,
                         pred_positions: np.ndarray) -> float:
    """Compute single global scale factor (paper protocol, following TartanVO [7]).

    s = sum(||delta_gt||) / sum(||delta_pred||)

    This scales the predicted trajectory so the total path length matches GT.

    Args:
        gt_positions: (N, 3) GT positions.
        pred_positions: (N, 3) predicted positions.

    Returns:
        scale: scalar global scale factor.
    """
    # Compute segment lengths
    gt_deltas = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    pred_deltas = np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)

    gt_path_length = gt_deltas.sum()
    pred_path_length = pred_deltas.sum()

    if pred_path_length < 1e-10:
        print("  WARNING: Predicted path length is near zero. Scale set to 1.0.")
        return 1.0

    scale = gt_path_length / pred_path_length
    return scale


def apply_scale(trajectory: np.ndarray, scale: float) -> np.ndarray:
    """Apply global scale factor to trajectory translations.

    Only scales the translation component — rotations are unaffected.

    Args:
        trajectory: (N, 4, 4) array of poses.
        scale: scalar scale factor.

    Returns:
        scaled: (N, 4, 4) scaled trajectory.
    """
    scaled = trajectory.copy()
    scaled[:, :3, 3] *= scale
    return scaled


# ------------------------------------------------------------------ #
#  ATE metric                                                          #
# ------------------------------------------------------------------ #

def compute_ate(gt_positions: np.ndarray,
                pred_positions: np.ndarray) -> dict:
    """Compute Absolute Trajectory Error (ATE).

    ATE = RMSE of position differences after alignment.

    Args:
        gt_positions: (N, 3) ground truth positions.
        pred_positions: (N, 3) predicted positions (already scaled).

    Returns:
        dict with: ate_rmse, ate_mean, ate_median, ate_std, ate_max, errors
    """
    errors = np.linalg.norm(gt_positions - pred_positions, axis=1)

    return {
        "ate_rmse": np.sqrt(np.mean(errors ** 2)),
        "ate_mean": np.mean(errors),
        "ate_median": np.median(errors),
        "ate_std": np.std(errors),
        "ate_max": np.max(errors),
        "errors": errors,
    }


# ------------------------------------------------------------------ #
#  Rotation error                                                      #
# ------------------------------------------------------------------ #

def compute_rotation_errors(gt_trajectory: np.ndarray,
                            pred_trajectory: np.ndarray) -> np.ndarray:
    """Compute per-frame rotation error in degrees.

    Args:
        gt_trajectory: (N, 4, 4) GT poses.
        pred_trajectory: (N, 4, 4) predicted poses.

    Returns:
        errors_deg: (N,) rotation error per frame in degrees.
    """
    N = gt_trajectory.shape[0]
    errors = np.zeros(N)

    for i in range(N):
        R_gt = gt_trajectory[i, :3, :3]
        R_pred = pred_trajectory[i, :3, :3]
        R_err = R_gt.T @ R_pred
        # Angle from rotation matrix: cos(theta) = (trace(R) - 1) / 2
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        errors[i] = np.degrees(np.arccos(cos_angle))

    return errors


# ------------------------------------------------------------------ #
#  Visualization                                                       #
# ------------------------------------------------------------------ #

def plot_trajectory(gt_positions: np.ndarray,
                    pred_positions: np.ndarray,
                    ate_metrics: dict,
                    rot_errors: np.ndarray,
                    output_path: str,
                    checkpoint_name: str = ""):
    """Plot GT vs predicted trajectory with metrics.

    Creates a 2x2 figure: 3D view, XY top-down, ATE over time, rotation error.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"DINO-VO Evaluation — {checkpoint_name}\n"
                 f"ATE RMSE: {ate_metrics['ate_rmse']:.4f} m | "
                 f"Mean: {ate_metrics['ate_mean']:.4f} m | "
                 f"Median: {ate_metrics['ate_median']:.4f} m",
                 fontsize=13)

    # 1. 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
             'b-', linewidth=1.5, label='Ground Truth')
    ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
             'r-', linewidth=1.5, label='Predicted (scaled)')
    ax1.scatter(*gt_positions[0], c='green', s=80, marker='o', label='Start')
    ax1.scatter(*gt_positions[-1], c='black', s=80, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend(fontsize=9)

    # 2. XY top-down view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1],
             'b-', linewidth=1.5, label='Ground Truth')
    ax2.plot(pred_positions[:, 0], pred_positions[:, 1],
             'r-', linewidth=1.5, label='Predicted (scaled)')
    ax2.scatter(gt_positions[0, 0], gt_positions[0, 1],
                c='green', s=80, marker='o', zorder=5, label='Start')
    ax2.scatter(gt_positions[-1, 0], gt_positions[-1, 1],
                c='black', s=80, marker='x', zorder=5, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 3. ATE over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(ate_metrics['errors'], 'r-', linewidth=1)
    ax3.axhline(y=ate_metrics['ate_rmse'], color='k', linestyle='--',
                linewidth=1, label=f"RMSE = {ate_metrics['ate_rmse']:.4f} m")
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Absolute Trajectory Error')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Rotation error over time
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(rot_errors, 'orange', linewidth=1)
    ax4.axhline(y=np.mean(rot_errors), color='k', linestyle='--',
                linewidth=1, label=f"Mean = {np.mean(rot_errors):.2f} deg")
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Rotation Error (deg)')
    ax4.set_title('Rotation Error')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trajectory plot saved: {output_path}")


# ------------------------------------------------------------------ #
#  Main evaluation                                                     #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(checkpoint_path: str, cfg: dict, max_pairs: int = None):
    """Run full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint .pth file.
        cfg: Configuration dictionary.
        max_pairs: If set, only evaluate first N pairs (for quick testing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = cfg["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Load dataset (sequential, no filtering) ---
    print("\nLoading evaluation dataset...")
    dataset = load_eval_dataset(
        sequence_path=cfg["data"]["sequence_path"],
        skip_frames=cfg["data"]["skip_frames"],
        target_h=cfg["data"]["target_h"],
        target_w=cfg["data"]["target_w"],
    )
    print(f"  Total pairs: {len(dataset)}")

    if max_pairs:
        n_pairs = min(max_pairs, len(dataset))
    else:
        n_pairs = len(dataset)
    print(f"  Evaluating: {n_pairs} pairs")

    # --- Load model ---
    print("\nLoading model...")
    model = DinoVO(
        top_k=cfg["model"]["top_k"],
        descriptor_dim=cfg["model"]["descriptor_dim"],
        matching_layers=cfg["model"]["matching_layers"],
    )
    model.load_dino(device)
    model = model.to(device)
    model.eval()

    # Load checkpoint weights
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_step = ckpt.get("global_step", "?")
    print(f"  Checkpoint from epoch {ckpt_epoch}, step {ckpt_step}")

    # --- Inference loop ---
    print(f"\nRunning inference on {n_pairs} pairs...")
    pred_relatives = []   # list of (4, 4) numpy arrays
    gt_relatives = []     # list of (4, 4) numpy arrays
    match_counts = []

    t_start = time.time()

    for idx in tqdm(range(n_pairs), desc="Inference"):
        sample = dataset[idx]

        image1 = sample["image1"].unsqueeze(0).to(device)      # (1, 3, H, W)
        image2 = sample["image2"].unsqueeze(0).to(device)      # (1, 3, H, W)
        K = sample["intrinsics"].unsqueeze(0).to(device)        # (1, 3, 3)
        T_gt = sample["relative_pose"].numpy()                  # (4, 4)

        # Forward pass
        out = model(image1, image2, K, return_all_assignments=False)

        R_pred = out["R"][0].cpu().numpy()   # (3, 3)
        t_pred = out["t"][0].cpu().numpy()   # (3,)
        n_matches = out["matches"][0].shape[0]

        # Build relative pose matrix
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

    # --- Accumulate trajectories ---
    print("\nAccumulating trajectories...")
    pred_traj = accumulate_trajectory(pred_relatives)
    gt_traj = accumulate_trajectory(gt_relatives)

    pred_positions = extract_positions(pred_traj)
    gt_positions = extract_positions(gt_traj)

    gt_path_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
    pred_path_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1))
    print(f"  GT path length: {gt_path_length:.3f} m")
    print(f"  Pred path length (unscaled): {pred_path_length:.3f} m")

    # --- Scale alignment (paper protocol: single global scale) ---
    scale = compute_global_scale(gt_positions, pred_positions)
    print(f"  Global scale factor: {scale:.4f}")

    pred_traj_scaled = apply_scale(pred_traj, scale)
    pred_positions_scaled = extract_positions(pred_traj_scaled)

    scaled_path_length = np.sum(np.linalg.norm(
        np.diff(pred_positions_scaled, axis=0), axis=1))
    print(f"  Pred path length (scaled): {scaled_path_length:.3f} m")

    # --- Compute ATE ---
    ate = compute_ate(gt_positions, pred_positions_scaled)
    print(f"\n{'='*50}")
    print(f"  ATE Results (checkpoint: epoch {ckpt_epoch})")
    print(f"{'='*50}")
    print(f"  RMSE:   {ate['ate_rmse']:.4f} m")
    print(f"  Mean:   {ate['ate_mean']:.4f} m")
    print(f"  Median: {ate['ate_median']:.4f} m")
    print(f"  Std:    {ate['ate_std']:.4f} m")
    print(f"  Max:    {ate['ate_max']:.4f} m")

    # --- Rotation error ---
    rot_errors = compute_rotation_errors(gt_traj, pred_traj_scaled)
    print(f"\n  Rotation Error:")
    print(f"  Mean:   {np.mean(rot_errors):.2f} deg")
    print(f"  Median: {np.median(rot_errors):.2f} deg")
    print(f"  Max:    {np.max(rot_errors):.2f} deg")

    # --- Plot ---
    ckpt_name = os.path.basename(checkpoint_path).replace(".pth", "")
    plot_path = os.path.join(output_dir, f"trajectory_{ckpt_name}.png")
    plot_trajectory(
        gt_positions, pred_positions_scaled,
        ate, rot_errors,
        plot_path,
        checkpoint_name=f"epoch {ckpt_epoch} ({n_pairs} pairs)",
    )

    # --- Save results to text file ---
    results_path = os.path.join(output_dir, f"eval_{ckpt_name}.txt")
    with open(results_path, "w") as f:
        f.write(f"DINO-VO Evaluation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {ckpt_epoch}\n")
        f.write(f"Sequence: {cfg['data']['sequence_path']}\n")
        f.write(f"Pairs evaluated: {n_pairs}\n")
        f.write(f"Skip frames: {cfg['data']['skip_frames']}\n")
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


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluate DINO-VO")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pth file (e.g., checkpoints/epoch_04.pth)"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Evaluate only first N pairs (for quick testing)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(args.checkpoint, cfg, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
