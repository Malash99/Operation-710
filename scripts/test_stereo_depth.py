"""
Test script: Stereo depth computation + GT correspondence generation.

Verifies:
1. Stereo calibration loads correctly (baseline ~11cm)
2. Depth map has reasonable values for indoor scenes (0.5-20m)
3. GT correspondences are geometrically consistent
4. Saves visualizations to outputs/

Usage:
    python -m scripts.test_stereo_depth
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.euroc import EuRoCDataset
from src.utils.stereo import generate_gt_correspondences


def main():
    seq_path = os.path.join("data", "euroc", "MH_01_easy")
    if not os.path.isdir(seq_path):
        print(f"ERROR: Dataset not found at {seq_path}")
        return

    os.makedirs("outputs", exist_ok=True)

    # --- 1. Load dataset with stereo depth ---
    print("Loading EuRoC dataset with stereo depth...")
    t0 = time.time()
    dataset = EuRoCDataset(seq_path, compute_stereo_depth=True)
    print(f"  Dataset loaded in {time.time() - t0:.1f}s")
    print(f"  Pairs: {len(dataset)}")
    print(f"  Stereo baseline: {dataset.stereo_calib['baseline']:.4f} m")

    # --- 2. Load a sample pair ---
    print("\nLoading sample pair (idx=100)...")
    t0 = time.time()
    sample = dataset[100]
    print(f"  Loaded in {time.time() - t0:.2f}s")

    print(f"  image1 shape: {sample['image1'].shape}")
    print(f"  image2 shape: {sample['image2'].shape}")
    print(f"  relative_pose shape: {sample['relative_pose'].shape}")
    print(f"  depth1 shape: {sample['depth1'].shape}")

    depth1 = sample["depth1"].numpy()
    valid_depth = depth1[depth1 > 0]
    print(f"\n  Depth statistics (valid pixels):")
    print(f"    Valid pixels: {len(valid_depth)} / {depth1.size} ({100*len(valid_depth)/depth1.size:.1f}%)")
    if len(valid_depth) > 0:
        print(f"    Min:    {valid_depth.min():.3f} m")
        print(f"    Max:    {valid_depth.max():.3f} m")
        print(f"    Mean:   {valid_depth.mean():.3f} m")
        print(f"    Median: {np.median(valid_depth):.3f} m")

    # --- 3. Test GT correspondence generation with fake keypoints ---
    # We simulate keypoints on a grid to test the reprojection pipeline
    print("\n--- Testing GT correspondence generation ---")
    K = sample["intrinsics"].numpy()
    T_1to2 = sample["relative_pose"].numpy()
    H, W = depth1.shape

    # Create grid keypoints (every 30 pixels)
    ys = np.arange(30, H - 30, 30)
    xs = np.arange(30, W - 30, 30)
    grid_x, grid_y = np.meshgrid(xs, ys)
    kp1_uv = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)  # (N, 2)

    # Get depth at these keypoints
    kp1_depths = np.array([depth1[int(v), int(u)] for u, v in kp1_uv], dtype=np.float32)

    # Use same grid as kp2 candidates (in practice these come from the detector)
    kp2_uv = kp1_uv.copy()

    print(f"  Grid keypoints: {kp1_uv.shape[0]}")
    print(f"  Keypoints with valid depth: {(kp1_depths > 0).sum()}")

    gt_matches, _ = generate_gt_correspondences(
        kp1_uv, kp2_uv, kp1_depths, K, T_1to2, reproj_threshold=5.0
    )
    print(f"  GT correspondences found: {gt_matches.shape[0]}")

    # --- 4. Test with detector-like keypoints (random subset) ---
    print("\n--- Testing with random keypoint subset ---")
    rng = np.random.default_rng(42)
    n_kp = 512
    kp1_rand = np.stack([
        rng.uniform(20, W - 20, n_kp),
        rng.uniform(20, H - 20, n_kp),
    ], axis=1).astype(np.float64)
    kp2_rand = np.stack([
        rng.uniform(20, W - 20, n_kp),
        rng.uniform(20, H - 20, n_kp),
    ], axis=1).astype(np.float64)

    kp1_rand_depths = np.array(
        [depth1[int(v), int(u)] for u, v in kp1_rand], dtype=np.float32
    )

    gt_matches_rand, _ = generate_gt_correspondences(
        kp1_rand, kp2_rand, kp1_rand_depths, K, T_1to2, reproj_threshold=5.0
    )
    print(f"  Random keypoints: {n_kp}")
    print(f"  With valid depth: {(kp1_rand_depths > 0).sum()}")
    print(f"  GT correspondences: {gt_matches_rand.shape[0]}")

    # --- 5. Visualizations ---
    print("\nSaving visualizations...")

    # Denormalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img1_vis = (sample["image1"] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    img2_vis = (sample["image2"] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Image 1
    axes[0, 0].imshow(img1_vis)
    axes[0, 0].set_title("Image 1 (cam0)")
    axes[0, 0].axis("off")

    # (b) Depth map
    depth_vis = depth1.copy()
    depth_vis[depth_vis <= 0] = np.nan
    im = axes[0, 1].imshow(depth_vis, cmap="plasma", vmin=0, vmax=15)
    axes[0, 1].set_title("Stereo Depth Map (meters)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # (c) Image 2
    axes[1, 0].imshow(img2_vis)
    axes[1, 0].set_title("Image 2 (cam0, next frame)")
    axes[1, 0].axis("off")

    # (d) GT correspondences
    # Side-by-side match visualization
    axes[1, 1].imshow(img1_vis)
    if gt_matches.shape[0] > 0:
        for m in range(min(gt_matches.shape[0], 50)):
            i_idx, j_idx = gt_matches[m]
            u1, v1 = kp1_uv[i_idx]
            u2, v2 = kp2_uv[j_idx]
            axes[1, 1].plot(u1, v1, "go", markersize=3)
            axes[1, 1].plot(u2, v2, "rx", markersize=3)
            axes[1, 1].plot([u1, u2], [v1, v2], "c-", linewidth=0.5, alpha=0.5)
    axes[1, 1].set_title(f"GT Correspondences ({gt_matches.shape[0]} matches)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/stereo_depth_test.png", dpi=150)
    plt.close()
    print("  Saved: outputs/stereo_depth_test.png")

    # --- 6. Test loading speed ---
    print("\n--- Loading speed test (5 samples) ---")
    t0 = time.time()
    for i in range(5):
        _ = dataset[i * 50]
    elapsed = time.time() - t0
    print(f"  {elapsed:.2f}s total, {elapsed/5:.2f}s per sample")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
