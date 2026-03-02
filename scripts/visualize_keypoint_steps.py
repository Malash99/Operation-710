"""
Generate step-by-step visualizations of the Salient Keypoint Detector pipeline.

Produces individual images for each stage to be embedded in the README:
  1. Original grayscale image
  2. Gaussian-smoothed image
  3. Sobel gradient magnitude
  4. Grid-based MaxPool candidates (with 14x14 grid overlay)
  5. After NMS (candidates that survive suppression)
  6. Final 512 keypoints (after threshold + top-K)

All images are saved to outputs/keypoint_steps/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.datasets.euroc import EuRoCDataset
from src.models.keypoint_detector import SalientKeypointDetector


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load one real EuRoC image pair ---
    dataset = EuRoCDataset(
        sequence_path="data/euroc/MH_01_easy",
        skip_frames=2,
    )
    sample = dataset[100]  # Pick a frame with interesting structure
    image = sample["image1"].unsqueeze(0).to(device)  # (1, 3, 476, 742)
    B, C, H, W = image.shape
    print(f"Image shape: {image.shape}")

    # --- Build detector and extract intermediates step-by-step ---
    det = SalientKeypointDetector().to(device)

    # Step 1: Recover grayscale
    gray = det._to_grayscale(image)  # (1, 1, H, W)

    # Step 2: Gaussian smoothing
    smoothed = F.conv2d(gray, det.gauss_kernel, padding=det.gauss_pad)

    # Step 3: Sobel gradient magnitude
    gx = F.conv2d(smoothed, det.sobel_x, padding=1)
    gy = F.conv2d(smoothed, det.sobel_y, padding=1)
    gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # Step 4: Grid MaxPool
    pooled_values, pooled_indices = F.max_pool2d(
        gradient_mag,
        kernel_size=det.pool_kernel_size,
        stride=det.pool_kernel_size,
        return_indices=True,
    )
    candidate_coords = det._indices_to_coordinates(pooled_indices, W)  # (1, N, 2)
    candidate_scores = pooled_values.view(B, -1)  # (1, N)

    # Step 5: NMS
    nms_results = det._nms(candidate_coords, candidate_scores, det.nms_radius)
    coords_nms, scores_nms = nms_results[0]  # first batch element

    # Step 6: Threshold + Top-K
    kp_final, sc_final, num_valid = det._threshold_and_topk(coords_nms, scores_nms)

    # --- Convert tensors to numpy for plotting ---
    gray_np = gray[0, 0].cpu().numpy()
    smoothed_np = smoothed[0, 0].cpu().numpy()
    grad_np = gradient_mag[0, 0].cpu().numpy()
    cand_xy = candidate_coords[0].cpu().numpy()  # (N, 2) as (x, y)
    cand_sc = candidate_scores[0].cpu().numpy()
    nms_xy = coords_nms.cpu().numpy()
    nms_sc = scores_nms.cpu().numpy()
    final_xy = kp_final[:num_valid].cpu().numpy()
    final_sc = sc_final[:num_valid].cpu().numpy()

    # --- Output directory ---
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "keypoint_steps"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ========== Step 1: Original Grayscale ==========
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Step 1: Original Grayscale (recovered from normalized RGB)", fontsize=13)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step1_grayscale.png"), dpi=140)
    plt.close(fig)
    print("  Saved step1_grayscale.png")

    # ========== Step 2: Gaussian Smoothed ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Before: Original Grayscale", fontsize=12)
    axes[1].imshow(smoothed_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("After: Gaussian Smoothed (k=5, σ=2.0)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Step 2: Gaussian Smoothing", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step2_gaussian.png"), dpi=140)
    plt.close(fig)
    print("  Saved step2_gaussian.png")

    # ========== Step 3: Gradient Magnitude ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(smoothed_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Smoothed Image", fontsize=12)
    im = axes[1].imshow(grad_np, cmap="hot")
    axes[1].set_title("Sobel Gradient Magnitude", fontsize=12)
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Step 3: Sobel Gradient Magnitude — G = √(Gx² + Gy²)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step3_gradient.png"), dpi=140)
    plt.close(fig)
    print("  Saved step3_gradient.png")

    # ========== Step 4: Grid MaxPool Candidates ==========
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(grad_np, cmap="gray", alpha=0.6)
    # Draw 14x14 grid
    for gx_line in range(0, W + 1, 14):
        ax.axvline(gx_line, color="cyan", linewidth=0.3, alpha=0.5)
    for gy_line in range(0, H + 1, 14):
        ax.axhline(gy_line, color="cyan", linewidth=0.3, alpha=0.5)
    # Plot candidates
    ax.scatter(
        cand_xy[:, 0], cand_xy[:, 1],
        c=cand_sc, cmap="hot", s=8, edgecolors="none", zorder=5,
    )
    n_cand = cand_xy.shape[0]
    ax.set_title(
        f"Step 4: Grid MaxPool Candidates — {n_cand} points (one per 14×14 patch)",
        fontsize=13,
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    grid_patch = mpatches.Patch(color="cyan", alpha=0.5, label="14×14 DINOv2 patch grid")
    ax.legend(handles=[grid_patch], loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step4_maxpool.png"), dpi=140)
    plt.close(fig)
    print(f"  Saved step4_maxpool.png  ({n_cand} candidates)")

    # ========== Step 5: After NMS ==========
    # Show which candidates survived vs. suppressed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: before NMS
    axes[0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter(
        cand_xy[:, 0], cand_xy[:, 1],
        c="lime", s=6, edgecolors="none", alpha=0.7,
    )
    axes[0].set_title(f"Before NMS: {n_cand} candidates", fontsize=12)
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)

    # Right: after NMS
    n_nms = nms_xy.shape[0]
    axes[1].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[1].scatter(
        nms_xy[:, 0], nms_xy[:, 1],
        c="lime", s=8, edgecolors="none", alpha=0.8,
    )
    axes[1].set_title(f"After NMS (r=8): {n_nms} keypoints", fontsize=12)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Step 5: Non-Maximum Suppression — {n_cand} → {n_nms} ({n_cand - n_nms} suppressed)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step5_nms.png"), dpi=140)
    plt.close(fig)
    print(f"  Saved step5_nms.png  ({n_cand} -> {n_nms})")

    # ========== Step 6: Final 512 Keypoints ==========
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    scatter = ax.scatter(
        final_xy[:, 0], final_xy[:, 1],
        c=final_sc, cmap="jet", s=12, edgecolors="black", linewidths=0.3,
        zorder=5,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02, label="Gradient score")
    ax.set_title(
        f"Step 6: Final Keypoints — top {num_valid} (threshold=0.01, K=512)",
        fontsize=13,
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step6_final.png"), dpi=140)
    plt.close(fig)
    print(f"  Saved step6_final.png  ({num_valid} keypoints)")

    # ========== Combined overview ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("1. Grayscale", fontsize=11)

    axes[0, 1].imshow(smoothed_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("2. Gaussian Smoothed", fontsize=11)

    axes[0, 2].imshow(grad_np, cmap="hot")
    axes[0, 2].set_title("3. Gradient Magnitude", fontsize=11)

    axes[1, 0].imshow(grad_np, cmap="gray", alpha=0.6)
    for gx_line in range(0, W + 1, 14):
        axes[1, 0].axvline(gx_line, color="cyan", linewidth=0.2, alpha=0.4)
    for gy_line in range(0, H + 1, 14):
        axes[1, 0].axhline(gy_line, color="cyan", linewidth=0.2, alpha=0.4)
    axes[1, 0].scatter(cand_xy[:, 0], cand_xy[:, 1], c="lime", s=4, edgecolors="none")
    axes[1, 0].set_title(f"4. MaxPool Candidates ({n_cand})", fontsize=11)

    axes[1, 1].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].scatter(nms_xy[:, 0], nms_xy[:, 1], c="lime", s=5, edgecolors="none")
    axes[1, 1].set_title(f"5. After NMS ({n_nms})", fontsize=11)

    axes[1, 2].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].scatter(
        final_xy[:, 0], final_xy[:, 1],
        c=final_sc, cmap="jet", s=8, edgecolors="black", linewidths=0.2,
    )
    axes[1, 2].set_title(f"6. Final Keypoints ({num_valid})", fontsize=11)

    for ax in axes.flat:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Salient Keypoint Detector — Full Pipeline (Paper Section III-A)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pipeline_overview.png"), dpi=150)
    plt.close(fig)
    print("  Saved pipeline_overview.png")

    print(f"\nAll visualizations saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
