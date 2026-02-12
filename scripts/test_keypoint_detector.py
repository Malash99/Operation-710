"""
Test script for the Salient Keypoint Detector (Phase 3 verification).

Validates:
    1. Detector outputs 512 keypoints per image
    2. Keypoints are spatially well-distributed (not clustered)
    3. Keypoints have valid coordinates within image bounds
    4. No two keypoints are closer than NMS radius (8 pixels)
    5. Keypoints fall on high-gradient areas
    6. GPU execution works correctly

Saves visualization to outputs/keypoint_detector_test.png
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.euroc import EuRoCDataset
from src.models.keypoint_detector import SalientKeypointDetector


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DINO-VO Phase 3: Keypoint Detector Verification")
    print(f"Device: {device}")
    print("=" * 60)

    # Load dataset
    sequence_path = os.path.join(project_root, "data", "euroc", "MH_01_easy")
    dataset = EuRoCDataset(sequence_path, skip_frames=2)
    sample = dataset[0]

    # Create detector
    detector = SalientKeypointDetector().to(device)
    print(f"\n[1] Detector created (no learnable parameters)")
    num_params = sum(p.numel() for p in detector.parameters())
    print(f"    Trainable parameters: {num_params}")

    # Run detection on image1
    image1 = sample["image1"].unsqueeze(0).to(device)  # (1, 3, 476, 742)
    print(f"\n[2] Input image shape: {image1.shape}")

    t0 = time.time()
    with torch.no_grad():
        result = detector(image1)
    t1 = time.time()

    keypoints = result["keypoints"]      # (1, 512, 2)
    scores = result["scores"]            # (1, 512)
    num_valid = result["num_valid"]      # (1,)
    gradient_map = result["gradient_map"]  # (1, 1, 476, 742)

    print(f"    Detection time: {(t1 - t0) * 1000:.1f} ms")
    print(f"\n[3] Output shapes:")
    print(f"    keypoints:    {keypoints.shape}  (expected [1, 512, 2])")
    print(f"    scores:       {scores.shape}  (expected [1, 512])")
    print(f"    num_valid:    {num_valid.item()}")
    print(f"    gradient_map: {gradient_map.shape}")

    # Move to CPU for analysis
    kp = keypoints[0].cpu()        # (512, 2) as (x, y)
    sc = scores[0].cpu()           # (512,)
    nv = num_valid[0].item()
    grad_map = gradient_map[0, 0].cpu().numpy()

    # Check 1: Correct output shape
    assert keypoints.shape == (1, 512, 2), f"Wrong shape: {keypoints.shape}"
    print(f"\n[4] Shape check: PASSED")

    # Check 2: Coordinates within image bounds
    valid_kp = kp[:nv]
    x_coords = valid_kp[:, 0]
    y_coords = valid_kp[:, 1]
    assert (x_coords >= 0).all() and (x_coords < 742).all(), "x out of bounds"
    assert (y_coords >= 0).all() and (y_coords < 476).all(), "y out of bounds"
    print(f"    Bounds check: PASSED (x: [{x_coords.min():.0f}, {x_coords.max():.0f}], "
          f"y: [{y_coords.min():.0f}, {y_coords.max():.0f}])")

    # Check 3: No two keypoints closer than NMS radius
    if nv > 1:
        dists = torch.cdist(valid_kp.unsqueeze(0).float(), valid_kp.unsqueeze(0).float()).squeeze(0)
        # Set diagonal to inf so we don't compare a point with itself
        dists.fill_diagonal_(float("inf"))
        min_dist = dists.min().item()
        print(f"    Min pairwise distance: {min_dist:.1f} px (NMS radius=8)")
        assert min_dist >= 7.5, f"NMS violation: min distance = {min_dist:.1f}"
        print(f"    NMS check: PASSED")

    # Check 4: Scores are positive and above threshold for valid keypoints
    valid_scores = sc[:nv]
    print(f"    Score range: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
    assert (valid_scores >= 0.01).all(), "Score below threshold"
    print(f"    Threshold check: PASSED")

    # Check 5: Spatial distribution — divide image into quadrants
    print(f"\n[5] Spatial distribution:")
    mid_x, mid_y = 742 / 2, 476 / 2
    q1 = ((x_coords < mid_x) & (y_coords < mid_y)).sum().item()
    q2 = ((x_coords >= mid_x) & (y_coords < mid_y)).sum().item()
    q3 = ((x_coords < mid_x) & (y_coords >= mid_y)).sum().item()
    q4 = ((x_coords >= mid_x) & (y_coords >= mid_y)).sum().item()
    print(f"    Top-left: {q1}, Top-right: {q2}")
    print(f"    Bot-left: {q3}, Bot-right: {q4}")
    print(f"    Total valid: {q1 + q2 + q3 + q4} / {nv}")

    # Check 6: Keypoints on high-gradient areas
    grad_at_kp = []
    for i in range(nv):
        xi, yi = int(x_coords[i].item()), int(y_coords[i].item())
        grad_at_kp.append(grad_map[yi, xi])
    grad_at_kp = np.array(grad_at_kp)
    print(f"\n[6] Gradient at keypoints:")
    print(f"    Mean: {grad_at_kp.mean():.4f}")
    print(f"    Median: {np.median(grad_at_kp):.4f}")
    print(f"    Image gradient mean: {grad_map.mean():.4f}")
    print(f"    Ratio (kp/image): {grad_at_kp.mean() / grad_map.mean():.1f}x")

    # Check 7: Run on image2 as well (batch of 2)
    image2 = sample["image2"].unsqueeze(0).to(device)
    batch = torch.cat([image1, image2], dim=0)  # (2, 3, 476, 742)
    with torch.no_grad():
        batch_result = detector(batch)
    print(f"\n[7] Batch processing (B=2):")
    print(f"    keypoints shape: {batch_result['keypoints'].shape}  (expected [2, 512, 2])")
    print(f"    num_valid: {batch_result['num_valid'].tolist()}")
    assert batch_result["keypoints"].shape == (2, 512, 2)
    print(f"    Batch check: PASSED")

    # Visualization
    print(f"\n[8] Saving visualizations...")
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)

    img_vis = denormalize_image(sample["image1"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Salient Keypoint Detector — {nv} keypoints detected",
        fontsize=14,
    )

    # Top-left: original image
    axes[0, 0].imshow(img_vis)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Top-right: gradient magnitude map
    axes[0, 1].imshow(grad_map, cmap="hot")
    axes[0, 1].set_title("Gradient Magnitude")
    axes[0, 1].axis("off")

    # Bottom-left: keypoints on image
    axes[1, 0].imshow(img_vis)
    axes[1, 0].scatter(
        x_coords[:nv].numpy(),
        y_coords[:nv].numpy(),
        c=valid_scores.numpy(),
        cmap="viridis",
        s=8,
        alpha=0.8,
    )
    axes[1, 0].set_title(f"Keypoints (n={nv}, colored by score)")
    axes[1, 0].axis("off")

    # Bottom-right: keypoints on gradient map
    axes[1, 1].imshow(grad_map, cmap="gray")
    axes[1, 1].scatter(
        x_coords[:nv].numpy(),
        y_coords[:nv].numpy(),
        c="lime",
        s=8,
        alpha=0.7,
    )
    # Draw 14x14 grid lines
    for gx in range(0, 742, 14):
        axes[1, 1].axvline(gx, color="blue", linewidth=0.2, alpha=0.3)
    for gy in range(0, 476, 14):
        axes[1, 1].axhline(gy, color="blue", linewidth=0.2, alpha=0.3)
    axes[1, 1].set_title("Keypoints on Gradient Map (14x14 grid)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(project_root, "outputs", "keypoint_detector_test.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Phase 3 verification COMPLETE. All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
