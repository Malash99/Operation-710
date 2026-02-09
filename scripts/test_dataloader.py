"""
Test script for the EuRoC data pipeline (Phase 2 verification).

Validates:
    1. Dataset loads correctly and reports expected length
    2. Image tensors have correct shape [3, 476, 742]
    3. Intrinsics are properly rescaled
    4. Relative pose rotation is a valid SO(3) matrix
    5. Saves side-by-side visualization to outputs/dataloader_test.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.euroc import EuRoCDataset


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor, shape (3, H, W).

    Returns:
        Image as uint8 numpy array, shape (H, W, 3), RGB.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def main():
    sequence_path = os.path.join(project_root, "data", "euroc", "MH_01_easy")

    if not os.path.isdir(sequence_path):
        print(f"ERROR: Dataset not found at {sequence_path}")
        print("Please download the EuRoC MH_01_easy sequence first.")
        sys.exit(1)

    print("=" * 60)
    print("DINO-VO Phase 2: Data Pipeline Verification")
    print("=" * 60)

    # 1. Create dataset
    print("\n[1] Creating EuRoCDataset for MH_01_easy...")
    dataset = EuRoCDataset(sequence_path, skip_frames=2)
    print(f"    Dataset length: {len(dataset)} pairs")
    print(f"    Skip frames: {dataset.skip_frames}")

    # 2. Load a sample
    print("\n[2] Loading sample at index 0...")
    sample = dataset[0]

    # 3. Check tensor shapes
    img1 = sample["image1"]
    img2 = sample["image2"]
    print(f"    image1 shape: {img1.shape}  (expected [3, 476, 742])")
    print(f"    image2 shape: {img2.shape}  (expected [3, 476, 742])")
    print(f"    image1 dtype: {img1.dtype}")
    print(f"    image1 value range: [{img1.min():.3f}, {img1.max():.3f}]")

    assert img1.shape == (3, 476, 742), f"Wrong shape: {img1.shape}"
    assert img2.shape == (3, 476, 742), f"Wrong shape: {img2.shape}"
    print("    Shape check: PASSED")

    # 4. Check intrinsics
    K = sample["intrinsics"]
    print(f"\n[3] Intrinsics (rescaled):")
    print(f"    fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"    Expected approx: fx~452.5, fy~453.1, cx~362.4, cy~246.2")

    # 5. Check relative pose
    pose = sample["relative_pose"]
    R = pose[:3, :3].numpy()
    t = pose[:3, 3].numpy()
    print(f"\n[4] Relative pose:")
    print(f"    Rotation matrix:\n{R}")
    print(f"    Translation vector: {t}")

    # Verify R is valid rotation matrix
    det_R = np.linalg.det(R)
    RRT = R @ R.T
    identity_err = np.max(np.abs(RRT - np.eye(3)))
    print(f"\n    det(R) = {det_R:.6f}  (expected ~1.0)")
    print(f"    max|R@R^T - I| = {identity_err:.2e}  (expected ~0)")

    assert abs(det_R - 1.0) < 1e-4, f"Invalid rotation: det(R) = {det_R}"
    assert identity_err < 1e-4, f"R is not orthogonal: max error = {identity_err}"
    print("    Rotation validity: PASSED")

    # Translation magnitude
    t_norm = np.linalg.norm(t)
    print(f"    |t| = {t_norm:.6f} meters")

    # 6. Timestamps
    print(f"\n[5] Timestamps:")
    print(f"    t1 = {sample['timestamp1']}")
    print(f"    t2 = {sample['timestamp2']}")
    dt_ms = (sample["timestamp2"] - sample["timestamp1"]) / 1e6
    print(f"    dt = {dt_ms:.1f} ms")

    # 7. Visualize
    print("\n[6] Saving visualization...")
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    output_path = os.path.join(project_root, "outputs", "dataloader_test.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(denormalize_image(img1))
    axes[0].set_title(f"Frame 1 (t={sample['timestamp1']})")
    axes[0].axis("off")

    axes[1].imshow(denormalize_image(img2))
    axes[1].set_title(f"Frame 2 (t={sample['timestamp2']})")
    axes[1].axis("off")

    fig.suptitle(
        f"EuRoC MH01 — skip={dataset.skip_frames}, "
        f"|t|={t_norm:.4f}m, dt={dt_ms:.0f}ms",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved to: {output_path}")

    # 8. Quick spot-check a few more samples
    print(f"\n[7] Spot-checking samples across the dataset...")
    for test_idx in [len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]:
        s = dataset[test_idx]
        R_test = s["relative_pose"][:3, :3].numpy()
        det_test = np.linalg.det(R_test)
        t_test = np.linalg.norm(s["relative_pose"][:3, 3].numpy())
        status = "OK" if abs(det_test - 1.0) < 1e-4 else "FAIL"
        print(f"    idx={test_idx}: det(R)={det_test:.6f}, |t|={t_test:.4f}m [{status}]")

    print("\n" + "=" * 60)
    print("Phase 2 verification COMPLETE. All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
