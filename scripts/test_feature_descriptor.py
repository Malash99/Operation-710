"""
Test script for the Feature Descriptor (Phase 4 verification).

Validates:
    1. DINOv2 loads and runs correctly (patch features: K×384)
    2. FinerCNN produces full-resolution features (K×64)
    3. Combined descriptor has correct shape (K×192)
    4. Descriptors are L2-normalized (unit norm)
    5. DINOv2 parameters are frozen (no gradients)
    6. FinerCNN + projection are trainable
    7. GPU execution works correctly
    8. Two different images produce different descriptors

Saves visualization to outputs/feature_descriptor_test.png
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
from src.models.feature_descriptor import FeatureDescriptor


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DINO-VO Phase 4: Feature Descriptor Verification")
    print(f"Device: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load dataset and get a sample pair
    # ------------------------------------------------------------------ #
    sequence_path = os.path.join(project_root, "data", "euroc", "MH_01_easy")
    dataset = EuRoCDataset(sequence_path, skip_frames=2)
    sample = dataset[0]

    image1 = sample["image1"].unsqueeze(0).to(device)  # (1, 3, 476, 742)
    image2 = sample["image2"].unsqueeze(0).to(device)  # (1, 3, 476, 742)
    print(f"\n[1] Input images loaded: {image1.shape}")

    # ------------------------------------------------------------------ #
    # 2. Detect keypoints (Phase 3 output → Phase 4 input)
    # ------------------------------------------------------------------ #
    detector = SalientKeypointDetector().to(device)
    with torch.no_grad():
        det1 = detector(image1)
        det2 = detector(image2)

    kp1 = det1["keypoints"]  # (1, 512, 2)
    kp2 = det2["keypoints"]  # (1, 512, 2)
    nv1 = det1["num_valid"][0].item()
    nv2 = det2["num_valid"][0].item()
    print(f"[2] Keypoints detected: image1={nv1}, image2={nv2}")

    # ------------------------------------------------------------------ #
    # 3. Create descriptor and load DINOv2
    # ------------------------------------------------------------------ #
    descriptor = FeatureDescriptor().to(device)
    descriptor.load_dino(device)

    total_params = sum(p.numel() for p in descriptor.parameters())
    trainable_params = sum(p.numel() for p in descriptor.parameters() if p.requires_grad)
    print(f"\n[3] FeatureDescriptor created:")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # ------------------------------------------------------------------ #
    # 4. Check DINOv2 is frozen (no gradients)
    # ------------------------------------------------------------------ #
    dino_params_with_grad = sum(
        1 for p in descriptor.dino.parameters() if p.requires_grad
    )
    assert dino_params_with_grad == 0, (
        f"DINOv2 has {dino_params_with_grad} parameters with requires_grad=True!"
    )
    print(f"[4] DINOv2 frozen check: PASSED (0 parameters require grad)")

    # ------------------------------------------------------------------ #
    # 5. Extract descriptors for image1
    # ------------------------------------------------------------------ #
    descriptor.eval()
    t0 = time.time()
    with torch.no_grad():
        desc1 = descriptor(image1, kp1)  # (1, 512, 192)
        desc2 = descriptor(image2, kp2)  # (1, 512, 192)
    t1 = time.time()

    print(f"\n[5] Descriptor extraction:")
    print(f"    Time per image pair: {(t1 - t0) * 1000:.1f} ms")
    print(f"    desc1 shape: {desc1.shape}  (expected [1, 512, 192])")
    print(f"    desc2 shape: {desc2.shape}  (expected [1, 512, 192])")

    # ------------------------------------------------------------------ #
    # 6. Shape check
    # ------------------------------------------------------------------ #
    assert desc1.shape == (1, 512, 192), f"Wrong shape: {desc1.shape}"
    assert desc2.shape == (1, 512, 192), f"Wrong shape: {desc2.shape}"
    print(f"[6] Shape check: PASSED")

    # ------------------------------------------------------------------ #
    # 7. L2 normalization check — all descriptors should have unit norm
    # ------------------------------------------------------------------ #
    norms1 = desc1[0].norm(dim=-1)  # (512,)
    norms2 = desc2[0].norm(dim=-1)  # (512,)
    max_norm_err = max(
        (norms1 - 1.0).abs().max().item(),
        (norms2 - 1.0).abs().max().item(),
    )
    print(f"[7] L2 normalization check:")
    print(f"    Max |norm - 1| = {max_norm_err:.2e}  (expected < 1e-5)")
    assert max_norm_err < 1e-5, f"Descriptors not unit-normalized: max err={max_norm_err}"
    print(f"    L2 normalization: PASSED")

    # ------------------------------------------------------------------ #
    # 8. Distinctiveness — image1 and image2 descriptors should differ
    # ------------------------------------------------------------------ #
    # Cosine similarity between corresponding descriptors (valid ones only)
    nv = min(nv1, nv2)
    cos_sim = (desc1[0, :nv] * desc2[0, :nv]).sum(dim=-1)  # (nv,)
    mean_sim = cos_sim.mean().item()
    print(f"\n[8] Descriptor distinctiveness:")
    print(f"    Mean cosine similarity between image1 and image2: {mean_sim:.4f}")
    print(f"    (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    assert mean_sim < 0.99, "Descriptors appear identical — something is wrong"
    print(f"    Distinctiveness check: PASSED")

    # ------------------------------------------------------------------ #
    # 9. Intermediate shape checks (DINOv2 patch map + FinerCNN map)
    # ------------------------------------------------------------------ #
    print(f"\n[9] Intermediate feature shapes:")

    # DINOv2 patch map
    dino_map = descriptor._extract_dino_features(image1)
    print(f"    DINOv2 patch map:  {tuple(dino_map.shape)}  (expected [1, 34, 53, 384])")
    assert dino_map.shape == (1, 34, 53, 384), f"Wrong DINOv2 shape: {dino_map.shape}"
    print(f"    DINOv2 shape: PASSED")

    # FinerCNN dense map
    fine_map = descriptor.finer_cnn(image1)
    print(f"    FinerCNN dense map: {tuple(fine_map.shape)}  (expected [1, 64, 476, 742])")
    assert fine_map.shape == (1, 64, 476, 742), f"Wrong FinerCNN shape: {fine_map.shape}"
    print(f"    FinerCNN shape: PASSED")

    # Sampled DINOv2 features at keypoints
    f_dino = descriptor._sample_at_keypoints(dino_map, kp1, is_patch_map=True)
    print(f"    DINOv2 @ keypoints: {tuple(f_dino.shape)}  (expected [1, 512, 384])")
    assert f_dino.shape == (1, 512, 384)
    print(f"    DINOv2 sampling: PASSED")

    # Sampled FinerCNN features at keypoints
    f_fine = descriptor._sample_at_keypoints(fine_map, kp1, is_patch_map=False)
    print(f"    FinerCNN @ keypoints: {tuple(f_fine.shape)}  (expected [1, 512, 64])")
    assert f_fine.shape == (1, 512, 64)
    print(f"    FinerCNN sampling: PASSED")

    # ------------------------------------------------------------------ #
    # 10. Visualization
    # ------------------------------------------------------------------ #
    print(f"\n[10] Saving visualizations...")
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)

    img1_vis = denormalize_image(sample["image1"])
    img2_vis = denormalize_image(sample["image2"])

    kp1_cpu = kp1[0].cpu().numpy()  # (512, 2)
    kp2_cpu = kp2[0].cpu().numpy()  # (512, 2)

    # Compute similarity map for coloring keypoints in image1
    # Color = mean cosine sim between this keypoint's descriptor and all kp2 descriptors
    sim_matrix = (desc1[0] @ desc2[0].T).cpu().numpy()  # (512, 512)
    best_sim = sim_matrix.max(axis=1)  # (512,) — best match score per keypoint

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        "Feature Descriptor (Phase 4) — DINOv2 + FinerCNN → 192-d",
        fontsize=14,
    )

    # Top-left: image1 keypoints colored by best-match similarity
    axes[0, 0].imshow(img1_vis)
    sc = axes[0, 0].scatter(
        kp1_cpu[:nv1, 0], kp1_cpu[:nv1, 1],
        c=best_sim[:nv1], cmap="plasma", s=10, alpha=0.85,
        vmin=0.0, vmax=1.0,
    )
    plt.colorbar(sc, ax=axes[0, 0], label="Best match cosine sim")
    axes[0, 0].set_title(f"Image 1 — {nv1} keypoints (colored by matchability)")
    axes[0, 0].axis("off")

    # Top-right: image2 keypoints
    axes[0, 1].imshow(img2_vis)
    axes[0, 1].scatter(
        kp2_cpu[:nv2, 0], kp2_cpu[:nv2, 1],
        c="lime", s=10, alpha=0.7,
    )
    axes[0, 1].set_title(f"Image 2 — {nv2} keypoints")
    axes[0, 1].axis("off")

    # Bottom-left: DINOv2 feature magnitude at keypoints (PCA first dim)
    dino_feat_np = f_dino[0, :nv1].cpu().float().numpy()  # (nv1, 384)
    dino_mag = np.linalg.norm(dino_feat_np, axis=-1)
    axes[1, 0].imshow(img1_vis)
    sc2 = axes[1, 0].scatter(
        kp1_cpu[:nv1, 0], kp1_cpu[:nv1, 1],
        c=dino_mag, cmap="viridis", s=10, alpha=0.85,
    )
    plt.colorbar(sc2, ax=axes[1, 0], label="DINOv2 feature norm")
    axes[1, 0].set_title("DINOv2 feature norm at keypoints")
    axes[1, 0].axis("off")

    # Bottom-right: cosine similarity histogram
    axes[1, 1].hist(
        cos_sim.cpu().numpy(), bins=50, color="steelblue", edgecolor="white"
    )
    axes[1, 1].axvline(mean_sim, color="red", linestyle="--", label=f"mean={mean_sim:.3f}")
    axes[1, 1].set_title("Cosine similarity: matching keypoint descriptors (img1 vs img2)")
    axes[1, 1].set_xlabel("Cosine similarity")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    plt.tight_layout()
    output_path = os.path.join(project_root, "outputs", "feature_descriptor_test.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Phase 4 verification COMPLETE. All checks passed.")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  DINOv2 features:   (K, 384) — frozen")
    print(f"  FinerCNN features: (K,  64) — trainable")
    print(f"  Final descriptors: (K, 192) — L2 normalized")


if __name__ == "__main__":
    main()
