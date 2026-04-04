"""Quick test of the TartanAir dataloader."""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.tartanair import build_tartanair_dataset

print("Building TartanAir dataset...")
dataset = build_tartanair_dataset(
    data_root="data/tartanair",
    skip_frames=1,
    target_h=476,
    target_w=742,
)
print(f"\nTotal pairs: {len(dataset)}")

# Test a few samples
for i in [0, len(dataset) // 2, len(dataset) - 1]:
    print(f"\n--- Sample {i} ---")
    sample = dataset[i]

    img1 = sample["image1"]
    img2 = sample["image2"]
    pose = sample["relative_pose"]
    K = sample["intrinsics"]
    depth = sample["depth1"]

    print(f"  image1:  {img1.shape}, range [{img1.min():.2f}, {img1.max():.2f}]")
    print(f"  image2:  {img2.shape}, range [{img2.min():.2f}, {img2.max():.2f}]")
    print(f"  depth1:  {depth.shape}, range [{depth.min():.2f}, {depth.max():.2f}] meters")
    print(f"  K:       fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    R = pose[:3, :3].numpy()
    t = pose[:3, 3].numpy()
    det = np.linalg.det(R)
    t_mag = np.linalg.norm(t)
    print(f"  R det:   {det:.6f} (should be 1.0)")
    print(f"  t mag:   {t_mag:.4f} meters")

# Test DataLoader
print("\n--- DataLoader test ---")
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"  Batch image1: {batch['image1'].shape}")
print(f"  Batch depth1: {batch['depth1'].shape}")
print(f"  Batch pose:   {batch['relative_pose'].shape}")
print(f"  Batch K:      {batch['intrinsics'].shape}")

print("\nAll checks passed!")
