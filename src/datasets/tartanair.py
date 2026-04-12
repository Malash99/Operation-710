"""
TartanAir Dataset loader for DINO-VO training.

Loads consecutive image pairs from TartanAir simulation environments
with GT relative poses and perfect depth maps for correspondence generation.

TartanAir format:
  env/Easy/P001/image_left/000000_left.png   — 640x480 RGB
  env/Easy/P001/depth_left/000000_left_depth.npy — 640x480 float32 (meters)
  env/Easy/P001/pose_left.txt — one line per frame: tx ty tz qx qy qz qw (NED)

Camera intrinsics (fixed for all TartanAir):
  fx = fy = 320.0, cx = 320.0, cy = 240.0, image = 640x480

Reference: https://theairlab.org/tartanair-dataset/
"""

import os
import glob

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, ConcatDataset

from src.datasets.transforms import get_rescaled_intrinsics

# ImageNet normalization (required by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# TartanAir fixed camera intrinsics
TARTANAIR_FX = 320.0
TARTANAIR_FY = 320.0
TARTANAIR_CX = 320.0
TARTANAIR_CY = 240.0
TARTANAIR_W = 640
TARTANAIR_H = 480


class TartanAirTrajectory(Dataset):
    """PyTorch Dataset for a single TartanAir trajectory.

    Loads pairs of consecutive frames and computes relative camera pose
    from GT poses for supervision. Depth maps are loaded for GT correspondence
    generation (matching loss, Eq. 12).

    Args:
        trajectory_path: Path to trajectory root, e.g.,
                         data/tartanair/carwelding/Easy/P001
        skip_frames: Frames to skip between pairs (default 1 = consecutive).
        target_h: Target image height after resize (476 as per paper).
        target_w: Target image width after resize (742 as per paper).
    """

    def __init__(
        self,
        trajectory_path: str,
        skip_frames: int = 1,
        target_h: int = 476,
        target_w: int = 742,
    ):
        self.trajectory_path = trajectory_path
        self.skip_frames = skip_frames
        self.target_h = target_h
        self.target_w = target_w

        self.image_dir = os.path.join(trajectory_path, "image_left")
        self.depth_dir = os.path.join(trajectory_path, "depth_left")
        self.pose_file = os.path.join(trajectory_path, "pose_left.txt")

        # Load image file list (sorted by frame number)
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        )
        self.n_frames = len(self.image_files)

        # Load poses: one line per frame, tx ty tz qx qy qz qw
        self.poses = self._load_poses(self.pose_file)
        assert len(self.poses) == self.n_frames, (
            f"Pose count ({len(self.poses)}) != image count ({self.n_frames}) "
            f"in {trajectory_path}"
        )

        # Build frame pairs
        self.pairs = []
        for i in range(self.n_frames - skip_frames):
            j = i + skip_frames
            self.pairs.append((i, j))

        # Original intrinsics
        self.K_orig = np.array([
            [TARTANAIR_FX, 0.0, TARTANAIR_CX],
            [0.0, TARTANAIR_FY, TARTANAIR_CY],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Rescaled intrinsics for target resolution
        self.K_scaled = get_rescaled_intrinsics(
            self.K_orig, TARTANAIR_W, TARTANAIR_H, self.target_w, self.target_h
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        frame_i, frame_j = self.pairs[idx]

        # Load images (already RGB)
        img1 = cv2.imread(
            os.path.join(self.image_dir, self.image_files[frame_i]),
            cv2.IMREAD_COLOR,
        )
        img2 = cv2.imread(
            os.path.join(self.image_dir, self.image_files[frame_j]),
            cv2.IMREAD_COLOR,
        )
        # BGR -> RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Resize to target resolution
        img1 = cv2.resize(img1, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # To tensor + normalize (no undistortion needed — synthetic data)
        tensor1 = self._to_tensor(img1)
        tensor2 = self._to_tensor(img2)

        # Load depth for frame 1 (for GT correspondence generation)
        depth_file = self.image_files[frame_i].replace("_left.png", "_left_depth.npy")
        depth1 = np.load(os.path.join(self.depth_dir, depth_file))  # (480, 640) float32
        # Resize depth to target resolution (nearest neighbor)
        depth1 = cv2.resize(
            depth1, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
        )

        # Compute relative pose
        T_w1 = self.poses[frame_i]  # world <- cam1
        T_w2 = self.poses[frame_j]  # world <- cam2
        # T_1to2 = inv(T_w2) @ T_w1  (cam1 -> cam2)
        T_1to2 = np.linalg.inv(T_w2) @ T_w1

        return {
            "image1": tensor1,                                         # (3, H, W)
            "image2": tensor2,                                         # (3, H, W)
            "relative_pose": torch.from_numpy(T_1to2).float(),        # (4, 4)
            "intrinsics": torch.from_numpy(self.K_scaled).float(),    # (3, 3)
            "depth1": torch.from_numpy(depth1).float(),               # (H, W)
        }

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HWC uint8 RGB -> CHW float32, ImageNet-normalized."""
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    # NED-to-camera frame conversion (see tartanair_tools/evaluation/trajectory_transform.py)
    # TartanAir NED: x=forward, y=right, z=down
    # Camera frame:  x=right,   y=down,  z=forward
    _T_ned2cam = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
    _T_cam2ned = np.linalg.inv(_T_ned2cam)

    def _load_poses(self, pose_file: str) -> list:
        """Load TartanAir poses as 4x4 homogeneous matrices in camera frame.

        Each line: tx ty tz qx qy qz qw (NED frame)
        Converted to standard camera frame via T_ned2cam conjugation.

        Returns:
            List of (4, 4) numpy arrays in camera frame.
        """
        poses = []
        with open(pose_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = [float(v) for v in line.split()]
                tx, ty, tz = values[0], values[1], values[2]
                qx, qy, qz, qw = values[3], values[4], values[5], values[6]

                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T_ned = np.eye(4, dtype=np.float64)
                T_ned[:3, :3] = R
                T_ned[:3, 3] = [tx, ty, tz]

                # Convert from NED frame to camera frame
                T_cam = self._T_ned2cam @ T_ned @ self._T_cam2ned
                poses.append(T_cam)
        return poses


def build_tartanair_dataset(
    data_root: str,
    skip_frames: int = 1,
    target_h: int = 476,
    target_w: int = 742,
    max_pairs: int = 0,
) -> ConcatDataset:
    """Build a combined dataset from all TartanAir trajectories found under data_root.

    Scans for: data_root/*/Easy/P*/image_left/ and data_root/*/Hard/P*/image_left/

    Args:
        data_root: Root path to tartanair data, e.g., data/tartanair
        skip_frames: Frames to skip between pairs.
        target_h: Target image height.
        target_w: Target image width.
        max_pairs: If > 0, limit total dataset to this many pairs (for quick experiments).

    Returns:
        ConcatDataset of all valid trajectories.
    """
    trajectories = []
    total_pairs = 0

    # Find all trajectory directories
    pattern = os.path.join(data_root, "*", "*", "P*")
    traj_dirs = sorted(glob.glob(pattern))

    for traj_dir in traj_dirs:
        # Validate: must have image_left, depth_left, pose_left.txt
        img_dir = os.path.join(traj_dir, "image_left")
        depth_dir = os.path.join(traj_dir, "depth_left")
        pose_file = os.path.join(traj_dir, "pose_left.txt")

        if not (os.path.isdir(img_dir) and os.path.isdir(depth_dir) and os.path.isfile(pose_file)):
            continue

        n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if n_imgs < 2:
            continue

        ds = TartanAirTrajectory(
            trajectory_path=traj_dir,
            skip_frames=skip_frames,
            target_h=target_h,
            target_w=target_w,
        )
        trajectories.append(ds)
        total_pairs += len(ds)

        rel_path = os.path.relpath(traj_dir, data_root)
        print(f"  {rel_path}: {n_imgs} frames, {len(ds)} pairs")

    print(f"\n  Total: {len(trajectories)} trajectories, {total_pairs} pairs")

    combined = ConcatDataset(trajectories)

    # Optionally limit dataset size for quick experiments
    if max_pairs > 0 and total_pairs > max_pairs:
        indices = list(range(0, total_pairs, total_pairs // max_pairs))[:max_pairs]
        combined = torch.utils.data.Subset(combined, indices)
        print(f"  Limited to {len(combined)} pairs (max_pairs={max_pairs})")

    return combined
