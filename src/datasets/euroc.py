"""
EuRoC MAV Dataset loader for DINO-VO.

Loads consecutive image pairs from the EuRoC Machine Hall sequences
with corresponding relative camera poses computed from ground truth.
Optionally computes stereo depth maps from cam0+cam1 for GT correspondence
generation (needed by the matching loss, Eq. 12).

Reference: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
"""

import os
import csv
import yaml

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from src.datasets.transforms import get_rescaled_intrinsics, preprocess_image
from src.utils.stereo import (
    load_stereo_calibration,
    compute_stereo_rectification,
    create_stereo_matcher,
    compute_depth_map,
)


class EuRoCDataset(Dataset):
    """PyTorch Dataset for EuRoC MAV sequences.

    Loads pairs of consecutive frames (separated by skip_frames) and computes
    the relative camera pose from ground truth for supervision.

    Args:
        sequence_path: Path to sequence root, e.g., data/euroc/MH_01_easy.
        skip_frames: Number of frames to skip between pairs (default 2,
                     i.e., alternate frames as noted in paper Section III-F).
        target_h: Target image height after resize (476 for EuRoC).
        target_w: Target image width after resize (742 for EuRoC).
    """

    def __init__(
        self,
        sequence_path: str,
        skip_frames: int = 2,
        target_h: int = 476,
        target_w: int = 742,
        compute_stereo_depth: bool = True,
        min_translation: float = 0.0,
        max_skip_multiplier: int = 5,
    ):
        self.sequence_path = sequence_path
        self.skip_frames = skip_frames
        self.target_h = target_h
        self.target_w = target_w
        self.compute_stereo_depth = compute_stereo_depth
        self.min_translation = min_translation
        self.max_skip_multiplier = max_skip_multiplier

        mav0_path = os.path.join(sequence_path, "mav0")

        # 1. Parse camera calibration
        cam0_sensor_path = os.path.join(mav0_path, "cam0", "sensor.yaml")
        self.K, self.dist_coeffs, self.T_BS = self._parse_sensor_yaml(cam0_sensor_path)
        self.orig_h, self.orig_w = 480, 752  # EuRoC cam0 resolution

        # 2. Parse image lists (cam0 + cam1)
        cam0_csv_path = os.path.join(mav0_path, "cam0", "data.csv")
        self.image_dir = os.path.join(mav0_path, "cam0", "data")
        self.image_list = self._parse_image_list(cam0_csv_path)

        # cam1 for stereo depth
        self.image_dir_cam1 = os.path.join(mav0_path, "cam1", "data")
        cam1_csv_path = os.path.join(mav0_path, "cam1", "data.csv")
        self.image_list_cam1 = self._parse_image_list(cam1_csv_path)
        # Build timestamp -> filename lookup for cam1
        self._cam1_lookup = {ts: fname for ts, fname in self.image_list_cam1}

        # 2b. Setup stereo depth computation
        if self.compute_stereo_depth:
            cam1_sensor_path = os.path.join(mav0_path, "cam1", "sensor.yaml")
            self.stereo_calib = load_stereo_calibration(
                cam0_sensor_path, cam1_sensor_path
            )
            self.stereo_rect = compute_stereo_rectification(
                self.stereo_calib,
                image_size=(self.orig_w, self.orig_h),
            )
            # Stereo matcher created lazily in _get_stereo_matcher() because
            # cv2.StereoSGBM is not picklable (breaks num_workers > 0).
            self._stereo_matcher = None

        # 3. Parse ground truth poses
        gt_csv_path = os.path.join(
            mav0_path, "state_groundtruth_estimate0", "data.csv"
        )
        self.gt_timestamps, self.gt_poses = self._parse_ground_truth(gt_csv_path)

        # 4. Match each image timestamp to nearest ground truth
        image_timestamps = np.array([ts for ts, _ in self.image_list], dtype=np.int64)
        gt_indices, valid_mask = self._find_nearest_gt(
            image_timestamps, self.gt_timestamps
        )

        # 5. Build valid frame indices (those with nearby GT)
        valid_frame_indices = np.where(valid_mask)[0]

        # 6. Build pairs with keyframe selection (Section III-F).
        #    Skip pairs where GT translation is below min_translation to avoid
        #    degenerate Essential matrix estimation from tiny baselines.
        self.pairs = []
        self.pair_gt_indices = []
        valid_set = set(valid_frame_indices.tolist())
        skipped_small_motion = 0

        for idx in valid_frame_indices:
            # Try increasing skip multiples until sufficient motion is found
            found = False
            for mult in range(1, self.max_skip_multiplier + 1):
                j = idx + skip_frames * mult
                if j not in valid_set:
                    continue

                if self.min_translation > 0:
                    # Check GT translation magnitude
                    T_rel = self._compute_relative_pose(
                        self.gt_poses[gt_indices[idx]],
                        self.gt_poses[gt_indices[j]],
                    )
                    t_mag = np.linalg.norm(T_rel[:3, 3])
                    if t_mag < self.min_translation:
                        continue

                self.pairs.append((int(idx), int(j)))
                self.pair_gt_indices.append(
                    (int(gt_indices[idx]), int(gt_indices[j]))
                )
                found = True
                break

            if not found and self.min_translation > 0:
                skipped_small_motion += 1

        if skipped_small_motion > 0:
            print(f"  Keyframe selection: skipped {skipped_small_motion} frames with insufficient motion")

        # 7. Precompute rescaled intrinsics
        self.K_scaled = get_rescaled_intrinsics(
            self.K, self.orig_w, self.orig_h, self.target_w, self.target_h
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        frame_i, frame_j = self.pairs[idx]
        gt_i, gt_j = self.pair_gt_indices[idx]

        # Load cam0 images
        ts_i, fname_i = self.image_list[frame_i]
        ts_j, fname_j = self.image_list[frame_j]
        img1 = cv2.imread(
            os.path.join(self.image_dir, fname_i), cv2.IMREAD_GRAYSCALE
        )
        img2 = cv2.imread(
            os.path.join(self.image_dir, fname_j), cv2.IMREAD_GRAYSCALE
        )

        if img1 is None:
            raise FileNotFoundError(
                f"Could not load image: {os.path.join(self.image_dir, fname_i)}"
            )
        if img2 is None:
            raise FileNotFoundError(
                f"Could not load image: {os.path.join(self.image_dir, fname_j)}"
            )

        # Preprocess images (undistort, resize, grayscale→RGB, normalize)
        tensor1 = preprocess_image(
            img1, self.K, self.dist_coeffs, self.target_h, self.target_w
        )
        tensor2 = preprocess_image(
            img2, self.K, self.dist_coeffs, self.target_h, self.target_w
        )

        # Compute relative pose in camera frame
        pose_wb1 = self.gt_poses[gt_i]  # T_WB at time 1
        pose_wb2 = self.gt_poses[gt_j]  # T_WB at time 2
        relative_pose = self._compute_relative_pose(pose_wb1, pose_wb2)

        result = {
            "image1": tensor1,                                      # (3, 476, 742)
            "image2": tensor2,                                      # (3, 476, 742)
            "relative_pose": torch.from_numpy(relative_pose).float(),  # (4, 4)
            "intrinsics": torch.from_numpy(self.K_scaled).float(),     # (3, 3)
            "timestamp1": ts_i,
            "timestamp2": ts_j,
        }

        # Compute stereo depth map for image 1 (original resolution)
        if self.compute_stereo_depth:
            depth1 = self._compute_stereo_depth_for_frame(ts_i, img1)
            # Resize depth map to target resolution (nearest neighbor to avoid interpolation artifacts)
            depth1_resized = cv2.resize(
                depth1,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            result["depth1"] = torch.from_numpy(depth1_resized).float()  # (476, 742)

        return result

    def _get_stereo_matcher(self):
        """Lazy-create stereo matcher (cv2.StereoSGBM is not picklable)."""
        if self._stereo_matcher is None:
            self._stereo_matcher = create_stereo_matcher()
        return self._stereo_matcher

    def _compute_stereo_depth_for_frame(
        self, timestamp_ns: int, img0_gray: np.ndarray
    ) -> np.ndarray:
        """Compute stereo depth for a single frame using cam0+cam1.

        Args:
            timestamp_ns: Timestamp of the frame to find matching cam1 image.
            img0_gray: cam0 grayscale image (original resolution), shape (H, W).

        Returns:
            depth: (H, W) float32 depth map in meters. Invalid pixels are 0.0.
        """
        # Find matching cam1 image by timestamp
        cam1_fname = self._cam1_lookup.get(timestamp_ns)
        if cam1_fname is None:
            # No exact match; return empty depth
            return np.zeros((self.orig_h, self.orig_w), dtype=np.float32)

        img1_gray = cv2.imread(
            os.path.join(self.image_dir_cam1, cam1_fname), cv2.IMREAD_GRAYSCALE
        )
        if img1_gray is None:
            return np.zeros((self.orig_h, self.orig_w), dtype=np.float32)

        depth = compute_depth_map(
            img0_gray,
            img1_gray,
            self.stereo_rect,
            self._get_stereo_matcher(),
            self.stereo_calib["baseline"],
        )
        return depth

    # ------------------------------------------------------------------ #
    #  Parsing helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_sensor_yaml(self, yaml_path: str):
        """Parse cam0/sensor.yaml to extract intrinsics, distortion, and extrinsics.

        Returns:
            K: Intrinsic matrix, shape (3, 3).
            dist_coeffs: Distortion coefficients, shape (4,).
            T_BS: Camera-to-body extrinsic transform, shape (4, 4).
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Intrinsics: [fu, fv, cu, cv]
        fu, fv, cu, cv = data["intrinsics"]
        K = np.array([
            [fu, 0.0, cu],
            [0.0, fv, cv],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Distortion coefficients: [k1, k2, p1, p2]
        dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64)

        # Camera-to-body extrinsic: T_BS (4x4 row-major)
        T_BS = np.array(data["T_BS"]["data"], dtype=np.float64).reshape(4, 4)

        return K, dist_coeffs, T_BS

    def _parse_image_list(self, csv_path: str):
        """Parse cam0/data.csv to get list of (timestamp_ns, filename) tuples.

        Returns:
            List of (int, str) tuples sorted by timestamp.
        """
        image_list = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip header and empty lines
                if not row or row[0].startswith("#"):
                    continue
                timestamp_ns = int(row[0])
                filename = row[1].strip()
                image_list.append((timestamp_ns, filename))

        image_list.sort(key=lambda x: x[0])
        return image_list

    def _parse_ground_truth(self, csv_path: str):
        """Parse ground truth CSV to get timestamps and 4x4 pose matrices.

        EuRoC GT format per row:
            timestamp, px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...

        The poses are T_WB (body frame in world frame).

        Returns:
            timestamps: Array of int64 timestamps in nanoseconds, shape (N,).
            poses: Array of 4x4 homogeneous transforms, shape (N, 4, 4).
        """
        timestamps = []
        poses = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue

                timestamp_ns = int(row[0])
                px, py, pz = float(row[1]), float(row[2]), float(row[3])
                qw, qx, qy, qz = (
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                )

                # scipy.Rotation.from_quat expects [qx, qy, qz, qw] order
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [px, py, pz]

                timestamps.append(timestamp_ns)
                poses.append(T)

        timestamps = np.array(timestamps, dtype=np.int64)
        poses = np.array(poses, dtype=np.float64)
        return timestamps, poses

    def _find_nearest_gt(
        self,
        image_timestamps: np.ndarray,
        gt_timestamps: np.ndarray,
        max_diff_ns: int = 25_000_000,
    ):
        """For each image timestamp, find the nearest ground truth timestamp.

        Uses np.searchsorted for efficient lookup in the sorted GT array.

        Args:
            image_timestamps: Array of image timestamps (ns), shape (M,).
            gt_timestamps: Array of GT timestamps (ns), shape (N,). Must be sorted.
            max_diff_ns: Maximum allowed time difference in nanoseconds (default 25ms).

        Returns:
            gt_indices: Array of nearest GT indices, shape (M,).
            valid_mask: Boolean array indicating which images have a close enough GT.
        """
        # Find insertion points
        insert_idx = np.searchsorted(gt_timestamps, image_timestamps)

        # Clamp to valid range
        insert_idx = np.clip(insert_idx, 1, len(gt_timestamps) - 1)

        # Compare with neighbor on each side, pick closer one
        diff_left = np.abs(image_timestamps - gt_timestamps[insert_idx - 1])
        diff_right = np.abs(image_timestamps - gt_timestamps[insert_idx])

        gt_indices = np.where(diff_left <= diff_right, insert_idx - 1, insert_idx)
        min_diffs = np.minimum(diff_left, diff_right)

        valid_mask = min_diffs <= max_diff_ns

        return gt_indices, valid_mask

    def _compute_relative_pose(
        self, pose_wb1: np.ndarray, pose_wb2: np.ndarray
    ) -> np.ndarray:
        """Compute relative pose from camera frame 1 to camera frame 2.

        Coordinate frame math:
            T_BS = camera-to-body extrinsic (from sensor.yaml)
            T_WB = body-in-world (from ground truth)
            T_WC = T_WB @ T_BS   (camera pose in world frame)
            T_1to2 = inv(T_WC2) @ T_WC1  (relative pose: cam1 → cam2)

        Args:
            pose_wb1: T_WB at time 1, shape (4, 4).
            pose_wb2: T_WB at time 2, shape (4, 4).

        Returns:
            T_1to2: Relative pose matrix, shape (4, 4).
        """
        # Camera pose in world frame
        T_WC1 = pose_wb1 @ self.T_BS
        T_WC2 = pose_wb2 @ self.T_BS

        # Relative pose: transforms points from cam1 frame to cam2 frame
        T_WC2_inv = np.linalg.inv(T_WC2)
        T_1to2 = T_WC2_inv @ T_WC1

        return T_1to2
