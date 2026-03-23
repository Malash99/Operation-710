"""
Stereo depth computation for EuRoC MAV dataset.

Uses cam0 (left) and cam1 (right) stereo pairs to compute dense disparity
maps via OpenCV StereoSGBM, then converts to depth.

The depth is used to generate ground truth correspondences for the
matching loss (Eq. 12) during training.

Stereo geometry:
    T_cam0_body = T_BS_cam0  (from cam0/sensor.yaml)
    T_cam1_body = T_BS_cam1  (from cam1/sensor.yaml)
    T_cam1_cam0 = inv(T_BS_cam1) @ T_BS_cam0  (cam0 -> cam1 transform)
    baseline = ||T_cam1_cam0[:3, 3]||  (~11cm for EuRoC VI-Sensor)
"""

import cv2
import numpy as np
import yaml


def load_stereo_calibration(cam0_yaml_path: str, cam1_yaml_path: str) -> dict:
    """Load and compute stereo calibration parameters from EuRoC sensor YAMLs.

    Args:
        cam0_yaml_path: Path to cam0/sensor.yaml.
        cam1_yaml_path: Path to cam1/sensor.yaml.

    Returns:
        dict with keys:
            K0, K1: (3,3) intrinsic matrices
            dist0, dist1: (4,) distortion coefficients
            T_BS0, T_BS1: (4,4) camera-to-body transforms
            T_10: (4,4) cam0-to-cam1 transform
            R_10: (3,3) rotation from cam0 to cam1
            t_10: (3,1) translation from cam0 to cam1
            baseline: scalar, stereo baseline in meters
    """
    def _parse_yaml(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        fu, fv, cu, cv = data["intrinsics"]
        K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array(data["distortion_coefficients"], dtype=np.float64)
        T_BS = np.array(data["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
        return K, dist, T_BS

    K0, dist0, T_BS0 = _parse_yaml(cam0_yaml_path)
    K1, dist1, T_BS1 = _parse_yaml(cam1_yaml_path)

    # Relative transform: cam0 frame -> cam1 frame
    # T_10 = inv(T_BS1) @ T_BS0
    T_10 = np.linalg.inv(T_BS1) @ T_BS0
    R_10 = T_10[:3, :3]
    t_10 = T_10[:3, 3:]  # (3, 1)
    baseline = np.linalg.norm(t_10)

    return {
        "K0": K0,
        "K1": K1,
        "dist0": dist0,
        "dist1": dist1,
        "T_BS0": T_BS0,
        "T_BS1": T_BS1,
        "T_10": T_10,
        "R_10": R_10,
        "t_10": t_10,
        "baseline": baseline,
    }


def compute_stereo_rectification(calib: dict, image_size: tuple) -> dict:
    """Compute stereo rectification maps for undistorted+rectified stereo pair.

    Args:
        calib: Output from load_stereo_calibration().
        image_size: (width, height) of the original images.

    Returns:
        dict with keys:
            map0x, map0y: Remap tables for cam0
            map1x, map1y: Remap tables for cam1
            Q: (4,4) disparity-to-depth reprojection matrix
            P0, P1: (3,4) projection matrices after rectification
            R0, R1: (3,3) rectification rotation for each camera
    """
    R_10 = calib["R_10"]
    t_10 = calib["t_10"]

    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        cameraMatrix1=calib["K0"],
        distCoeffs1=calib["dist0"],
        cameraMatrix2=calib["K1"],
        distCoeffs2=calib["dist1"],
        imageSize=image_size,
        R=R_10,
        T=t_10,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,  # Crop to valid pixels only
    )

    map0x, map0y = cv2.initUndistortRectifyMap(
        calib["K0"], calib["dist0"], R0, P0, image_size, cv2.CV_32FC1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"], calib["dist1"], R1, P1, image_size, cv2.CV_32FC1
    )

    return {
        "map0x": map0x,
        "map0y": map0y,
        "map1x": map1x,
        "map1y": map1y,
        "Q": Q,
        "P0": P0,
        "P1": P1,
        "R0": R0,
        "R1": R1,
    }


def create_stereo_matcher(
    min_disparity: int = 0,
    num_disparities: int = 96,
    block_size: int = 5,
    p1_multiplier: int = 8,
    p2_multiplier: int = 32,
    disp12_max_diff: int = 1,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
) -> cv2.StereoSGBM:
    """Create a StereoSGBM matcher with reasonable defaults for EuRoC.

    Args:
        min_disparity: Minimum possible disparity value.
        num_disparities: Range of disparity search (must be divisible by 16).
        block_size: Matched block size (odd number, >=1).
        p1_multiplier: Penalty multiplier for P1 (small disparity changes).
        p2_multiplier: Penalty multiplier for P2 (large disparity changes).
        disp12_max_diff: Maximum allowed difference in left-right disparity check.
        uniqueness_ratio: Margin in percentage for best match vs second best.
        speckle_window_size: Maximum size of smooth disparity regions.
        speckle_range: Maximum disparity variation within a connected component.

    Returns:
        cv2.StereoSGBM matcher object.
    """
    channels = 1  # grayscale

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1_multiplier * channels * block_size ** 2,
        P2=p2_multiplier * channels * block_size ** 2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    return matcher


def compute_depth_map(
    img0_gray: np.ndarray,
    img1_gray: np.ndarray,
    rect: dict,
    matcher: cv2.StereoSGBM,
    baseline: float,
) -> np.ndarray:
    """Compute depth map from a stereo image pair.

    Pipeline: rectify both images -> compute disparity -> convert to depth.

    Args:
        img0_gray: Left (cam0) grayscale image, shape (H, W), uint8.
        img1_gray: Right (cam1) grayscale image, shape (H, W), uint8.
        rect: Output from compute_stereo_rectification().
        matcher: StereoSGBM matcher from create_stereo_matcher().
        baseline: Stereo baseline in meters.

    Returns:
        depth: (H, W) float32 depth map in meters. Invalid pixels are 0.0.
    """
    # Rectify both images
    img0_rect = cv2.remap(img0_gray, rect["map0x"], rect["map0y"], cv2.INTER_LINEAR)
    img1_rect = cv2.remap(img1_gray, rect["map1x"], rect["map1y"], cv2.INTER_LINEAR)

    # Compute disparity (SGBM returns fixed-point: disparity * 16)
    disparity_raw = matcher.compute(img0_rect, img1_rect)

    # Convert to float disparity
    disparity = disparity_raw.astype(np.float32) / 16.0

    # Invalid disparity -> 0
    disparity[disparity <= 0] = 0.0

    # depth = f * baseline / disparity
    # After rectification, focal length is P0[0,0]
    f = rect["P0"][0, 0]
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = f * baseline / disparity[valid]

    # Clamp unreasonable depths (> 100m is noise for indoor EuRoC)
    depth[depth > 100.0] = 0.0

    return depth


def get_depth_at_keypoints(
    depth_map: np.ndarray,
    keypoints_uv: np.ndarray,
    rect: dict,
) -> np.ndarray:
    """Sample depth values at keypoint locations.

    Since the depth map is in rectified coordinates but keypoints are detected
    in the original (undistorted+resized) image, we need to account for the
    rectification mapping. However, since we compute depth on original-resolution
    rectified images and keypoints are on resized images, the caller must handle
    coordinate scaling before calling this function.

    Args:
        depth_map: (H, W) depth map from compute_depth_map().
        keypoints_uv: (K, 2) keypoint pixel coordinates (u, v) in depth map coords.
        rect: Rectification dict (unused here but kept for API consistency).

    Returns:
        depths: (K,) depth values. 0.0 for invalid/out-of-bounds keypoints.
    """
    K = keypoints_uv.shape[0]
    depths = np.zeros(K, dtype=np.float32)

    H, W = depth_map.shape
    for i in range(K):
        u, v = int(round(keypoints_uv[i, 0])), int(round(keypoints_uv[i, 1]))
        if 0 <= u < W and 0 <= v < H:
            depths[i] = depth_map[v, u]

    return depths


def generate_gt_correspondences(
    kp1_uv: np.ndarray,
    kp2_uv: np.ndarray,
    depth1: np.ndarray,
    K: np.ndarray,
    T_1to2: np.ndarray,
    reproj_threshold: float = 4.0,
) -> tuple:
    """Generate ground truth correspondences by reprojecting keypoints.

    For each keypoint in image 1 with valid depth:
      1. Back-project to 3D: X = Z * K^{-1} * [u, v, 1]^T
      2. Transform to camera 2: X' = T_1to2 @ X
      3. Project to image 2: [u', v'] = K @ X'[:3] / X'[2]
      4. Find nearest keypoint in image 2 within reproj_threshold

    Args:
        kp1_uv: (N1, 2) keypoint pixel coords in image 1 (u, v).
        kp2_uv: (N2, 2) keypoint pixel coords in image 2 (u, v).
        depth1: (N1,) depth values for each keypoint in image 1.
        K: (3, 3) camera intrinsics (for the resized image).
        T_1to2: (4, 4) GT relative pose from camera 1 to camera 2.
        reproj_threshold: Max pixel distance to accept a correspondence.

    Returns:
        gt_matches: (M, 2) array of (idx_in_kp1, idx_in_kp2) pairs.
        gt_mask_indices: indices into kp1 that have valid matches.
    """
    N1 = kp1_uv.shape[0]
    N2 = kp2_uv.shape[0]

    if N1 == 0 or N2 == 0:
        return np.zeros((0, 2), dtype=np.int64), np.array([], dtype=np.int64)

    K_inv = np.linalg.inv(K)
    R = T_1to2[:3, :3]
    t = T_1to2[:3, 3]

    matches = []

    for i in range(N1):
        z = depth1[i]
        if z <= 0:
            continue

        # Back-project to 3D in camera 1 frame
        uv1_h = np.array([kp1_uv[i, 0], kp1_uv[i, 1], 1.0])
        p3d_cam1 = z * (K_inv @ uv1_h)  # (3,)

        # Transform to camera 2 frame
        p3d_cam2 = R @ p3d_cam1 + t  # (3,)

        # Skip points behind camera 2
        if p3d_cam2[2] <= 0:
            continue

        # Project to image 2
        proj = K @ p3d_cam2
        u2 = proj[0] / proj[2]
        v2 = proj[1] / proj[2]

        # Find nearest keypoint in image 2
        dists = np.sqrt((kp2_uv[:, 0] - u2) ** 2 + (kp2_uv[:, 1] - v2) ** 2)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist < reproj_threshold:
            matches.append((i, int(min_idx)))

    if len(matches) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.array([], dtype=np.int64)

    gt_matches = np.array(matches, dtype=np.int64)  # (M, 2)

    # Deduplicate: if multiple kp1 match to the same kp2, keep the closest
    # Group by kp2 index, keep only the first occurrence (simplest approach)
    seen_j = {}
    unique_matches = []
    for m in range(gt_matches.shape[0]):
        j = gt_matches[m, 1]
        if j not in seen_j:
            seen_j[j] = m
            unique_matches.append(gt_matches[m])

    gt_matches = np.array(unique_matches, dtype=np.int64) if unique_matches else np.zeros((0, 2), dtype=np.int64)

    return gt_matches, gt_matches[:, 0] if gt_matches.shape[0] > 0 else np.array([], dtype=np.int64)
