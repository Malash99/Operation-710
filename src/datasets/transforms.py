"""
Image preprocessing transforms for DINO-VO.

Pipeline: undistort → resize (752x480 → 742x476) → grayscale-to-RGB → tensor → normalize.
Resolution 476x742 (HxW) is specified in Section IV-A of the paper.
"""

import cv2
import numpy as np
import torch


# ImageNet normalization constants (required by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def undistort_image(image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Remove lens distortion using camera intrinsics and distortion coefficients.

    Args:
        image: Grayscale image, shape (H, W), uint8.
        K: Camera intrinsic matrix, shape (3, 3).
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2), shape (4,).

    Returns:
        Undistorted image, same shape and dtype as input.
    """
    return cv2.undistort(image, K, dist_coeffs)


def resize_image(image: np.ndarray, target_h: int = 476, target_w: int = 742) -> np.ndarray:
    """Resize image to paper-specified resolution.

    Paper Section IV-A specifies 476x742 for EuRoC.

    Args:
        image: Input image, shape (H, W) or (H, W, C), uint8.
        target_h: Target height (476 for EuRoC).
        target_w: Target width (742 for EuRoC).

    Returns:
        Resized image, shape (target_h, target_w, ...).
    """
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def get_rescaled_intrinsics(
    K: np.ndarray, orig_w: int, orig_h: int, new_w: int, new_h: int
) -> np.ndarray:
    """Scale camera intrinsics after image resize.

    Args:
        K: Original intrinsic matrix, shape (3, 3).
        orig_w: Original image width (752 for EuRoC).
        orig_h: Original image height (480 for EuRoC).
        new_w: New image width (742 for EuRoC).
        new_h: New image height (476 for EuRoC).

    Returns:
        Rescaled intrinsic matrix, shape (3, 3).
    """
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    K_new = K.copy()
    K_new[0, 0] *= scale_x  # fx
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 1] *= scale_y  # fy
    K_new[1, 2] *= scale_y  # cy
    return K_new


def preprocess_image(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    target_h: int = 476,
    target_w: int = 742,
) -> torch.Tensor:
    """Full preprocessing pipeline: undistort → resize → grayscale-to-RGB → tensor → normalize.

    Args:
        image: Raw grayscale image from EuRoC, shape (H, W), uint8.
        K: Camera intrinsic matrix, shape (3, 3).
        dist_coeffs: Distortion coefficients, shape (4,).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Preprocessed image tensor, shape (3, target_h, target_w), float32.
        Values are ImageNet-normalized (roughly in range [-2, 3]).
    """
    # 1. Undistort
    image = undistort_image(image, K, dist_coeffs)

    # 2. Resize
    image = resize_image(image, target_h, target_w)

    # 3. Grayscale → RGB (repeat single channel 3 times)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)  # (H, W, 3)

    # 4. To tensor: HWC uint8 → CHW float32 [0, 1]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # 5. Normalize with ImageNet mean/std (required by DINOv2)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor
