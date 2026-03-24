"""
DINO-VO: Unified model wrapper combining all pipeline stages.

Wires together:
  1. SalientKeypointDetector  (Section III-A)
  2. FeatureDescriptor        (Section III-B)
  3. FeatureMatching          (Section III-C)
  4. PoseEstimation           (Section III-D)

Usage:
    model = DinoVO()
    model.load_dino(device)
    model = model.to(device)

    out = model(image1, image2, intrinsics, return_all_assignments=True)
    # out keys: kp1, kp2, desc1, desc2, all_assignments, all_sigma1,
    #           all_sigma2, matches, weights, R, t, E
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.models.keypoint_detector import SalientKeypointDetector
from src.models.feature_descriptor import FeatureDescriptor
from src.models.feature_matching import FeatureMatching
from src.models.pose_estimation import PoseEstimation


def _pad_matches(
    kp1: torch.Tensor,
    kp2: torch.Tensor,
    matches: List[torch.Tensor],
    weights: List[torch.Tensor],
    min_matches: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length per-batch matches to a fixed-length tensor batch.

    PoseEstimation expects batched inputs of shape (B, M, 2), but the matcher
    returns a list of variable-length match tensors per batch element.
    We pad each element to the same length M using the last valid match.

    Args:
        kp1:      (B, K, 2) keypoints in image 1.
        kp2:      (B, K, 2) keypoints in image 2.
        matches:  list of B tensors, each (M_b, 2) — (idx_in_kp1, idx_in_kp2).
        weights:  list of B tensors, each (M_b,) — confidence per match.
        min_matches: minimum padded length (avoids degenerate 8-pt algorithm).

    Returns:
        kp1_matched: (B, M, 2) matched keypoints in image 1.
        kp2_matched: (B, M, 2) matched keypoints in image 2.
        w_matched:   (B, M)    per-match confidence weights.
    """
    B = kp1.shape[0]
    device = kp1.device

    # Determine padded length M
    lengths = [max(m.shape[0], min_matches) for m in matches]
    M = max(lengths)

    kp1_out = torch.zeros(B, M, 2, device=device, dtype=kp1.dtype)
    kp2_out = torch.zeros(B, M, 2, device=device, dtype=kp2.dtype)
    w_out   = torch.zeros(B, M,    device=device, dtype=kp1.dtype)

    for b in range(B):
        m = matches[b]   # (M_b, 2)
        w = weights[b]   # (M_b,)
        M_b = m.shape[0]

        if M_b == 0:
            # No matches: fill with first keypoint pair (degenerate, low weight)
            kp1_out[b] = kp1[b, 0:1].expand(M, 2)
            kp2_out[b] = kp2[b, 0:1].expand(M, 2)
            # weights stay 0
            continue

        i_idx = m[:, 0].long()   # indices into kp1[b]
        j_idx = m[:, 1].long()   # indices into kp2[b]

        kp1_matched_b = kp1[b, i_idx]   # (M_b, 2)
        kp2_matched_b = kp2[b, j_idx]   # (M_b, 2)

        # Fill valid range
        kp1_out[b, :M_b] = kp1_matched_b
        kp2_out[b, :M_b] = kp2_matched_b
        w_out[b, :M_b]   = w

        # Pad remainder by repeating last valid match
        if M_b < M:
            kp1_out[b, M_b:] = kp1_matched_b[-1]
            kp2_out[b, M_b:] = kp2_matched_b[-1]
            w_out[b, M_b:]   = 0.0   # zero weight so padded matches don't affect SVD

    return kp1_out, kp2_out, w_out


class DinoVO(nn.Module):
    """DINO-VO end-to-end visual odometry pipeline.

    Combines keypoint detection, feature extraction, transformer-based
    matching, and differentiable pose estimation into a single module.

    Args:
        top_k: Number of keypoints to detect per image (default 512).
        descriptor_dim: Feature descriptor dimension (default 192).
        matching_layers: Number of transformer layers in matcher (default 12).
        match_threshold: Minimum assignment score to extract a match (default 0.1).
    """

    def __init__(
        self,
        top_k: int = 512,
        descriptor_dim: int = 192,
        matching_layers: int = 12,
        match_threshold: float = 0.1,
    ):
        super().__init__()

        self.detector = SalientKeypointDetector(top_k=top_k)
        self.descriptor = FeatureDescriptor(out_dim=descriptor_dim)
        self.matcher = FeatureMatching(
            descriptor_dim=descriptor_dim,
            num_layers=matching_layers,
            match_threshold=match_threshold,
        )
        self.pose_est = PoseEstimation()

    def load_dino(self, device: torch.device) -> None:
        """Load and freeze DINOv2-ViT-S/14. Must be called before training."""
        self.descriptor.load_dino(device)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        intrinsics: torch.Tensor,
        return_all_assignments: bool = False,
    ) -> Dict:
        """Run the full DINO-VO pipeline on a stereo pair.

        Args:
            image1:     (B, 3, H, W) ImageNet-normalized image at time t.
            image2:     (B, 3, H, W) ImageNet-normalized image at time t+1.
            intrinsics: (B, 3, 3) camera intrinsic matrix K (rescaled).
            return_all_assignments: If True, return per-layer P/sigma lists
                                    needed for the deep-supervision matching
                                    loss (Eq. 12). Set True during training.

        Returns:
            dict with:
                kp1, kp2:          (B, K, 2) detected keypoints
                desc1, desc2:      (B, K, 192) descriptors
                matches:           list of B tensors (M_b, 2) match index pairs
                weights:           list of B tensors (M_b,) confidence
                R:                 (B, 3, 3) estimated rotation
                t:                 (B, 3) estimated translation (unit vector)
                E:                 (B, 3, 3) Essential matrix
                assignment:        (B, K, K) final soft assignment P
                sigma1, sigma2:    (B, K, 1) matchability scores

                If return_all_assignments=True:
                all_assignments:   list of 12 × (B, K, K)
                all_sigma1:        list of 12 × (B, K, 1)
                all_sigma2:        list of 12 × (B, K, 1)
        """
        H, W = image1.shape[2], image1.shape[3]

        # --- Step 1: Detect keypoints ---
        kp1_dict = self.detector(image1)
        kp2_dict = self.detector(image2)
        kp1 = kp1_dict["keypoints"]   # (B, K, 2)
        kp2 = kp2_dict["keypoints"]   # (B, K, 2)

        # --- Step 2: Extract descriptors ---
        desc1 = self.descriptor(image1, kp1)   # (B, K, 192)
        desc2 = self.descriptor(image2, kp2)   # (B, K, 192)

        # --- Step 3: Match features ---
        match_out = self.matcher(
            desc1, desc2, kp1, kp2,
            img_h=H, img_w=W,
            return_all_assignments=return_all_assignments,
        )

        # --- Step 4: Estimate pose ---
        matches = match_out["matches"]   # list[tensor(M_b, 2)]
        weights = match_out["weights"]   # list[tensor(M_b,)]

        # Sanitize weights before SVD — NaN/Inf from unstable softmax causes
        # degenerate Essential matrix and NaN gradients in the backward pass.
        weights_clean = [
            torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            for w in weights
        ]

        kp1_matched, kp2_matched, w_matched = _pad_matches(
            kp1, kp2, matches, weights_clean
        )
        pose_out = self.pose_est(kp1_matched, kp2_matched, w_matched, intrinsics)

        return {
            "kp1": kp1,
            "kp2": kp2,
            "desc1": desc1,
            "desc2": desc2,
            **match_out,
            **pose_out,
        }
