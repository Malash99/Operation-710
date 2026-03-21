"""
Loss functions for DINO-VO training.

Paper Section III-E: Two loss terms combined with a scheduling weight.

Equation 12 -- Matching Loss (L_m):
    Averaged over all transformer layers l = 1..L for deep supervision.
    Three terms per layer:
      (1) NLL of assignment at GT correspondences: log(P_ij)
      (2) Unmatchable keypoints in image t:   log(1 - sigma_i)  for i in K_bar_t
      (3) Unmatchable keypoints in image t+1: log(1 - sigma_j)  for j in K_bar_t+1

    L_m = -(1/L) * Sum_l [
        (1/|M|) * Sum_{(i,j) in M} log(P_ij^l)
      + (1/(2*|K_bar_t|)) * Sum_{i in K_bar_t} log(1 - sigma_i^l)
      + (1/(2*|K_bar_t+1|)) * Sum_{j in K_bar_t+1} log(1 - sigma_j^l)
    ]

Equation 13 -- Pose Loss (L_p):
    L_p = lambda_t * ||t_hat/max(||t_hat||, eps) - t_gt/max(||t_gt||, eps)||
        + lambda_r * ||Log(R_hat) - Log(R_gt)||

    where eps = 1e-6, lambda_r = 180, lambda_t = 400.

Equation 14 -- Total Loss:
    L_total = (1 - lambda_p) * L_m + lambda_p * L_p

    Training schedule:
        Epochs 1-4:   lambda_p = 0.0 (matching loss only)
        Epochs 5-14:  lambda_p increases 0.0 -> 0.9 (increment 1.5e-4 per step)
"""

import torch
import torch.nn as nn
from typing import List, Optional


def _log_rotation(R: torch.Tensor) -> torch.Tensor:
    """Compute the matrix logarithm of a rotation matrix (SO(3) -> so(3)).

    Uses the Rodrigues formula to extract the rotation vector.

    Args:
        R: (B, 3, 3) -- rotation matrix.

    Returns:
        (B, 3) -- rotation vector (axis * angle).
    """
    # Rotation angle: cos(theta) = (trace(R) - 1) / 2
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_angle)  # (B,)

    sin_theta = torch.sin(theta).clamp(min=1e-7)

    # Skew-symmetric part: (R - R^T) / 2
    skew = (R - R.transpose(-1, -2)) / 2.0  # (B, 3, 3)

    # Scale factor: theta / (2 * sin(theta))
    scale = theta / (2.0 * sin_theta)  # (B,)

    # Extract rotation vector from skew-symmetric matrix
    wx = skew[:, 2, 1]  # (B,)
    wy = skew[:, 0, 2]  # (B,)
    wz = skew[:, 1, 0]  # (B,)

    rot_vec = torch.stack([wx, wy, wz], dim=-1)  # (B, 3)
    rot_vec = rot_vec * scale.unsqueeze(-1)  # (B, 3)

    return rot_vec


class MatchingLoss(nn.Module):
    """Matching loss (Eq. 12): full formulation with deep supervision.

    Three terms per layer, averaged over all L layers:
      1. NLL at GT correspondences
      2. Unmatchable penalty for keypoints in image t without a match
      3. Unmatchable penalty for keypoints in image t+1 without a match

    Args:
        eps: Small constant to avoid log(0).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        all_assignments: List[torch.Tensor],
        all_sigma1: List[torch.Tensor],
        all_sigma2: List[torch.Tensor],
        gt_matches: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_assignments: list of L tensors, each (B, K, K) -- P at each layer.
            all_sigma1:      list of L tensors, each (B, K, 1) -- matchability img1.
            all_sigma2:      list of L tensors, each (B, K, 1) -- matchability img2.
            gt_matches:      (B, M, 2) -- GT correspondence indices (i, j).
                             Padded with -1 for invalid entries.
            gt_mask:         (B, M) -- boolean mask, True for valid GT matches.

        Returns:
            loss: scalar -- mean NLL over layers and valid matches.
        """
        L = len(all_assignments)
        B = all_assignments[0].shape[0]
        K = all_assignments[0].shape[1]
        device = all_assignments[0].device

        total_layer_loss = torch.tensor(0.0, device=device)

        for l_idx in range(L):
            P_l = all_assignments[l_idx]      # (B, K, K)
            sigma1_l = all_sigma1[l_idx]      # (B, K, 1)
            sigma2_l = all_sigma2[l_idx]      # (B, K, 1)

            layer_loss = torch.tensor(0.0, device=device)
            count = 0

            for b in range(B):
                valid = gt_mask[b]  # (M,)
                if valid.sum() == 0:
                    continue

                indices = gt_matches[b, valid]  # (N_valid, 2)
                i_idx = indices[:, 0].long()
                j_idx = indices[:, 1].long()

                # --- Term 1: NLL at GT correspondences ---
                p_values = P_l[b, i_idx, j_idx]  # (N_valid,)
                nll_match = -torch.log(p_values + self.eps).mean()

                # --- Term 2 & 3: Unmatchable keypoint penalties ---
                # K_bar_t: keypoints in image 1 that have NO GT match
                matched_i = torch.zeros(K, dtype=torch.bool, device=device)
                matched_i[i_idx] = True
                unmatched_i = ~matched_i  # (K,)

                # K_bar_t+1: keypoints in image 2 that have NO GT match
                matched_j = torch.zeros(K, dtype=torch.bool, device=device)
                matched_j[j_idx] = True
                unmatched_j = ~matched_j  # (K,)

                # sigma values for unmatchable keypoints
                sig1 = sigma1_l[b, :, 0]  # (K,)
                sig2 = sigma2_l[b, :, 0]  # (K,)

                # log(1 - sigma) for unmatchable keypoints (Eq. 12 terms 2 & 3)
                nll_unmatch1 = torch.tensor(0.0, device=device)
                nll_unmatch2 = torch.tensor(0.0, device=device)

                n_unmatched_i = unmatched_i.sum()
                if n_unmatched_i > 0:
                    nll_unmatch1 = -torch.log(
                        1.0 - sig1[unmatched_i] + self.eps
                    ).mean()

                n_unmatched_j = unmatched_j.sum()
                if n_unmatched_j > 0:
                    nll_unmatch2 = -torch.log(
                        1.0 - sig2[unmatched_j] + self.eps
                    ).mean()

                # Combine three terms (Eq. 12 inside the layer sum)
                layer_loss = layer_loss + nll_match + 0.5 * nll_unmatch1 + 0.5 * nll_unmatch2
                count += 1

            if count > 0:
                total_layer_loss = total_layer_loss + layer_loss / count

        # Average over L layers (Eq. 12: -1/L * Sum_l)
        if L > 0:
            total_layer_loss = total_layer_loss / L

        return total_layer_loss


class PoseLoss(nn.Module):
    """Pose loss (Eq. 13): rotation + translation error.

    L_p = lambda_r * ||log(R_hat) - log(R_gt)||
        + lambda_t * ||t_hat/max(||t_hat||, eps) - t_gt/max(||t_gt||, eps)||

    where eps = 1e-6 (paper Eq. 13).

    Args:
        lambda_r: Weight for rotation error (default 180, from paper).
        lambda_t: Weight for translation error (default 400, from paper).
        eps: Numerical stability for translation normalization (1e-6, from paper).
    """

    def __init__(
        self,
        lambda_r: float = 180.0,
        lambda_t: float = 400.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_t = lambda_t
        self.eps = eps

    def forward(
        self,
        R_est: torch.Tensor,
        t_est: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            R_est: (B, 3, 3) -- estimated rotation from Phase 6.
            t_est: (B, 3)    -- estimated translation.
            R_gt:  (B, 3, 3) -- ground truth rotation.
            t_gt:  (B, 3)    -- ground truth translation.

        Returns:
            loss: scalar -- weighted sum of rotation and translation errors.
        """
        # Rotation error: ||log(R_est) - log(R_gt)||
        log_R_est = _log_rotation(R_est)  # (B, 3)
        log_R_gt = _log_rotation(R_gt)    # (B, 3)
        rot_err = (log_R_est - log_R_gt).norm(dim=-1)  # (B,)

        # Translation error: ||t_hat/max(||t_hat||, eps) - t_gt/max(||t_gt||, eps)||
        # Paper Eq. 13: uses max(||t||, eps) with eps=1e-6
        t_est_unit = t_est / torch.clamp(t_est.norm(dim=-1, keepdim=True), min=self.eps)
        t_gt_unit = t_gt / torch.clamp(t_gt.norm(dim=-1, keepdim=True), min=self.eps)
        trans_err = (t_est_unit - t_gt_unit).norm(dim=-1)  # (B,)

        # Eq. 13: weighted sum (note: paper puts lambda_t first)
        loss = self.lambda_r * rot_err + self.lambda_t * trans_err

        return loss.mean()


class DinoVOLoss(nn.Module):
    """Combined DINO-VO loss (Eq. 14).

    L_total = (1 - lambda_p) * L_matching + lambda_p * L_pose

    Training schedule (paper Section IV-A):
        Epochs 1-4:   lambda_p = 0.0 (matching loss only)
        Epochs 5-14:  lambda_p ramps 0.0 -> 0.9 (increment 1.5e-4 per step)

    Args:
        lambda_r: Rotation loss weight (180).
        lambda_t: Translation loss weight (400).
        lambda_p_increment: Per-step increment for lambda_p (1.5e-4).
        lambda_p_max: Maximum value of lambda_p (0.9).
    """

    def __init__(
        self,
        lambda_r: float = 180.0,
        lambda_t: float = 400.0,
        lambda_p_increment: float = 1.5e-4,
        lambda_p_max: float = 0.9,
    ):
        super().__init__()

        self.matching_loss = MatchingLoss()
        self.pose_loss = PoseLoss(lambda_r=lambda_r, lambda_t=lambda_t)

        self.lambda_p_increment = lambda_p_increment
        self.lambda_p_max = lambda_p_max

        # Current lambda_p value (updated externally by the training loop)
        self.register_buffer(
            "lambda_p", torch.tensor(0.0, dtype=torch.float32)
        )

    def step_lambda_p(self) -> float:
        """Increment lambda_p by one step. Called once per training step.

        Returns:
            Current lambda_p value after increment.
        """
        new_val = min(
            self.lambda_p.item() + self.lambda_p_increment,
            self.lambda_p_max,
        )
        self.lambda_p.fill_(new_val)
        return new_val

    def set_lambda_p(self, value: float) -> None:
        """Directly set lambda_p (e.g., to 0.0 for first 4 epochs)."""
        self.lambda_p.fill_(min(value, self.lambda_p_max))

    def forward(
        self,
        all_assignments: List[torch.Tensor],
        all_sigma1: List[torch.Tensor],
        all_sigma2: List[torch.Tensor],
        gt_matches: torch.Tensor,
        gt_mask: torch.Tensor,
        R_est: torch.Tensor,
        t_est: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            all_assignments: list of L tensors (B, K, K) -- P at each layer.
            all_sigma1:      list of L tensors (B, K, 1) -- matchability img1.
            all_sigma2:      list of L tensors (B, K, 1) -- matchability img2.
            gt_matches:      (B, M, 2) -- GT correspondence indices.
            gt_mask:         (B, M) -- valid GT match mask.
            R_est:           (B, 3, 3) -- estimated rotation.
            t_est:           (B, 3) -- estimated translation.
            R_gt:            (B, 3, 3) -- GT rotation.
            t_gt:            (B, 3) -- GT translation.

        Returns:
            dict with:
                'total':    scalar -- combined loss (Eq. 14)
                'matching': scalar -- matching loss (Eq. 12)
                'pose':     scalar -- pose loss (Eq. 13)
                'lambda_p': float  -- current lambda_p value
        """
        lp = self.lambda_p.item()

        # Matching loss (Eq. 12) -- always computed, with deep supervision
        L_m = self.matching_loss(
            all_assignments, all_sigma1, all_sigma2, gt_matches, gt_mask
        )

        # Pose loss (Eq. 13) -- only meaningful when lambda_p > 0
        if lp > 0:
            L_p = self.pose_loss(R_est, t_est, R_gt, t_gt)
        else:
            L_p = torch.tensor(0.0, device=all_assignments[0].device)

        # Combined loss (Eq. 14)
        L_total = (1.0 - lp) * L_m + lp * L_p

        return {
            "total": L_total,
            "matching": L_m,
            "pose": L_p,
            "lambda_p": lp,
        }
