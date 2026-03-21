"""
Loss functions for DINO-VO training.

Paper Section III-E: Two loss terms combined with a scheduling weight.

Equation 12 — Matching Loss (L_m):
    Negative log-likelihood of the assignment matrix P at ground truth
    correspondence locations. Encourages the matching transformer to
    assign high probability to correct matches.

    L_m = -(1/|M|) * Sum_{(i,j) in M} log(P_ij^l)

    Averaged over all transformer layers l = 1..L for deep supervision.

Equation 13 — Pose Loss (L_p):
    Compares estimated rotation R_hat and translation t_hat with ground
    truth. Uses geodesic rotation distance and normalized translation error.

    L_p = lambda_r * ||log(R_hat) - log(R_gt)||
        + lambda_t * ||t_hat/||t_hat|| - t_gt/||t_gt||||

    where lambda_r = 180, lambda_t = 400.

Equation 14 — Total Loss:
    L_total = (1 - lambda_p) * L_m + lambda_p * L_p

    Training schedule:
        Epochs 1-4:   lambda_p = 0.0 (matching loss only)
        Epochs 5-14:  lambda_p increases 0.0 -> 0.9 (increment 1.5e-4 per step)
"""

import torch
import torch.nn as nn


def _log_rotation(R: torch.Tensor) -> torch.Tensor:
    """Compute the matrix logarithm of a rotation matrix (SO(3) -> so(3)).

    Uses the Rodrigues formula to extract the rotation vector.

    For R with rotation angle theta:
        log(R) = (theta / (2 * sin(theta))) * (R - R^T)

    The output is the skew-symmetric matrix; we return the 3-vector.

    Args:
        R: (B, 3, 3) — rotation matrix.

    Returns:
        (B, 3) — rotation vector (axis * angle).
    """
    # Rotation angle: cos(theta) = (trace(R) - 1) / 2
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_angle)  # (B,)

    # Handle small angles (theta ~ 0): log(R) ~ (R - R^T) / 2
    # Handle near-pi angles carefully
    sin_theta = torch.sin(theta).clamp(min=1e-7)  # avoid division by zero

    # Skew-symmetric part: (R - R^T) / 2
    skew = (R - R.transpose(-1, -2)) / 2.0  # (B, 3, 3)

    # Scale factor: theta / (2 * sin(theta))
    # For small theta, this approaches 0.5
    scale = theta / (2.0 * sin_theta)  # (B,)

    # Extract rotation vector from skew-symmetric matrix
    # skew = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
    wx = skew[:, 2, 1]  # (B,)
    wy = skew[:, 0, 2]  # (B,)
    wz = skew[:, 1, 0]  # (B,)

    rot_vec = torch.stack([wx, wy, wz], dim=-1)  # (B, 3)
    rot_vec = rot_vec * scale.unsqueeze(-1)  # (B, 3)

    return rot_vec


class MatchingLoss(nn.Module):
    """Matching loss (Eq. 12): negative log-likelihood of assignment matrix.

    Supervises the matching transformer to assign high probability
    to ground truth correspondences.

    For DINO-VO, ground truth correspondences are obtained by reprojecting
    keypoints using the known relative pose and camera intrinsics, then
    finding the nearest keypoint in the other image.

    Args:
        eps: Small constant to avoid log(0).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        assignment: torch.Tensor,
        gt_matches: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            assignment: (B, K, K) — soft assignment matrix P from Phase 5.
            gt_matches: (B, M, 2) — ground truth correspondence indices (i, j).
                        Padded with -1 for invalid entries.
            gt_mask:    (B, M) — boolean mask, True for valid GT matches.

        Returns:
            loss: scalar — mean negative log-likelihood over valid matches.
        """
        B = assignment.shape[0]
        total_loss = torch.tensor(0.0, device=assignment.device)
        count = 0

        for b in range(B):
            valid = gt_mask[b]  # (M,)
            if valid.sum() == 0:
                continue

            indices = gt_matches[b, valid]  # (N_valid, 2)
            i_idx = indices[:, 0].long()
            j_idx = indices[:, 1].long()

            # Extract P[i, j] for each GT correspondence
            p_values = assignment[b, i_idx, j_idx]  # (N_valid,)

            # Negative log-likelihood (Eq. 12)
            nll = -torch.log(p_values + self.eps)
            total_loss = total_loss + nll.mean()
            count += 1

        if count == 0:
            return total_loss  # return 0 if no valid matches

        return total_loss / count


class PoseLoss(nn.Module):
    """Pose loss (Eq. 13): rotation + translation error.

    L_p = lambda_r * ||log(R_hat) - log(R_gt)||
        + lambda_t * ||t_hat/||t_hat|| - t_gt/||t_gt||||

    Args:
        lambda_r: Weight for rotation error (default 180, from paper).
        lambda_t: Weight for translation error (default 400, from paper).
    """

    def __init__(self, lambda_r: float = 180.0, lambda_t: float = 400.0):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_t = lambda_t

    def forward(
        self,
        R_est: torch.Tensor,
        t_est: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            R_est: (B, 3, 3) — estimated rotation from Phase 6.
            t_est: (B, 3)    — estimated translation (unit vector).
            R_gt:  (B, 3, 3) — ground truth rotation.
            t_gt:  (B, 3)    — ground truth translation.

        Returns:
            loss: scalar — weighted sum of rotation and translation errors.
        """
        # Rotation error: ||log(R_est) - log(R_gt)||
        log_R_est = _log_rotation(R_est)  # (B, 3)
        log_R_gt = _log_rotation(R_gt)    # (B, 3)
        rot_err = (log_R_est - log_R_gt).norm(dim=-1)  # (B,)

        # Translation error: ||t_est_unit - t_gt_unit||
        t_est_unit = t_est / (t_est.norm(dim=-1, keepdim=True) + 1e-8)
        t_gt_unit = t_gt / (t_gt.norm(dim=-1, keepdim=True) + 1e-8)
        trans_err = (t_est_unit - t_gt_unit).norm(dim=-1)  # (B,)

        # Eq. 13: weighted sum
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
        assignment: torch.Tensor,
        gt_matches: torch.Tensor,
        gt_mask: torch.Tensor,
        R_est: torch.Tensor,
        t_est: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            assignment: (B, K, K) — soft assignment matrix P.
            gt_matches: (B, M, 2) — GT correspondence indices.
            gt_mask:    (B, M) — valid GT match mask.
            R_est:      (B, 3, 3) — estimated rotation.
            t_est:      (B, 3) — estimated translation.
            R_gt:       (B, 3, 3) — GT rotation.
            t_gt:       (B, 3) — GT translation.

        Returns:
            dict with:
                'total':    scalar — combined loss (Eq. 14)
                'matching': scalar — matching loss (Eq. 12)
                'pose':     scalar — pose loss (Eq. 13)
                'lambda_p': float  — current lambda_p value
        """
        lp = self.lambda_p.item()

        # Matching loss (Eq. 12) — always computed
        L_m = self.matching_loss(assignment, gt_matches, gt_mask)

        # Pose loss (Eq. 13) — only meaningful when lambda_p > 0
        if lp > 0:
            L_p = self.pose_loss(R_est, t_est, R_gt, t_gt)
        else:
            L_p = torch.tensor(0.0, device=assignment.device)

        # Combined loss (Eq. 14)
        L_total = (1.0 - lp) * L_m + lp * L_p

        return {
            "total": L_total,
            "matching": L_m,
            "pose": L_p,
            "lambda_p": lp,
        }
