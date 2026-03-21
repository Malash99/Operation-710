"""
Differentiable Pose Estimation for DINO-VO.

Paper Section III-D: Recovers the relative camera pose (R, t) from
weighted keypoint correspondences using the Essential matrix.

Pipeline:
    1. Convert matched pixel coordinates to normalized camera coordinates
       using the camera intrinsics K.
    2. Build the weighted epipolar constraint system (Eq. 10-11).
    3. Solve via SVD for the Essential matrix E (weighted 8-point algorithm).
    4. Decompose E into four (R, t) candidates.
    5. Select the correct (R, t) via cheirality check (positive depth).

Equations (paper):
    Eq. 10:  x_j^T E x_i = 0        (epipolar constraint)
    Eq. 11:  diag(w) Phi flat(E) = 0  (weighted linear system)

Key property: The entire pipeline is differentiable — gradients flow
from the pose loss (Eq. 13) back through SVD, through the weights,
and into the matching transformer.

Input:
    kp1, kp2:    (B, M, 2) matched keypoint pixel coordinates
    weights:     (B, M)    per-match confidence weights from Phase 5
    intrinsics:  (B, 3, 3) camera intrinsic matrix K

Output:
    R: (B, 3, 3) rotation matrix (SO(3))
    t: (B, 3)    unit translation vector (up to scale)
"""

import torch
import torch.nn as nn


class PoseEstimation(nn.Module):
    """Differentiable pose estimation via weighted 8-point algorithm.

    No learnable parameters — this is a geometric layer. Gradients
    flow through the SVD and the confidence weights.
    """

    def __init__(self):
        super().__init__()

    def _pixel_to_normalized(
        self,
        kp: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Convert pixel coordinates to normalized camera coordinates.

        x_norm = K^{-1} @ [u, v, 1]^T

        Args:
            kp: (B, M, 2) — pixel coordinates (u, v).
            K:  (B, 3, 3) — camera intrinsic matrix.

        Returns:
            (B, M, 3) — normalized coordinates [x, y, 1].
        """
        B, M, _ = kp.shape

        # Extract intrinsic parameters
        fx = K[:, 0, 0]  # (B,)
        fy = K[:, 1, 1]  # (B,)
        cx = K[:, 0, 2]  # (B,)
        cy = K[:, 1, 2]  # (B,)

        # Normalize: x = (u - cx) / fx,  y = (v - cy) / fy
        x = (kp[..., 0] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # (B, M)
        y = (kp[..., 1] - cy.unsqueeze(1)) / fy.unsqueeze(1)  # (B, M)
        ones = torch.ones_like(x)

        return torch.stack([x, y, ones], dim=-1)  # (B, M, 3)

    def _build_epipolar_constraint(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Build the epipolar constraint matrix Phi (Eq. 11).

        For each correspondence (x1_i, x2_i), the epipolar constraint is:
            x2^T E x1 = 0

        This can be written as:
            phi_i^T @ flat(E) = 0

        where phi_i is the Kronecker product of x1 and x2.

        Args:
            x1: (B, M, 3) — normalized coords in image 1.
            x2: (B, M, 3) — normalized coords in image 2.

        Returns:
            Phi: (B, M, 9) — constraint matrix.
        """
        # x1 = [x1, y1, 1], x2 = [x2, y2, 1]
        # phi_i = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        # This is the Kronecker product x1 (x) x2, flattened
        x1_x = x1[..., 0]  # (B, M)
        x1_y = x1[..., 1]
        x2_x = x2[..., 0]
        x2_y = x2[..., 1]

        Phi = torch.stack([
            x1_x * x2_x,   # e11 coefficient
            x1_y * x2_x,   # e21 coefficient
            x2_x,           # e31 coefficient
            x1_x * x2_y,   # e12 coefficient
            x1_y * x2_y,   # e22 coefficient
            x2_y,           # e32 coefficient
            x1_x,           # e13 coefficient
            x1_y,           # e23 coefficient
            torch.ones_like(x1_x),  # e33 coefficient
        ], dim=-1)  # (B, M, 9)

        return Phi

    def _weighted_eight_point(
        self,
        Phi: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the weighted 8-point algorithm for the Essential matrix (Eq. 11).

        Minimizes:  || diag(w) @ Phi @ flat(E) ||^2
        Solution:   flat(E) = last right singular vector of (diag(sqrt(w)) @ Phi)

        The SVD is differentiable in PyTorch, allowing gradients to flow
        from the pose loss back through the weights.

        Args:
            Phi:     (B, M, 9) — epipolar constraint matrix.
            weights: (B, M)    — per-match confidence weights.

        Returns:
            E: (B, 3, 3) — estimated Essential matrix.
        """
        # Apply weights: scale each row of Phi by sqrt(w_i)
        # Using sqrt so that ||diag(sqrt(w)) @ Phi @ e||^2 = e^T Phi^T diag(w) Phi e
        w_sqrt = weights.sqrt().unsqueeze(-1)  # (B, M, 1)
        Phi_w = Phi * w_sqrt  # (B, M, 9)

        # SVD of weighted constraint matrix
        # U: (B, M, M), S: (B, min(M,9)), Vt: (B, 9, 9)
        _, _, Vt = torch.linalg.svd(Phi_w, full_matrices=True)

        # Essential matrix = last column of V = last row of Vt
        E_flat = Vt[:, -1, :]  # (B, 9)
        E = E_flat.view(-1, 3, 3)  # (B, 3, 3)

        return E

    def _enforce_essential_constraint(
        self, E: torch.Tensor,
    ) -> torch.Tensor:
        """Project E onto the Essential matrix manifold.

        An Essential matrix has two equal singular values and one zero:
            E = U @ diag(1, 1, 0) @ Vt

        This projection ensures E is a valid Essential matrix,
        which is required for the decomposition into (R, t).

        Args:
            E: (B, 3, 3) — raw Essential matrix from 8-point.

        Returns:
            E_proj: (B, 3, 3) — projected Essential matrix.
        """
        U, S, Vt = torch.linalg.svd(E)

        # Force singular values to (1, 1, 0)
        # Use mean of top-2 singular values for numerical stability
        s_mean = (S[:, 0] + S[:, 1]) / 2.0  # (B,)
        S_new = torch.zeros_like(S)
        S_new[:, 0] = s_mean
        S_new[:, 1] = s_mean
        # S_new[:, 2] = 0  (already zero)

        # Reconstruct
        E_proj = U @ torch.diag_embed(S_new) @ Vt

        return E_proj

    def _decompose_essential(
        self, E: torch.Tensor,
    ) -> tuple:
        """Decompose Essential matrix into four (R, t) candidates.

        E = U @ diag(1, 1, 0) @ Vt

        The four solutions are:
            R1 = U @ W   @ Vt,  t1 = +U[:, 2]
            R2 = U @ W   @ Vt,  t2 = -U[:, 2]
            R3 = U @ W^T @ Vt,  t3 = +U[:, 2]
            R4 = U @ W^T @ Vt,  t4 = -U[:, 2]

        where W = [[0,-1,0],[1,0,0],[0,0,1]] is a 90-degree rotation.

        Args:
            E: (B, 3, 3) — Essential matrix (should satisfy essential constraint).

        Returns:
            R_candidates: (B, 4, 3, 3) — four rotation candidates.
            t_candidates: (B, 4, 3)     — four translation candidates.
        """
        U, _, Vt = torch.linalg.svd(E)

        # Ensure proper rotation (det = +1)
        # If det(U) < 0, flip sign of last column
        det_U = torch.linalg.det(U)
        U = U * torch.where(det_U < 0, -1.0, 1.0).unsqueeze(-1).unsqueeze(-1)

        det_Vt = torch.linalg.det(Vt)
        Vt = Vt * torch.where(det_Vt < 0, -1.0, 1.0).unsqueeze(-1).unsqueeze(-1)

        # W matrix: 90-degree rotation around z-axis
        W = torch.tensor(
            [[0.0, -1.0, 0.0],
             [1.0,  0.0, 0.0],
             [0.0,  0.0, 1.0]],
            device=E.device, dtype=E.dtype,
        ).unsqueeze(0).expand(E.shape[0], -1, -1)  # (B, 3, 3)

        Wt = W.transpose(-1, -2)  # W^T

        # Two rotation candidates
        R1 = U @ W @ Vt    # (B, 3, 3)
        R2 = U @ Wt @ Vt   # (B, 3, 3)

        # Translation is the last column of U (up to sign)
        t_pos = U[:, :, 2]   # (B, 3)
        t_neg = -U[:, :, 2]  # (B, 3)

        # Stack four candidates: (R1,+t), (R1,-t), (R2,+t), (R2,-t)
        R_candidates = torch.stack([R1, R1, R2, R2], dim=1)  # (B, 4, 3, 3)
        t_candidates = torch.stack([t_pos, t_neg, t_pos, t_neg], dim=1)  # (B, 4, 3)

        return R_candidates, t_candidates

    def _cheirality_check(
        self,
        R_candidates: torch.Tensor,
        t_candidates: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple:
        """Select the (R, t) pair where most 3D points have positive depth.

        For each candidate (R, t), triangulate points and count how many
        have z > 0 in both camera frames. The candidate with the most
        positive-depth points is selected.

        Args:
            R_candidates: (B, 4, 3, 3) — rotation candidates.
            t_candidates: (B, 4, 3)    — translation candidates.
            x1: (B, M, 3) — normalized coords in image 1.
            x2: (B, M, 3) — normalized coords in image 2.

        Returns:
            R_best: (B, 3, 3) — selected rotation.
            t_best: (B, 3)    — selected translation (unit norm).
        """
        B = R_candidates.shape[0]
        best_counts = torch.zeros(B, dtype=torch.long, device=R_candidates.device)
        best_idx = torch.zeros(B, dtype=torch.long, device=R_candidates.device)

        for c in range(4):
            R = R_candidates[:, c]  # (B, 3, 3)
            t = t_candidates[:, c]  # (B, 3)

            # Triangulate using the DLT (linear) method
            # Camera 1: P1 = [I | 0],  Camera 2: P2 = [R | t]
            # For each point, check if depth > 0 in both cameras

            # Depth in camera 1 frame:
            # Using the relation: depth1 * x1 = X  (3D point in cam1)
            # and depth2 * x2 = R @ X + t  (3D point in cam2)
            #
            # From the second: X = R^T @ (depth2 * x2 - t)
            # Substituting: depth1 * x1 = R^T @ (depth2 * x2 - t)
            #
            # Cross product elimination to solve for depths:
            # x2 x (R @ x1 * d1 + t) = 0  →  solve for d1
            # We use a simplified check: compute depth via the z-component

            # Method: for each point, compute the 3D position via mid-point
            # triangulation and check z > 0 in both frames.

            # Simplified approach: use the constraint that z-depth must be
            # positive in both cameras. Compute depth in cam1 via:
            #   [x2]x @ (R @ x1 @ d1 + t) = 0
            # where [x2]x is the skew-symmetric matrix of x2.

            # For efficiency, use a batch-compatible approach:
            # depth1 = -(t x x2) . (R@x1 x x2) / |R@x1 x x2|^2

            Rx1 = (R.unsqueeze(1) @ x1.unsqueeze(-1)).squeeze(-1)  # (B, M, 3)

            # Cross products
            Rx1_cross_x2 = torch.cross(Rx1, x2, dim=-1)  # (B, M, 3)
            t_cross_x2 = torch.cross(
                t.unsqueeze(1).expand_as(x2), x2, dim=-1
            )  # (B, M, 3)

            # Depth in camera 1
            num = (t_cross_x2 * Rx1_cross_x2).sum(dim=-1)      # (B, M)
            denom = (Rx1_cross_x2 * Rx1_cross_x2).sum(dim=-1)  # (B, M)
            depth1 = num / (denom + 1e-8)                        # (B, M)

            # Depth in camera 2: X = depth1 * x1, then X_cam2 = R @ X + t
            X = depth1.unsqueeze(-1) * x1  # (B, M, 3)
            X_cam2 = (R.unsqueeze(1) @ X.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1)
            depth2 = X_cam2[..., 2]  # (B, M)

            # Count points with positive depth in both cameras
            positive = (depth1 > 0) & (depth2 > 0)  # (B, M)
            count = positive.sum(dim=-1)  # (B,)

            # Update best
            improved = count > best_counts
            best_counts = torch.where(improved, count, best_counts)
            best_idx = torch.where(
                improved,
                torch.tensor(c, device=R_candidates.device),
                best_idx,
            )

        # Gather best R and t
        # best_idx: (B,) indices into dim=1 of candidates
        b_idx = torch.arange(B, device=R_candidates.device)
        R_best = R_candidates[b_idx, best_idx]  # (B, 3, 3)
        t_best = t_candidates[b_idx, best_idx]  # (B, 3)

        # Normalize translation to unit vector
        t_best = t_best / (t_best.norm(dim=-1, keepdim=True) + 1e-8)

        return R_best, t_best

    def forward(
        self,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        weights: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        """Estimate relative pose from weighted correspondences.

        Args:
            kp1:        (B, M, 2) — matched keypoint pixel coords in image 1.
            kp2:        (B, M, 2) — matched keypoint pixel coords in image 2.
            weights:    (B, M)    — per-match confidence weights from Phase 5.
            intrinsics: (B, 3, 3) — camera intrinsic matrix K.

        Returns:
            dict with:
                'R':         (B, 3, 3) — estimated rotation matrix
                't':         (B, 3)    — estimated unit translation vector
                'E':         (B, 3, 3) — Essential matrix (projected)
                'E_raw':     (B, 3, 3) — Essential matrix (before projection)
        """
        # Step 1: Pixel → normalized camera coordinates
        x1 = self._pixel_to_normalized(kp1, intrinsics)  # (B, M, 3)
        x2 = self._pixel_to_normalized(kp2, intrinsics)  # (B, M, 3)

        # Step 2: Build epipolar constraint matrix (Eq. 11)
        Phi = self._build_epipolar_constraint(x1, x2)  # (B, M, 9)

        # Step 3: Weighted 8-point algorithm → Essential matrix
        E_raw = self._weighted_eight_point(Phi, weights)  # (B, 3, 3)

        # Step 4: Project onto Essential manifold (enforce rank-2, equal SVs)
        E = self._enforce_essential_constraint(E_raw)  # (B, 3, 3)

        # Step 5: Decompose E → four (R, t) candidates
        R_candidates, t_candidates = self._decompose_essential(E)

        # Step 6: Cheirality check → select best (R, t)
        R, t = self._cheirality_check(R_candidates, t_candidates, x1, x2)

        return {
            "R": R,           # (B, 3, 3)
            "t": t,           # (B, 3)
            "E": E,           # (B, 3, 3)
            "E_raw": E_raw,   # (B, 3, 3)
        }
