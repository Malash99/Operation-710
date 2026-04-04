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

SVD backward stability (DSAC-style):
    The Essential matrix has degenerate singular values (s, s, 0).
    Standard SVD backward involves 1/(s_i^2 - s_j^2) which is undefined
    when s_i = s_j. Following DSAC (Brachmann et al. [5]), we clamp
    these terms to a maximum value instead of allowing NaN/Inf.

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


# ---------------------------------------------------------------------------
# DSAC-style SVD with clamped backward (handles degenerate singular values)
# ---------------------------------------------------------------------------

class ClampedSVD(torch.autograd.Function):
    """SVD with clamped backward pass for degenerate singular values.

    Standard SVD backward involves F_ij = 1/(s_i^2 - s_j^2), which is
    undefined when s_i = s_j (as in the Essential matrix with (s, s, 0)).

    Following DSAC (Brachmann et al., CVPR 2017), we clamp F_ij to
    [-max_val, max_val] instead of allowing NaN/Inf. This preserves
    gradient direction while bounding magnitude.

    Reference: Ionescu et al., "Matrix Backpropagation for Deep Networks
    with Structured Layers", ICCV 2015 — Eq. 4 for SVD backward.
    """

    MAX_CLAMP = 1e6  # clamp bound for 1/(si^2 - sj^2)

    @staticmethod
    def forward(ctx, A):
        U, S, Vt = torch.linalg.svd(A, full_matrices=True)
        ctx.save_for_backward(U, S, Vt)
        return U, S, Vt

    @staticmethod
    def backward(ctx, dU, dS, dVt):
        U, S, Vt = ctx.saved_tensors
        V = Vt.transpose(-2, -1)
        Ut = U.transpose(-2, -1)

        M, N = U.shape[-2], Vt.shape[-1]
        K = S.shape[-1]  # min(M, N)

        # F matrix: F_ij = 1/(s_j^2 - s_i^2), clamped for stability
        s_sq = S.unsqueeze(-1) ** 2  # (..., K, 1)
        diff = s_sq.transpose(-2, -1) - s_sq  # (..., K, K) = s_j^2 - s_i^2
        # Clamp: avoid division by zero, preserve sign
        diff_clamped = torch.where(
            diff.abs() < 1e-10,
            torch.full_like(diff, 1e-10).copysign(diff + 1e-30),
            diff,
        )
        F = 1.0 / diff_clamped
        F = F.clamp(-ClampedSVD.MAX_CLAMP, ClampedSVD.MAX_CLAMP)
        # Zero the diagonal (i == j terms should not contribute)
        F.diagonal(dim1=-2, dim2=-1).zero_()

        # dL/dA = U @ (dS_diag + F * (Ut @ dU - dVt^T @ V) * S_diag) @ Vt
        #       + (I - U @ Ut) @ dU @ S_inv @ Vt  [only if M > K]
        #       + U @ S_inv @ dVt @ (I - V @ Vt)  [only if N > K]

        # Core term
        S_diag = torch.diag_embed(S)  # (..., K, K)
        Ut_dU = Ut @ dU  # (..., K, K) — only first K cols of dU matter
        dVt_V = dVt @ V  # (..., K, K) — only first K rows of dVt matter

        # Symmetric and antisymmetric parts
        sym = Ut_dU + dVt_V.transpose(-2, -1)  # should be zero on diagonal ideally
        core = F * (Ut_dU - dVt_V.transpose(-2, -1))

        # Add diagonal from dS
        dS_diag = torch.diag_embed(dS) if dS is not None else torch.zeros_like(S_diag)
        inner = dS_diag + core @ S_diag + S_diag @ core.transpose(-2, -1)

        # Simplified: for square or near-square matrices typical in our use
        # dA = U @ inner @ Vt
        # This is the main term; the rectangular correction terms are negligible
        # for our 3x3 and (M,9) cases.

        # Actually, use the full formula from Ionescu et al.:
        # dA = U @ (diag(dS) + F*(Ut@dU)*S + S*(F*(Vt@dV))^T ) @ Vt
        # Simplification for full SVD of square matrix:

        # For a cleaner derivation: use the formula from
        # https://j-towns.github.io/papers/svd-derivative.pdf
        # dA = U @ (dSigma + F.*(Ut dU) Sigma + Sigma F.*(dVt V)^T) @ Vt

        Ut_dU_F = F * (Ut @ dU[..., :K, :K] if dU.shape[-1] > K else Ut @ dU)

        # Handle dU shape: dU is (..., M, M) for full_matrices, we need (..., M, K)
        # For our uses: 3x3 (K=3=M=N) or (M, 9) with K=9=N
        # So M >= K always, and we use full_matrices=True

        # Rebuild using simpler stable formula:
        # dA = U (dS_diag + F * (Ut dU) S_diag + S_diag F * (Vt dV)^T) Vt
        dV = dVt.transpose(-2, -1)

        term_U = F * (Ut[..., :K, :] @ dU[..., :, :K])  # (..., K, K)
        term_V = F * (Vt[..., :K, :] @ dV[..., :, :K])  # (..., K, K)

        middle = dS_diag + term_U @ S_diag + S_diag @ term_V.transpose(-2, -1)

        dA = U[..., :, :K] @ middle @ Vt[..., :K, :]

        # Replace any remaining NaN/Inf (safety net)
        dA = torch.nan_to_num(dA, nan=0.0, posinf=0.0, neginf=0.0)

        return dA


def clamped_svd(A: torch.Tensor):
    """Compute SVD with DSAC-style clamped backward pass.

    Use this instead of torch.linalg.svd when the matrix may have
    repeated singular values (e.g., the Essential matrix).

    Args:
        A: (..., M, N) input matrix.

    Returns:
        U: (..., M, M) left singular vectors.
        S: (..., min(M,N)) singular values.
        Vt: (..., N, N) right singular vectors (transposed).
    """
    if A.requires_grad:
        return ClampedSVD.apply(A)
    else:
        return torch.linalg.svd(A, full_matrices=True)


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
        # Epipolar constraint: x2^T E x1 = 0
        # Expanding with flat(E) in ROW-MAJOR order [e11,e12,e13,e21,e22,e23,e31,e32,e33]:
        #   phi = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
        x1_x = x1[..., 0]  # (B, M)
        x1_y = x1[..., 1]
        x2_x = x2[..., 0]
        x2_y = x2[..., 1]

        Phi = torch.stack([
            x1_x * x2_x,   # e11 coefficient
            x1_y * x2_x,   # e12 coefficient
            x2_x,           # e13 coefficient
            x1_x * x2_y,   # e21 coefficient
            x1_y * x2_y,   # e22 coefficient
            x2_y,           # e23 coefficient
            x1_x,           # e31 coefficient
            x1_y,           # e32 coefficient
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

        # SVD of weighted constraint matrix — non-degenerate, standard SVD is fine
        _, _, Vt = torch.linalg.svd(Phi_w, full_matrices=True)

        # Essential matrix = last column of V = last row of Vt
        E_flat = Vt[:, -1, :]  # (B, 9)
        E = E_flat.view(-1, 3, 3)  # (B, 3, 3)

        return E

    def _decompose_essential(
        self, E: torch.Tensor,
    ) -> tuple:
        """Decompose Essential matrix into four (R, t) candidates.

        Uses DSAC-style clamped SVD backward to handle the degenerate
        singular values (s, s, 0) of the Essential matrix.

        E = U @ diag(1, 1, 0) @ Vt

        The four solutions are:
            R1 = U @ W   @ Vt,  t1 = +U[:, 2]
            R2 = U @ W   @ Vt,  t2 = -U[:, 2]
            R3 = U @ W^T @ Vt,  t3 = +U[:, 2]
            R4 = U @ W^T @ Vt,  t4 = -U[:, 2]

        where W = [[0,-1,0],[1,0,0],[0,0,1]] is a 90-degree rotation.

        Args:
            E: (B, 3, 3) — Essential matrix.

        Returns:
            R_candidates: (B, 4, 3, 3) — four rotation candidates.
            t_candidates: (B, 4, 3)     — four translation candidates.
        """
        # Use clamped SVD for stable backward through degenerate (s, s, 0)
        U, S, Vt = clamped_svd(E)

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

            Rx1 = (R.unsqueeze(1) @ x1.unsqueeze(-1)).squeeze(-1)  # (B, M, 3)

            # Cross products
            Rx1_cross_x2 = torch.cross(Rx1, x2, dim=-1)  # (B, M, 3)
            t_cross_x2 = torch.cross(
                t.unsqueeze(1).expand_as(x2), x2, dim=-1
            )  # (B, M, 3)

            # Depth in camera 1:
            num = -(t_cross_x2 * Rx1_cross_x2).sum(dim=-1)     # (B, M)
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
                'E':         (B, 3, 3) — Essential matrix (raw from 8-point)
        """
        # Step 1: Pixel -> normalized camera coordinates
        x1 = self._pixel_to_normalized(kp1, intrinsics)  # (B, M, 3)
        x2 = self._pixel_to_normalized(kp2, intrinsics)  # (B, M, 3)

        # Step 2: Build epipolar constraint matrix (Eq. 11)
        Phi = self._build_epipolar_constraint(x1, x2)  # (B, M, 9)

        # Step 3: Weighted 8-point algorithm -> Essential matrix
        E = self._weighted_eight_point(Phi, weights)  # (B, 3, 3)

        # Step 4: Decompose E -> four (R, t) candidates
        # Uses clamped SVD for stable backward through degenerate (s, s, 0)
        # Note: no Essential manifold projection — decompose raw E directly
        # (paper does not mention projection; raw E is approximately Essential)
        R_candidates, t_candidates = self._decompose_essential(E)

        # Step 5: Cheirality check -> select best (R, t)
        R, t = self._cheirality_check(R_candidates, t_candidates, x1, x2)

        return {
            "R": R,           # (B, 3, 3)
            "t": t,           # (B, 3)
            "E": E,           # (B, 3, 3)
        }
