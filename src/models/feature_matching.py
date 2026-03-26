"""
Feature Matching for DINO-VO.

Paper Section III-C: Transformer-based feature matching inspired by LightGlue.
Establishes correspondences between keypoints in an image pair using
alternating self- and cross-attention with rotary positional encoding.

Architecture:
    L = 12 transformer layers, each containing:
        1. Self-attention with 2D Rotary Positional Encoding (within each image)
        2. Cross-attention (between the image pair, key-key dot product per Eq. 4)
        3. Feed-forward update (Eq. 2)
    Followed by:
        - Soft assignment matrix via dual-softmax (Eq. 5-8)
        - Confidence prediction MLP (Eq. 9)

Equations (paper):
    Eq. 2:  f_i^T  <- f_i^T + MLP([f_i^T | m_i^{T<-S}])
    Eq. 3:  m_i^{T<-S} = Sum_j softmax(a_ij)_j * v_j
    Eq. 4:  a_ij = { q_i^T R(p_j-p_i) k_j,   for self-attention
                    { k_i^T k_j,                for cross-attention
    Eq. 5:  P_ij = sigma_i * sigma_j * softmax_row(S)_ij * softmax_col(S)_ij
    Eq. 6:  S_ij = Linear(f_i)^T * Linear(f_j)
    Eq. 7:  sigma_i = sigmoid(Linear(f_i))
    Eq. 9:  w_ij = ConfMLP([f_i | f_j])

Input:
    desc1, desc2: (B, K, 192) descriptors from Phase 4
    kp1, kp2:     (B, K, 2)   keypoint (x, y) pixel coords from Phase 3
    img_h, img_w: image height and width (for normalizing coords to [0,1])

Output:
    assignment:  (B, K, K)   soft assignment matrix P
    matches:     list of (M_b, 2) tensors -- matched index pairs per batch
    weights:     list of (M_b,)  tensors -- confidence weight per match
"""


from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (2D) -- used in self-attention only
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rearrange adjacent pairs for rotary encoding.

    [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    """
    x = x.unflatten(-1, (-1, 2))        # (..., D/2, 2)
    x1, x2 = x.unbind(dim=-1)           # each (..., D/2)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _compute_rope_2d(
    pos: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute 2D rotary positional encoding (cos, sin).

    Paper Eq. 4: p_i := (x_i, y_i) in [0, 1]^2 (normalized coordinates).

    Splits head_dim into two halves:
        - First half  (head_dim/2 dims) encodes the x-coordinate.
        - Second half (head_dim/2 dims) encodes the y-coordinate.

    Args:
        pos: (B, K, 2) -- keypoint (x, y) coordinates, already normalized to [0, 1].
        head_dim: Dimension per attention head (64 in this model).

    Returns:
        cos, sin: each (B, 1, K, head_dim) -- broadcastable over heads.
    """
    half_dim = head_dim // 2  # 32 dims per coordinate axis

    # Frequency bases: one per pair of dimensions -> half_dim/2 values
    freq_idx = torch.arange(0, half_dim, 2, device=pos.device, dtype=pos.dtype)
    freqs = 1.0 / (10000.0 ** (freq_idx / half_dim))  # (half_dim/2,)

    # Compute rotation angles with normalized [0,1] coordinates
    angles_x = pos[..., 0:1] * freqs  # (B, K, half_dim/2)
    angles_y = pos[..., 1:2] * freqs  # (B, K, half_dim/2)

    # Repeat-interleave so each frequency covers a pair of dims
    angles_x = angles_x.repeat_interleave(2, dim=-1)  # (B, K, half_dim)
    angles_y = angles_y.repeat_interleave(2, dim=-1)  # (B, K, half_dim)

    # Concatenate x-encoding and y-encoding -> full head_dim
    angles = torch.cat([angles_x, angles_y], dim=-1)  # (B, K, head_dim)

    # Add head-broadcast dimension: (B, K, D) -> (B, 1, K, D)
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    return cos, sin


# ---------------------------------------------------------------------------
# Multi-Head Attention (Self and Cross variants)
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention with 2D RoPE (Eq. 3-4, self-attention case).

    a_ij = (R(p_i) * q_i)^T (R(p_j) * k_j) / sqrt(d)

    Args:
        dim: Feature dimension (192).
        num_heads: Number of attention heads (3).
        head_dim: Dimension per head (64).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, K, D) -- features for one image.
            rope: (cos, sin) each (B, 1, K, head_dim) -- 2D RoPE.

        Returns:
            (B, K, D)
        """
        B, K, _ = x.shape
        cos, sin = rope

        q = self.q_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE to Q and K (Eq. 4 self-attention case)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, K, K)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, K, -1)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with key-key dot product (Eq. 4, cross case).

    Paper Eq. 4: a_ij = k_i^T k_j (NOT q^T k) for cross-attention.
    No positional encoding is applied in cross-attention.

    Args:
        dim: Feature dimension (192).
        num_heads: Number of attention heads (3).
        head_dim: Dimension per head (64).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target: (B, K, D) -- features of the target image (provides keys for attention scores).
            source: (B, K, D) -- features of the source image (provides keys + values).

        Returns:
            (B, K, D) -- aggregated message from source to target.
        """
        B, K_t, _ = target.shape
        K_s = source.shape[1]

        # Eq. 4 cross-attention: a_ij = k_T_i^T k_S_j (key-key dot product)
        k_target = self.k_proj(target).view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        k_source = self.k_proj(source).view(B, K_s, self.num_heads, self.head_dim).transpose(1, 2)
        v_source = self.v_proj(source).view(B, K_s, self.num_heads, self.head_dim).transpose(1, 2)

        # Key-key attention scores (no RoPE for cross-attention)
        attn = (k_target @ k_source.transpose(-2, -1)) * self.scale  # (B, H, K_t, K_s)
        attn = attn.softmax(dim=-1)
        out = attn @ v_source

        out = out.transpose(1, 2).contiguous().view(B, K_t, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single Transformer Layer
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """One layer of the matching transformer.

    Each layer performs (for both images symmetrically):
        1. Pre-norm self-attention with RoPE (spatial context within image)
        2. Pre-norm cross-attention with key-key dot product (inter-image, Eq. 4)
        3. Feed-forward update with residual (Eq. 2)

    Weights are shared between image 1 and image 2 processing.

    Args:
        dim: Feature dimension (192).
        num_heads: Number of attention heads (3).
        head_dim: Dimension per head (64).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()

        # Self-attention block (with RoPE)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, head_dim)

        # Cross-attention block (key-key, no RoPE, Eq. 4)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, head_dim)

        # Feed-forward network (Eq. 2)
        # Input: [f_i | m_i] (concatenation) -> 2*dim
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        rope1: Tuple[torch.Tensor, torch.Tensor],
        rope2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat1: (B, K, D) -- features for image 1 keypoints
            feat2: (B, K, D) -- features for image 2 keypoints
            rope1: (cos, sin) -- 2D RoPE for image 1 positions
            rope2: (cos, sin) -- 2D RoPE for image 2 positions

        Returns:
            Updated (feat1, feat2), each (B, K, D).
        """
        # --- Self-attention with RoPE (within each image) ---
        f1_norm = self.norm_self(feat1)
        f2_norm = self.norm_self(feat2)

        feat1 = feat1 + self.self_attn(f1_norm, rope=rope1)
        feat2 = feat2 + self.self_attn(f2_norm, rope=rope2)

        # --- Cross-attention with key-key (Eq. 4) ---
        f1_norm = self.norm_cross(feat1)
        f2_norm = self.norm_cross(feat2)

        # Image 1 target attends to image 2 source (key-key)
        m1 = self.cross_attn(target=f1_norm, source=f2_norm)
        # Image 2 target attends to image 1 source (key-key)
        m2 = self.cross_attn(target=f2_norm, source=f1_norm)

        # --- Feed-forward update (Eq. 2) ---
        # f_i <- f_i + MLP([f_i | m_i])
        feat1 = feat1 + self.ffn(torch.cat([feat1, m1], dim=-1))
        feat2 = feat2 + self.ffn(torch.cat([feat2, m2], dim=-1))

        return feat1, feat2


# ---------------------------------------------------------------------------
# Feature Matching Module
# ---------------------------------------------------------------------------


class FeatureMatching(nn.Module):
    """Transformer-based feature matching (Paper Section III-C).

    Processes descriptors from an image pair through L alternating self-
    and cross-attention layers, then produces a soft assignment matrix
    and per-match confidence weights.

    Supports deep supervision: when return_all_assignments=True, returns
    the assignment matrix from every layer (needed for Eq. 12 loss).

    Args:
        descriptor_dim: Input/output descriptor dimension (192).
        num_layers: Number of transformer layers (L=12).
        num_heads: Attention heads per layer (3).
        head_dim: Dimension per attention head (64).
        match_threshold: Minimum assignment score for match extraction.
    """

    def __init__(
        self,
        descriptor_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        head_dim: int = 64,
        match_threshold: float = 0.1,
    ):
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.match_threshold = match_threshold

        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            TransformerLayer(descriptor_dim, num_heads, head_dim)
            for _ in range(num_layers)
        ])

        # Final normalization after all layers
        self.final_norm = nn.LayerNorm(descriptor_dim)

        # --- Assignment head (Eq. 5-8) ---

        # Eq. 6: Score projection -- S_ij = (W*f_i)^T (W*f_j) / sqrt(d)
        self.score_proj = nn.Linear(descriptor_dim, descriptor_dim, bias=False)

        # Eq. 7: Matchability -- sigma_i = sigmoid(W*f_i + b)
        self.matchability = nn.Linear(descriptor_dim, 1)

        # --- Confidence head (Eq. 9) ---
        # w_ij = sigmoid(ConfMLP([f_i | f_j]))
        self.conf_mlp = nn.Sequential(
            nn.Linear(2 * descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, 1),
        )

    def _compute_assignment(
        self, feat1: torch.Tensor, feat2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute soft assignment matrix P, sigma values (Eq. 5-8).

        Args:
            feat1: (B, K, D) -- refined features for image 1.
            feat2: (B, K, D) -- refined features for image 2.

        Returns:
            P: (B, K, K) -- soft assignment probabilities.
            S: (B, K, K) -- raw (scaled) score matrix.
            sigma1: (B, K, 1) -- matchability scores for image 1.
            sigma2: (B, K, 1) -- matchability scores for image 2.
        """
        # Eq. 6: score matrix via projected dot product
        f1_proj = self.score_proj(feat1)  # (B, K, D)
        f2_proj = self.score_proj(feat2)  # (B, K, D)
        S = torch.bmm(f1_proj, f2_proj.transpose(1, 2))  # (B, K, K)

        # Eq. 7-8: per-keypoint matchability
        sigma1 = torch.sigmoid(self.matchability(feat1))  # (B, K, 1)
        sigma2 = torch.sigmoid(self.matchability(feat2))  # (B, K, 1)

        # Eq. 5: dual-softmax assignment weighted by matchability
        P_row = F.softmax(S, dim=-1)   # softmax over j (columns)
        P_col = F.softmax(S, dim=-2)   # softmax over i (rows)

        # P_ij = sigma_i * sigma_j * softmax_row * softmax_col
        P = sigma1 * sigma2.transpose(1, 2) * P_row * P_col  # (B, K, K)

        return P, S, sigma1, sigma2

    def _extract_matches(
        self, P: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract mutual nearest-neighbor matches from assignment matrix.

        A pair (i, j) is a match if:
            - j = argmax_j P[i, :] (best match for i in image 2)
            - i = argmax_i P[:, j] (best match for j in image 1)
            - P[i, j] > match_threshold

        Args:
            P: (B, K1, K2) -- soft assignment matrix.

        Returns:
            matches_list: list of (M_b, 2) long tensors -- (idx1, idx2) pairs.
            scores_list:  list of (M_b,) float tensors -- P[i, j] for each match.
        """
        B, K1, K2 = P.shape

        max_j = P.argmax(dim=-1)  # (B, K1)
        max_i = P.argmax(dim=-2)  # (B, K2)

        i_indices = torch.arange(K1, device=P.device).unsqueeze(0).expand(B, -1)
        mutual = max_i.gather(1, max_j) == i_indices

        scores = P.gather(2, max_j.unsqueeze(-1)).squeeze(-1)
        valid = mutual & (scores > self.match_threshold)

        matches_list = []
        scores_list = []
        for b in range(B):
            valid_idx = valid[b].nonzero(as_tuple=True)[0]
            if valid_idx.shape[0] > 0:
                i_matched = valid_idx
                j_matched = max_j[b, valid_idx]
                matches_list.append(
                    torch.stack([i_matched, j_matched], dim=-1)
                )
                scores_list.append(scores[b, valid_idx])
            else:
                matches_list.append(
                    torch.zeros(0, 2, dtype=torch.long, device=P.device)
                )
                scores_list.append(
                    torch.zeros(0, dtype=P.dtype, device=P.device)
                )

        return matches_list, scores_list

    def _compute_confidence(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        matches_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Compute per-match confidence weights (Eq. 9).

        w_ij = sigmoid(ConfMLP([f_i | f_j]))

        Args:
            feat1: (B, K, D) -- refined features for image 1.
            feat2: (B, K, D) -- refined features for image 2.
            matches_list: list of (M_b, 2) -- matched index pairs per batch.

        Returns:
            weights_list: list of (M_b,) -- confidence in [0, 1] per match.
        """
        weights_list = []
        for b, matches in enumerate(matches_list):
            if matches.shape[0] == 0:
                weights_list.append(
                    torch.zeros(0, dtype=feat1.dtype, device=feat1.device)
                )
                continue

            idx1 = matches[:, 0]
            idx2 = matches[:, 1]
            f1 = feat1[b, idx1]  # (M, D)
            f2 = feat2[b, idx2]  # (M, D)

            w = self.conf_mlp(torch.cat([f1, f2], dim=-1))  # (M, 1)
            w = torch.sigmoid(w.squeeze(-1))                  # (M,)
            weights_list.append(w)

        return weights_list

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        img_h: int = 476,
        img_w: int = 742,
        return_all_assignments: bool = False,
    ) -> Dict[str, object]:
        """Run the full matching pipeline.

        Args:
            desc1: (B, K, 192) -- descriptors for image 1 keypoints.
            desc2: (B, K, 192) -- descriptors for image 2 keypoints.
            kp1:   (B, K, 2)   -- (x, y) pixel coordinates, image 1.
            kp2:   (B, K, 2)   -- (x, y) pixel coordinates, image 2.
            img_h: Image height (for normalizing coords to [0,1] per paper).
            img_w: Image width (for normalizing coords to [0,1] per paper).
            return_all_assignments: If True, return per-layer P and sigma
                                    for deep supervision (Eq. 12).

        Returns:
            dict with:
                'assignment':   (B, K, K)      -- soft assignment matrix P (final layer)
                'score_matrix': (B, K, K)      -- raw score matrix S
                'matches':      list of (M, 2) -- matched index pairs
                'match_scores': list of (M,)   -- assignment score per match
                'weights':      list of (M,)   -- confidence weight per match
                'feat1':        (B, K, D)      -- refined features (for loss)
                'feat2':        (B, K, D)      -- refined features (for loss)
                'sigma1':       (B, K, 1)      -- matchability, image 1 (final)
                'sigma2':       (B, K, 1)      -- matchability, image 2 (final)
            If return_all_assignments:
                'all_assignments': list of (B, K, K) -- P at each layer
                'all_sigma1':      list of (B, K, 1) -- sigma1 at each layer
                'all_sigma2':      list of (B, K, 1) -- sigma2 at each layer
        """
        # Normalize keypoint coordinates to [0, 1] as per paper (Eq. 4, line 419)
        # Paper: p_i := (x_i, y_i) in [0, 1]^2
        kp1_norm = kp1.clone()
        kp1_norm[..., 0] = kp1[..., 0] / (img_w - 1)  # x (column) -> [0, 1]
        kp1_norm[..., 1] = kp1[..., 1] / (img_h - 1)  # y (row)    -> [0, 1]

        kp2_norm = kp2.clone()
        kp2_norm[..., 0] = kp2[..., 0] / (img_w - 1)
        kp2_norm[..., 1] = kp2[..., 1] / (img_h - 1)

        # Precompute 2D rotary positional encoding with normalized coords
        rope1 = _compute_rope_2d(kp1_norm, self.head_dim)
        rope2 = _compute_rope_2d(kp2_norm, self.head_dim)

        # Storage for deep supervision
        all_assignments = [] if return_all_assignments else None
        all_sigma1 = [] if return_all_assignments else None
        all_sigma2 = [] if return_all_assignments else None

        # Process through L=12 transformer layers
        feat1, feat2 = desc1, desc2
        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2, rope1, rope2)

            # Deep supervision: compute assignment at each layer (Eq. 12)
            if return_all_assignments:
                with torch.no_grad() if not self.training else torch.enable_grad():
                    f1_n = self.final_norm(feat1)
                    f2_n = self.final_norm(feat2)
                    P_l, _, s1_l, s2_l = self._compute_assignment(f1_n, f2_n)
                    all_assignments.append(P_l)
                    all_sigma1.append(s1_l)
                    all_sigma2.append(s2_l)

        # Final normalization
        feat1 = self.final_norm(feat1)
        feat2 = self.final_norm(feat2)

        # Soft assignment matrix (Eq. 5-8) from final layer
        P, S, sigma1, sigma2 = self._compute_assignment(feat1, feat2)

        # Extract mutual nearest-neighbor matches
        matches_list, match_scores_list = self._extract_matches(P)

        # Confidence weights for matched pairs (Eq. 9)
        weights_list = self._compute_confidence(feat1, feat2, matches_list)

        result = {
            "assignment": P,
            "score_matrix": S,
            "matches": matches_list,
            "match_scores": match_scores_list,
            "weights": weights_list,
            "feat1": feat1,
            "feat2": feat2,
            "sigma1": sigma1,
            "sigma2": sigma2,
        }

        if return_all_assignments:
            # Replace last layer's entry with the actual final-layer assignment
            all_assignments[-1] = P
            all_sigma1[-1] = sigma1
            all_sigma2[-1] = sigma2
            result["all_assignments"] = all_assignments
            result["all_sigma1"] = all_sigma1
            result["all_sigma2"] = all_sigma2

        return result
