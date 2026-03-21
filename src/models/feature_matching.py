"""
Feature Matching for DINO-VO.

Paper Section III-C: Transformer-based feature matching inspired by LightGlue.
Establishes correspondences between keypoints in an image pair using
alternating self- and cross-attention with rotary positional encoding.

Architecture:
    L = 12 transformer layers, each containing:
        1. Self-attention with 2D Rotary Positional Encoding (within each image)
        2. Cross-attention (between the image pair, no positional encoding)
        3. Feed-forward update (Eq. 2)
    Followed by:
        - Soft assignment matrix via dual-softmax (Eq. 5-8)
        - Confidence prediction MLP (Eq. 9)

Equations (paper):
    Eq. 2:  f_i^T  ← f_i^T + MLP([f_i^T | m_i^{T←S}])
    Eq. 3:  m_i^{T←S} = Σ_j softmax(a_ij)_j · v_j
    Eq. 4:  a_ij = (R(p_i)·q_i)^T (R(p_j)·k_j) / sqrt(d)
    Eq. 5:  P_ij = σ_i · σ_j · softmax_row(S)_ij · softmax_col(S)_ij
    Eq. 6:  S_ij = Linear(f_i)^T · Linear(f_j)
    Eq. 7:  σ_i  = sigmoid(Linear(f_i))
    Eq. 9:  w_ij = ConfMLP([f_i | f_j])

Input:
    desc1, desc2: (B, K, 192) descriptors from Phase 4
    kp1, kp2:     (B, K, 2)   keypoint (x, y) pixel coords from Phase 3

Output:
    assignment:  (B, K, K)   soft assignment matrix P
    matches:     list of (M_b, 2) tensors — matched index pairs per batch
    weights:     list of (M_b,)  tensors — confidence weight per match
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (2D) — used in self-attention only
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rearrange adjacent pairs for rotary encoding.

    [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
    """
    x = x.unflatten(-1, (-1, 2))        # (..., D/2, 2)
    x1, x2 = x.unbind(dim=-1)           # each (..., D/2)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _compute_rope_2d(
    pos: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute 2D rotary positional encoding (cos, sin).

    Splits head_dim into two halves:
        - First half  (head_dim/2 dims) encodes the x-coordinate.
        - Second half (head_dim/2 dims) encodes the y-coordinate.

    Within each half, pairs of consecutive dimensions share a frequency
    base following the standard RoPE formula:
        θ_i = 1 / (10000^(2i / half_dim)),   i = 0 … half_dim/2 - 1

    Args:
        pos: (B, K, 2) — keypoint (x, y) pixel coordinates.
        head_dim: Dimension per attention head (64 in this model).

    Returns:
        cos, sin: each (B, 1, K, head_dim) — broadcastable over heads.
    """
    half_dim = head_dim // 2  # 32 dims per coordinate axis

    # Frequency bases: one per pair of dimensions → half_dim/2 values
    freq_idx = torch.arange(0, half_dim, 2, device=pos.device, dtype=pos.dtype)
    freqs = 1.0 / (10000.0 ** (freq_idx / half_dim))  # (half_dim/2,)

    # Compute rotation angles
    # pos[..., 0:1] = x-coord, (B, K, 1) × (half_dim/2,) → (B, K, half_dim/2)
    angles_x = pos[..., 0:1] * freqs
    angles_y = pos[..., 1:2] * freqs

    # Repeat-interleave so each frequency covers a pair of dims
    # (B, K, half_dim/2) → (B, K, half_dim)
    angles_x = angles_x.repeat_interleave(2, dim=-1)
    angles_y = angles_y.repeat_interleave(2, dim=-1)

    # Concatenate x-encoding and y-encoding → full head_dim
    angles = torch.cat([angles_x, angles_y], dim=-1)  # (B, K, head_dim)

    # Add head-broadcast dimension: (B, K, D) → (B, 1, K, D)
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    return cos, sin


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional 2D RoPE (Eq. 3-4).

    Used for both self-attention (with RoPE) and cross-attention (no RoPE).

    Args:
        dim: Feature dimension (192).
        num_heads: Number of attention heads (3).
        head_dim: Dimension per head (64).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5  # 1/sqrt(64) = 0.125

        inner_dim = num_heads * head_dim  # 192
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, K_q, D)
            key:   (B, K_k, D)
            value: (B, K_k, D)
            rope:  (cos, sin) each (B, 1, K, head_dim), or None.
                   Provided for self-attention (same pos for Q and K).
                   None for cross-attention.

        Returns:
            (B, K_q, D)
        """
        B, K_q, _ = query.shape
        K_k = key.shape[1]

        # Project and reshape to (B, H, K, D_h)
        q = self.q_proj(query).view(B, K_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE to Q and K if provided (Eq. 4)
        # Self-attention: positions encode spatial layout within one image
        # Cross-attention: no positional encoding (different coordinate frames)
        if rope is not None:
            cos, sin = rope
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        # Scaled dot-product attention (Eq. 3-4)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, K_q, K_k)
        attn = attn.softmax(dim=-1)

        # Aggregate values
        out = attn @ v                          # (B, H, K_q, D_h)
        out = out.transpose(1, 2).contiguous()  # (B, K_q, H, D_h)
        out = out.view(B, K_q, -1)              # (B, K_q, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single Transformer Layer
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """One layer of the matching transformer.

    Each layer performs (for both images symmetrically):
        1. Pre-norm self-attention with RoPE (spatial context within image)
        2. Pre-norm cross-attention without RoPE (inter-image context)
        3. Feed-forward update with residual (Eq. 2)

    Weights are shared between image 1 and image 2 processing.

    Args:
        dim: Feature dimension (192).
        num_heads: Number of attention heads (3).
        head_dim: Dimension per head (64).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()

        # Self-attention block
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, head_dim)

        # Cross-attention block
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, head_dim)

        # Feed-forward network (Eq. 2)
        # Input: [f_i | m_i] (concatenation) → 2*dim
        # Architecture follows LightGlue: hidden_dim = 2*dim
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
            feat1: (B, K, D) — features for image 1 keypoints
            feat2: (B, K, D) — features for image 2 keypoints
            rope1: (cos, sin) — 2D RoPE for image 1 positions
            rope2: (cos, sin) — 2D RoPE for image 2 positions

        Returns:
            Updated (feat1, feat2), each (B, K, D).
        """
        # --- Self-attention with RoPE (within each image) ---
        f1_norm = self.norm_self(feat1)
        f2_norm = self.norm_self(feat2)

        feat1 = feat1 + self.self_attn(f1_norm, f1_norm, f1_norm, rope=rope1)
        feat2 = feat2 + self.self_attn(f2_norm, f2_norm, f2_norm, rope=rope2)

        # --- Cross-attention without RoPE (between images) ---
        f1_norm = self.norm_cross(feat1)
        f2_norm = self.norm_cross(feat2)

        # Image 1 queries attend to image 2 keys/values
        m1 = self.cross_attn(f1_norm, f2_norm, f2_norm)
        # Image 2 queries attend to image 1 keys/values
        m2 = self.cross_attn(f2_norm, f1_norm, f1_norm)

        # --- Feed-forward update (Eq. 2) ---
        # f_i ← f_i + MLP([f_i | m_i])
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

        # Eq. 6: Score projection — S_ij = (W·f_i)^T (W·f_j) / sqrt(d)
        self.score_proj = nn.Linear(descriptor_dim, descriptor_dim, bias=False)

        # Eq. 7: Matchability — σ_i = sigmoid(W·f_i + b)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft assignment matrix P and raw score matrix S (Eq. 5-8).

        Args:
            feat1: (B, K, D) — refined features for image 1.
            feat2: (B, K, D) — refined features for image 2.

        Returns:
            P: (B, K, K) — soft assignment probabilities.
            S: (B, K, K) — raw (scaled) score matrix.
        """
        # Eq. 6: score matrix via projected dot product
        f1_proj = self.score_proj(feat1)  # (B, K, D)
        f2_proj = self.score_proj(feat2)  # (B, K, D)
        S = torch.bmm(f1_proj, f2_proj.transpose(1, 2))  # (B, K, K)
        S = S / math.sqrt(self.descriptor_dim)

        # Eq. 7: per-keypoint matchability (probability of having a match)
        sigma1 = torch.sigmoid(self.matchability(feat1))  # (B, K, 1)
        sigma2 = torch.sigmoid(self.matchability(feat2))  # (B, K, 1)

        # Eq. 5: dual-softmax assignment weighted by matchability
        P_row = F.softmax(S, dim=-1)   # softmax over j (columns)
        P_col = F.softmax(S, dim=-2)   # softmax over i (rows)

        # P_ij = σ_i · σ_j · softmax_row · softmax_col
        P = sigma1 * sigma2.transpose(1, 2) * P_row * P_col  # (B, K, K)

        return P, S

    def _extract_matches(
        self, P: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract mutual nearest-neighbor matches from assignment matrix.

        A pair (i, j) is a match if:
            - j = argmax_j P[i, :] (best match for i in image 2)
            - i = argmax_i P[:, j] (best match for j in image 1)
            - P[i, j] > match_threshold

        Args:
            P: (B, K1, K2) — soft assignment matrix.

        Returns:
            matches_list: list of (M_b, 2) long tensors — (idx1, idx2) pairs.
            scores_list:  list of (M_b,) float tensors — P[i, j] for each match.
        """
        B, K1, K2 = P.shape

        # Best match in image 2 for each keypoint in image 1
        max_j = P.argmax(dim=-1)  # (B, K1)

        # Best match in image 1 for each keypoint in image 2
        max_i = P.argmax(dim=-2)  # (B, K2)

        # Mutual consistency check (vectorized)
        i_indices = torch.arange(K1, device=P.device).unsqueeze(0).expand(B, -1)
        mutual = max_i.gather(1, max_j) == i_indices  # (B, K1)

        # Gather assignment scores for proposed matches
        scores = P.gather(2, max_j.unsqueeze(-1)).squeeze(-1)  # (B, K1)

        # Valid = mutual AND above threshold
        valid = mutual & (scores > self.match_threshold)

        matches_list = []
        scores_list = []
        for b in range(B):
            valid_idx = valid[b].nonzero(as_tuple=True)[0]  # indices in image 1
            if valid_idx.shape[0] > 0:
                i_matched = valid_idx
                j_matched = max_j[b, valid_idx]
                matches_list.append(
                    torch.stack([i_matched, j_matched], dim=-1)  # (M, 2)
                )
                scores_list.append(scores[b, valid_idx])  # (M,)
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

        These weights are used by the weighted 8-point algorithm (Phase 6)
        to down-weight unreliable correspondences.

        Args:
            feat1: (B, K, D) — refined features for image 1.
            feat2: (B, K, D) — refined features for image 2.
            matches_list: list of (M_b, 2) — matched index pairs per batch.

        Returns:
            weights_list: list of (M_b,) — confidence in [0, 1] per match.
        """
        weights_list = []
        for b, matches in enumerate(matches_list):
            if matches.shape[0] == 0:
                weights_list.append(
                    torch.zeros(0, dtype=feat1.dtype, device=feat1.device)
                )
                continue

            idx1 = matches[:, 0]  # (M,)
            idx2 = matches[:, 1]  # (M,)

            f1 = feat1[b, idx1]  # (M, D)
            f2 = feat2[b, idx2]  # (M, D)

            # Eq. 9: confidence MLP on concatenated features
            w = self.conf_mlp(torch.cat([f1, f2], dim=-1))  # (M, 1)
            w = torch.sigmoid(w.squeeze(-1))                 # (M,)

            weights_list.append(w)

        return weights_list

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
    ) -> Dict[str, object]:
        """Run the full matching pipeline.

        Args:
            desc1: (B, K, 192) — descriptors for image 1 keypoints.
            desc2: (B, K, 192) — descriptors for image 2 keypoints.
            kp1:   (B, K, 2)   — (x, y) pixel coordinates, image 1.
            kp2:   (B, K, 2)   — (x, y) pixel coordinates, image 2.

        Returns:
            dict with:
                'assignment':   (B, K, K)      — soft assignment matrix P
                'score_matrix': (B, K, K)      — raw score matrix S
                'matches':      list of (M, 2) — matched index pairs
                'match_scores': list of (M,)   — assignment score per match
                'weights':      list of (M,)   — confidence weight per match
                'feat1':        (B, K, D)      — refined features (for loss)
                'feat2':        (B, K, D)      — refined features (for loss)
                'sigma1':       (B, K, 1)      — matchability, image 1
                'sigma2':       (B, K, 1)      — matchability, image 2
        """
        # Precompute 2D rotary positional encoding for keypoint positions
        rope1 = _compute_rope_2d(kp1, self.head_dim)
        rope2 = _compute_rope_2d(kp2, self.head_dim)

        # Process through L=12 transformer layers
        feat1, feat2 = desc1, desc2
        for layer in self.layers:
            feat1, feat2 = layer(feat1, feat2, rope1, rope2)

        # Final normalization
        feat1 = self.final_norm(feat1)
        feat2 = self.final_norm(feat2)

        # Soft assignment matrix (Eq. 5-8)
        P, S = self._compute_assignment(feat1, feat2)

        # Matchability scores (for output / debugging)
        sigma1 = torch.sigmoid(self.matchability(feat1))  # (B, K, 1)
        sigma2 = torch.sigmoid(self.matchability(feat2))  # (B, K, 1)

        # Extract mutual nearest-neighbor matches
        matches_list, match_scores_list = self._extract_matches(P)

        # Confidence weights for matched pairs (Eq. 9)
        weights_list = self._compute_confidence(feat1, feat2, matches_list)

        return {
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
