"""
Feature Descriptor for DINO-VO.

Paper Section III-B: Fuses frozen DINOv2 patch features with fine-grained
FinerCNN features, then projects to a 192-dimensional descriptor.

Pipeline for one image:
    1. DINOv2-ViT-S/14 (frozen) → patch feature map (B, H/14, W/14, 384)
    2. FinerCNN (trainable)      → dense feature map  (B, H, W, 64)
    3. Sample both maps at keypoint locations          → (B, K, 384), (B, K, 64)
    4. Concatenate                                     → (B, K, 448)
    5. Linear projection                               → (B, K, 192)

Equation 1 (paper):
    f_i = Linear([f_DINO_i | f_FINE_i]) ∈ R^192

DINOv2 note:
    - Loaded once from torch.hub (facebookresearch/dinov2, dinov2_vits14)
    - All parameters are frozen — no gradients flow through DINOv2
    - DINOv2 forward is wrapped in torch.no_grad() to avoid storing
      intermediate activations (saves ~300MB VRAM during training)
    - Input images must have H and W divisible by 14 (476×742 satisfies this)
"""

import torch
import torch.nn as nn

from .finer_cnn import FinerCNN


class FeatureDescriptor(nn.Module):
    """Fused feature descriptor combining DINOv2 and FinerCNN.

    Args:
        dino_dim: DINOv2-ViT-S output dimension (384, fixed by architecture).
        fine_dim: FinerCNN output channels (64, as per paper).
        out_dim: Final descriptor dimension (192, as per paper Eq. 1).
        patch_size: DINOv2 patch size (14, fixed by ViT-S/14).
    """

    def __init__(
        self,
        dino_dim: int = 384,
        fine_dim: int = 64,
        out_dim: int = 192,
        patch_size: int = 14,
    ):
        super().__init__()

        self.dino_dim = dino_dim
        self.fine_dim = fine_dim
        self.patch_size = patch_size

        # FinerCNN — trainable fine-grained encoder
        self.finer_cnn = FinerCNN(in_channels=1, out_channels=fine_dim)

        # Linear projection: concat(DINOv2, FinerCNN) → 192-d descriptor
        # Equation 1: f_i = Linear([f_DINO_i | f_FINE_i]) ∈ R^192
        self.proj = nn.Linear(dino_dim + fine_dim, out_dim)

        # DINOv2 is loaded separately via load_dino() to allow device control
        self.dino: nn.Module | None = None

    def load_dino(self, device: torch.device) -> None:
        """Load pretrained DINOv2-ViT-S/14 from torch.hub and freeze it.

        Downloads ~330MB on first call; cached to ~/.cache/torch/hub afterward.

        Args:
            device: Target device for the DINOv2 model.
        """
        print("Loading DINOv2-ViT-S/14 from torch.hub...")
        dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
        )

        # Freeze all parameters — DINOv2 is used as a fixed feature extractor
        for param in dino.parameters():
            param.requires_grad = False

        dino.eval()
        dino = dino.to(device)

        self.dino = dino
        print(
            f"DINOv2-ViT-S/14 loaded and frozen. "
            f"Parameters: {sum(p.numel() for p in dino.parameters()):,}"
        )

    def _extract_dino_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features for the full image.

        Calls DINOv2 in no_grad mode (frozen) and reshapes the flat patch
        token sequence into a 2D spatial feature map.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized tensor.
                   H and W must be divisible by 14.

        Returns:
            (B, H//14, W//14, 384) — spatial patch feature map.
        """
        assert self.dino is not None, (
            "DINOv2 not loaded. Call load_dino(device) first."
        )

        B, _, H, W = image.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Image size ({H}×{W}) must be divisible by patch_size={self.patch_size}"
        )

        Gh = H // self.patch_size  # grid height (34 for 476px)
        Gw = W // self.patch_size  # grid width  (53 for 742px)

        # DINOv2 is frozen — no gradients needed
        with torch.no_grad():
            features = self.dino.forward_features(image)

        # 'x_norm_patchtokens': (B, Gh*Gw, 384) — patch tokens (no CLS)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N, 384)

        # Reshape to spatial grid
        patch_map = patch_tokens.view(B, Gh, Gw, self.dino_dim)  # (B, Gh, Gw, 384)
        return patch_map

    def _sample_at_keypoints(
        self,
        feature_map: torch.Tensor,
        keypoints: torch.Tensor,
        is_patch_map: bool,
    ) -> torch.Tensor:
        """Sample features from a spatial feature map at keypoint locations.

        For DINOv2 patch features (is_patch_map=True):
            Convert pixel (x, y) → patch grid index (x//14, y//14), then index.

        For FinerCNN dense features (is_patch_map=False):
            Index directly at pixel coordinates (x, y), clamped to image bounds.

        Args:
            feature_map: Spatial feature map.
                - DINOv2: (B, Gh, Gw, D) with D=384
                - FinerCNN: (B, D, H, W) with D=64
            keypoints: (B, K, 2) — (x, y) pixel coordinates (float).
            is_patch_map: True for DINOv2 map, False for FinerCNN map.

        Returns:
            (B, K, D) — features sampled at each keypoint.
        """
        B, K, _ = keypoints.shape

        # Round to integer pixel coordinates
        kp = keypoints.long()  # (B, K, 2) — (x, y)
        x = kp[..., 0]         # (B, K)
        y = kp[..., 1]         # (B, K)

        if is_patch_map:
            # DINOv2: convert pixel → patch grid index
            # feature_map: (B, Gh, Gw, D)
            _, Gh, Gw, D = feature_map.shape
            gx = (x // self.patch_size).clamp(0, Gw - 1)  # (B, K)
            gy = (y // self.patch_size).clamp(0, Gh - 1)  # (B, K)

            # Gather: for each batch and keypoint, index [gy, gx]
            # Expand indices for gather
            b_idx = torch.arange(B, device=keypoints.device).view(B, 1).expand(B, K)
            sampled = feature_map[b_idx, gy, gx, :]  # (B, K, D)

        else:
            # FinerCNN: index at pixel coordinates
            # feature_map: (B, D, H, W) → transpose for indexing → (B, H, W, D)
            feat = feature_map.permute(0, 2, 3, 1)  # (B, H, W, D)
            _, H, W, D = feat.shape
            cx = x.clamp(0, W - 1)  # (B, K)
            cy = y.clamp(0, H - 1)  # (B, K)

            b_idx = torch.arange(B, device=keypoints.device).view(B, 1).expand(B, K)
            sampled = feat[b_idx, cy, cx, :]  # (B, K, D)

        return sampled  # (B, K, D)

    def forward(
        self, image: torch.Tensor, keypoints: torch.Tensor
    ) -> torch.Tensor:
        """Compute 192-dimensional descriptors at detected keypoint locations.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized image pair element.
            keypoints: (B, K, 2) — (x, y) pixel coords from keypoint detector.

        Returns:
            descriptors: (B, K, 192) — L2-normalized fused descriptors.
        """
        # Step 1: Extract DINOv2 patch feature map (frozen, no_grad)
        # Output: (B, Gh, Gw, 384)
        dino_map = self._extract_dino_features(image)

        # Step 2: Extract FinerCNN dense feature map (trainable)
        # Output: (B, 64, H, W)
        fine_map = self.finer_cnn(image)

        # Step 3: Sample both maps at keypoint locations
        # DINOv2 features: (B, K, 384)
        f_dino = self._sample_at_keypoints(dino_map, keypoints, is_patch_map=True)
        # FinerCNN features: (B, K, 64)
        f_fine = self._sample_at_keypoints(fine_map, keypoints, is_patch_map=False)

        # Step 4: Concatenate → (B, K, 448)
        f_cat = torch.cat([f_dino, f_fine], dim=-1)

        # Step 5: Linear projection → (B, K, 192)  [Equation 1]
        descriptors = self.proj(f_cat)

        # L2-normalize so matching operates on unit-sphere descriptors
        descriptors = nn.functional.normalize(descriptors, dim=-1)

        return descriptors  # (B, K, 192)
