"""
FinerCNN — Lightweight fine-grained feature encoder for DINO-VO.

Paper Section III-B, Fig. 4:
    "FinerCNN is built from CNN blocks called basic layers, inspired by
    XFeat [29]. We adopt the feature pyramid approach to expand the
    receptive field until the resolution is reduced to H/16 x W/16.
    The intermediate features are then fused at H/4 x W/4 and H x W
    resolution levels."

Architecture (from Fig. 4):
    Downsample path (feature pyramid):
        Block 0: Input(1ch)  -> H   x W   x 64
        Block 1: stride-2    -> H/2 x W/2 x 64
        Block 2: stride-2    -> H/4 x W/4 x 64
        Block 3: stride-2    -> H/8 x W/8 x 64
        Block 4: stride-2    -> H/16x W/16x 64

    Fusion path:
        1. Upsample H/16 to H/4, fuse with H/4 skip -> H/4 x W/4 x 64
        2. 1x1 Conv + Upsample H/4 to H x W, fuse with H x W skip -> H x W x 64

    Output: F_FINE in R^{H x W x 64}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _basic_layer(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """XFeat-style basic layer: Conv3x3 + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FinerCNN(nn.Module):
    """Fine-grained local feature encoder with feature pyramid (XFeat-style).

    Produces a 64-channel dense feature map at full image resolution via
    a downsample pyramid to H/16 with fusion at H/4 and H x W levels.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        out_channels: Output feature channels (64 as per paper).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()

        # --- Downsample path (feature pyramid) ---
        # Block 0: Input -> H x W x 64
        self.block0 = _basic_layer(in_channels, 64, stride=1)

        # Block 1: H -> H/2
        self.block1 = _basic_layer(64, 64, stride=2)

        # Block 2: H/2 -> H/4
        self.block2 = _basic_layer(64, 64, stride=2)

        # Block 3: H/4 -> H/8
        self.block3 = _basic_layer(64, 64, stride=2)

        # Block 4: H/8 -> H/16
        self.block4 = _basic_layer(64, 64, stride=2)

        # --- Fusion at H/4 level ---
        # 1x1 conv on H/16 features before upsampling to H/4
        self.up_16_to_4 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # Refine fused H/4 features
        self.fuse_quarter = _basic_layer(64, 64, stride=1)

        # --- Fusion at H x W level ---
        # 1x1 conv on H/4 features before upsampling to H x W (as in Fig. 4)
        self.up_4_to_1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # Refine fused full-resolution features
        self.fuse_full = _basic_layer(64, out_channels, stride=1)

        # ImageNet normalization constants (for recovering grayscale)
        self.register_buffer(
            "img_mean", torch.tensor(0.485, dtype=torch.float32)
        )
        self.register_buffer(
            "img_std", torch.tensor(0.229, dtype=torch.float32)
        )

    def _recover_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """Recover grayscale [0, 1] from ImageNet-normalized 3-channel tensor.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized tensor.

        Returns:
            (B, 1, H, W) — grayscale values in [0, 1].
        """
        gray = image[:, 0:1] * self.img_std + self.img_mean
        return gray.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract fine-grained features at full resolution via feature pyramid.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized RGB tensor.

        Returns:
            (B, 64, H, W) — dense feature map at full image resolution.
        """
        gray = self._recover_grayscale(image)  # (B, 1, H, W)

        # --- Downsample path ---
        f0 = self.block0(gray)   # (B, 64, H,    W)
        f1 = self.block1(f0)     # (B, 64, H/2,  W/2)
        f2 = self.block2(f1)     # (B, 64, H/4,  W/4)
        f3 = self.block3(f2)     # (B, 64, H/8,  W/8)
        f4 = self.block4(f3)     # (B, 64, H/16, W/16)

        # --- Fusion at H/4 level ---
        # Upsample H/16 features to H/4, add with H/4 skip connection
        f4_up = F.interpolate(
            self.up_16_to_4(f4), size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        f_quarter = self.fuse_quarter(f2 + f4_up)  # (B, 64, H/4, W/4)

        # --- Fusion at H x W level ---
        # 1x1 Conv + Upsample H/4 to H x W, add with full-res skip (Fig. 4)
        f_quarter_up = F.interpolate(
            self.up_4_to_1(f_quarter), size=f0.shape[2:], mode="bilinear", align_corners=False
        )
        f_full = self.fuse_full(f0 + f_quarter_up)  # (B, 64, H, W)

        return f_full
