"""
FinerCNN — Lightweight fine-grained feature encoder for DINO-VO.

Paper Section III-B:
    "We use a lightweight CNN encoder based on XFeat to extract fine-grained
    local features at full image resolution, producing 64-dimensional features."

Architecture:
    Input : (B, 1, H, W)  — grayscale image in [0, 1]
    Output: (B, 64, H, W) — dense feature map, same spatial resolution

All convolutions use padding=1 (kernel 3×3) so spatial size is preserved.
No downsampling — this is a fully dense feature map.
"""

import torch
import torch.nn as nn


class FinerCNN(nn.Module):
    """Fine-grained local feature encoder (XFeat-style).

    Produces a 64-channel dense feature map at full image resolution.
    Used alongside frozen DINOv2 features in the descriptor fusion step.

    Args:
        in_channels: Number of input channels (1 for grayscale, default).
        out_channels: Output feature channels (64 as per paper).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 1 → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 32 → 32
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 64 → 64 (out_channels)
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ImageNet normalization constants (channel 0, grayscale repeated)
        # Used to recover grayscale from the 3-channel normalized input
        self.register_buffer(
            "img_mean", torch.tensor(0.485, dtype=torch.float32)
        )
        self.register_buffer(
            "img_std", torch.tensor(0.229, dtype=torch.float32)
        )

    def _recover_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """Recover grayscale [0, 1] from ImageNet-normalized 3-channel tensor.

        EuRoC images are grayscale repeated to 3 channels, so channel 0
        carries the full signal. We reverse the normalization on that channel.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized tensor.

        Returns:
            (B, 1, H, W) — grayscale values in [0, 1].
        """
        gray = image[:, 0:1] * self.img_std + self.img_mean
        return gray.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract fine-grained features at full resolution.

        Args:
            image: (B, 3, H, W) — ImageNet-normalized RGB tensor.

        Returns:
            (B, 64, H, W) — dense feature map.
        """
        gray = self._recover_grayscale(image)  # (B, 1, H, W)
        return self.encoder(gray)              # (B, 64, H, W)
