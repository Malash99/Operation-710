"""
Salient Keypoint Detector for DINO-VO.

Paper Section III-A: Detects keypoints using gradient-based saliency
aligned to DINOv2's 14x14 patch grid.

Pipeline:
    1. Gaussian smoothing (kernel=5, sigma=2.0)
    2. Sobel gradient magnitude
    3. Grid-based MaxPooling (kernel=14, stride=14)
    4. Non-Maximum Suppression (radius=8)
    5. Gradient thresholding (0.01)
    6. Top-K selection (K=512)

No learnable parameters — all filters are fixed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalientKeypointDetector(nn.Module):
    """Gradient-based keypoint detector aligned to DINOv2 patch grid.

    Args:
        gauss_kernel_size: Gaussian smoothing kernel size (default 5).
        gauss_sigma: Gaussian smoothing standard deviation (default 2.0).
        pool_kernel_size: MaxPool kernel matching DINOv2 patch size (default 14).
        nms_radius: Non-maximum suppression radius in pixels (default 8).
        gradient_threshold: Minimum gradient magnitude to keep (default 0.01).
        top_k: Number of keypoints to select (default 512).
    """

    def __init__(
        self,
        gauss_kernel_size: int = 5,
        gauss_sigma: float = 2.0,
        pool_kernel_size: int = 14,
        nms_radius: int = 8,
        gradient_threshold: float = 0.01,
        top_k: int = 512,
    ):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.nms_radius = nms_radius
        self.gradient_threshold = gradient_threshold
        self.top_k = top_k

        # Build fixed Gaussian kernel (registered as buffer for auto GPU transfer)
        gauss_kernel = self._make_gaussian_kernel(gauss_kernel_size, gauss_sigma)
        self.register_buffer("gauss_kernel", gauss_kernel)  # (1, 1, K, K)

        # Build fixed Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # ImageNet normalization constants for recovering grayscale
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.gauss_pad = gauss_kernel_size // 2

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Args:
            kernel_size: Size of the kernel (must be odd).
            sigma: Standard deviation.

        Returns:
            Gaussian kernel, shape (1, 1, kernel_size, kernel_size).
        """
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """Recover grayscale [0, 1] from ImageNet-normalized 3-channel tensor.

        Since EuRoC images are grayscale repeated to 3 channels, all channels
        contain the same value. We reverse normalization on channel 0.

        Args:
            image: Normalized tensor, shape (B, 3, H, W).

        Returns:
            Grayscale tensor, shape (B, 1, H, W), values in [0, 1].
        """
        gray = image[:, 0:1, :, :] * self.img_std[:, 0:1, :, :] + self.img_mean[:, 0:1, :, :]
        return gray.clamp(0.0, 1.0)

    def _compute_gradient_magnitude(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude after Gaussian smoothing.

        Steps:
            1. Apply Gaussian filter (kernel=5, sigma=2.0)
            2. Apply Sobel operators in x and y
            3. Compute magnitude: G = sqrt(Gx^2 + Gy^2)

        Args:
            gray: Grayscale image, shape (B, 1, H, W).

        Returns:
            Gradient magnitude, shape (B, 1, H, W).
        """
        # Gaussian smoothing
        smoothed = F.conv2d(gray, self.gauss_kernel, padding=self.gauss_pad)

        # Sobel gradients
        gx = F.conv2d(smoothed, self.sobel_x, padding=1)
        gy = F.conv2d(smoothed, self.sobel_y, padding=1)

        # Gradient magnitude
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return magnitude

    def _grid_maxpool(self, magnitude: torch.Tensor):
        """Extract one keypoint candidate per 14x14 patch via MaxPooling.

        Uses F.max_pool2d with return_indices to get both the max value
        and its position within each patch.

        Args:
            magnitude: Gradient magnitude, shape (B, 1, H, W).

        Returns:
            pooled_values: Max gradient per patch, shape (B, 1, H//14, W//14).
            pooled_indices: Flat index of max pixel in original image,
                            shape (B, 1, H//14, W//14).
        """
        pooled_values, pooled_indices = F.max_pool2d(
            magnitude,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_kernel_size,
            return_indices=True,
        )
        return pooled_values, pooled_indices

    def _indices_to_coordinates(
        self, indices: torch.Tensor, image_width: int
    ) -> torch.Tensor:
        """Convert flat MaxPool indices to (x, y) pixel coordinates.

        F.max_pool2d returns flat indices into the (H*W) space of the input.

        Args:
            indices: Flat indices, shape (B, 1, Gh, Gw).
            image_width: Width of the original image (W).

        Returns:
            Coordinates as (x, y) pairs, shape (B, Gh*Gw, 2).
        """
        B = indices.shape[0]
        flat_indices = indices.view(B, -1)  # (B, Gh*Gw)

        y = flat_indices // image_width  # row
        x = flat_indices % image_width   # col

        coords = torch.stack([x, y], dim=-1)  # (B, N, 2) as (x, y)
        return coords.float()

    def _nms(
        self, coords: torch.Tensor, scores: torch.Tensor, radius: int
    ) -> tuple:
        """Non-maximum suppression to remove nearby keypoints.

        Processes each image in the batch independently. Uses vectorized
        distance computation with greedy suppression.

        Args:
            coords: Keypoint coordinates, shape (B, N, 2).
            scores: Keypoint scores, shape (B, N).
            radius: Suppression radius in pixels.

        Returns:
            List of (coords_kept, scores_kept) tuples, one per batch element.
            coords_kept: shape (M, 2), scores_kept: shape (M,), where M varies.
        """
        B = coords.shape[0]
        results = []

        for b in range(B):
            pts = coords[b]    # (N, 2)
            sc = scores[b]     # (N,)

            # Sort by score descending
            order = sc.argsort(descending=True)
            pts_sorted = pts[order]
            sc_sorted = sc[order]

            # Compute pairwise distance matrix
            dists = torch.cdist(
                pts_sorted.unsqueeze(0), pts_sorted.unsqueeze(0)
            ).squeeze(0)  # (N, N)

            N = pts_sorted.shape[0]
            keep = torch.ones(N, dtype=torch.bool, device=coords.device)

            for i in range(N):
                if not keep[i]:
                    continue
                # Suppress all lower-scored points within radius
                suppress = dists[i, i + 1:] < radius
                keep[i + 1:] = keep[i + 1:] & ~suppress

            results.append((pts_sorted[keep], sc_sorted[keep]))

        return results

    def _threshold_and_topk(
        self, coords: torch.Tensor, scores: torch.Tensor
    ) -> tuple:
        """Apply gradient threshold and select top-K keypoints.

        Args:
            coords: Keypoint coordinates after NMS, shape (M, 2).
            scores: Keypoint scores after NMS, shape (M,).

        Returns:
            coords_topk: shape (K, 2), padded with zeros if M < K.
            scores_topk: shape (K,), padded with zeros if M < K.
            num_valid: int, number of valid keypoints (min(M_thresholded, K)).
        """
        # Threshold
        mask = scores >= self.gradient_threshold
        coords = coords[mask]
        scores = scores[mask]

        num_valid = min(coords.shape[0], self.top_k)

        if coords.shape[0] >= self.top_k:
            # Already sorted by score from NMS, take top K
            coords = coords[: self.top_k]
            scores = scores[: self.top_k]
        else:
            # Pad with zeros
            pad_size = self.top_k - coords.shape[0]
            coords = torch.cat(
                [coords, torch.zeros(pad_size, 2, device=coords.device)], dim=0
            )
            scores = torch.cat(
                [scores, torch.zeros(pad_size, device=scores.device)], dim=0
            )

        return coords, scores, num_valid

    def forward(self, image: torch.Tensor) -> dict:
        """Detect salient keypoints in the input image.

        Args:
            image: ImageNet-normalized RGB tensor, shape (B, 3, H, W).
                   (From the EuRoC dataloader — grayscale repeated to 3 channels.)

        Returns:
            dict with:
                'keypoints': (B, K, 2) — (x, y) pixel coordinates
                'scores': (B, K) — gradient magnitude at each keypoint
                'num_valid': (B,) — number of valid keypoints per image
                'gradient_map': (B, 1, H, W) — full gradient magnitude map
        """
        B, _, H, W = image.shape

        # Step 1-2: Recover grayscale and compute gradient magnitude
        gray = self._to_grayscale(image)
        gradient_mag = self._compute_gradient_magnitude(gray)

        # Step 3: Grid-based MaxPooling (14x14 patches)
        pooled_values, pooled_indices = self._grid_maxpool(gradient_mag)

        # Convert flat indices to (x, y) coordinates
        candidate_coords = self._indices_to_coordinates(pooled_indices, W)
        candidate_scores = pooled_values.view(B, -1)  # (B, Gh*Gw)

        # Step 4: Non-Maximum Suppression (radius=8)
        nms_results = self._nms(candidate_coords, candidate_scores, self.nms_radius)

        # Step 5-6: Threshold and Top-K selection (per batch element)
        all_keypoints = []
        all_scores = []
        all_num_valid = []

        for coords_nms, scores_nms in nms_results:
            kp, sc, nv = self._threshold_and_topk(coords_nms, scores_nms)
            all_keypoints.append(kp)
            all_scores.append(sc)
            all_num_valid.append(nv)

        keypoints = torch.stack(all_keypoints, dim=0)    # (B, K, 2)
        scores = torch.stack(all_scores, dim=0)           # (B, K)
        num_valid = torch.tensor(all_num_valid, device=image.device)  # (B,)

        return {
            "keypoints": keypoints,
            "scores": scores,
            "num_valid": num_valid,
            "gradient_map": gradient_mag,
        }
