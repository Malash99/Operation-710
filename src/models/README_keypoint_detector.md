# Salient Keypoint Detector — DINO-VO

**Paper Reference:** Section III-A, "Salient Keypoint Detector"

---

## What Is It?

The Salient Keypoint Detector selects **512 interest points** per image that are likely to be reliably matched across consecutive frames. Instead of using learned keypoint detectors (like SuperPoint), DINO-VO uses a classical gradient-based approach that is fast, deterministic, and aligned with DINOv2's patch grid.

---

## Why Does It Exist?

DINO-VO needs sparse correspondences between two frames to estimate camera motion. Rather than matching every pixel (expensive), we select a small set of keypoints that:

1. **Have strong gradients** — edges and corners are easier to match
2. **Are well-distributed** — spread across the image for robust pose estimation
3. **Align to the DINOv2 patch grid** — each 14x14 patch produces one feature vector, so keypoints should align to these patches for clean feature extraction

---

## Where It Fits in the DINO-VO Pipeline

```
                        DINO-VO Pipeline
                        ================

  Image_t  ──┬──> [Salient Keypoint Detector] ──> 512 keypoints (x,y)
             │              ▼
             ├──> [DINOv2 ViT-S] ──────────────> patch features (384-dim)
             │              ▼
             ├──> [FinerCNN] ──────────────────> fine features (64-dim)
             │              ▼
             │   [Feature Fusion] ─────────────> descriptors (192-dim)
             │              ▼
  Image_t+1 ─┘   [Feature Matching] ──────────> correspondences + weights
                            ▼
                  [Pose Estimation] ──────────> relative pose (R, t)
```

The keypoint detector is the **first module** in the pipeline. Its output (pixel coordinates) is used by:
- **Feature Descriptor** (Phase 4) — extracts DINOv2 + FinerCNN features at keypoint locations
- **Feature Matching** (Phase 5) — matches keypoints between the two images
- **Pose Estimation** (Phase 6) — uses matched keypoints to compute Essential matrix

---

## How It Works (Step by Step)

### Step 1: Gaussian Smoothing
Apply a Gaussian filter to reduce noise before gradient computation.

- **Kernel size:** 5x5
- **Standard deviation:** 2.0
- **Implementation:** `torchvision.transforms.GaussianBlur` or `torch.nn.Conv2d` with Gaussian kernel

```
I_smooth = GaussianFilter(I_gray, kernel=5, sigma=2.0)
```

### Step 2: Sobel Gradient Magnitude
Compute image gradients in x and y directions using Sobel operators, then compute the gradient magnitude.

```
G_x = Sobel_x(I_smooth)
G_y = Sobel_y(I_smooth)
G = sqrt(G_x² + G_y²)
```

This produces a gradient magnitude map where edges and corners have high values.

### Step 3: Grid-Based MaxPooling
Pool the gradient magnitude into a grid that matches DINOv2's patch size. DINOv2-ViT-S uses 14x14 patches, so we apply MaxPool with:

- **Kernel size:** 14x14
- **Stride:** 14

This produces a coarse grid where each cell contains the maximum gradient value from its corresponding 14x14 image patch. For a 476x742 image:
- Output size: 34x53 = 1,802 cells

```
G_pooled = MaxPool2d(G, kernel_size=14, stride=14)  # shape: (34, 53)
```

### Step 4: Non-Maximum Suppression (NMS)
Within each grid cell that was selected as a local maximum, find the exact pixel with the highest gradient. Then apply NMS with a radius to avoid keypoints that are too close together.

- **NMS radius:** r_NMS = 8 pixels

This ensures keypoints are spatially well-distributed.

### Step 5: Gradient Thresholding
Remove keypoints with very weak gradients (likely in textureless regions).

- **Threshold:** 0.01 (on normalized gradient magnitude)

### Step 6: Top-K Selection
Select the top 512 keypoints by gradient magnitude.

- **K:** 512 keypoints per image

If fewer than 512 keypoints pass the threshold, all surviving keypoints are kept.

---

## Hyperparameters Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gaussian kernel | 5x5 | Standard smoothing before gradients |
| Gaussian sigma | 2.0 | Moderate smoothing |
| MaxPool kernel | 14x14 | Matches DINOv2 ViT-S patch size |
| MaxPool stride | 14 | Non-overlapping patches |
| NMS radius | 8 pixels | Prevents keypoint clustering |
| Gradient threshold | 0.01 | Removes textureless regions |
| Top-K | 512 | Balance between coverage and compute |

---

## Expected Input / Output

**Input:**
- Grayscale image tensor, shape `(1, 1, 476, 742)` or `(B, 1, 476, 742)`
- Values in `[0, 1]` range (before ImageNet normalization)

**Output:**
- Keypoint coordinates: `(B, 512, 2)` — (x, y) pixel coordinates
- Keypoint scores: `(B, 512)` — gradient magnitude at each keypoint

---

## Verification Criteria

After implementation, verify:
1. Exactly 512 keypoints are output per image (or fewer if thresholding removes too many)
2. Keypoints are distributed across the image (not clustered in one region)
3. Keypoints tend to fall on edges and corners (high gradient areas)
4. Keypoints roughly align to the 14x14 DINOv2 patch grid
5. No keypoints appear within r_NMS=8 pixels of each other

---

## Connection to Other Modules

| Module | Interaction |
|--------|-------------|
| **transforms.py** | Provides preprocessed images; keypoint detector needs the pre-normalization grayscale |
| **feature_descriptor.py** | Takes keypoint coordinates → samples DINOv2 and FinerCNN features at those locations |
| **feature_matching.py** | Receives keypoints + descriptors from both frames → produces correspondences |
| **losses.py** | Matching loss supervises correspondence quality; pose loss supervises estimated motion |

---

## Implementation Notes

- The detector is **not learned** — it uses fixed Gaussian/Sobel filters, no trainable parameters
- It operates on **single-channel** (grayscale) images, not the 3-channel normalized tensors
- The grayscale image should be extracted from the original image before ImageNet normalization
- All operations should be implemented as **torch operations** for GPU compatibility
- The detector processes both frames of a pair independently
