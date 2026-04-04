# Phase 4: Feature Descriptor — DINO-VO

**Paper Reference:** Section III-B, "Feature Descriptor"
**Status:** COMPLETE — all verification checks passed

---

## What Is It?

The Feature Descriptor module extracts a **192-dimensional descriptor vector** for each detected keypoint. It does this by fusing two complementary feature sources:

1. **DINOv2-ViT-S/14** — a frozen visual foundation model that provides rich, semantically-aware patch features (384-dim)
2. **FinerCNN** — a lightweight trainable CNN encoder that provides fine-grained, pixel-level local features (64-dim)

The two feature sets are concatenated and projected to the final 192-dim descriptor space via a learned linear layer (Equation 1 of the paper).

---

## Why Does It Exist?

Matching keypoints between frames requires descriptors that are:λp

- **Discriminative** — each keypoint has a unique signature so wrong matches can be rejected
- **Robust** — similar appearance in different lighting or viewing angles should produce similar descriptors

DINOv2 alone provides global semantic context but lacks fine spatial precision (it operates on 14×14 patches). FinerCNN provides pixel-level detail but lacks semantic understanding. The fusion of both gives the best of both worlds.

---

## Where It Fits in the DINO-VO Pipeline

```
                        DINO-VO Pipeline
                        ================

  Image_t  ──┬──> [Phase 3: Salient Keypoint Detector] ──> 512 keypoints (x,y)
             │                    │
             │                    ▼
             ├──> [Phase 4: Feature Descriptor] ────────> descriptors (192-dim)
             │         │                                       │
             │    ┌────┴──────────────┐                        │
             │    │ DINOv2-ViT-S/14   │ (frozen, 384-dim)      │
             │    │ FinerCNN          │ (trainable, 64-dim)     │
             │    │ Linear Projection │ → 192-dim               │
             │    └───────────────────┘                        │
             │                                                  ▼
  Image_t+1 ─┘              [Phase 5: Feature Matching] ──> correspondences + weights
                                          ▼
                             [Phase 6: Pose Estimation] ──> relative pose (R, t)
```

The feature descriptor receives **keypoint coordinates** from Phase 3 and produces **descriptors** consumed by Phase 5 (Feature Matching).

---

## Paper Equation

**Equation 1:**
```
f_i = Linear([f_DINO_i | f_FINE_i]) ∈ R^192
```

Where:
- `f_DINO_i` — 384-dim DINOv2 patch token for the patch containing keypoint i
- `f_FINE_i` — 64-dim FinerCNN feature at pixel location of keypoint i
- `[· | ·]` — concatenation → 448-dim
- `Linear(·)` — learned projection 448 → 192
- Output is L2-normalized to unit norm

---

## How It Works (Step by Step)

### Step 1: DINOv2 Patch Feature Extraction

DINOv2-ViT-S/14 divides the image into non-overlapping 14×14 patches and produces one 384-dim token per patch. For a 476×742 image:

```
Grid size: 476/14 × 742/14 = 34 × 53 = 1,802 patches
Output: (B, 34, 53, 384) patch feature map
```

DINOv2 is called with `torch.no_grad()` — all its parameters are frozen. No gradients flow through it during training.

### Step 2: FinerCNN Dense Feature Extraction

FinerCNN is a feature pyramid encoder (XFeat-style, Fig. 4) that processes the full grayscale image through a downsample path and fuses features at multiple scales:

```
Input:  (B, 1, H, W)  — grayscale [0, 1]
Output: (B, 64, H, W) — dense feature map (same resolution)
```

**Downsample path** (feature pyramid, expands receptive field to H/16):
- Block 0: Input(1ch) → H x W x 64
- Block 1: stride-2 → H/2 x W/2 x 64
- Block 2: stride-2 → H/4 x W/4 x 64
- Block 3: stride-2 → H/8 x W/8 x 64
- Block 4: stride-2 → H/16 x W/16 x 64

**Fusion path** (combines multi-scale information):
1. Upsample H/16 → H/4, fuse with H/4 skip (1x1 Conv + bilinear upsample + add)
2. Upsample H/4 → H x W, fuse with H x W skip (1x1 Conv + bilinear upsample + add)

Since the dataset loader gives 3-channel ImageNet-normalized tensors, FinerCNN first recovers grayscale by reversing channel 0 normalization:
```
gray = image[:, 0:1] * 0.229 + 0.485  (clamped to [0, 1])
```

### Step 3: Sample Features at Keypoint Locations

For each of the 512 keypoints `(x, y)`:

**DINOv2 sampling** — convert pixel coords to patch grid index:
```
gx = x // 14,  gy = y // 14
f_DINO_i = patch_map[gy, gx]   → shape (384,)
```

**FinerCNN sampling** — index directly at pixel:
```
f_FINE_i = dense_map[:, :, y, x]   → shape (64,)
```

Both use integer rounding and clamping to stay in bounds.

### Step 4: Concatenate and Project

```
f_cat = concat(f_DINO_i, f_FINE_i)   → shape (448,)
f_i   = Linear(f_cat)                 → shape (192,)
f_i   = L2_normalize(f_i)             → unit norm
```

---

## Architecture: FinerCNN (Fig. 4)

```
Input: (B, 1, H, W)
    ↓
Block 0: Conv2d(1→64, k=3, s=1, p=1) + BN + ReLU     → H x W x 64
    ↓
Block 1: Conv2d(64→64, k=3, s=2, p=1) + BN + ReLU    → H/2 x W/2 x 64
    ↓
Block 2: Conv2d(64→64, k=3, s=2, p=1) + BN + ReLU    → H/4 x W/4 x 64  (skip →)
    ↓
Block 3: Conv2d(64→64, k=3, s=2, p=1) + BN + ReLU    → H/8 x W/8 x 64
    ↓
Block 4: Conv2d(64→64, k=3, s=2, p=1) + BN + ReLU    → H/16 x W/16 x 64
    ↓
    ↓  Upsample to H/4, 1x1 Conv, add with Block 2 skip
    ↓  Conv2d(64→64, k=3, p=1) + BN + ReLU             → H/4 x W/4 x 64
    ↓
    ↓  Upsample to H×W, 1x1 Conv, add with Block 0 skip
    ↓  Conv2d(64→64, k=3, p=1) + BN + ReLU             → H x W x 64
    ↓
Output: (B, 64, H, W)   — full resolution via feature pyramid fusion
```

**Trainable parameters:** ~231K (lightweight)

---

## Hyperparameters Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| DINOv2 model | ViT-S/14 | Paper Sec III-B |
| DINOv2 output dim | 384 | ViT-S architecture |
| DINOv2 patch size | 14×14 | ViT-S/14 architecture |
| FinerCNN output dim | 64 | Paper Sec III-B |
| Final descriptor dim | 192 | Paper Sec III-B, Eq. 1 |
| FinerCNN blocks | 4 | XFeat-style encoder |
| L2 normalization | yes | Unit-sphere matching |

---

## Expected Input / Output

**Input:**
- `image`: `(B, 3, H, W)` — ImageNet-normalized tensor (H=476, W=742)
- `keypoints`: `(B, K, 2)` — (x, y) pixel coordinates from Phase 3

**Output:**
- `descriptors`: `(B, K, 192)` — L2-normalized descriptor vectors

---

## Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| DINOv2-ViT-S/14 | 22,056,576 | No (frozen) |
| FinerCNN | ~37,000 | Yes |
| Linear projection (448→192) | ~86,000 | Yes |
| **Total trainable** | **~151,392** | Yes |

Only ~0.7% of parameters are trained — the frozen DINOv2 provides free high-quality features.

---

## Verification Results (Phase 4 Test)

All checks passed on EuRoC MH_01_easy, first image pair:

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Output shape | `(1, 512, 192)` | `(1, 512, 192)` | PASSED |
| L2 norm error | < 1e-5 | 1.19e-07 | PASSED |
| DINOv2 frozen params | 0 | 0 | PASSED |
| DINOv2 patch map shape | `(1, 34, 53, 384)` | `(1, 34, 53, 384)` | PASSED |
| FinerCNN dense map shape | `(1, 64, 476, 742)` | `(1, 64, 476, 742)` | PASSED |
| DINOv2 @ keypoints | `(1, 512, 384)` | `(1, 512, 384)` | PASSED |
| FinerCNN @ keypoints | `(1, 512, 64)` | `(1, 512, 64)` | PASSED |
| Descriptor distinctiveness | sim < 0.99 | 0.38 | PASSED |
| Extraction time (GPU) | — | 264 ms/pair | — |

Visualization saved to: `outputs/feature_descriptor_test.png`

---

## Files

| File | Description |
|------|-------------|
| `src/models/feature_descriptor.py` | Main `FeatureDescriptor` class |
| `src/models/finer_cnn.py` | `FinerCNN` lightweight encoder |
| `scripts/test_feature_descriptor.py` | Phase 4 verification script |

---

## Connection to Other Modules

| Module | Interaction |
|--------|-------------|
| **Phase 3: keypoint_detector.py** | Provides `(B, K, 2)` keypoint coordinates as input |
| **Phase 5: feature_matching.py** | Receives `(B, K, 192)` descriptors for both frames → produces correspondences |
| **Phase 7: losses.py** | Matching loss supervises whether the right keypoints were matched using these descriptors |

---

## Implementation Notes

- DINOv2 is loaded via `load_dino(device)` separately from `__init__` to allow device control and avoid auto-download at import time
- DINOv2 is always called inside `torch.no_grad()` to avoid storing intermediate activations (~300MB VRAM saving during training)
- FinerCNN operates on **single-channel** grayscale recovered from the 3-channel normalized input
- The `_sample_at_keypoints` method handles both patch-level (DINOv2) and pixel-level (FinerCNN) sampling with proper coordinate conversion
- Input images must have H and W divisible by 14 — the paper's 476×742 satisfies this exactly
