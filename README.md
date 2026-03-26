# DINO-VO: Feature-based Visual Odometry with Visual Foundation Model

Implementation of **DINO-VO** from the paper:
> "DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model"
> Azhari & Shim, IEEE Robotics and Automation Letters (RA-L), July 2025
> arXiv: [2507.13145v1](https://arxiv.org/abs/2507.13145)

---

## Project Overview

This is a from-scratch reimplementation of DINO-VO, a monocular visual odometry system that combines:
- **DINOv2-ViT-S** (frozen visual foundation model) for robust feature extraction
- **FinerCNN** (lightweight trainable encoder) for fine-grained local features
- **Transformer-based matching** (inspired by LightGlue) for feature correspondence
- **Differentiable pose estimation** via weighted 8-point algorithm

The paper trains on **TartanAir** (synthetic dataset with GT depth) and evaluates on **EuRoC MAV**. Our current implementation trains on **EuRoC** using stereo-derived depth for GT correspondences.

---

## Current Status: Training Run 2 Complete (2026-03-26)

### What Works
- **Matching loss converges**: 8.9 -> 3.0 over 14 epochs (significant improvement)
- **Keyframe selection**: Filters degenerate small-motion pairs (504/3637 filtered)
- **Bug fixes**: 5 critical/medium bugs found and fixed via paper audit (see below)
- **All pipeline components** verified individually

### What Doesn't Work Yet
- **Pose loss does not converge**: pose_raw stays ~370-440 (random pose territory, max ~1365)
- **NaN gradients from SVD backward**: 1953/5488 steps (36%) produce NaN gradients
- **NaN rate increases with lambda_p**: As pose loss weight grows, more steps are NaN. Epoch 14 was ~100% NaN

### Root Cause
The **SVD backward pass** in pose estimation is numerically unstable when matches are imperfect. The gradient through `torch.linalg.svd` contains terms like `1/(sigma_i - sigma_j)` that explode when singular values are close. This is a known PyTorch issue, not an implementation bug. The paper likely avoids this by training on TartanAir (larger motions, better-conditioned Essential matrices).

### Evaluation Results (v0.1, epoch 4 checkpoint, 50 pairs)
| Metric | Value |
|--------|-------|
| ATE RMSE | 0.4121 m |
| ATE Mean | 0.3744 m |
| Rotation Mean | 9.18 deg |
| Scale factor | 0.0491 |
| Avg matches/pair | 249 |

Trajectory shape does not follow GT — expected, since pose loss never converged and we trained on EuRoC (not TartanAir).

### Next Steps
1. **Download TartanAir dataset** for proper training (as the paper does) — this is the #1 priority
2. **Build TartanAir dataloader** — adapt pipeline for TartanAir's format (RGB images, GT depth, GT poses)
3. **Retrain on TartanAir** — expect pose loss to converge (larger motions, better-conditioned E matrices)
4. **Re-evaluate on EuRoC** — compare with paper's reported ATE results
5. **Multi-sequence evaluation** — test on MH_02, MH_03, etc.

---

## Bugs Found and Fixed (Paper Audit, 2026-03-25)

A full audit comparing every equation and hyperparameter against the paper revealed 5 bugs:

### Bug 1 (CRITICAL): RoPE Coordinate Normalization Swapped
- **File**: `src/models/feature_matching.py` lines 522-527
- **Problem**: x-coordinate (column) was divided by `img_h`, y-coordinate (row) was divided by `img_w` — swapped
- **Impact**: Corrupted ALL spatial positional encoding in self-attention. The transformer couldn't reason about spatial relationships correctly
- **Fix**: `kp[..., 0] / (img_w - 1)` and `kp[..., 1] / (img_h - 1)`

### Bug 2 (CRITICAL): Epipolar Constraint Matrix Column Order
- **File**: `src/models/pose_estimation.py` lines 108-118
- **Problem**: Phi columns were in column-major order `[e11, e21, e31, e12, ...]` but `view(-1, 3, 3)` assumes row-major. This produced **E-transpose** instead of E
- **Impact**: Estimated the INVERSE relative pose (cam2-to-cam1 instead of cam1-to-cam2)
- **Fix**: Reordered to row-major `[e11, e12, e13, e21, e22, e23, e31, e32, e33]`

### Bug 3 (CRITICAL): Cheirality Check Depth Sign
- **File**: `src/models/pose_estimation.py`
- **Problem**: Missing negative sign in triangulated depth computation
- **Impact**: Selected the wrong (R, t) candidate from the 4 Essential matrix decompositions — specifically, picked the opposite translation direction
- **Why it was hidden**: Bugs #2 and #3 **cancelled each other out**. The transposed E gave the inverse pose, and the wrong depth sign flipped the cheirality check, so the final result was approximately correct. Fixing Bug #2 alone exposed Bug #3
- **Fix**: `num = -(t_cross_x2 * Rx1_cross_x2).sum(dim=-1)`
- **Verification**: After both fixes, clean correspondences give 0.0000 deg rotation error and 0.000005 translation error

### Bug 4 (MEDIUM): L2 Normalization Not in Paper
- **File**: `src/models/feature_descriptor.py` line 217
- **Problem**: Applied `nn.functional.normalize(descriptors, dim=-1)` after Linear projection. Paper Eq. 1 does NOT include L2 normalization
- **Impact**: Constrained all descriptors to unit sphere, reducing expressiveness before matching transformer
- **Fix**: Removed L2 normalization

### Bug 5 (MEDIUM): Score Matrix Scaling Not in Paper
- **File**: `src/models/feature_matching.py` line 380
- **Problem**: Applied `S / sqrt(192)` scaling to score matrix. Paper Eq. 6 is a raw dot product without scaling
- **Impact**: Changed the sharpness of the dual-softmax assignment probabilities
- **Fix**: Removed scaling factor

---

## Training History

### Training Run 1 (Pre-Audit, 2026-03-24)
- **Config**: batch_size=1, skip_frames=2, pose_loss_clamp=50
- **Result**: Matching loss 0.76 -> 0.64 (slow improvement). Pose loss = 50.0 constant (clamped). 2874/5488 steps NaN (52%)
- **Problems**:
  - Pose loss clamp at 50 blocked ALL gradient (pose_raw was always 600-1072 >> 50)
  - Per-parameter NaN gradient check was O(6M) — dominated step time (6.25s/step)
  - No keyframe selection — pairs with ~1cm motion made Essential matrix degenerate
  - All 5 bugs above were present

### Training Run 2 (Post-Audit, 2026-03-25)
- **Config**: batch_size=8, skip_frames=2 + keyframe selection (min_translation=0.03m), no pose clamp
- **Fixes applied**: All 5 bugs fixed, pose clamp removed, fast NaN check, keyframe selection
- **Result**: Matching loss 8.9 -> 3.0 (huge improvement). Pose loss ~370 (not converging). 1953/5488 NaN (36%)
- **Key insight**: Matching loss starts higher (8.9 vs 0.76) because removing L2 normalization and score scaling changed the loss landscape — but it converges much further (3.0 vs 0.64)
- **Speed**: 7.3s/step with batch=8 vs 6.25s/step with batch=1. But 8x more samples per step, so ~8x more efficient

---

## Key Design Decisions and Lessons

### 1. Paper trains on TartanAir, NOT EuRoC
The paper only evaluates on EuRoC. Training on EuRoC was our adaptation because we didn't have TartanAir. EuRoC has very small inter-frame motions (~1-5cm) which makes the Essential matrix poorly conditioned for SVD-based training.

### 2. GT Correspondences Require Depth
The matching loss (Eq. 12) is supervised — it needs "keypoint i in image 1 matches keypoint j in image 2." Generating these requires:
1. Back-project keypoint from image 1 to 3D using **depth**
2. Transform by GT pose
3. Project into image 2
4. Find nearest detected keypoint within threshold

The paper uses TartanAir's GT depth. We compute depth from EuRoC's stereo cameras using StereoSGBM.

### 3. SVD Backward is Fundamentally Unstable
`torch.linalg.svd` backward pass contains `1/(sigma_i - sigma_j)` terms that produce NaN when singular values are close. This is not fixable without replacing the SVD with a custom backward implementation. The paper likely doesn't encounter this because TartanAir has larger, more diverse motions.

### 4. Cancelling Bugs Can Hide Problems
Bugs #2 and #3 (transposed E + wrong depth sign) cancelled each other out, passing unit tests. This is why integration tests with known poses are essential — they would have caught the individual errors.

### 5. Keyframe Selection is Critical
Without keyframe selection, ~14% of EuRoC pairs have translation < 3cm, making the Essential matrix degenerate. The paper specifies this in Section III-F (24px mean pixel displacement threshold).

---

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM) or equivalent
  - RTX 50 series requires PyTorch nightly with CUDA 12.8 support
  - Training uses ~12GB VRAM with batch_size=8
- **RAM**: 16GB+ recommended
- **Storage**: ~15GB for EuRoC dataset + models

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8+
- **CUDA**: 12.8 (for RTX 50 series) or 11.8+ (for older GPUs)
- **PyTorch**: 2.11+ (nightly) or 2.0+ (stable for older GPUs)

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Operation-710.git
cd Operation-710
```

### 2. Install PyTorch

#### For RTX 50 Series (sm_120):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### For Other GPUs:
Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install appropriate version.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU
```bash
python scripts/verify_gpu.py
```

### 5. Download EuRoC Dataset

Download from [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/690084):

```bash
unzip machine_hall.zip -d data/euroc/
cd data/euroc/MH_01_easy
unzip MH_01_easy.zip
```

Verify:
```bash
python scripts/verify_dataset.py
```

---

## Usage

### Training
```bash
python -m scripts.train --config configs/default.yaml
```

Resume from checkpoint:
```bash
python -m scripts.train --config configs/default.yaml --resume checkpoints/epoch_04.pth
```

Quick test (N steps only):
```bash
python -m scripts.train --config configs/default.yaml --max_steps 10
```

### Config (`configs/default.yaml`)
```yaml
data:
  sequence_path: data/euroc/MH_01_easy
  skip_frames: 2
  min_translation: 0.03    # keyframe selection threshold (meters)
  max_skip_multiplier: 5

training:
  epochs: 14
  batch_size: 8            # 12GB VRAM peak
  learning_rate: 1.0e-4
  lambda_p_start_epoch: 5  # matching-only for epochs 1-4
  lambda_p_increment: 1.5e-4
  lambda_p_max: 0.9
  grad_clip: 1.0
```

---

## Architecture

### Pipeline Overview
```
Image Pair (It, It+1)
    |
[1] Salient Keypoint Detector (Sec III-A)
    - Gaussian smoothing + Sobel gradients
    - Grid-based MaxPooling (14x14)
    - Non-Maximum Suppression + Top-k (512 keypoints)
    |
[2] Feature Descriptor (Sec III-B)
    - DINOv2-ViT-S: 384-dim features (frozen)
    - FinerCNN: 64-dim features (trainable)
    - Fusion: 192-dim final descriptors
    |
[3] Feature Matching (Sec III-C)
    - Transformer (L=12 layers, 3 heads)
    - Soft assignment matrix with dual-softmax
    - Confidence prediction MLP
    |
[4] Pose Estimation (Sec III-D)
    - Weighted 8-point algorithm
    - Essential matrix decomposition
    - Cheirality check
    |
Relative Pose (R, t)
```

### Key Components

| Component | Implementation | Location |
|-----------|----------------|----------|
| Keypoint Detector | Gradient-based saliency detection | `src/models/keypoint_detector.py` |
| Feature Descriptor | DINOv2 + FinerCNN fusion | `src/models/feature_descriptor.py` |
| Feature Matching | Transformer with rotary encoding | `src/models/feature_matching.py` |
| Pose Estimation | Weighted 8-point + SVD | `src/models/pose_estimation.py` |
| Loss Functions | Matching + Pose losses | `src/losses/losses.py` |
| Unified Model | End-to-end pipeline wrapper | `src/models/dino_vo.py` |
| Dataset Loader | EuRoC image + pose + stereo depth | `src/datasets/euroc.py` |
| Stereo Depth | StereoSGBM + GT correspondence gen | `src/utils/stereo.py` |

### Trainable Parameters
| Module | Parameters |
|--------|-----------|
| DINOv2-ViT-S/14 (frozen) | 22,056,576 |
| FinerCNN + Fusion | ~151,000 |
| Feature Matching Transformer | ~6,044,000 |
| **Total trainable** | **6,195,522** |

---

## Loss Function

```
L_total = (1 - lambda_p) * L_matching + lambda_p * L_pose
```

**Matching loss** (Eq. 12): NLL of soft assignment matrix at GT correspondence locations, with deep supervision over all 12 transformer layers. Includes unmatchable keypoint penalties.

**Pose loss** (Eq. 13): `180 * ||log(R_est) - log(R_gt)|| + 400 * ||t_est_unit - t_gt_unit||`
- Theoretical max: ~1365 (180*pi + 400*2)

**Schedule**: lambda_p = 0 for epochs 1-4, then increments by 1.5e-4 per step (caps at 0.9).

---

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Environment Setup | COMPLETE | PyTorch nightly + CUDA 12.8 + EuRoC dataset |
| 2. Data Pipeline | COMPLETE | EuRoC loader with transforms, GT poses, stereo depth |
| 3. Keypoint Detector | COMPLETE | Gaussian + Sobel + MaxPool + NMS + Top-k |
| 4. Feature Descriptor | COMPLETE | DINOv2 + FinerCNN fusion to 192-dim |
| 5. Feature Matching | COMPLETE | 12-layer transformer with RoPE + dual-softmax |
| 6. Pose Estimation | COMPLETE | Weighted 8-point + Essential matrix + cheirality |
| 7. Loss Functions | COMPLETE | Matching (Eq. 12) + Pose (Eq. 13) + Combined (Eq. 14) |
| 8. Training Pipeline | COMPLETE | Training loop + keyframe selection + checkpointing |
| 9. Evaluation | COMPLETE | Trajectory accumulation + global scale + ATE metric |
| 10. TartanAir Data | **NEXT** | Download TartanAir + build dataloader |
| 11. TartanAir Training | TODO | Train on proper dataset as the paper does |
| 12. Final Evaluation | TODO | Re-evaluate on EuRoC, compare with paper |

---

## Repository Structure

```
.
├── CLAUDE.md              # Detailed implementation guide
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── configs/
│   └── default.yaml       # Training hyperparameters
│
├── data/                  # Dataset directory (not committed)
│   └── euroc/
│       └── MH_01_easy/
│
├── src/
│   ├── models/
│   │   ├── dino_vo.py           # Unified model wrapper
│   │   ├── keypoint_detector.py # Salient Keypoint Detector (Sec III-A)
│   │   ├── feature_descriptor.py# DINOv2 + FinerCNN (Sec III-B)
│   │   ├── finer_cnn.py         # Lightweight CNN encoder
│   │   ├── feature_matching.py  # Transformer matching (Sec III-C)
│   │   └── pose_estimation.py   # Weighted 8-point (Sec III-D)
│   ├── datasets/
│   │   ├── euroc.py             # EuRoC dataset loader
│   │   └── transforms.py        # Image preprocessing
│   ├── utils/
│   │   └── stereo.py            # Stereo depth + GT correspondences
│   └── losses/
│       └── losses.py            # Matching + Pose losses (Eq. 12-14)
│
├── scripts/
│   ├── train.py                 # Training script
│   ├── verify_gpu.py            # GPU verification
│   ├── verify_dataset.py        # Dataset verification
│   └── test_stereo_depth.py     # Stereo depth verification
│
├── checkpoints/           # Saved models (epoch_01.pth ... epoch_14.pth)
└── outputs/               # Loss curves, visualizations
```

---

## Citation

```bibtex
@article{azhari2025dinovo,
  title={DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model},
  author={Azhari and Shim},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  month={July},
  note={arXiv:2507.13145v1}
}
```

---

## License

This implementation is for educational and research purposes. Please refer to the original paper and DINOv2 repository for licensing information.

---

## Acknowledgments

- **Original Paper**: Azhari & Shim (IEEE RA-L 2025)
- **DINOv2**: Meta AI Research ([facebookresearch/dinov2](https://github.com/facebookresearch/dinov2))
- **EuRoC Dataset**: ETH Zurich ASL ([EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets))
- **Inspiration**: LightGlue, XFeat, ORB-SLAM3
