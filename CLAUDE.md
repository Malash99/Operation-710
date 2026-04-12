# CLAUDE.md — DINO-VO Reimplementation Project

## Project Overview

**Objective:** Reimplement DINO-VO from the paper "DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model" [Azhari & Shim, IEEE RA-L, July 2025].

**Paper Reference:** arXiv:2507.13145v1

**Dataset:** EuRoC MAV Dataset (starting with single sequence, e.g., MH01)

**Hardware:** RTX 5060 Ti (16GB VRAM), Intel i5-14400F

---

## CRITICAL RULES — READ BEFORE ANY ACTION

### Rule 1: NO FAKE IMPLEMENTATIONS
- **NEVER** create mock data, synthetic placeholders, or dummy implementations
- **NEVER** generate fake sensor readings, fake images, or fake ground truth
- **NEVER** use `random.rand()` or similar to simulate real data
- If data is missing, **STOP and ask the user** how to proceed
- If a component cannot be implemented, **STOP and explain why**
- Every piece of data must come from the actual EuRoC dataset

### Rule 2: EXPLAIN BEFORE CODING
- Before writing ANY code, explain:
  1. **WHAT** we are implementing (component name, purpose)
  2. **WHY** this component exists (its role in the pipeline)
  3. **HOW** it connects to other components
  4. **REFERENCE** to the specific section/equation in the paper
- Wait for user acknowledgment before proceeding with implementation

### Rule 3: INCREMENTAL DEVELOPMENT
- Implement ONE component at a time
- Test each component before moving to the next
- Never implement multiple modules in a single step
- If a test fails, fix it before proceeding

### Rule 4: GPU UTILIZATION
- All tensor operations must be GPU-compatible
- Use `torch.cuda.is_available()` checks
- Implement proper device management (`.to(device)`)
- Monitor VRAM usage — stay under 14GB to leave headroom

### Rule 5: NO DOCKER
- All setup is local, no containerization
- Use conda/pip for dependencies
- Document all system requirements clearly

---

## Project Structure

```
dino-vo/
├── CLAUDE.md                 # This file
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup (optional)
│
├── configs/
│   └── default.yaml          # Training/inference configuration
│
├── data/
│   ├── euroc/                # EuRoC dataset (downloaded, not committed)
│   │   └── MH_01_easy/
│   │       ├── mav0/
│   │       │   ├── cam0/     # Left camera images
│   │       │   ├── cam1/     # Right camera images (not used for mono)
│   │       │   ├── imu0/     # IMU data (not used in DINO-VO)
│   │       │   └── state_groundtruth_estimate0/  # Ground truth poses
│   │       └── ...
│   └── README.md             # Dataset download instructions
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dino_vo.py        # Main DINO-VO model class
│   │   ├── keypoint_detector.py    # Salient Keypoint Detector (Sec III-A)
│   │   ├── feature_descriptor.py   # DINOv2 + FinerCNN (Sec III-B)
│   │   ├── finer_cnn.py      # Lightweight CNN encoder
│   │   ├── feature_matching.py     # Transformer-based matching (Sec III-C)
│   │   └── pose_estimation.py      # Differentiable pose layer (Sec III-D)
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── euroc.py          # EuRoC dataset loader
│   │   └── transforms.py     # Image preprocessing
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py       # Geometric utilities (Essential matrix, SVD, etc.)
│   │   ├── visualization.py  # Trajectory plotting, match visualization
│   │   └── metrics.py        # ATE computation
│   │
│   └── losses/
│       ├── __init__.py
│       └── losses.py         # Matching loss (Eq. 12) and Pose loss (Eq. 13)
│
├── scripts/
│   ├── download_euroc.py     # Dataset download script
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── visualize.py          # Visualization script
│
├── notebooks/                # Jupyter notebooks for exploration (optional)
│
├── checkpoints/              # Saved model weights
│
├── outputs/                  # Evaluation outputs (trajectories, plots)
│
└── tests/                    # Unit tests for components
    ├── test_detector.py
    ├── test_descriptor.py
    ├── test_matching.py
    └── test_pose.py
```

---

## Implementation Order

Follow this order strictly. Do not skip steps.

### Phase 1: Environment Setup
1. Create project directory structure
2. Create `requirements.txt` with dependencies
3. Verify GPU availability
4. Download EuRoC MH01 sequence

### Phase 2: Data Pipeline
5. Implement EuRoC dataset loader (images + ground truth)
6. Implement image preprocessing (resize to 476×742 as per paper)
7. Test: Load and visualize a single image pair

### Phase 3: Keypoint Detector (Paper Section III-A)
8. Implement Gaussian filter + Sobel gradient computation
9. Implement grid-based MaxPooling (kernel=14, stride=14)
10. Implement Non-Maximum Suppression (NMS) with radius=8
11. Implement gradient thresholding + top-k selection
12. Test: Visualize detected keypoints on sample image

### Phase 4: Feature Descriptor (Paper Section III-B)
13. Load pretrained DINOv2-ViT-S (frozen, from torch hub)
14. Implement FinerCNN encoder (basic layers from XFeat architecture)
15. Implement feature fusion (concatenation + linear projection)
16. Test: Extract features for detected keypoints

### Phase 5: Feature Matching (Paper Section III-C)
17. Implement transformer-based matching layer (based on LightGlue)
18. Implement self-attention with rotary positional encoding
19. Implement cross-attention between image pairs
20. Implement soft assignment matrix computation (Eq. 5-8)
21. Implement confidence prediction MLP (Eq. 9)
22. Test: Match keypoints between consecutive frames

### Phase 6: Pose Estimation (Paper Section III-D)
23. Implement weighted 8-point algorithm
24. Implement Essential matrix decomposition
25. Implement cheirality check for pose selection
26. Test: Estimate relative pose from matches

### Phase 7: Loss Functions (Paper Section III-E)
27. Implement matching loss (Eq. 12)
28. Implement pose loss (Eq. 13)
29. Implement combined objective (Eq. 14)

### Phase 8: Training Pipeline
30. Implement training loop with proper logging
31. Implement keyframe selection logic (24px threshold)
32. Implement learning rate scheduling
33. Test: Run single training iteration

### Phase 9: Evaluation
34. Implement trajectory accumulation
35. Implement ATE metric computation
36. Run evaluation on EuRoC MH01
37. Compare with paper's reported results

### Phase 10: Documentation
38. Write comprehensive README.md
39. Document all functions and classes
40. Create usage examples

---

## Key Implementation Details from Paper

### Salient Keypoint Detector (Section III-A)
- GaussianFilter: kernel_size=5, std=2.0
- MaxPooling: kernel_size=14 (matches DINOv2 patch size), stride=14
- NMS radius: rNMS=8
- Gradient threshold: 0.01
- Top-k keypoints: 512

### Feature Descriptor (Section III-B)
- DINOv2-ViT-S encoder: output shape H/14 × W/14 × 384
- FinerCNN: outputs H × W × 64
- Final descriptor dimension: 192 (after Linear projection)

### Feature Matching (Section III-C)
- Transformer layers: L=12
- Attention heads: 3
- Head dimension: 64
- Uses rotary positional encoding

### Pose Estimation (Section III-D)
- Weighted 8-point algorithm
- SVD for Essential matrix computation
- Cheirality check for pose disambiguation

### Training Details (Section IV-A)
- Image resolution for EuRoC: 476×742
- First 4 epochs: matching loss only
- Next 10 epochs: combined loss
- λr=180, λt=400
- λp: 0.0 → 0.9 (increment 1.5e-4 per step from epoch 5)

### Keyframe Selection (Section III-F)
- Keyframe when mean pixel displacement > 24px
- For EuRoC: process alternate frames

---

## Dependencies

```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
scipy>=1.10.0

# DINOv2
# Loaded via torch.hub from facebookresearch/dinov2

# Configuration
pyyaml>=6.0
omegaconf>=2.3.0

# Visualization
matplotlib>=3.7.0
tqdm>=4.65.0

# Evaluation
evo>=1.20.0  # For trajectory evaluation (optional, can implement manually)

# Development
jupyter>=1.0.0
pytest>=7.3.0
```

---

## GPU Memory Considerations

**RTX 5060 Ti (16GB VRAM) Budget:**

| Component | Estimated VRAM |
|-----------|----------------|
| DINOv2-ViT-S (frozen) | ~350MB |
| FinerCNN | ~50MB |
| Feature Matching (12 layers) | ~200MB |
| Image pair (476×742) | ~10MB |
| Intermediate activations | ~2-4GB |
| **Total (training)** | ~4-6GB |
| **Headroom** | ~10GB |

**Optimization if needed:**
- Use FP16 (automatic mixed precision)
- Gradient checkpointing for transformer layers
- Reduce batch size (paper doesn't specify, likely 1-4)

---

## Checkpoints and Verification

After each component, verify:

1. **Keypoint Detector:**
   - Outputs 512 keypoints per image
   - Keypoints are distributed across image (not clustered)
   - Keypoints align to 14×14 grid

2. **Feature Descriptor:**
   - DINOv2 features: shape (K, 384)
   - FinerCNN features: shape (K, 64)
   - Combined features: shape (K, 192)

3. **Feature Matching:**
   - Assignment matrix P: shape (K, K)
   - Correspondences: list of (i, j) pairs
   - Confidences: values in [0, 1]

4. **Pose Estimation:**
   - Rotation R: valid SO(3) matrix
   - Translation t: unit vector (up-to-scale)
   - Passes cheirality check

---

## Communication Protocol

When implementing each component:

1. **Claude says:** "I'm about to implement [COMPONENT NAME]. This is [DESCRIPTION]. In the paper, this is described in [SECTION]. It connects to [OTHER COMPONENTS] because [REASON]. Should I proceed?"

2. **User reviews and approves**

3. **Claude implements with comments referencing paper equations**

4. **Claude proposes a test:** "To verify this works, I'll [TEST DESCRIPTION]. This should output [EXPECTED RESULT]."

5. **User runs test and confirms**

6. **Move to next component**

---

## What NOT To Do

- ❌ Generate synthetic images or fake data
- ❌ Use random numbers to simulate features
- ❌ Skip components or implement stubs
- ❌ Combine multiple implementation steps
- ❌ Proceed without user confirmation
- ❌ Ignore paper specifications (use exact hyperparameters)
- ❌ Use Docker or containerization
- ❌ Download models without user consent
- ❌ Exceed GPU memory limits
- ❌ Implement features not in the paper

---

## Reference Equations from Paper

### Equation 1: Feature Concatenation
```
f_i = Linear([f_DINO_i | f_FINE_i]) ∈ R^192
```

### Equations 2-4: Attention Mechanism
```
f_i^T ← f_i^T + MLP([f_i^T | m_i^(T←S)])
m_i^(T←S) = Σ_j Softmax(a_ij^TS)_j * v_j
```

### Equations 5-8: Assignment Matrix
```
P_ij = σ_i * σ_j * Softmax_k(S_kj)_i * Softmax_k(S_ik)_j
S_ij = Linear(f_i^It)^T * Linear(f_j^It+1)
σ_i = Sigmoid(Linear(f_i^It))
```

### Equation 9: Confidence
```
w_ij = ConfMLP([f_i^It | f_j^It+1])
```

### Equations 10-11: Essential Matrix
```
x_j^T * E * x_i = 0
diag(w) * Φ * flat(E) = 0
```

### Equation 12: Matching Loss
```
L_m = -(1/L) Σ_l [ (1/|M|) Σ_(i,j)∈M log(P_ij) + ... ]
```

### Equation 13: Pose Loss
```
L_p = λ_t * ||t̂/||t̂|| - t/||t||| + λ_r * ||Log(R̂) - Log(R)||
```

### Equation 14: Total Loss
```
L_t = (1 - λ_p) * L_m + λ_p * L_p
```

---

## Current Implementation Status

### ✅ Phase 1: Environment Setup (COMPLETED)

**Date Completed**: February 5, 2026

#### Completed Tasks:
1. ✅ Created complete project directory structure
2. ✅ Created `requirements.txt` with all dependencies
3. ✅ Installed PyTorch nightly (2.11.0.dev20260205+cu128) for RTX 5060 Ti
4. ✅ Verified GPU functionality (16GB VRAM available)
5. ✅ Downloaded EuRoC Machine Hall dataset (all 5 sequences)
6. ✅ Verified MH_01_easy structure (3,682 images, ground truth, calibration)

#### Important Notes:

**RTX 50 Series GPU Support (CRITICAL)**:
- RTX 5060 Ti uses compute capability sm_120 (Blackwell architecture)
- Stable PyTorch releases (≤2.9.x) do NOT support sm_120
- **SOLUTION**: Use PyTorch nightly with CUDA 12.8:
  ```bash
  pip uninstall torch torchvision torchaudio -y
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- This is a known issue tracked at: https://github.com/pytorch/pytorch/issues/164342
- For new Claude instances: Always check PyTorch version and GPU compatibility first

**Dataset Setup**:
- EuRoC dataset moved to ETH Research Collection
- Old server (robotics.ethz.ch/~asl-datasets) may timeout
- Download from: https://www.research-collection.ethz.ch/handle/20.500.11850/690084
- machine_hall.zip contains nested zips - extract MH_XX_easy.zip files individually
- Dataset location: `data/euroc/MH_01_easy/mav0/`

**Verified Hardware Configuration**:
- GPU: NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)
- CUDA: 12.8
- PyTorch: 2.11.0.dev20260205+cu128
- Compute Capability: 12.0 (sm_120)
- Estimated VRAM usage during training: 4-6 GB
- Available headroom: ~10 GB

**Dataset Statistics (MH_01_easy)**:
- Images (cam0): 3,682 frames @ 752×480 grayscale
- Images (cam1): 3,682 frames (for stereo extension)
- IMU measurements: 36,820 @ 200Hz (for VIO extension)
- Ground truth poses: 36,382
- Duration: ~184 seconds
- Trajectory length: 0.25 meters (straight-line distance)

#### Scripts Created:
- `scripts/verify_gpu.py` - GPU capability verification
- `scripts/install_pytorch_rtx50.py` - Automated PyTorch installation for RTX 50 series
- `scripts/download_euroc.py` - Dataset downloader with progress bar
- `scripts/verify_dataset.py` - Dataset structure verification
- `scripts/visualize_sample.py` - Quick dataset visualization

---

### ✅ Phase 2: Data Pipeline (COMPLETED)

**Date Completed**: February 10, 2026

#### Completed Tasks:
1. ✅ Implemented image transforms (`src/datasets/transforms.py`)
   - Undistortion via `cv2.undistort` using calibration from sensor.yaml
   - Resize: 752×480 → 742×476 (W×H) as per paper Section IV-A
   - Grayscale → RGB (3-channel repeat for DINOv2)
   - ImageNet normalization (mean/std required by DINOv2)
   - Intrinsics rescaling after resize

2. ✅ Implemented EuRoC dataset loader (`src/datasets/euroc.py`)
   - Parses cam0/sensor.yaml for K, distortion, T_BS extrinsics
   - Parses cam0/data.csv for image timestamps and filenames
   - Parses state_groundtruth_estimate0/data.csv for GT poses
   - Matches image timestamps to nearest GT via np.searchsorted (25ms threshold)
   - Computes relative camera pose: T_1to2 = inv(T_WC2) @ T_WC1
   - Coordinate frames: T_WC = T_WB @ T_BS (body→world @ cam→body)
   - skip_frames=2 (alternate frames, as per paper Section III-F)

3. ✅ Verified with test script (`scripts/test_dataloader.py`)
   - Dataset length: 3,637 pairs
   - Image tensors: shape [3, 476, 742], values in [-2.1, 2.6] (normalized)
   - Rescaled intrinsics: fx=452.55, fy=453.49, cx=362.33, cy=246.31
   - Rotation validity: det(R)=1.000000, max|R@R^T - I|=5.96e-08
   - Translation magnitude: ~0.079m (first pair, 100ms apart)
   - Visualization saved: outputs/dataloader_test.png

#### Important Notes:

**EuRoC Coordinate Frames**:
- Ground truth gives T_WB (body in world frame)
- cam0/sensor.yaml T_BS transforms camera→body (NOT identity!)
- Camera in world: T_WC = T_WB @ T_BS
- Relative pose: T_1to2 = inv(T_WC2) @ T_WC1

**EuRoC Quaternion Convention**:
- EuRoC stores (qw, qx, qy, qz) in CSV columns 5-8
- scipy.Rotation.from_quat expects (qx, qy, qz, qw) — reorder required

**Dataset GPU Notes**:
- Dataloader returns CPU tensors (standard PyTorch practice)
- `.to(device)` is applied in the training loop (Phase 8)
- All model computation (Phases 3-9) will run on GPU

#### Files Created:
- `src/datasets/transforms.py` — Image preprocessing pipeline
- `src/datasets/euroc.py` — EuRoC PyTorch Dataset class
- `scripts/test_dataloader.py` — Data pipeline verification script

---

### ✅ Phases 3-9: All Pipeline Components (COMPLETED)

**Date Range**: February–March 2026

All core pipeline phases are complete:
- **Phase 3**: Keypoint Detector — Gaussian + Sobel + MaxPool + NMS + Top-k (512)
- **Phase 4**: Feature Descriptor — DINOv2-ViT-S/14 (frozen) + FinerCNN + Linear → 192-d
- **Phase 5**: Feature Matching — 12-layer transformer with RoPE + dual-softmax
- **Phase 6**: Pose Estimation — Weighted 8-point + Essential matrix + cheirality check
- **Phase 7**: Loss Functions — Matching (Eq. 12) + Pose (Eq. 13) + Combined (Eq. 14)
- **Phase 8**: Training Pipeline — Training loop + keyframe selection + checkpointing
- **Phase 9**: Evaluation — Trajectory accumulation + global scale + ATE metric

#### Paper Audit (2026-03-25): 5 bugs found and fixed
1. **CRITICAL**: RoPE coordinate normalization swapped (x÷h, y÷w → x÷w, y÷h)
2. **CRITICAL**: Epipolar Phi matrix in column-major → fixed to row-major
3. **CRITICAL**: Cheirality check missing negative sign (hidden by bug #2)
4. **MEDIUM**: L2 normalization on descriptors removed (not in paper)
5. **MEDIUM**: Score matrix √d scaling removed (not in paper)

#### Training Run 2 Results (EuRoC MH_01, 14 epochs)
- Matching loss: 8.9 → 3.0 (converges)
- Pose loss: ~370 (does NOT converge — SVD backward NaN, 36% of steps)
- Root cause: Small EuRoC motions + SVD backward instability

#### Evaluation Results (v0.1, epoch 4, 50 pairs)
- ATE RMSE: 0.41 m, Rotation: 9.2 deg mean
- Trajectory shape does not follow GT (expected without pose loss convergence)

#### Tagged Version
- **`v0.1-euroc-baseline`** — all code + 14 checkpoints saved
- See `VERSIONS.md` for details

---

### ✅ Phase 10: TartanAir Dataset + Training (COMPLETED — partial)

**Why TartanAir**: The paper trains ONLY on TartanAir (Section IV-A). EuRoC is evaluation-only.

#### Completed:
- Downloaded TartanAir (5 environments: carwelding, japanesealley, ocean, office, office2)
- Built TartanAir dataloader (`src/datasets/tartanair.py`)
- 41 trajectories, 34309 pairs total, training uses 8500 pairs
- Ran 8 epochs of training (checkpoints in `checkpoints_v02_tartanair/`)

#### Training Results (v0.2, 8 epochs):
- Epochs 1-4 (matching only): loss converges 6.29 → 4.52, zero NaN
- Epochs 5-8 (pose added): pose_raw stays flat ~540, matching degrades to 6.6
- **Pose loss does NOT converge** — root cause identified (see Debug Audit below)

---

### 🔧 Debug Audit (2026-03-31): 4 bugs found

See `DEBUG_AUDIT_2026-03-31.md` for full details.

#### Bug 1 — CRITICAL (ROOT CAUSE): SVD backward degeneracy
- Essential matrix has singular values (s, s, 0)
- PyTorch SVD backward involves 1/(si²-sj²) → NaN when s1=s2
- Old workaround (`_sanitize_svd_grad`) zeroed NaN → destroyed gradient entirely
- **FIX APPLIED**: Implemented DSAC-style `ClampedSVD` autograd function
  that clamps 1/(si²-sj²) to [-1e6, 1e6] instead of NaN
- Verified: pose loss gradient now flows to all weights (was dead before)

#### Bug 2 — `_log_rotation` factor-of-2 error
- `scale = theta / (2 * sin_theta)` should be `theta / sin_theta`
- The `/ 2` in `skew = (R - R^T) / 2` already accounts for the factor
- Result was exactly half correct → effective λ_r=90 instead of 180
- **FIX APPLIED**: Changed scale formula

#### Bug 3 — `_enforce_essential_constraint` unnecessary
- Paper does NOT mention projecting E onto Essential manifold
- This step forced exact SVD degeneracy, making Bug 1 worse
- **FIX APPLIED**: Removed — raw E from 8-point is decomposed directly

#### Bug 4 — FinerCNN grayscale for color images
- Used channel 0 (red only) instead of proper grayscale
- **FIX APPLIED**: Uses ITU-R BT.601 luminance (0.299R + 0.587G + 0.114B)

#### Config fix:
- `lambda_p_max` restored from 0.3 to 0.9 (paper value)

---

### ✅ Phase 10b: Retrain with Bug Fixes (COMPLETED)

**Date Completed**: April 4, 2026

All four bugs fixed. Retrained from scratch on TartanAir (v0.3), 14 epochs.
Checkpoints saved in `checkpoints_v03_tartanair/`.

#### v0.3 Training Results (14 epochs, 1417 steps/epoch, lambda_p_max=0.9):

| Epoch | avg_match | avg_pose | lp | nan_skipped |
|-------|-----------|----------|----|-------------|
| 1 | 5.42 | 0.0 | 0.0 | 0 |
| 2 | 4.72 | 0.0 | 0.0 | 0 |
| 3 | 4.47 | 0.0 | 0.0 | 0 |
| 4 | 4.21 | 0.0 | 0.0 | 0 |
| 5-8 | ~4.0 | ~510 | 0.0→0.9 | 30 |
| 9 | 4.06 | 512.8 | 0.9 | 30 |
| 10 | 4.10 | 506.7 | 0.9 | 30 |
| 11 | 4.08 | 501.2 | 0.9 | 30 |
| 12 | 4.04 | 506.5 | 0.9 | 30 |
| 13 | 4.00 | 502.2 | 0.9 | 30 |
| 14 | ~4.1 | ~500 | 0.9 | 30 |

**What improved vs v0.2**:
- Matching loss: 4.0 (v0.3) vs 6.6 (v0.2) — bug fixes helped matching
- NaN: 30/epoch stable (v0.3) vs 0 but only because lp was capped at 0.3 (v0.2)
- ClampedSVD working — gradients flow, no explosion
- Training 2x faster (~3.2s/it vs ~6.7s/it)

**What did NOT improve**:
- **Pose loss stays flat at ~500 for 10 epochs** — NOT converging
- 30 NaN steps/epoch constant (degenerate pairs)
- Bug fixes were necessary but NOT sufficient for pose convergence

#### v0.3 Evaluation on EuRoC MH_01 (epoch 13, 200 pairs):

| Metric | v0.3 Result | Paper Target |
|--------|-------------|-------------|
| ATE RMSE | 0.503 m | ~0.05 m |
| ATE Mean | 0.423 m | — |
| Rotation Error (mean) | 25.5 deg | ~2-5 deg |
| Avg matches/pair | 257 | ~512 |
| Scale factor | 0.036 | ~1.0 |

- Trajectory plot: `outputs/trajectory_epoch_13.png`
- Results text: `outputs/eval_epoch_13.txt`
- Predicted trajectory has rough shape but diverges significantly from GT
- ~10x off from paper's reported results

---

### 🔧 Diagnosis (2026-04-04): Pose Pipeline Foundation Unverified

**Core finding**: Bug fixes solved NaN/gradient flow, but pose loss (~500) is flat.
This means the gradient FROM pose loss reaches the matching weights but does NOT
produce useful learning signal. The matches improve (loss 4.0) but are not
geometrically accurate enough for the 8-point algorithm.

**Critical question**: Does our weighted 8-point algorithm work correctly at all?
We have NEVER tested it with known-correct inputs.

#### Diagnostic Tests Needed (before any further training):

1. **Test 1 — 8-point with GT correspondences**: Feed perfect pixel correspondences
   (from GT pose + depth reprojection) with uniform weights. Should recover correct R, t.
   If fails → implementation bug in pose_estimation.py.

2. **Test 2 — Pose loss sanity check**: Compute L_pose(R_gt, t_gt, R_gt, t_gt).
   Must be exactly 0. Small perturbations → small loss proportional to perturbation.

3. **Test 3 — Compare with OpenCV**: Take model's matched keypoints, run
   cv2.findEssentialMat with RANSAC. If OpenCV gets good poses but ours doesn't
   → our 8-point is wrong. If both fail → matches are bad.

4. **Test 4 — Gradient flow magnitude**: Check torch.autograd.grad(pose_loss,
   matching_weights) — is gradient nonzero and reasonable magnitude?

**Why this matters**: If the 8-point algorithm itself is broken, no amount of
training will fix pose convergence. Must verify foundation before ANY enhancements.

---

### ✅ Phase 11: Foundation Verification (COMPLETED)

**Date Completed**: April 4, 2026

Ran 4 diagnostic tests (script: `scripts/diagnostic_tests.py`).
Full report: `outputs/DIAGNOSTIC_REPORT_2026-04-04.md`

#### Results: 10 PASS, 1 FAIL, 1 SKIP

**8-point algorithm: CORRECT on clean data**
- Synthetic (100 perfect correspondences): 0.04 deg rotation, 0.00 deg translation
- The implementation is mathematically correct — NOT the root cause

**8-point algorithm: FAILS with outliers or noisy matches**
- 20% outlier matches: 3.8 deg rot, 14.7 deg trans (OpenCV RANSAC: 0.0 deg)
- Real EuRoC optical flow matches: 3.2 deg rot, **112 deg trans** (catastrophic)
- Translation is the weakest link — extremely sensitive to outlier matches

**Pose loss: CORRECT**
- L(R_gt, t_gt, R_gt, t_gt) = 0.0000 (exact)
- _log_rotation: exact for 5/30/90/170 deg
- Proportional scaling verified for both rotation and translation

**Gradient flow: WORKS but partial**
- 67.7% of parameters receive nonzero gradient from pose loss
- Deeper transformer layers (8-11) get ZERO gradient
- Gradient reaches conf_mlp (largest) and early matching layers

#### Root Cause Identified: Chicken-and-Egg Problem

The paper relies on **learned confidence weights** (ConfMLP) to replace RANSAC.
The 8-point algorithm needs good weights to downweight outlier matches.
But the weights can only learn from pose loss, which needs good weights to
produce meaningful poses. **Circular dependency.**

Training observation: pose_raw ~500 corresponds to ~90 deg translation error
(confirmed by Test 2c: 90 deg -> loss 565). The model predicts near-random
translation directions because matches contain outliers that the uniform
confidence weights can't filter.

---

### ✅ Phase 12: RANSAC Pre-filtering + NED Fix + Evaluation (COMPLETED)

**Date Completed**: April 12, 2026

#### Bug 5 — CRITICAL: TartanAir NED-to-Camera Frame Conversion Missing

TartanAir poses are in NED frame (x=forward, y=right, z=down) but our code
loaded them directly as camera frame (x=right, y=down, z=forward). This caused
~90° axis permutation in GT poses, making pose loss plateau at ~500.

**Fix**: Added conjugation in `src/datasets/tartanair.py:_load_poses()`:
```python
T_cam = T_ned2cam @ T_ned @ T_cam2ned
```
where `T_ned2cam = [[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]]`

**Verified** with depth consistency test (`scripts/verify_ned_frame_v3.py`):
0.18% median depth error WITH NED conversion vs 7.8% WITHOUT.

#### RANSAC Pre-filtering (v0.4)

- Config: `configs/tartanair_v04.yaml`
- Checkpoints: `checkpoints_v04_tartanair/` (14 epochs)
- OpenCV RANSAC zeros outlier weights before 8-point (epochs 5+)
- RANSAC mask detached — gradient flows through inlier weights only

#### v0.4 Training Results (14 epochs):

| Metric | v0.3 (no NED fix) | v0.4 (NED fix + RANSAC) |
|--------|-------------------|------------------------|
| Matching loss (ep 4) | 4.21 | 2.04 |
| Matching loss (ep 10) | 4.10 | 1.35 |
| Pose loss (ep 10) | 506.7 | ~230 |
| NaN/epoch | 30 | ~30 |

NED fix dramatically improved matching (4.0 → 1.35) and reduced pose loss
(500 → 230), but pose loss plateaued at ~230 — did NOT fully converge.

#### v0.4 Evaluation on EuRoC MH_01 (epoch 11, 200 pairs):

| Metric | v0.4 Result | Paper Target |
|--------|-------------|-------------|
| ATE RMSE | 0.615 m | 0.150 m |
| Rotation Error (mean) | 27.15 deg | ~2-5 deg |
| Avg matches/pair | 314 | ~512 |
| Scale factor | 0.036 | — |

#### v0.4 Evaluation on TartanAir carwelding/Easy/P001 (epoch 14, 200 pairs):

| Metric | v0.4 Result | Observation |
|--------|-------------|-------------|
| ATE RMSE | 4.940 m | 15% of 33m path length |
| Rotation Error (mean) | 29.35 deg | Similar to EuRoC |
| Avg matches/pair | 299 | — |
| Scale factor | 0.166 | — |

#### Key Findings from Cross-Dataset Evaluation:

1. **TartanAir trajectory shape is roughly correct** — predicted trajectory
   follows GT shape with systematic drift (visible in plots)
2. **EuRoC trajectory is chaotic** — random-looking predictions, not drift
3. Rotation error is ~27-29° on BOTH datasets but failure modes differ:
   - TartanAir: smooth monotonic drift (model partially learned)
   - EuRoC: chaotic oscillation (domain gap)
4. **Two separate problems identified**:
   - Model-level: systematic translation direction bias (~33° per pair)
   - Domain gap: synthetic TartanAir → real EuRoC breaks the matcher

#### Diagnostic Scripts Created:
- `scripts/verify_ned_frame.py` — NED verification attempt 1
- `scripts/verify_ned_frame_v2.py` — NED verification attempt 2
- `scripts/verify_ned_frame_v3.py` — **Correct** depth consistency verification
- `scripts/diagnose_pose_plateau.py` — Per-pair pose breakdown on TartanAir
- `scripts/evaluate_tartanair.py` — TartanAir trajectory evaluation

#### Evaluation Outputs:
- `outputs/trajectory_epoch_11.png` — EuRoC trajectory plot (v0.4)
- `outputs/trajectory_tartanair_P001_epoch_14.png` — TartanAir trajectory plot (v0.4)
- `outputs/eval_epoch_11.txt` — EuRoC metrics
- `outputs/eval_tartanair_P001_epoch_14.txt` — TartanAir metrics

---

### 🚧 Phase 13: Coordinate Normalization Fix + Data Augmentation (v0.5)

#### Problem: Two remaining issues identified

1. **Coordinate normalization discrepancy** — Paper Eq. 10 specifies `x, y ∈ [0,1]`
   for the 8-point algorithm. Our `pose_estimation.py:_pixel_to_normalized()` uses
   K^-1 camera coordinates instead. This is the only confirmed code discrepancy
   with the paper that hasn't been fixed.

2. **No data augmentation** — TartanAir is visually clean (synthetic). EuRoC has
   motion blur, variable lighting, aggressive motion. The matcher overfits to
   clean synthetic visual properties.

#### Planned Fixes for v0.5:
1. Change `_pixel_to_normalized()` to use [0,1] normalization (divide by image dimensions)
2. Add color jitter + gaussian noise augmentation in TartanAir dataloader
3. Disable RANSAC (paper doesn't use it — and v0.4 proved it didn't help)
4. Use full dataset (34,309 pairs instead of 8,500)
5. Retrain from scratch

#### Config: `configs/tartanair_v05.yaml`
#### Checkpoints: `checkpoints_v05_tartanair/`
#### Outputs: `outputs_v05_tartanair/`

All v0.4 files are preserved for comparison. Nothing is overwritten.

---

## Version History

| Version | Config | Checkpoints | Key Change | Pose Loss |
|---------|--------|-------------|------------|-----------|
| v0.1 | configs/default.yaml | checkpoints/ | EuRoC baseline | ~370 (NaN) |
| v0.2 | configs/tartanair_v02.yaml | checkpoints_v02_tartanair/ | TartanAir training | ~540 |
| v0.3 | configs/tartanair_v03.yaml | checkpoints_v03_tartanair/ | 4 bug fixes (SVD, log_rot, etc.) | ~500 |
| v0.4 | configs/tartanair_v04.yaml | checkpoints_v04_tartanair/ | NED fix + RANSAC | ~230 |
| v0.5 | configs/tartanair_v05.yaml | checkpoints_v05_tartanair/ | [0,1] coords + augmentation | TBD |

---

## Start Command

"Phase 13 in progress (v0.5). Fixes to apply:
  1. [0,1] coordinate normalization in pose_estimation.py
  2. Data augmentation in tartanair.py
  3. Disable RANSAC, use full dataset
  Config: configs/tartanair_v05.yaml
  Previous versions preserved: v0.4 checkpoints in checkpoints_v04_tartanair/
  Run: python -m scripts.train --config configs/tartanair_v05.yaml
  Evaluate EuRoC: python -m scripts.evaluate --checkpoint checkpoints_v05_tartanair/epoch_14.pth --config configs/default.yaml --max_pairs 200
  Evaluate TartanAir: python -m scripts.evaluate_tartanair --checkpoint checkpoints_v05_tartanair/epoch_14.pth --trajectory data/tartanair/carwelding/Easy/P001 --max_pairs 200"