# DINO-VO Debug Audit — 2026-03-31

## Training Symptom
- Epochs 1-4 (matching only): loss converges 6.29 -> 4.52, zero NaN. GOOD.
- Epochs 5-8 (pose added): pose_raw stays flat ~540, matching degrades 4.52 -> 6.60. BAD.
- Pose loss contributes NO learning signal. Model is effectively only doing matching.

---

## BUG #1 (ROOT CAUSE): SVD Backward Degeneracy — Pose Gradients Are Dead

**File**: `src/models/pose_estimation.py`

The Essential matrix always has singular values `(s, s, 0)`. PyTorch's SVD backward
involves `1/(s_i^2 - s_j^2)`, which is `1/0 = NaN` when `s_1 = s_2`.

The gradient path from pose loss:

```
pose_loss -> dL/d(R,t)
  -> decompose_essential SVD backward   [NaN - degenerate]
  -> enforce_essential SVD backward     [NaN - degenerate]  
  -> weighted_eight_point SVD backward  [FINE - non-degenerate]
  -> weights (matching network)
```

Our workaround `_sanitize_svd_grad` (line 37-49) replaces NaN with 0.
This **destroys the gradient** — the pose loss cannot backpropagate to the model.

**The paper references DSAC (Brachmann et al.)** which uses **gradient clamping**
(clamp the `1/(si^2-sj^2)` terms to a max value) instead of zeroing. Clamping
preserves gradient DIRECTION while bounding magnitude. Zeroing removes the
gradient entirely.

**Fix approach**: Implement a custom SVD autograd function with clamped gradients
(DSAC-style). This is a reasonable inference from the paper's DSAC reference.

---

## BUG #2: Rotation Logarithm Factor-of-2 Error

**File**: `src/losses/losses.py`, function `_log_rotation` (lines 38-70)

**Correct Rodrigues formula**:
```
[omega]_x = (R - R^T) * theta / (2 * sin(theta))
```

**Our code**:
```python
skew = (R - R.transpose(-1, -2)) / 2.0       # divides by 2
scale = theta / (2.0 * sin_theta)              # divides by 2 AGAIN
rot_vec = [wx, wy, wz] * scale
```

**Result**: rotation vector = `(R - R^T) * theta / (4 * sin(theta))` — exactly HALF correct.

**Proof**: For rotation by angle theta around z-axis:
- Our code gives wz = theta/2
- Correct answer: wz = theta

**Impact**: Effective `lambda_r = 90` instead of 180. Rotation loss contribution halved.
Not the root cause (gradient direction is correct), but loss magnitude is wrong.

**Fix**: Change `scale = theta / (2.0 * sin_theta)` to `scale = theta / sin_theta`

---

## BUG #3: `_enforce_essential_constraint` — Likely Not in Paper

**File**: `src/models/pose_estimation.py`, lines 180-213

The paper says "we use SVD decomposition to solve Eq. 11 for E" — it does NOT mention
projecting E onto the Essential manifold. Our code forces `S_new = (s_mean, s_mean, 0)`,
which GUARANTEES exact SVD degeneracy (making bug #1 worse).

**Fix**: Consider removing this step entirely. The raw E from 8-point is already
approximately Essential.

---

## BUG #4: FinerCNN Grayscale for Color Images

**File**: `src/models/finer_cnn.py`, line 99

```python
gray = image[:, 0:1] * self.img_std + self.img_mean  # uses channel 0 only
```

For TartanAir RGB images, channel 0 is just the red channel, not proper grayscale.

**Fix**: Use standard luminance: `0.299*R + 0.587*G + 0.114*B`

---

## Config Mismatch: lambda_p_max = 0.3 vs Paper's 0.9

**File**: `configs/tartanair_v02.yaml`, line 28

Intentionally capped as a workaround. Should be restored to 0.9 once pose
gradients work.

---

## What We CAN Fix (Without Contacting Authors)

| Fix | Confidence | Effort |
|-----|-----------|--------|
| `_log_rotation` factor-of-2 bug | 100% certain bug | 1 line change |
| FinerCNN proper grayscale | 100% correct improvement | 3 line change |
| Remove `_enforce_essential_constraint` | ~80% confidence paper doesn't use it | Remove function + 1 line |
| DSAC-style SVD clamping backward | ~85% confidence this is what paper does (they cite DSAC) | Custom autograd function, ~50 lines |
| Restore lambda_p_max to 0.9 | 100% matches paper | 1 line change |

---

## Questions for Paper Authors (REQUIRED)

Email to: Yassine Azhari and/or Dongwon Shim (authors of arXiv:2507.13145)

### Question 1: SVD backward for Essential matrix
The Essential matrix has degenerate singular values (sigma, sigma, 0). Standard PyTorch
SVD backward produces NaN via 1/(si^2 - sj^2) terms. How do you differentiate through
the E -> (R, t) decomposition? We noticed you reference DSAC [Brachmann et al.] —
do you use their gradient clamping approach? Or a custom autograd function?

### Question 2: Essential matrix projection
After the weighted 8-point algorithm, do you project E onto the Essential manifold
(force singular values to (1, 1, 0)) before decomposition? Or do you decompose the
raw E from the null space directly?

### Question 3: TartanAir training resolution
What image resolution do you resize TartanAir images to during training? The paper
specifies 476x742 for EuRoC but does not state the TartanAir resolution. The native
640x480 is not divisible by 14 (DINOv2 patch size).

### Question 4: TartanAir environments
Which TartanAir environments did you use for training? All of them, or a specific subset?

---

## Recommended Fix Order

1. Fix `_log_rotation` (trivial, certain)
2. Fix FinerCNN grayscale (trivial, certain)
3. Implement DSAC-style SVD backward (main engineering task)
4. Remove `_enforce_essential_constraint` (simplifies gradient path)
5. Restore lambda_p_max = 0.9
6. Retrain and evaluate

Steps 1-5 can be done without waiting for author response.
Step 3 is the critical one that should unlock pose loss convergence.

---

## Post-Fix Results (2026-04-04): v0.3 Retrain + Evaluation

All 4 bugs were fixed. Retrained from scratch on TartanAir, 14 epochs (v0.3).

### What the fixes solved:
- **Matching loss improved**: 4.0 (v0.3) vs 6.6 (v0.2 at same stage). FinerCNN grayscale
  fix and Essential projection removal helped matching learn better features.
- **NaN eliminated**: ClampedSVD works — 30 NaN/epoch is stable (degenerate pairs),
  not cascading. Gradients flow without explosion.
- **Training 2x faster**: ~3.2s/it vs ~6.7s/it.

### What the fixes did NOT solve:
- **Pose loss stays flat at ~500 for 10 consecutive epochs (5-14)**. NOT converging.
- With λ_t=400, λ_r=180, pose_raw≈500 means ~30 deg rotation error and ~random
  translation direction. The model is NOT learning pose.
- 30 NaN steps/epoch stays constant — never decreases.

### Evaluation on EuRoC MH_01 (epoch 13, 200 pairs):
- ATE RMSE: 0.503 m (paper target: ~0.05 m) — **10x off**
- Rotation error: 25.5 deg mean (paper target: ~2-5 deg)
- Only 257 matches/pair (out of 512 keypoints)
- Scale factor: 0.036 (translations 27x too large)
- Trajectory has rough shape but significant drift and wrong turns

### Root cause assessment:
The bug fixes were **necessary but not sufficient**. Gradient now flows from pose loss
to matching weights, but the signal is not producing useful learning. This points to
either:
1. **Weighted 8-point algorithm implementation bug** — never tested with known-correct inputs
2. **Match geometric accuracy too low** — matching loss optimizes assignment probability,
   not pixel-level accuracy needed for Essential matrix estimation
3. **Confidence weights (ConfMLP) not meaningful** — treating outliers same as inliers

### CRITICAL: Foundation never verified
The weighted 8-point algorithm has NEVER been tested with perfect GT correspondences.
If it can't recover correct pose from perfect inputs, no amount of training will help.
This is the #1 priority before any further work.

### Diagnostic tests required:
1. 8-point with GT correspondences → should recover correct R, t
2. Pose loss sanity check → L(R_gt, R_gt) must be 0
3. Compare with OpenCV findEssentialMat on same matches
4. Gradient magnitude check through full pose pipeline

---

## BUG #5 (2026-04-10): TartanAir NED-to-Camera Frame Conversion Missing

**File**: `src/datasets/tartanair.py`, function `_load_poses`

TartanAir camera motion is defined in the NED (North-East-Down) frame:
- NED: x=forward, y=right, z=down
- Camera: x=right, y=down, z=forward

Our dataloader loaded poses directly WITHOUT converting from NED to camera frame.
This caused a ~90° axis permutation in all GT poses, making every GT relative
pose fundamentally wrong. The matching loss still converged because GT correspondences
were generated from the SAME wrong poses (errors cancelled in reprojection), but
the pose loss target was garbage.

**Verification**: `scripts/verify_ned_frame_v3.py` — cross-frame depth consistency:
- Without NED conversion: 7.8% median relative depth error
- With NED conversion: 0.18% median relative depth error (100x better)

**Fix applied**: Conjugation `T_cam = T_ned2cam @ T_ned @ T_cam2ned` in `_load_poses()`.
The conversion matrix `T_ned2cam = [[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]]` was
confirmed from `tartanair_tools/evaluation/trajectory_transform.py`.

**Impact**: Pose loss dropped from ~500 to ~230. Matching loss improved from 4.0 to 1.35.
However, pose loss plateaued at ~230 — NED fix was necessary but NOT sufficient.

---

## Remaining Issue: Coordinate Normalization (2026-04-10)

**File**: `src/models/pose_estimation.py`, function `_pixel_to_normalized`

The paper (Eq. 10) specifies `x, y ∈ [0, 1]` — coordinates normalized by dividing
by image dimensions. Our code uses K^-1 camera coordinates (standard textbook approach
but different from what the paper describes).

**Status**: Confirmed discrepancy, not yet fixed. Will be applied in v0.5.

---

## Cross-Dataset Evaluation Summary (2026-04-11)

After NED fix + RANSAC training (v0.4), evaluated on both datasets:

| Dataset | ATE RMSE | Rotation Mean | Failure Mode |
|---------|----------|---------------|-------------|
| EuRoC MH_01 (200 pairs) | 0.615 m | 27.15° | Chaotic predictions |
| TartanAir carwelding/P001 (200 pairs) | 4.940 m | 29.35° | Smooth drift |

**Key insight**: TartanAir shows the model IS learning (trajectory shape is recognizable,
drift is smooth/monotonic). EuRoC shows chaotic breakdown (domain gap from synthetic
to real images). Two separate problems require two fixes:
1. Translation direction bias → [0,1] coordinate normalization
2. Domain gap → data augmentation during training

---

## v0.5 Results (2026-04-13): Coordinate Normalization Made Things WORSE

v0.5 applied [0,1] coordinate normalization (paper Eq. 10) + data augmentation
+ removed RANSAC. **Results were worse than v0.4 across the board.**

### v0.5 Training (7 completed epochs):

| Metric | v0.4 (ep 5-10) | v0.5 (ep 5-7) | Verdict |
|--------|---------------|---------------|---------|
| Matching loss | 1.81 → 1.35 | 1.88 → 1.83 | v0.5 worse |
| Pose loss (raw) | ~230 (flat) | ~390 (flat) | v0.5 much worse |
| NaN/epoch | ~30 | 0 | v0.5 better |

**Analysis**:
- [0,1] normalization throws away camera calibration info (focal length, principal
  point). K_inv normalization gives the 8-point algorithm geometrically meaningful
  coordinates. The paper's notation may not reflect the actual implementation.
- Removing RANSAC without trained confidence weights is premature — the ConfMLP
  can't filter outliers, so the 8-point algorithm sees garbage matches.
- Data augmentation is the one change worth keeping (for domain gap).

**Decision**: Revert to K_inv normalization and RANSAC for v0.6. Keep augmentation.

---

## Root Cause Confirmed: Chicken-and-Egg in ConfMLP (2026-04-14)

After 3 training versions (v0.3: ~500, v0.4: ~230, v0.5: ~390), the pose loss
plateau is definitively caused by the ConfMLP chicken-and-egg problem:

1. ConfMLP receives NO training signal during epochs 1-4 (only matching loss runs)
2. At epoch 5, confidence weights are random → all matches get equal weight
3. The ~50 outlier matches (out of ~300) corrupt the 8-point algorithm
4. Garbage poses → garbage pose loss gradients → ConfMLP gets meaningless signal
5. Weights stay random → cycle never starts

**Evidence**:
- Matching loss converges well (3.7 → 1.35) — model learns WHICH points match
- But ~17% of matches are still outliers, which is catastrophic for 8-point
- RANSAC partially breaks the cycle (pose 500→230) but isn't differentiable,
  so it can't teach the ConfMLP to replace it
- The fundamental problem: ConfMLP has NO supervision signal that doesn't go
  through the broken 8-point algorithm

---

## v0.6 Solution: Inlier Classification Loss (Confidence Pretraining)

**File**: `src/losses/losses.py`, class `InlierLoss`

Add a BCE (Binary Cross-Entropy) loss that trains the ConfMLP directly, using
depth-based reprojection to label each predicted match as inlier or outlier:

```
For each predicted match (p1 in image 1 → p2 in image 2):
  1. Back-project p1 to 3D using depth map
  2. Transform by GT pose: P' = R_gt @ P + t_gt
  3. Project to image 2: p2_expected = K @ P' / z
  4. If ||p2_expected - p2|| < 5px → inlier (label = 1)
  5. Else → outlier (label = 0)
  6. BCE(confidence_weight, label)
```

**Why this works**: Gives ConfMLP a direct training signal from epoch 1, without
going through the 8-point algorithm. By epoch 5, confidence weights already
distinguish inliers from outliers → 8-point gets clean weighted matches →
poses improve → pose loss provides useful fine-tuning signal.

**Proven technique**: SuperGlue and LightGlue both pretrain confidence/matching
before adding pose supervision.

**Config**: `configs/tartanair_v06.yaml`
**Checkpoints**: `checkpoints_v06_tartanair/`
