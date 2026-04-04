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
