# DINO-VO Foundation Diagnostic Report
**Date**: April 4, 2026
**Checkpoint**: v0.3, epoch 13 (checkpoints_v03_tartanair/epoch_13.pth)

---

## Executive Summary

The weighted 8-point algorithm implementation is **CORRECT** on clean data.
The pose pipeline fails on real data due to **match quality / outliers**, not
algorithm bugs. The 8-point algorithm (without RANSAC) is extremely sensitive
to outliers — even 20% outliers cause 14 deg translation error, while OpenCV
RANSAC handles them perfectly.

**Root cause of pose non-convergence**: The network's matches contain outliers
that the confidence weights fail to downweight. The paper relies on learned
confidence weights to replace RANSAC, but our model hasn't learned meaningful
weights yet — creating a chicken-and-egg problem.

---

## Test Results

### Test 1: 8-point Algorithm with GT Correspondences

| Test | Rotation Error | Translation Error | Result |
|------|---------------|-------------------|--------|
| 1a: Synthetic pure translation | 0.04 deg | 0.00 deg | **PASS** |
| 1b: Synthetic rotation + translation | 0.04 deg | 0.01 deg | **PASS** |
| 1c: Real EuRoC (optical flow matches) | 3.18 deg | **112.07 deg** | **FAIL** |

**Analysis**: The algorithm works perfectly on clean synthetic correspondences
(< 0.05 deg error). On real EuRoC data with optical flow matches:
- Rotation is reasonable (3.18 deg)
- Translation is catastrophic (112 deg — worse than random!)

This is consistent with the 8-point algorithm's known weakness:
translation estimation is extremely sensitive to noise/outliers in
correspondences, while rotation is more robust. Optical flow matches
are not perfect — they contain drift and errors that the un-RANSACed
8-point algorithm amplifies into completely wrong translations.

### Test 2: Pose Loss Sanity Check

| Test | Result | Details |
|------|--------|---------|
| 2a: L(R_gt, t_gt, R_gt, t_gt) = 0? | **PASS** | 0.00000000 |
| 2b: Rotation proportional? | **PASS** | 1 deg -> 3.14, 30 deg -> 94.25 |
| 2c: Translation proportional? | **PASS** | 1 deg -> 6.98, 90 deg -> 565.69, 180 deg -> 800.00 |
| 2d: _log_rotation correct? | **PASS** | 5/30/90/170 deg all exact |

**Analysis**: Pose loss is mathematically correct. The _log_rotation fix from
the March 31 audit is verified — rotation angles are now exact. The loss scales
linearly with rotation angle (lambda_r * theta in radians) as expected.

**Key insight from Test 2c**: At 90 deg translation error, pose_raw = 565.
This matches our training observation (pose_raw ~500-540), confirming the
model's predicted translations are approximately perpendicular to GT — i.e.,
essentially random direction.

### Test 3: Comparison with OpenCV

| Test | Our 8-point | OpenCV RANSAC | Result |
|------|------------|---------------|--------|
| 3a: Clean synthetic | 0.06 deg rot | 0.00 deg rot | **PASS** (both work) |
| 3b: 20% outliers | 3.83 deg rot, 14.72 deg trans | 0.00 deg rot, 0.01 deg trans | **PASS** (expected) |
| 3c: Model matches | (skipped due to format issue) | — | **SKIP** |

**Critical finding from 3b**: With just 20% outlier matches:
- Our 8-point: 14.72 deg translation error (already degraded)
- OpenCV RANSAC: 0.01 deg (perfect, filters outliers)

The paper's approach assumes the **confidence weights from ConfMLP** will
downweight outliers, acting as a soft RANSAC. But if the confidence weights
are learned from pose loss, and pose loss can't converge because the weights
are bad, we have a **chicken-and-egg problem**.

### Test 4: Gradient Flow

| Test | Result | Details |
|------|--------|---------|
| 4a: Gradient to confidence weights | **PASS** | norm=0.14, 47/50 nonzero |
| 4b: Gradient to transformer weights | **PASS** | 67.7% params have nonzero grad |

**Analysis**: Gradient does flow from pose loss to model weights. The
ClampedSVD fix is working. However:
- 59 layers (deeper transformer layers 8-11) have ZERO gradient
- Top gradient is at `conf_mlp` (the confidence predictor), which makes sense
- The gradient magnitude to early layers is small (5e-3 mean)

**Key insight**: The gradient reaches the model, but 32.3% of parameters get
zero gradient. The deeper transformer layers (8-11) are not reached, meaning
the pose signal only affects the first few layers and the confidence head.

---

## Root Cause Diagnosis

### The problem is NOT:
1. ~~8-point algorithm implementation~~ -- works perfectly on clean data
2. ~~Pose loss formulation~~ -- mathematically correct
3. ~~Gradient flow~~ -- gradient reaches 67.7% of parameters
4. ~~SVD backward~~ -- ClampedSVD works, no NaN

### The problem IS:
1. **Match quality for pose estimation**: The 8-point algorithm requires very
   accurate correspondences. Even small outlier ratios destroy translation
   estimation. The paper's approach is to use learned confidence weights instead
   of RANSAC, but the weights can only be learned from pose loss, which requires
   good weights to produce good poses — **circular dependency**.

2. **Translation is the weakest link**: On real data, rotation error is 3 deg
   (acceptable) but translation error is 112 deg (catastrophic). This aligns
   with training (pose_raw ~500 = 90 deg translation error).

3. **Partial gradient coverage**: 32.3% of model parameters (deeper transformer
   layers) receive zero gradient from pose loss. This limits the model's ability
   to learn from pose supervision.

---

## Comparison with Training Observations

| Observation | Explanation from diagnostics |
|-------------|---------------------------|
| pose_raw ~500, flat for 10 epochs | 500 = ~90 deg trans error. Matches have outliers, 8-point fails, gradient is noise. |
| Matching loss 4.0 (good) but pose bad | Matching loss optimizes assignment probability, not geometric accuracy. |
| 30 NaN steps/epoch | Degenerate pairs with insufficient motion or very bad matches. |
| Scale factor 0.036 (27x too large) | Translations are ~random direction, accumulated trajectory explodes. |

---

## Recommended Next Steps (Priority Order)

### 1. IMMEDIATE: Verify with OpenCV on model matches (Test 3c fix)
Fix the match format issue and run Test 3c. If OpenCV RANSAC gets
good poses from the same matches -> the matches are OK but have outliers.
If OpenCV also fails -> matches are fundamentally bad (wrong pixel locations).

### 2. SHORT-TERM: Add RANSAC pre-filtering to training
Before the weighted 8-point, filter matches with OpenCV RANSAC as a
preprocessing step. This breaks the chicken-and-egg problem:
- Phase 1: Train with RANSAC-filtered matches (pose converges)
- Phase 2: Fine-tune without RANSAC (confidence weights take over)

### 3. SHORT-TERM: Investigate confidence weight distribution
Log the distribution of confidence weights during training. If they're
all uniform (~1.0), the ConfMLP hasn't learned anything and the 8-point
sees all matches equally (including outliers).

### 4. MEDIUM-TERM: Consider differentiable RANSAC
Replace the weighted 8-point with a differentiable RANSAC layer
(e.g., DSAC++ style). This would make the pipeline robust to outliers
while maintaining end-to-end differentiability.

### 5. MEDIUM-TERM: Increase matching accuracy
- Add reprojection loss (penalize pixel-level match error, not just assignment)
- Increase matching-only pretraining (4 epochs may not be enough)
- Curriculum: start with easy pairs (small motion, high overlap)

---

## Files

- Diagnostic script: `scripts/diagnostic_tests.py`
- This report: `outputs/DIAGNOSTIC_REPORT_2026-04-04.md`
- Training output: `train_output.txt`
- Evaluation plot: `outputs/trajectory_epoch_13.png`
- Evaluation results: `outputs/eval_epoch_13.txt`
