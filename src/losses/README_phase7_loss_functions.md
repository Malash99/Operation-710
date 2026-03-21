# Phase 7: Loss Functions — DINO-VO

**Paper Reference:** Section III-E, "Loss Functions"
**Status:** COMPLETE — all verification checks passed

---

## What Is It?

Three loss functions that drive the end-to-end training of DINO-VO:

1. **Matching Loss (Eq. 12)** — teaches the transformer to produce correct correspondences
2. **Pose Loss (Eq. 13)** — teaches it to produce geometrically useful correspondences
3. **Combined Loss (Eq. 14)** — balances both with a scheduled weighting

---

## Why Does It Exist?

The matching transformer (Phase 5) starts with random weights and produces meaningless assignment matrices. The loss functions provide the learning signal:

- **Matching loss** directly supervises the assignment matrix P — "did you assign high probability to the correct correspondence?"
- **Pose loss** provides indirect supervision — "did your correspondences lead to the correct camera pose?" This is critical because some correspondences are geometrically more informative than others.
- The **scheduled combination** trains matching first (stable, easy), then gradually introduces pose supervision (harder, requires good matches).

---

## Paper Equations

### Eq. 12: Matching Loss (Full Formulation with Deep Supervision)
```
L_m = -(1/L) * Sum_l [
    (1/|M|) * Sum_{(i,j) in M} log(P_ij^l)
  + (1/(2*|K_bar_t|))   * Sum_{i in K_bar_t}   log(1 - sigma_i^l)
  + (1/(2*|K_bar_t+1|)) * Sum_{j in K_bar_t+1} log(1 - sigma_j^l)
]
```
Three terms, averaged over all L transformer layers:
1. **NLL at GT correspondences** — pushes P to assign high probability to correct matches
2. **Unmatchable penalty (image t)** — pushes sigma toward 0 for keypoints without a match
3. **Unmatchable penalty (image t+1)** — same for the second image

### Eq. 13: Pose Loss
```
L_p = lambda_t * ||t_hat/max(||t_hat||, eps) - t_gt/max(||t_gt||, eps)||
    + lambda_r * ||Log(R_hat) - Log(R_gt)||
```
- **Translation error:** L2 distance between unit translation vectors, using `max(||t||, eps)` with `eps=1e-6` for numerical stability. Weighted by `lambda_t = 400`.
- **Rotation error:** Geodesic distance on SO(3) via the matrix logarithm (rotation vector representation). Weighted by `lambda_r = 180`.

### Eq. 14: Combined Loss
```
L_total = (1 - lambda_p) * L_m + lambda_p * L_p
```

**Training schedule (Section IV-A):**
- Epochs 1-4: `lambda_p = 0.0` (matching loss only)
- Epochs 5-14: `lambda_p` ramps from 0.0 to 0.9 (increment 1.5e-4 per training step)

---

## How It Works

### Matching Loss (Deep Supervision)

For each transformer layer l = 1..L:
1. Compute assignment matrix P^l and matchability sigma^l from layer features
2. **Term 1:** For GT matches (i,j) in M, compute `-log(P^l[i,j])` and average
3. **Term 2:** For unmatchable keypoints in image t (K_bar_t), compute `-log(1 - sigma_i^l)` and average, weighted by 0.5
4. **Term 3:** For unmatchable keypoints in image t+1 (K_bar_t+1), compute `-log(1 - sigma_j^l)` and average, weighted by 0.5
5. Sum three terms for this layer
6. Average over all L layers

### Pose Loss

1. Receive estimated `(R_est, t_est)` from Phase 6
2. Receive ground truth `(R_gt, t_gt)` from the dataset
3. Compute rotation error via matrix logarithm:
   - `log(R)` extracts the rotation vector (axis * angle)
   - Error = L2 distance between rotation vectors
4. Compute translation error:
   - Normalize both to unit vectors (monocular = up-to-scale)
   - Error = L2 distance between unit vectors
5. Weight and sum: `180 * rot_err + 400 * trans_err`

### Rotation Matrix Logarithm

Converts SO(3) rotation matrix to so(3) rotation vector using Rodrigues formula:
```
theta = arccos((trace(R) - 1) / 2)
[wx, wy, wz] = (theta / (2*sin(theta))) * [R32-R23, R13-R31, R21-R12] / 2
```
This is differentiable, enabling backpropagation through the rotation error.

### lambda_p Scheduling

```
Step 0-N1:     lambda_p = 0.0      (matching loss only)
Step N1+1:     lambda_p = 0.00015
Step N1+2:     lambda_p = 0.00030
   ...
Step N1+6000:  lambda_p = 0.9      (maximum, clamped)
```

Where N1 is the total steps in the first 4 epochs. After step N1, `lambda_p` increases by `1.5e-4` per step until reaching 0.9.

---

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| lambda_r (rotation weight) | 180 | Paper Sec IV-A |
| lambda_t (translation weight) | 400 | Paper Sec IV-A |
| lambda_p initial | 0.0 | Paper Sec IV-A |
| lambda_p max | 0.9 | Paper Sec IV-A |
| lambda_p increment | 1.5e-4 per step | Paper Sec IV-A |
| Matching-only epochs | 1-4 | Paper Sec IV-A |
| Combined loss epochs | 5-14 | Paper Sec IV-A |

---

## Expected Input / Output

### MatchingLoss
- **Input:** `all_assignments: list of L (B,K,K)`, `all_sigma1: list of L (B,K,1)`, `all_sigma2: list of L (B,K,1)`, `gt_matches (B,M,2)`, `gt_mask (B,M)`
- **Output:** scalar loss (averaged over L layers, 3 terms per layer)

### PoseLoss
- **Input:** `R_est (B,3,3)`, `t_est (B,3)`, `R_gt (B,3,3)`, `t_gt (B,3)`
- **Output:** scalar loss

### DinoVOLoss (combined)
- **Input:** all_assignments, all_sigma1, all_sigma2, gt_matches, gt_mask, R_est, t_est, R_gt, t_gt
- **Output:** dict with `total`, `matching`, `pose`, `lambda_p`

---

## Verification Results

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| MatchingLoss 3 terms + deep supervision | > 0 | 1.60 | PASSED |
| Unmatchable penalty (sigma~0.99) | high | 5.50 | PASSED |
| Unmatchable penalty (sigma~0.01) | low | 0.91 | PASSED |
| PoseLoss eps=1e-6 (paper Eq. 13) | finite | 396.0 | PASSED |
| PoseLoss near-zero t_est | no NaN | finite | PASSED |
| lambda_p=0: total = matching | equal | equal | PASSED |
| lambda_p=0.5: total = 0.5*m + 0.5*p | correct | correct | PASSED |
| Scheduling caps at 0.9 | 0.9 | 0.9 | PASSED |
| Deep supervision per-layer P | L tensors | L tensors | PASSED |

---

## Files

| File | Description |
|------|-------------|
| `src/losses/losses.py` | `MatchingLoss`, `PoseLoss`, `DinoVOLoss` |
| `src/losses/__init__.py` | Module exports |
| `scripts/test_losses.py` | Phase 7 verification script |

---

## Connection to Other Modules

| Module | Interaction |
|--------|-------------|
| **Phase 5: feature_matching.py** | Provides assignment matrix P for matching loss |
| **Phase 6: pose_estimation.py** | Provides estimated (R, t) for pose loss |
| **Phase 8: train.py** | Training loop calls DinoVOLoss and manages lambda_p scheduling |

---

## Implementation Notes

- **Differentiable rotation logarithm:** Uses Rodrigues formula with clamped arccos to avoid NaN gradients at theta=0 and theta=pi.
- **Translation comparison:** Both estimated and GT translations are normalized to unit vectors before comparison, since monocular VO cannot recover absolute scale.
- **Epsilon in log:** `log(P + eps)` with `eps=1e-8` prevents `-inf` when P is exactly 0.
- **Translation normalization:** Uses `max(||t||, eps)` with `eps=1e-6` per paper Eq. 13, instead of `||t|| + eps`.
- **Deep supervision:** MatchingLoss averages over all L transformer layers (Eq. 12: `1/L * Sum_l`).
- **Unmatchable keypoints:** Loss terms 2 and 3 penalize high sigma for keypoints without GT matches.
- **Batch handling:** MatchingLoss iterates over batch elements to handle variable-length GT match lists via masking.
- **lambda_p as buffer:** Stored as a registered buffer so it persists in checkpoints and moves with `.to(device)`.
