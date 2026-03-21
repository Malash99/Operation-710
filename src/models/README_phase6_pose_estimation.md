# Phase 6: Pose Estimation — DINO-VO

**Paper Reference:** Section III-D, "Pose Estimation"
**Status:** COMPLETE — all verification checks passed

---

## What Is It?

The Pose Estimation module is a **differentiable geometric layer** that recovers the relative camera pose (rotation R, translation t) from weighted keypoint correspondences. It implements the **weighted 8-point algorithm** with SVD-based Essential matrix decomposition and cheirality check.

This module has **zero learnable parameters** — it is pure geometry. However, it is fully differentiable, so gradients from the pose loss (Eq. 13) flow back through the SVD and weights into the matching transformer.

---

## Why Does It Exist?

The entire DINO-VO pipeline exists to estimate camera motion. Phases 3-5 produce matched keypoints with confidence weights. This module converts those correspondences into the actual rotation and translation between frames — the final output of the pipeline.

The **weighted** 8-point algorithm is key: the confidence weights from Phase 5 allow the model to learn which correspondences are reliable for pose estimation, down-weighting outliers and ambiguous matches.

---

## Where It Fits in the DINO-VO Pipeline

```
                        DINO-VO Pipeline
                        ================

  Image_t  ──> [Phase 3: Keypoint Detector] ──> 512 keypoints (x,y)
                         |
                         v
           [Phase 4: Feature Descriptor] ──> descriptors (192-dim)
                         |
                         v
           [Phase 5: Feature Matching] ──> matched pairs + weights
                         |
                         v
           [Phase 6: Pose Estimation] ──> relative pose (R, t)
                |                  |
           ┌────┴──────────────┐   |
           | 1. Normalize coords|   |
           | 2. Build Phi       |   |
           | 3. Weighted SVD    |   |
           | 4. Enforce E rank  |   |
           | 5. Decompose E     |   |
           | 6. Cheirality      |   |
           └───────────────────┘   |
                                    v
                        Pose Loss (Eq. 13) ──> backprop to matching transformer
```

Phase 6 is the **final inference step** of the pipeline. During training, the pose loss (Phase 7) compares the estimated (R, t) with ground truth and sends gradients back through the entire chain.

---

## Paper Equations

### Eq. 10: Epipolar Constraint
```
x_j^T  E  x_i = 0
```
For each correspondence (x_i, x_j) in normalized camera coordinates, the Essential matrix E encodes the geometric relationship between the two camera views.

### Eq. 11: Weighted Linear System
```
diag(w) * Phi * flat(E) = 0
```
The epipolar constraints for all correspondences form a linear system. The weights w (from Phase 5's confidence MLP) scale each constraint, giving more influence to reliable matches.

---

## How It Works (Step by Step)

### Step 1: Pixel to Normalized Camera Coordinates

Convert pixel coordinates (u, v) to normalized coordinates using the intrinsic matrix K:
```
x = (u - cx) / fx
y = (v - cy) / fy
x_norm = [x, y, 1]
```

### Step 2: Build Epipolar Constraint Matrix Phi

For each correspondence (x1_i, x2_i), construct the constraint vector:
```
phi_i = kron(x1, x2) = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
```
This is a (M, 9) matrix where M is the number of correspondences.

### Step 3: Weighted 8-Point Algorithm (SVD)

Apply confidence weights and solve via SVD:
```
Phi_weighted = diag(sqrt(w)) @ Phi
U, S, Vt = SVD(Phi_weighted)
E = reshape(Vt[-1], (3, 3))   # last right singular vector
```
The Essential matrix is the null space of the weighted constraint system.

### Step 4: Enforce Essential Matrix Constraint

Project onto the Essential matrix manifold (rank 2, equal singular values):
```
U, S, Vt = SVD(E)
s_mean = (S[0] + S[1]) / 2
E_proj = U @ diag(s_mean, s_mean, 0) @ Vt
```

### Step 5: Decompose Essential Matrix

Extract four (R, t) candidates:
```
W = [[0,-1,0],[1,0,0],[0,0,1]]   # 90-degree rotation

R1 = U @ W @ Vt,    t1 = +U[:, 2]
R2 = U @ W @ Vt,    t2 = -U[:, 2]
R3 = U @ W^T @ Vt,  t3 = +U[:, 2]
R4 = U @ W^T @ Vt,  t4 = -U[:, 2]
```
Determinant signs of U and Vt are corrected to ensure proper rotations (det = +1).

### Step 6: Cheirality Check

For each candidate, triangulate 3D points and count how many have positive depth in both camera frames. The candidate with the most positive-depth points is selected.

Triangulation uses cross-product elimination:
```
depth1 = (t x x2) . (Rx1 x x2) / |Rx1 x x2|^2
```
Then check `depth1 > 0` and `depth2 > 0` for each point.

---

## Hyperparameters Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| Algorithm | Weighted 8-point | Paper Sec III-D |
| SVD solver | torch.linalg.svd (differentiable) | PyTorch |
| Essential projection | Enforce rank-2, equal SVs | Standard |
| Cheirality candidates | 4 | Standard (2 rotations x 2 translations) |
| Translation output | Unit norm (up-to-scale) | Monocular VO |

---

## Expected Input / Output

**Input:**
- `kp1`, `kp2`: `(B, M, 2)` — matched keypoint pixel coordinates
- `weights`: `(B, M)` — per-match confidence weights from Phase 5
- `intrinsics`: `(B, 3, 3)` — camera intrinsic matrix K

**Output (dict):**
- `R`: `(B, 3, 3)` — rotation matrix (valid SO(3), det=1)
- `t`: `(B, 3)` — unit translation vector (up-to-scale for monocular)
- `E`: `(B, 3, 3)` — Essential matrix (projected, rank-2)
- `E_raw`: `(B, 3, 3)` — Essential matrix (before projection)

---

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| **Total** | **0** |

This is a purely geometric layer with no learnable parameters. All operations (coordinate normalization, Kronecker products, SVD, cross products) are fixed computations.

Gradients flow through the differentiable SVD in PyTorch, enabling end-to-end training.

---

## Verification Results (Phase 6 Test)

Tested with synthetic correspondences (100 points, known 3D structure projected through ground truth pose):

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| R shape | `(1, 3, 3)` | `(1, 3, 3)` | PASSED |
| t shape | `(1, 3)` | `(1, 3)` | PASSED |
| det(R) | 1.0 | 1.000001 | PASSED |
| max\|R@R^T - I\| | < 1e-4 | 9.54e-07 | PASSED |
| \|\|t\|\| | 1.0 | 1.000000 | PASSED |
| Essential SVs | [s, s, 0] | [0.7071, 0.7071, 0.0000] | PASSED |
| Epipolar error | ~0 | mean=7.64e-08 | PASSED |
| Rotation error | ~0 deg | 0.0000 deg | PASSED |
| Translation error | ~0 deg | 0.0198 deg | PASSED |
| Gradient flow | gradients exist | PASSED | PASSED |
| 0 parameters | 0 | 0 | PASSED |

**Note on noise sensitivity:** The 8-point algorithm is inherently sensitive to noise, especially with small baselines (0.05-0.06m in EuRoC). With clean correspondences the error is essentially zero. With noise, the confidence weights from Phase 5 become critical — they down-weight unreliable matches, which is why end-to-end training (Phase 8) is essential for good performance.

Visualization saved to: `outputs/pose_estimation_test.png`

---

## Files

| File | Description |
|------|-------------|
| `src/models/pose_estimation.py` | `PoseEstimation` class |
| `scripts/test_pose_estimation.py` | Phase 6 verification script |

---

## Connection to Other Modules

| Module | Interaction |
|--------|-------------|
| **Phase 5: feature_matching.py** | Provides matched keypoint pairs + confidence weights |
| **Phase 7: losses.py** | Pose loss (Eq. 13) compares estimated (R, t) with ground truth |
| **Phase 9: evaluation** | Accumulates relative poses into a trajectory for ATE computation |

---

## Implementation Notes

- **Differentiable SVD:** PyTorch's `torch.linalg.svd` supports backpropagation. Gradients from the pose loss flow through the SVD, through the weighted constraint matrix, and into the confidence weights — allowing the matching transformer to learn which correspondences are most useful for pose estimation.
- **Translation is up-to-scale:** Monocular VO cannot recover absolute scale. The translation vector is normalized to unit length. Scale must be recovered from other sources (stereo, IMU, or known scene dimensions).
- **Essential matrix projection:** The raw 8-point solution may not satisfy the rank-2 constraint exactly. Projecting onto the Essential manifold (two equal SVs, one zero) ensures valid decomposition.
- **Determinant correction:** After SVD of E, the signs of U and Vt are corrected to ensure det(U) > 0 and det(Vt) > 0, guaranteeing proper rotation matrices (not reflections).
- **Small baseline sensitivity:** With EuRoC's small inter-frame translations (~0.05m), the 8-point algorithm becomes ill-conditioned. The keyframe selection strategy (Phase 8, 24px threshold) mitigates this by ensuring sufficient baseline.
