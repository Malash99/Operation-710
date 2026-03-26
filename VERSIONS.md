# Version History — DINO-VO

This file tracks tagged versions of the project for easy reference and comparison.
To return to any version: `git checkout <tag>`
To compare two versions: `git diff <tag1> <tag2>`

---

## Tags

### `v0.1-euroc-baseline` — 2026-03-26
**First complete pipeline: trained on EuRoC MH_01_easy**

| Item | Details |
|------|---------|
| Commit | `e074286` |
| PR | #10 |
| Dataset | EuRoC MH_01_easy (stereo-derived depth for GT correspondences) |
| Training | 14 epochs, batch=8, skip_frames=2, min_translation=0.03m |
| Checkpoints | `checkpoints/epoch_01.pth` — `epoch_14.pth` |
| Best checkpoint | `epoch_04.pth` (end of matching-only phase) |

**Training results:**
- Matching loss: 8.9 → 3.0 (converges well)
- Pose loss: ~370 (does not converge — SVD backward NaN in 36% of steps)

**Evaluation results (epoch 4, MH_01_easy):**
| Metric | 50 pairs | Full sequence |
|--------|----------|---------------|
| ATE RMSE | 0.4121 m | *pending* |
| ATE Mean | 0.3744 m | *pending* |
| Rotation Mean | 9.18 deg | *pending* |
| Scale factor | 0.0491 | *pending* |
| Avg matches/pair | 249 | *pending* |

**Known issues:**
- SVD backward produces NaN gradients (36% of steps during pose phase)
- Pose loss never converges due to small EuRoC motions (~1-5cm baselines)
- Training on EuRoC instead of TartanAir (paper's training dataset)
- Scale factor ~0.05 means predicted translations are 20x too large

**What was fixed in this version (paper audit):**
1. RoPE coordinate normalization swapped (x÷height, y÷width → x÷width, y÷height)
2. Epipolar constraint Phi matrix in column-major instead of row-major
3. Cheirality check missing negative sign in depth computation
4. L2 normalization on descriptors (not in paper)
5. Score matrix √d scaling (not in paper)

```bash
# Return to this version
git checkout v0.1-euroc-baseline

# Compare with a future version
git diff v0.1-euroc-baseline v0.2-tartanair
```

---

## Checkpoint Storage

Checkpoints are **not** tracked in git (too large). Each version's checkpoints
are stored locally in separate folders:

| Version | Checkpoint folder | Size (approx) |
|---------|-------------------|---------------|
| v0.1 | `checkpoints/` | ~1.4 GB (14 epochs × ~100MB) |

When retraining (e.g., on TartanAir), save to a new folder:
```bash
# In configs/default.yaml, change:
checkpoint_dir: checkpoints_v02_tartanair
```

---

## How to Add a New Version

1. Complete your training/changes and merge to main
2. Create an annotated tag:
   ```bash
   git tag -a v0.X-description -m "Brief summary of what changed"
   git push origin v0.X-description
   ```
3. Add an entry to this file following the format above
4. Update the checkpoint storage table
