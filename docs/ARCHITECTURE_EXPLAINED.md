# DINO-VO: Architecture, Training, and Debugging -- Explained Simply

This document explains the full DINO-VO pipeline, how training works, the problems
we encountered, and the solution we're implementing. Technical terms are used but
always explained.

---

## What Does DINO-VO Do?

DINO-VO takes **two consecutive camera images** and figures out: "How did the camera
move between these two photos?" That movement has two parts:

- **Rotation (R)**: Which direction did the camera turn? (left, right, up, down, tilt)
- **Translation (t)**: Which direction did the camera move? (forward, left, up, etc.)

Important: monocular (single-camera) VO can only recover translation **direction**,
not distance. This is called **scale ambiguity** -- you can't tell from images alone
whether an object moved 1 meter or 10 meters.

---

## The Pipeline: 4 Stations

Think of the system as an assembly line with 4 stations. Each station processes the
data and passes it to the next.

### Station 1: Keypoint Detector (Paper Section III-A)

**Purpose**: Find "interesting points" in each image -- corners, edges, textured areas.
Plain walls and sky are boring and useless for tracking.

**How it works**:
1. **Gaussian blur**: Slightly blur the image (kernel=5, std=2.0) to reduce noise
2. **Sobel gradients**: Compute how quickly brightness changes at each pixel.
   High gradient = edge or corner = interesting
3. **Grid-based MaxPool**: Divide image into 14x14 pixel blocks (matching DINOv2's
   patch size). Pick the strongest point in each block. This ensures points are
   spread across the image, not all clustered in one corner.
4. **NMS** (Non-Maximum Suppression, radius=8): Remove points that are too close
   together. Keep only the local maximum within an 8-pixel radius.
5. **Top-k**: Keep the 512 strongest points

**Output**: 512 **keypoints** per image -- (x, y) pixel coordinates of distinctive spots.

### Station 2: Feature Descriptor (Paper Section III-B)

**Purpose**: Give each keypoint a "fingerprint" so we can recognize the same physical
point in both images.

**Two descriptor sources**:

1. **DINOv2** (by Meta): A pretrained vision foundation model. It understands
   "what" is in the image at a high level. Produces a 384-number description per
   keypoint. This network is **frozen** -- we never change its weights during training.

2. **FinerCNN**: A small CNN (Convolutional Neural Network) that captures fine
   pixel-level detail that DINOv2 misses (DINOv2 works at 14x14 patch level,
   losing fine spatial detail). Produces a 64-number description per keypoint.

**Fusion**: We **concatenate** (join) both: 384 + 64 = 448 numbers. Then a
**linear layer** (a learned multiplication matrix) compresses to **192 numbers**.
That's the final descriptor for each keypoint.

**Output**: 512 keypoints x 192-dimensional descriptors per image.

### Station 3: Feature Matching (Paper Section III-C)

**Purpose**: Figure out which keypoint in image 1 is the same physical point as
which keypoint in image 2.

**How**: A **transformer** (same architecture family as ChatGPT, but for keypoints
instead of words) with **12 layers**. Each layer does:

1. **Self-attention**: Points within the SAME image exchange context. "I'm a table
   corner, and you're also on this table -- let's share context." This helps each
   point understand its surroundings.

2. **Cross-attention**: Points in image 1 look at points in image 2. "I'm a corner
   in image 1 -- which point in image 2 looks most like me?" This is where
   matching happens.

After 12 rounds, the transformer outputs:

- **Assignment matrix P** (512 x 512): Each cell P[i][j] is a probability -- "how
  likely is point i in image 1 the same as point j in image 2?"

- **Confidence weights w**: For each selected match, how certain are we? This comes
  from the **ConfMLP** (Confidence Multi-Layer Perceptron -- a small neural network).
  It takes two matched descriptors and outputs a weight between 0 and 1.

**Output**: ~300 matched point pairs with confidence weights.

### Station 4: Pose Estimation (Paper Section III-D)

**Purpose**: From the matched points, compute the camera rotation R and translation t.

**The Weighted 8-Point Algorithm**:

A classic geometry formula (from the 1980s). If I know that point A in image 1
corresponds to point B in image 2, this gives a geometric constraint on camera
motion. With **at least 8** correspondences, I can solve for the **Essential matrix E**
(a 3x3 matrix encoding the geometric relationship between two camera views).

From E, we extract R (rotation) and t (translation direction) using **SVD**
(Singular Value Decomposition -- a way to break a matrix into simpler components,
like factoring 12 = 3 x 4 but for matrices).

**Why "weighted"?** Not all matches are equally reliable. The confidence weights
from the ConfMLP tell the algorithm: "Trust this match heavily" (weight ~0.9) or
"This match might be wrong, mostly ignore it" (weight ~0.1). This is crucial
because even a few bad matches (**outliers**) can completely destroy the result.

**Cheirality check**: SVD gives 4 possible solutions for (R, t). Only one is
physically correct -- where 3D points are in front of both cameras, not behind.
"Cheirality" means "handedness" -- checking which solution is physically sensible.

**Output**: R (3x3 rotation matrix), t (3-element translation direction vector).

---

## Training: How the Network Learns

### What Is a Loss Function?

A **loss** measures "how wrong is our prediction." Lower = better. Training works by:
1. Make a prediction
2. Compute the loss (how wrong)
3. Use **backpropagation** (chain rule of calculus) to compute gradients -- "which
   direction should each weight change to reduce the loss?"
4. Update weights slightly in that direction
5. Repeat thousands of times

### Loss 1: Matching Loss (L_m, Paper Eq. 12)

**Question**: "Did you match the right points together?"

We know the **ground truth correspondences** -- which point in image 1 truly
corresponds to which in image 2. We know this because TartanAir provides **depth
maps** (how far each pixel is from the camera) and **ground truth poses** (the
actual camera motion). With depth + pose, we can project any point from image 1
into image 2 and find where it should land.

The loss says: the assignment matrix P should assign high probability to correct
matches. Technically, it's the **negative log probability (NLL)** of correct matches:
- P[i][j] = 0.99 for correct match -> loss = -log(0.99) = 0.01 (small, good)
- P[i][j] = 0.01 for correct match -> loss = -log(0.01) = 4.6 (large, bad)

**This loss works well.** It drops from ~3.7 to ~1.35 across training.

### Loss 2: Pose Loss (L_p, Paper Eq. 13)

**Question**: "Did you estimate the camera motion correctly?"

Two components:

- **Rotation part**: lambda_r x ||log(R_pred) - log(R_gt)||
  The **matrix logarithm** converts a rotation matrix to an axis-angle vector,
  then we measure the vector difference. lambda_r = 180 is a scaling weight.

- **Translation part**: lambda_t x ||t_pred/||t_pred|| - t_gt/||t_gt||||
  Normalize both to unit vectors (direction only), then measure how different
  the directions are. lambda_t = 400 is a scaling weight.

**This loss does NOT converge.** It stays flat at ~230-390 for 10+ epochs.

### Combined Loss (L_total, Paper Eq. 14)

    L_total = (1 - lambda_p) x L_matching + lambda_p x L_pose

lambda_p starts at 0 (epochs 1-4: matching only) and slowly ramps to 0.9
(by ~epoch 10: 90% pose, 10% matching). This is a **curriculum** -- teach
matching first, then gradually focus on pose.

### Training Schedule

| Epochs | lambda_p | What's Happening |
|--------|----------|------------------|
| 1-4    | 0.0      | Matching only. Network learns which points correspond. |
| 5-14   | 0.0->0.9 | Pose loss added. Network should learn camera motion. |

---

## The Problem: Why Pose Loss Doesn't Converge

### The Chicken-and-Egg Cycle

```
Good confidence weights
        |
        v
8-point gets clean matches --> Good poses
                                    |
                                    v
                              Pose loss gives
                              useful gradient
                                    |
                                    v
                              ConfMLP learns
                              better weights ----> (back to top)
```

**The problem**: This cycle needs to START somewhere, but it can't:

1. At epoch 5, confidence weights are **random** (untrained during epochs 1-4)
2. Random weights give ~equal importance to good AND bad matches
3. The 8-point algorithm sees outlier matches with high weight -> **garbage poses**
4. Garbage poses -> garbage pose loss gradients -> ConfMLP gets **nonsense signal**
5. Confidence weights stay random -> back to step 1

**Nothing improves. The cycle never starts spinning.**

### Why Doesn't Low Matching Loss Fix This?

The matching loss being low (1.35) means **on average, correct matches get high
probability**. But "on average" hides a critical detail:

Out of ~300 selected matches:
- ~250 are correct (good matches)
- ~50 are wrong (outliers)

The matching loss averages over correct matches, so 250 good ones with high
probability = low loss. It looks great.

But the 8-point algorithm uses ALL 300 matches. Those 50 outliers with equal
weight destroy the result. It's like computing the average of
{1, 2, 1, 2, 1, 2, 1000} -- one outlier drags the average to 144.

**The matching loss and the confidence weights are two separate things:**

| | Assignment Matrix P | Confidence Weight w |
|---|---|---|
| Question | Which points correspond? | How much to trust each match? |
| Trained by | Matching loss (works) | Pose loss only (broken) |
| Network | Transformer dual-softmax | Separate ConfMLP |

The ConfMLP is a separate small network that has NO training signal during
epochs 1-4, and gets garbage signal from epoch 5+.

### What We Tried (and Why It Didn't Work)

| Version | Fix | Pose Loss | Why It Failed |
|---------|-----|-----------|---------------|
| v0.3 | SVD gradient fix, log_rot fix | ~500 | Gradients flow but signal is meaningless |
| v0.4 | NED frame fix + RANSAC | ~230 | RANSAC helped but can't teach ConfMLP (not differentiable) |
| v0.5 | [0,1] coords + augmentation - RANSAC | ~390 | Made things worse: wrong normalization + removed RANSAC crutch |

RANSAC (Random Sample Consensus) is an outlier rejection algorithm. In v0.4, it
filtered bad matches before the 8-point algorithm, which helped reduce pose loss
from ~500 to ~230. But RANSAC is a blunt tool -- it's not differentiable (gradients
can't flow through it), so it can't teach the ConfMLP to do the same job.

---

## The Solution: Confidence Pretraining (v0.6)

### Core Idea

Give the ConfMLP its own **direct training signal** that doesn't go through the
8-point algorithm. Teach it directly: "This match is geometrically correct (inlier),
output high confidence" or "This match is wrong (outlier), output low confidence."

### How It Works

We have TartanAir **depth maps** for every frame. For any predicted match
(point A in image 1 matched to point B in image 2), we can verify if it's correct:

```
1. Take point A at pixel (x1, y1) in image 1
2. Look up depth d1 from depth map
3. Convert to 3D: P = d1 x K_inv x (x1, y1, 1)
4. Transform using GT pose: P' = R_gt x P + t_gt
5. Project to image 2: (x2', y2') = K x P'
6. Compare (x2', y2') with actual matched point B at (x2, y2)
7. If distance < 5 pixels --> INLIER (good match)
8. If distance > 5 pixels --> OUTLIER (bad match)
```

### New Training Schedule

| Epochs | Losses Active | What ConfMLP Learns |
|--------|--------------|---------------------|
| 1-4 | Matching + **Inlier (NEW)** | Which matches to trust (from depth labels) |
| 5-14 | Matching + Inlier + Pose | Fine-tunes further from pose signal |

### Why This Breaks the Cycle

```
BEFORE (broken):
  ConfMLP --> 8-point --> pose --> pose loss --> gradient to ConfMLP
  (too indirect, too noisy, garbage signal)

AFTER (direct):
  ConfMLP --> compare with depth-based inlier label --> BCE loss --> gradient to ConfMLP
  (direct supervision, clean signal, works from epoch 1)
```

At epoch 5, the ConfMLP **already knows** which matches are good. The 8-point
algorithm gets well-weighted input. Poses are reasonable from the start. The
virtuous cycle can begin.

### Why We're Confident

1. **Directly targets the diagnosed root cause** -- not a parameter tweak or guess
2. **We have all the data** -- TartanAir depth maps are already loaded
3. **Proven in literature** -- SuperGlue and LightGlue use confidence pretraining
4. **Matching already works** (loss 1.35) -- the model CAN find correspondences,
   it just can't WEIGHT them properly
5. **Simple implementation** -- one new loss, one change to training loop

---

## Version History

| Version | Config | Key Change | Pose Loss | Outcome |
|---------|--------|-----------|-----------|---------|
| v0.1 | default.yaml | EuRoC baseline | ~370 (NaN) | SVD backward broken |
| v0.2 | tartanair_v02.yaml | TartanAir training | ~540 | Pose flat |
| v0.3 | tartanair_v03.yaml | 4 bug fixes | ~500 | Still flat |
| v0.4 | tartanair_v04.yaml | NED fix + RANSAC | ~230 | Best so far, still flat |
| v0.5 | tartanair_v05.yaml | [0,1] coords + augment | ~390 | Worse, reverted |
| v0.6 | tartanair_v06.yaml | Inlier pretraining | TBD | Expected: decreasing |

---

## Key Terms Glossary

| Term | Meaning |
|------|---------|
| **Keypoint** | A distinctive point in an image (corner, edge) that can be recognized in another image |
| **Descriptor** | A numerical fingerprint for a keypoint -- used to find the same point in another image |
| **Transformer** | A neural network architecture using attention to relate elements to each other |
| **Self-attention** | Points within one image exchange context with each other |
| **Cross-attention** | Points in image 1 look at points in image 2 to find matches |
| **Assignment matrix** | A probability grid showing how likely each pair of points matches |
| **ConfMLP** | Small neural network that predicts how reliable each match is (0 to 1) |
| **8-point algorithm** | Classic formula to compute camera motion from point correspondences |
| **Essential matrix** | 3x3 matrix encoding the geometric relationship between two camera views |
| **SVD** | Singular Value Decomposition -- breaks a matrix into simpler factors |
| **Cheirality** | Physical validity check -- are 3D points in front of both cameras? |
| **NLL** | Negative Log Likelihood -- a loss that penalizes low probability on correct answers |
| **BCE** | Binary Cross-Entropy -- a loss for binary (yes/no) predictions |
| **RANSAC** | Random Sample Consensus -- an algorithm to find inliers and reject outliers |
| **Inlier** | A correct match (reprojection error < threshold) |
| **Outlier** | A wrong match that would corrupt the pose estimate |
| **Backpropagation** | Algorithm to compute how each weight should change to reduce loss |
| **Gradient** | The direction and magnitude of change needed to reduce loss |
| **Scale ambiguity** | Monocular VO can't determine translation magnitude, only direction |
| **Domain gap** | Performance drop when a model trained on synthetic data is tested on real data |
| **NED frame** | North-East-Down coordinate system used by TartanAir |
| **Camera frame** | Right-Down-Forward coordinate system used in computer vision |

---

*Document created: 2026-04-16*
*Project: DINO-VO Reimplementation (arXiv:2507.13145v1)*
