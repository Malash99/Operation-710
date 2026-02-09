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

### 🚧 Phase 2: Data Pipeline (NEXT)

**Next Steps**:
1. Implement EuRoC dataset loader (`src/datasets/euroc.py`)
   - Load consecutive image pairs
   - Load ground truth poses
   - Load camera intrinsics from sensor.yaml

2. Implement image transforms (`src/datasets/transforms.py`)
   - Resize: 752×480 → 476×742 (as per paper Section IV-A)
   - Normalization: mean/std for DINOv2 input
   - Optional augmentation: brightness, contrast (if needed)

3. Test and visualize
   - Load sample image pairs
   - Verify preprocessing
   - Display overlaid features

---

## Start Command

When ready to begin, say:

"Let's start implementing DINO-VO. Phase 1: Environment Setup. First, I'll create the project directory structure."

Then follow the implementation order step by step.

**For resuming from Phase 2**:

"Phase 1 is complete. Let's proceed with Phase 2: Data Pipeline. First, I'll implement the EuRoC dataset loader."