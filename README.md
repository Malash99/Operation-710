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

The system is trained end-to-end on the **EuRoC MAV dataset** and evaluated on real robot trajectories.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM) or equivalent
  - Minimum: GPU with 8GB VRAM + CUDA compute capability ≥ 5.0
  - RTX 50 series requires PyTorch nightly with CUDA 12.8 support
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

Download the **Machine Hall** sequences from [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/690084):

```bash
# After downloading machine_hall.zip, extract to data/euroc/
# Manual extraction:
unzip machine_hall.zip -d data/euroc/
cd data/euroc/MH_01_easy
unzip MH_01_easy.zip
```

Verify dataset:
```bash
python scripts/verify_dataset.py
```

Expected structure:
```
data/euroc/MH_01_easy/
└── mav0/
    ├── cam0/data/*.png          # 3,682 left camera images
    ├── cam1/data/*.png          # 3,682 right camera images (for stereo)
    ├── imu0/data.csv            # IMU measurements (for VIO)
    └── state_groundtruth_estimate0/data.csv  # Ground truth poses
```

---

## Usage

### Training
```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --sequence MH_01_easy
```

### Visualization
```bash
python scripts/visualize.py --trajectory outputs/MH_01_easy_trajectory.txt
```

---

## Architecture

### Pipeline Overview
```
Image Pair (It, It+1)
    ↓
[1] Salient Keypoint Detector (Sec III-A)
    - Gaussian smoothing + Sobel gradients
    - Grid-based MaxPooling (14×14)
    - Non-Maximum Suppression + Top-k (512 keypoints)
    ↓
[2] Feature Descriptor (Sec III-B)
    - DINOv2-ViT-S: 384-dim features (frozen)
    - FinerCNN: 64-dim features (trainable)
    - Fusion: 192-dim final descriptors
    ↓
[3] Feature Matching (Sec III-C)
    - Transformer (L=12 layers, 3 heads)
    - Soft assignment matrix with dual-softmax
    - Confidence prediction MLP
    ↓
[4] Pose Estimation (Sec III-D)
    - Weighted 8-point algorithm
    - Essential matrix decomposition
    - Cheirality check
    ↓
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
| Dataset Loader | EuRoC image + pose loader | `src/datasets/euroc.py` |

---

## Training Details

### Loss Function
```
L_total = (1 - λ_p) * L_matching + λ_p * L_pose

L_matching: Supervised correspondence loss (Eq. 12)
L_pose: Rotation + Translation loss (Eq. 13)
```

### Training Schedule
- **Epochs 1-4**: λ_p = 0.0 (matching loss only)
- **Epochs 5-14**: λ_p increases 0.0 → 0.9 (gradual pose loss introduction)
- **Learning rate**: Adam with scheduling
- **Image resolution**: 476×742 (resized from 752×480)

---

## Current Progress

### ✅ Phase 1: Environment Setup (COMPLETE)
- [x] Project structure created
- [x] PyTorch nightly installed with RTX 5060 Ti support (CUDA 12.8, sm_120)
- [x] GPU verified and working (15.93 GB VRAM)
- [x] EuRoC MH_01_easy dataset downloaded and verified (3,682 frames)

### ✅ Phase 2: Data Pipeline (COMPLETE)
- [x] EuRoC dataset loader (`src/datasets/euroc.py`)
- [x] Image preprocessing and transforms (`src/datasets/transforms.py`)
- [x] Undistortion, resize 752×480 → 742×476, ImageNet normalization
- [x] Ground truth pose loading with coordinate frame correction
- [x] Verified: 3,637 image pairs, intrinsics rescaled correctly

### ✅ Phase 3: Keypoint Detector (COMPLETE)
- [x] Gaussian smoothing (kernel=5, sigma=2.0)
- [x] Sobel gradient magnitude computation
- [x] Grid-based MaxPooling (kernel=14, stride=14 — matches DINOv2 patch size)
- [x] Non-Maximum Suppression (radius=8)
- [x] Gradient thresholding (0.01) + Top-k selection (512 keypoints)
- [x] Verified: 512 keypoints per image, spatially distributed
- See: [`src/models/README_phase3_keypoint_detector.md`](src/models/README_phase3_keypoint_detector.md)

### ✅ Phase 4: Feature Descriptor (COMPLETE)
- [x] DINOv2-ViT-S/14 loaded from torch.hub, fully frozen (22M params)
- [x] FinerCNN encoder implemented (XFeat-style, 4 blocks, 64-dim output)
- [x] Feature fusion: concat(384-dim, 64-dim) → Linear → 192-dim
- [x] L2 normalization on final descriptors
- [x] Verified: shape `(B, 512, 192)`, norm error < 1.2e-07, 151K trainable params
- See: [`src/models/README_phase4_feature_descriptor.md`](src/models/README_phase4_feature_descriptor.md)

### ✅ Phase 5: Feature Matching (COMPLETE)
- [x] Transformer-based matching (L=12 layers, 3 heads, head_dim=64)
- [x] Self-attention with 2D Rotary Positional Encoding (encodes keypoint x,y)
- [x] Cross-attention between image pairs (no positional encoding)
- [x] Dual-softmax soft assignment matrix (Eq. 5-8)
- [x] Confidence prediction MLP (Eq. 9)
- [x] Mutual nearest-neighbor match extraction
- [x] Verified: 6.3M trainable params, 110ms forward, 169 MB GPU, gradients flow
- See: [`src/models/README_phase5_feature_matching.md`](src/models/README_phase5_feature_matching.md)

### 📋 Phase 6–9: Upcoming
- Phase 6: Pose Estimation (Weighted 8-point, Essential matrix, cheirality check)
- Phase 7: Loss Functions (Matching loss Eq.12 + Pose loss Eq.13)
- Phase 8: Training Pipeline
- Phase 9: Evaluation (ATE on EuRoC MH_01_easy)

For detailed implementation order, see [CLAUDE.md](CLAUDE.md).

---

## Results (Target)

Based on the paper, expected performance on EuRoC MH_01_easy:

| Metric | Target Value |
|--------|--------------|
| ATE (Absolute Trajectory Error) | ~0.05-0.10 m |
| Processing Speed | 15-20 FPS |
| Scale Drift | < 2% |

---

## Extensions

### Stereo Visual Odometry
Use both cam0 (left) and cam1 (right) for:
- Absolute scale recovery (no scale ambiguity)
- Improved depth estimation
- More robust tracking

### Visual-Inertial Odometry (VIO)
Fuse IMU measurements with visual odometry:
- High-frequency motion estimation (200 Hz)
- Scale recovery via IMU integration
- Robust to motion blur and fast motion

See implementation details in [CLAUDE.md](CLAUDE.md) Section: "Sensor Fusion Extensions"

---

## Repository Structure

```
.
├── CLAUDE.md              # Detailed implementation guide
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── configs/               # Configuration files
│   └── default.yaml
│
├── data/                  # Dataset directory (not committed)
│   └── euroc/
│       └── MH_01_easy/
│
├── src/                   # Source code
│   ├── models/            # VO pipeline components
│   ├── datasets/          # Data loaders
│   ├── utils/             # Helper functions
│   └── losses/            # Loss functions
│
├── scripts/               # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── download_euroc.py
│
├── checkpoints/           # Saved models
├── outputs/               # Results and visualizations
└── tests/                 # Unit tests
```

---

## Citation

If you use this implementation, please cite the original paper:

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

---

## Contact

For questions or issues, please open an issue on GitHub.
