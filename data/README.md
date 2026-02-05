# Dataset Directory

This directory contains the EuRoC MAV Dataset sequences used for training and evaluation.

## EuRoC Dataset Structure

The dataset is not committed to version control. Download instructions will be provided in `scripts/download_euroc.py`.

Expected structure after download:
```
data/
└── euroc/
    └── MH_01_easy/
        └── mav0/
            ├── cam0/
            │   ├── data/           # Left camera images (*.png)
            │   └── data.csv        # Timestamps
            ├── cam1/               # Right camera (not used for monocular VO)
            ├── imu0/               # IMU data (not used in DINO-VO)
            └── state_groundtruth_estimate0/
                └── data.csv        # Ground truth poses (timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z)
```

## Download Instructions

Run the download script once it's implemented:
```bash
python scripts/download_euroc.py --sequence MH_01_easy
```

Or manually download from: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
