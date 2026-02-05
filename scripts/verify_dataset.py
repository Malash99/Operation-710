"""
Dataset Verification Script
Verifies EuRoC dataset structure and displays summary.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def verify_euroc_sequence(sequence_dir):
    """Verify EuRoC sequence structure and display info."""

    sequence_dir = Path(sequence_dir)

    print("=" * 60)
    print(f"EuRoC Dataset Verification: {sequence_dir.name}")
    print("=" * 60)

    mav0_dir = sequence_dir / 'mav0'

    if not mav0_dir.exists():
        print(f"\nERROR: mav0 directory not found at {mav0_dir}")
        return False

    # Check required directories
    required_dirs = {
        'cam0': mav0_dir / 'cam0',
        'cam1': mav0_dir / 'cam1',
        'imu0': mav0_dir / 'imu0',
        'ground_truth': mav0_dir / 'state_groundtruth_estimate0'
    }

    print("\n--- Directory Structure ---")
    all_exist = True
    for name, path in required_dirs.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path.relative_to(sequence_dir)}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\nERROR: Some required directories are missing!")
        return False

    # Check cam0 images
    print("\n--- Camera 0 (Left) ---")
    cam0_data = required_dirs['cam0'] / 'data'
    cam0_images = sorted(list(cam0_data.glob('*.png')))
    num_images = len(cam0_images)
    print(f"  Number of images: {num_images}")

    if num_images > 0:
        # Load first image to check resolution
        first_img = cv2.imread(str(cam0_images[0]), cv2.IMREAD_GRAYSCALE)
        print(f"  Image resolution: {first_img.shape[1]}x{first_img.shape[0]}")
        print(f"  Image format: Grayscale (1 channel)")
        print(f"  First timestamp: {cam0_images[0].stem}")
        print(f"  Last timestamp: {cam0_images[-1].stem}")

    # Check calibration
    cam0_yaml = required_dirs['cam0'] / 'sensor.yaml'
    if cam0_yaml.exists():
        print(f"  Calibration file: OK")

        # Parse intrinsics (simple parsing)
        with open(cam0_yaml, 'r') as f:
            content = f.read()
            if 'intrinsics:' in content:
                # Extract intrinsics line
                for line in content.split('\n'):
                    if 'intrinsics:' in line:
                        intrinsics_str = line.split('[')[1].split(']')[0]
                        fx, fy, cx, cy = map(float, intrinsics_str.split(','))
                        print(f"  Camera intrinsics:")
                        print(f"    fx={fx:.2f}, fy={fy:.2f}")
                        print(f"    cx={cx:.2f}, cy={cy:.2f}")

    # Check cam1 (stereo)
    print("\n--- Camera 1 (Right) ---")
    cam1_data = required_dirs['cam1'] / 'data'
    cam1_images = list(cam1_data.glob('*.png'))
    print(f"  Number of images: {len(cam1_images)}")
    print(f"  Status: Available for stereo VO extension")

    # Check IMU
    print("\n--- IMU ---")
    imu_csv = required_dirs['imu0'] / 'data.csv'
    if imu_csv.exists():
        with open(imu_csv, 'r') as f:
            lines = f.readlines()
            num_imu = len(lines) - 1  # Subtract header
            print(f"  Number of measurements: {num_imu}")

            # Estimate frequency
            if num_imu > 1:
                first_line = lines[1].strip().split(',')
                last_line = lines[-1].strip().split(',')
                t_start = int(first_line[0])
                t_end = int(last_line[0])
                duration = (t_end - t_start) / 1e9  # Convert ns to seconds
                freq = num_imu / duration
                print(f"  Estimated frequency: {freq:.1f} Hz")

            print(f"  Status: Available for VIO extension")

    # Check ground truth
    print("\n--- Ground Truth ---")
    gt_csv = required_dirs['ground_truth'] / 'data.csv'
    if gt_csv.exists():
        with open(gt_csv, 'r') as f:
            lines = f.readlines()
            num_poses = len(lines) - 1  # Subtract header
            print(f"  Number of poses: {num_poses}")

            # Parse first and last pose positions
            first_pose = lines[1].strip().split(',')
            last_pose = lines[-1].strip().split(',')

            # Position is columns 1-3 (px, py, pz)
            p_start = np.array([float(first_pose[1]), float(first_pose[2]), float(first_pose[3])])
            p_end = np.array([float(last_pose[1]), float(last_pose[2]), float(last_pose[3])])

            trajectory_length = np.linalg.norm(p_end - p_start)
            print(f"  Start position: [{p_start[0]:.2f}, {p_start[1]:.2f}, {p_start[2]:.2f}]")
            print(f"  End position: [{p_end[0]:.2f}, {p_end[1]:.2f}, {p_end[2]:.2f}]")
            print(f"  Straight-line distance: {trajectory_length:.2f} meters")

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset is ready for DINO-VO training and evaluation!")
    print(f"  - Monocular VO: Use cam0 images")
    print(f"  - Stereo VO: Use cam0 + cam1 images")
    print(f"  - Visual-Inertial: Use cam0 + imu0 data")

    return True


if __name__ == '__main__':
    # Verify MH_01_easy
    project_root = Path(__file__).parent.parent
    sequence_dir = project_root / 'data' / 'euroc' / 'MH_01_easy'

    if not sequence_dir.exists():
        print(f"ERROR: Sequence directory not found: {sequence_dir}")
        print("Please download the EuRoC dataset first.")
        sys.exit(1)

    verify_euroc_sequence(sequence_dir)
