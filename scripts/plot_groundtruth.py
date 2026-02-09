"""
Plot the ground truth trajectory from EuRoC MH_01_easy.

Generates a 3D trajectory plot and 2D projections (XY, XZ, YZ) from
the state_groundtruth_estimate0 data. This serves as the reference
trajectory that DINO-VO will attempt to estimate.

Outputs:
    outputs/groundtruth_trajectory.png
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_ground_truth(csv_path: str):
    """Load ground truth positions and orientations from EuRoC CSV.

    Returns:
        timestamps: Array of timestamps in seconds (relative to start).
        positions: Array of (x, y, z) positions in meters, shape (N, 3).
    """
    timestamps = []
    positions = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
            positions.append([float(row[1]), float(row[2]), float(row[3])])

    timestamps = np.array(timestamps, dtype=np.float64)
    timestamps = (timestamps - timestamps[0]) / 1e9  # convert to seconds from start
    positions = np.array(positions, dtype=np.float64)

    return timestamps, positions


def main():
    gt_csv = os.path.join(
        project_root,
        "data", "euroc", "MH_01_easy", "mav0",
        "state_groundtruth_estimate0", "data.csv",
    )

    if not os.path.isfile(gt_csv):
        print(f"ERROR: Ground truth file not found at {gt_csv}")
        sys.exit(1)

    print("Loading ground truth poses...")
    timestamps, positions = load_ground_truth(gt_csv)
    print(f"  Loaded {len(positions)} poses over {timestamps[-1]:.1f} seconds")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Compute total path length
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)
    print(f"  Total path length: {total_length:.2f} meters")
    print(f"  Start position: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
    print(f"  End position:   ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")

    # Create figure with 3D plot + three 2D projections
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"EuRoC MH_01_easy — Ground Truth Trajectory\n"
        f"{len(positions)} poses, {timestamps[-1]:.1f}s, path length: {total_length:.1f}m",
        fontsize=14,
    )

    # 3D trajectory
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(x, y, z, linewidth=0.5, color="steelblue", alpha=0.8)
    ax3d.scatter(x[0], y[0], z[0], color="green", s=80, zorder=5, label="Start")
    ax3d.scatter(x[-1], y[-1], z[-1], color="red", s=80, zorder=5, label="End")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

    # XY projection (top-down view)
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(x, y, linewidth=0.5, color="steelblue", alpha=0.8)
    ax_xy.scatter(x[0], y[0], color="green", s=80, zorder=5, label="Start")
    ax_xy.scatter(x[-1], y[-1], color="red", s=80, zorder=5, label="End")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.set_aspect("equal")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    # XZ projection (side view)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(x, z, linewidth=0.5, color="steelblue", alpha=0.8)
    ax_xz.scatter(x[0], z[0], color="green", s=80, zorder=5, label="Start")
    ax_xz.scatter(x[-1], z[-1], color="red", s=80, zorder=5, label="End")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Side View (XZ)")
    ax_xz.set_aspect("equal")
    ax_xz.legend()
    ax_xz.grid(True, alpha=0.3)

    # YZ projection (front view)
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(y, z, linewidth=0.5, color="steelblue", alpha=0.8)
    ax_yz.scatter(y[0], z[0], color="green", s=80, zorder=5, label="Start")
    ax_yz.scatter(y[-1], z[-1], color="red", s=80, zorder=5, label="End")
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.set_title("Front View (YZ)")
    ax_yz.set_aspect("equal")
    ax_yz.legend()
    ax_yz.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    output_path = os.path.join(project_root, "outputs", "groundtruth_trajectory.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved trajectory plot to: {output_path}")


if __name__ == "__main__":
    main()
