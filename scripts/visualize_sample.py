"""
Quick visualization of EuRoC dataset sample.
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load first image
project_root = Path(__file__).parent.parent
cam0_data = project_root / 'data' / 'euroc' / 'MH_01_easy' / 'mav0' / 'cam0' / 'data'

images = sorted(list(cam0_data.glob('*.png')))

if len(images) > 0:
    # Load first 3 images
    img1 = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(images[100]), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(str(images[500]), cv2.IMREAD_GRAYSCALE)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'Frame 0\n{images[0].stem}')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Frame 100\n{images[100].stem}')
    axes[1].axis('off')

    axes[2].imshow(img3, cmap='gray')
    axes[2].set_title(f'Frame 500\n{images[500].stem}')
    axes[2].axis('off')

    plt.suptitle('EuRoC MH_01_easy - Sample Images (Machine Hall)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = project_root / 'outputs' / 'dataset_sample.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to: {output_path}")

    plt.close()

    print("\nDataset verification complete!")
    print(f"Total images: {len(images)}")
    print(f"Image resolution: {img1.shape[1]}x{img1.shape[0]}")
    print("\nReady to proceed with Phase 2: Data Pipeline implementation!")
