"""Unzip all TartanAir zip files in place, deleting each zip after extraction to save space."""
import os
import zipfile
import glob
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tartanair")

# First, delete already-extracted carwelding to free space (it was extracted before disk ran out)
for env in ["carwelding"]:
    for diff in ["Easy", "Hard"]:
        for dtype in ["image_left", "depth_left"]:
            extracted = os.path.join(DATA_DIR, env, diff, dtype)
            if os.path.isdir(extracted):
                # Check if it has actual content (subdirs like P000, P001)
                subdirs = [d for d in os.listdir(extracted) if os.path.isdir(os.path.join(extracted, d))]
                if subdirs:
                    print(f"Deleting already-extracted {env}/{diff}/{dtype} ({len(subdirs)} trajectories) to free space...")
                    shutil.rmtree(extracted)
                    os.makedirs(extracted, exist_ok=True)

print("Freed space from previous partial extraction.\n")

zips = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.zip"), recursive=True))
print(f"Found {len(zips)} zip files to extract.\n")

for zf in zips:
    extract_dir = os.path.dirname(zf)
    rel = os.path.relpath(zf, DATA_DIR)
    zip_size_gb = os.path.getsize(zf) / 1e9

    print(f"Extracting: {rel} ({zip_size_gb:.2f} GB)")
    try:
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(extract_dir)
        print(f"  Extracted OK. Deleting zip to free space...")
        os.remove(zf)
        print(f"  Deleted {rel} (+{zip_size_gb:.2f} GB free)")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Keeping zip file for retry.")

print("\nDone! Checking results...\n")

# Verify extraction
total_imgs_all = 0
total_trajs_all = 0
for env_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    if not os.path.isdir(env_dir):
        continue
    env = os.path.basename(env_dir)
    for diff in ["Easy", "Hard"]:
        img_dir = os.path.join(env_dir, diff, "image_left")
        if os.path.isdir(img_dir):
            trajectories = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
            total_imgs = sum(len(os.listdir(os.path.join(img_dir, t))) for t in trajectories)
            total_imgs_all += total_imgs
            total_trajs_all += len(trajectories)
            print(f"  {env}/{diff}: {len(trajectories)} trajectories, {total_imgs} images")

print(f"\nTotal: {total_trajs_all} trajectories, {total_imgs_all} images")
