"""Fix double-nested TartanAir directory structure.
Moves carwelding/Easy/carwelding/Easy/P001/... -> carwelding/Easy/P001/...
Also removes empty env dirs that have no data, and cleans up .cache and leftover zips.
"""
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tartanair")

# Step 1: Fix double-nested directories
print("Step 1: Fixing double-nested directories...\n")
for env in os.listdir(DATA_DIR):
    if env.startswith("."):
        continue
    env_path = os.path.join(DATA_DIR, env)
    if not os.path.isdir(env_path):
        continue

    for diff in ["Easy", "Hard"]:
        nested = os.path.join(env_path, diff, env, diff)
        target = os.path.join(env_path, diff)

        if os.path.isdir(nested):
            # Move all contents from nested to target
            for item in os.listdir(nested):
                src = os.path.join(nested, item)
                dst = os.path.join(target, item)
                if os.path.exists(dst) and os.path.isdir(dst) and not os.listdir(dst):
                    # Remove empty target dir first
                    os.rmdir(dst)
                elif os.path.exists(dst):
                    print(f"  SKIP (already exists): {dst}")
                    continue
                print(f"  Moving: {env}/{diff}/{item}")
                shutil.move(src, dst)

            # Remove leftover empty nested dirs
            leftover = os.path.join(env_path, diff, env)
            if os.path.isdir(leftover):
                shutil.rmtree(leftover)
                print(f"  Cleaned up: {env}/{diff}/{env}/")

# Step 2: Delete remaining zip files
print("\nStep 2: Deleting leftover zip files...\n")
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".zip"):
            zf = os.path.join(root, f)
            size_gb = os.path.getsize(zf) / 1e9
            print(f"  Deleting: {os.path.relpath(zf, DATA_DIR)} ({size_gb:.2f} GB)")
            os.remove(zf)

# Step 3: Delete .cache directory
cache_dir = os.path.join(DATA_DIR, ".cache")
if os.path.isdir(cache_dir):
    print(f"\nStep 3: Deleting .cache directory...")
    shutil.rmtree(cache_dir)

# Step 4: Remove empty environment directories
print("\nStep 4: Removing empty environments...\n")
for env in os.listdir(DATA_DIR):
    env_path = os.path.join(DATA_DIR, env)
    if not os.path.isdir(env_path):
        continue
    # Check if any actual data files exist
    has_files = False
    for root, dirs, files in os.walk(env_path):
        if any(f.endswith(('.png', '.npy', '.txt')) for f in files):
            has_files = True
            break
    if not has_files:
        print(f"  Removing empty: {env}/")
        shutil.rmtree(env_path)

# Step 5: Summary
print("\n" + "="*60)
print("Final structure:")
print("="*60)
total_imgs = 0
total_depths = 0
total_trajs = 0
for env in sorted(os.listdir(DATA_DIR)):
    env_path = os.path.join(DATA_DIR, env)
    if not os.path.isdir(env_path):
        continue
    for diff in ["Easy", "Hard"]:
        img_dir = os.path.join(env_path, diff, "image_left")
        depth_dir = os.path.join(env_path, diff, "depth_left")
        if not os.path.isdir(img_dir):
            continue
        trajs = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
        imgs = sum(len([f for f in os.listdir(os.path.join(img_dir, t)) if f.endswith('.png')]) for t in trajs)
        depths = 0
        if os.path.isdir(depth_dir):
            depths = sum(len([f for f in os.listdir(os.path.join(depth_dir, t)) if f.endswith('.npy')]) for t in trajs if os.path.isdir(os.path.join(depth_dir, t)))

        pose_file = os.path.join(env_path, diff, "pose_left.txt")
        has_pose = "YES" if os.path.isfile(pose_file) else "NO"

        total_imgs += imgs
        total_depths += depths
        total_trajs += len(trajs)
        print(f"  {env}/{diff}: {len(trajs)} trajs, {imgs} imgs, {depths} depths, pose: {has_pose}")

print(f"\nTotal: {total_trajs} trajectories, {total_imgs} images, {total_depths} depth maps")
