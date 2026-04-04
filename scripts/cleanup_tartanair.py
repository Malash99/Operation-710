"""
Clean up TartanAir dataset:
1. Remove broken environments (mismatched image/depth counts)
2. Remove empty difficulty dirs (Easy/Hard with no trajectories)
3. Remove environments with no usable data
4. Print final summary
"""
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tartanair")


def count_traj(diff_path):
    """Count trajectories, images, depths, poses in a difficulty dir."""
    trajs = [t for t in os.listdir(diff_path)
             if t.startswith("P") and os.path.isdir(os.path.join(diff_path, t))]
    imgs = 0
    depths = 0
    poses = 0
    for t in trajs:
        img_dir = os.path.join(diff_path, t, "image_left")
        dep_dir = os.path.join(diff_path, t, "depth_left")
        pose_f = os.path.join(diff_path, t, "pose_left.txt")
        if os.path.isdir(img_dir):
            imgs += len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if os.path.isdir(dep_dir):
            depths += len([f for f in os.listdir(dep_dir) if f.endswith(".npy")])
        if os.path.isfile(pose_f):
            poses += 1
    return len(trajs), imgs, depths, poses


print("=" * 60)
print("TartanAir Dataset Cleanup")
print("=" * 60)

# Step 1: Scan everything
print("\nStep 1: Scanning all environments...\n")
env_status = {}
for env in sorted(os.listdir(DATA_DIR)):
    env_path = os.path.join(DATA_DIR, env)
    if not os.path.isdir(env_path):
        continue
    env_status[env] = {}
    for diff in ["Easy", "Hard"]:
        diff_path = os.path.join(env_path, diff)
        if not os.path.isdir(diff_path):
            continue
        trajs, imgs, depths, poses = count_traj(diff_path)
        env_status[env][diff] = {
            "trajs": trajs, "imgs": imgs, "depths": depths, "poses": poses
        }

# Step 2: Identify problems and fix
print("Step 2: Fixing problems...\n")

to_delete_envs = []
for env, diffs in env_status.items():
    env_path = os.path.join(DATA_DIR, env)
    env_has_good_data = False

    for diff, stats in diffs.items():
        diff_path = os.path.join(env_path, diff)

        # Case 1: Empty difficulty dir (no trajectories)
        if stats["trajs"] == 0:
            print(f"  REMOVING empty dir: {env}/{diff}/")
            shutil.rmtree(diff_path)
            continue

        # Case 2: Broken data (big mismatch between images and depths)
        if stats["imgs"] > 0 and stats["depths"] > 0:
            ratio = min(stats["imgs"], stats["depths"]) / max(stats["imgs"], stats["depths"])
        elif stats["imgs"] == 0 and stats["depths"] == 0:
            ratio = 0
        else:
            ratio = 0

        if ratio < 0.5:
            print(f"  REMOVING broken: {env}/{diff}/ "
                  f"({stats['imgs']} imgs vs {stats['depths']} depths - mismatch)")
            shutil.rmtree(diff_path)
            continue

        # Case 3: Check individual trajectories for completeness
        for t in os.listdir(diff_path):
            t_path = os.path.join(diff_path, t)
            if not t.startswith("P") or not os.path.isdir(t_path):
                continue
            img_dir = os.path.join(t_path, "image_left")
            dep_dir = os.path.join(t_path, "depth_left")
            pose_f = os.path.join(t_path, "pose_left.txt")

            t_imgs = len([f for f in os.listdir(img_dir) if f.endswith(".png")]) if os.path.isdir(img_dir) else 0
            t_deps = len([f for f in os.listdir(dep_dir) if f.endswith(".npy")]) if os.path.isdir(dep_dir) else 0

            if t_imgs == 0 or t_deps == 0 or not os.path.isfile(pose_f):
                print(f"  REMOVING incomplete trajectory: {env}/{diff}/{t}/ "
                      f"({t_imgs} imgs, {t_deps} depths, pose: {os.path.isfile(pose_f)})")
                shutil.rmtree(t_path)
                continue

            if t_imgs != t_deps:
                print(f"  WARNING: {env}/{diff}/{t}/ has {t_imgs} imgs vs {t_deps} depths (keeping anyway)")

        env_has_good_data = True

    # Check if env has anything left
    remaining = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
    if not remaining:
        to_delete_envs.append(env)

# Step 3: Remove empty environments
print("\nStep 3: Removing empty environments...\n")
for env in to_delete_envs:
    env_path = os.path.join(DATA_DIR, env)
    print(f"  REMOVING empty environment: {env}/")
    shutil.rmtree(env_path)

# Step 4: Delete any leftover zip files
print("\nStep 4: Cleaning up zip files and cache...\n")
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".zip"):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp) / 1e9
            print(f"  Deleting zip: {os.path.relpath(fp, DATA_DIR)} ({sz:.2f} GB)")
            os.remove(fp)

cache = os.path.join(DATA_DIR, ".cache")
if os.path.isdir(cache):
    print("  Deleting .cache/")
    shutil.rmtree(cache)

# Step 5: Final summary
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)
grand_imgs = 0
grand_depths = 0
grand_trajs = 0
for env in sorted(os.listdir(DATA_DIR)):
    env_path = os.path.join(DATA_DIR, env)
    if not os.path.isdir(env_path):
        continue
    for diff in ["Easy", "Hard"]:
        diff_path = os.path.join(env_path, diff)
        if not os.path.isdir(diff_path):
            continue
        trajs, imgs, depths, poses = count_traj(diff_path)
        if trajs > 0:
            grand_imgs += imgs
            grand_depths += depths
            grand_trajs += trajs
            print(f"  {env}/{diff}: {trajs} trajs, {imgs} imgs, {depths} depths, {poses} poses  OK")

print(f"\nTOTAL: {grand_trajs} trajectories, {grand_imgs} images, {grand_depths} depth maps")
print("=" * 60)
