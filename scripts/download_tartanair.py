"""
Download TartanAir v1 dataset — selective environments (~50GB).
Uses AirLab Minio server (same as tartanair_tools).

Usage:
    python scripts/download_tartanair.py                    # Download ~50GB selection
    python scripts/download_tartanair.py --max-gb 30        # Limit to ~30GB
    python scripts/download_tartanair.py --list             # Show available environments
"""

import os
import sys
import argparse
from os.path import isdir, isfile, join

# Add tartanair_tools to path for the downloader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TOOLS_DIR = os.path.join(PROJECT_DIR, "tartanair_tools")
ZIPFILE_LIST = os.path.join(TOOLS_DIR, "download_training_zipfiles.txt")

DATA_DIR = os.path.join(PROJECT_DIR, "data", "tartanair")


def read_zipfile_list():
    """Read the master file list from tartanair_tools."""
    with open(ZIPFILE_LIST) as f:
        lines = f.readlines()
    return [ll.strip().split() for ll in lines if ll.strip() and ll.strip().split()[0].endswith('.zip')]


def get_env_plan(max_gb=50.0):
    """Select environments that fit within the GB budget (Easy, left, image+depth)."""
    entries = read_zipfile_list()

    # Filter: Easy difficulty, left camera, image or depth only
    from collections import defaultdict
    env_sizes = defaultdict(float)
    env_files = defaultdict(list)

    for name, size in entries:
        parts = name.split('/')
        env, diff, fname = parts[0], parts[1], parts[2]
        ftype = fname.split('_')[0]
        cam = fname.split('.')[0].split('_')[-1]

        if diff == 'Easy' and cam == 'left' and ftype in ('image', 'depth'):
            env_sizes[env] += float(size)
            env_files[env].append(name)

    # Select environments sorted by size (smallest first) until budget
    selected_envs = []
    cumulative = 0
    for env, size in sorted(env_sizes.items(), key=lambda x: x[1]):
        if cumulative + size > max_gb:
            break
        selected_envs.append(env)
        cumulative += size

    # Collect all files for selected environments
    selected_files = []
    for env in selected_envs:
        selected_files.extend(env_files[env])

    return selected_envs, selected_files, cumulative, env_sizes


def main():
    parser = argparse.ArgumentParser(description="Download TartanAir v1 (selective)")
    parser.add_argument("--max-gb", type=float, default=50.0, help="Max download size in GB (default: 50)")
    parser.add_argument("--list", action="store_true", help="List all environments with sizes")
    parser.add_argument("--unzip", action="store_true", default=True, help="Unzip after download (default: True)")
    parser.add_argument("--no-unzip", action="store_true", help="Don't unzip after download")
    args = parser.parse_args()

    if not os.path.isfile(ZIPFILE_LIST):
        print(f"ERROR: tartanair_tools not found at {TOOLS_DIR}")
        print("Run: git clone https://github.com/castacks/tartanair_tools.git")
        sys.exit(1)

    if args.list:
        entries = read_zipfile_list()
        from collections import defaultdict
        env_sizes = defaultdict(float)
        for name, size in entries:
            parts = name.split('/')
            env, diff, fname = parts[0], parts[1], parts[2]
            ftype = fname.split('_')[0]
            cam = fname.split('.')[0].split('_')[-1]
            if diff == 'Easy' and cam == 'left' and ftype in ('image', 'depth'):
                env_sizes[env] += float(size)

        print(f"\n{'Environment':<30} {'Size (GB)':>10}")
        print("-" * 42)
        total = 0
        for env, size in sorted(env_sizes.items(), key=lambda x: x[1]):
            total += size
            print(f"  {env:<28} {size:>10.2f}")
        print("-" * 42)
        print(f"  {'TOTAL':<28} {total:>10.2f}")
        return

    selected_envs, selected_files, total_gb, all_env_sizes = get_env_plan(args.max_gb)

    print("TartanAir v1 Selective Downloader")
    print(f"  Budget:       {args.max_gb} GB")
    print(f"  Download:     image_left + depth_left (Easy only)")
    print(f"  Output:       {DATA_DIR}")
    print(f"\n  Selected environments ({total_gb:.1f} GB):")
    for env in selected_envs:
        print(f"    - {env:<28} {all_env_sizes[env]:.2f} GB")
    print(f"\n  Total files: {len(selected_files)} zip files")

    resp = input("\nProceed with download? [y/N] ")
    if resp.lower() != "y":
        print("Aborted.")
        return

    # Create output dir
    os.makedirs(DATA_DIR, exist_ok=True)

    # Use Hugging Face downloader (AirLab and Cloudflare are down/expired)
    sys.path.insert(0, TOOLS_DIR)
    from download_training import HuggingfaceDownloader

    downloader = HuggingfaceDownloader()

    print(f"\nStarting download of {len(selected_files)} files from Hugging Face...")
    print("(Files download to the output dir with folder structure preserved)")
    success, downloaded_files = downloader.download(selected_files, DATA_DIR)

    if success:
        print(f"\nAll files downloaded to {DATA_DIR}")
    else:
        print(f"\nSome files may have failed. Check {DATA_DIR}")

    # Unzip — HuggingFace preserves folder structure: env/Easy/depth_left.zip
    if not args.no_unzip and downloaded_files:
        import zipfile
        print("\nUnzipping files...")
        for zf in downloaded_files:
            if isfile(zf) and zf.endswith('.zip'):
                print(f"  Unzipping {zf}...")
                try:
                    extract_dir = os.path.dirname(zf)
                    with zipfile.ZipFile(zf, 'r') as z:
                        z.extractall(extract_dir)
                    print(f"    -> {extract_dir}")
                except Exception as e:
                    print(f"    ERROR: {e}")

        print("\nDone! Dataset ready at:", DATA_DIR)


if __name__ == "__main__":
    main()
