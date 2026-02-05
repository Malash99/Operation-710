"""
PyTorch Installation Script for RTX 50 Series GPUs
Installs PyTorch nightly with CUDA 12.8 support for sm_120 compute capability.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and display output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\n⚠ Warning: Command exited with code {result.returncode}")
        return False

    print(f"\n✓ {description} completed")
    return True

def main():
    """Main installation process."""

    print("="*60)
    print("PyTorch Installation for RTX 5060 Ti (sm_120)")
    print("="*60)
    print("\nThis script will:")
    print("1. Uninstall existing PyTorch packages")
    print("2. Install PyTorch nightly with CUDA 12.8 support")
    print("3. Verify GPU functionality")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        sys.exit(0)

    # Step 1: Uninstall existing PyTorch
    print("\n" + "="*60)
    print("STEP 1: Uninstalling existing PyTorch packages")
    print("="*60)

    packages_to_remove = ["torch", "torchvision", "torchaudio"]
    for package in packages_to_remove:
        run_command(
            f"pip uninstall {package} -y",
            f"Uninstalling {package}"
        )

    # Step 2: Install PyTorch nightly with CUDA 12.8
    print("\n" + "="*60)
    print("STEP 2: Installing PyTorch Nightly (CUDA 12.8)")
    print("="*60)
    print("\nThis may take several minutes...")

    install_command = (
        "pip install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/nightly/cu128"
    )

    success = run_command(install_command, "Installing PyTorch nightly")

    if not success:
        print("\n❌ Installation failed!")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try running the command manually:")
        print(f"   {install_command}")
        print("3. Visit: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # Step 3: Verify installation
    print("\n" + "="*60)
    print("STEP 3: Verifying GPU Functionality")
    print("="*60)

    verify_script = os.path.join(
        os.path.dirname(__file__),
        "verify_gpu.py"
    )

    if os.path.exists(verify_script):
        print("\nRunning GPU verification script...\n")
        subprocess.run([sys.executable, verify_script])
    else:
        print("\nManual verification...")
        try:
            import torch
            print(f"\n✓ PyTorch imported successfully")
            print(f"  Version: {torch.__version__}")
            print(f"  CUDA Available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"  CUDA Version: {torch.version.cuda}")
                print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

                # Test GPU computation
                device = torch.device("cuda:0")
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor)
                print(f"\n✓ GPU computation test PASSED")
                print(f"  Test tensor created on: {result.device}")
            else:
                print("\n❌ CUDA not available after installation")

        except Exception as e:
            print(f"\n❌ Verification failed: {e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("✓ Installation Complete!")
    print("="*60)
    print("\nYour RTX 5060 Ti should now work with PyTorch.")
    print("You can continue with DINO-VO development.")

if __name__ == "__main__":
    main()
