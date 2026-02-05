"""
GPU Verification Script for DINO-VO
Checks CUDA availability and displays GPU specifications.
"""

import torch
import sys

def verify_gpu():
    """Verify GPU availability and display specifications."""

    print("=" * 60)
    print("GPU Verification for DINO-VO")
    print("=" * 60)

    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ ERROR: CUDA is not available!")
        print("Please install CUDA-enabled PyTorch:")
        print("Visit: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # Display CUDA version
    print(f"CUDA Version: {torch.version.cuda}")

    # Get GPU device count
    device_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {device_count}")

    # Display GPU details
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")

        # Get memory info
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024**3)
        print(f"Total VRAM: {total_memory_gb:.2f} GB")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"Currently Allocated: {allocated:.2f} GB")
        print(f"Currently Reserved: {reserved:.2f} GB")

        # Compute capability
        print(f"Compute Capability: {props.major}.{props.minor}")

    # Test GPU computation
    print("\n--- Testing GPU Computation ---")
    try:
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        print("✓ GPU computation test passed")
        print(f"Test tensor device: {result.device}")
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        sys.exit(1)

    # Memory budget check for DINO-VO
    print("\n--- DINO-VO Memory Budget ---")
    print("Estimated VRAM usage during training: 4-6 GB")
    if total_memory_gb >= 14:
        print(f"✓ Your GPU ({total_memory_gb:.2f} GB) has sufficient VRAM")
        print(f"  Available headroom: ~{total_memory_gb - 6:.2f} GB")
    else:
        print(f"⚠ Warning: Limited VRAM ({total_memory_gb:.2f} GB)")
        print("  Consider using FP16 or reducing batch size")

    print("\n" + "=" * 60)
    print("✓ GPU verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    verify_gpu()
