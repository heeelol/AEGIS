"""Quick setup guide for local GPU training."""

import torch

print("\n" + "=" * 60)
print("GPU SETUP VERIFICATION")
print("=" * 60 + "\n")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("\n✓ Your GPU is ready for training!")
else:
    print("\n❌ CUDA not available. Install PyTorch with CUDA support:")
    print("\npip uninstall torch torchvision torchaudio -y")
    print(
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    print("\nThen verify again with this script.")

print("\n" + "=" * 60)
