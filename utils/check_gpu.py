import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("REASON: PyTorch cannot see your GPU.")
    print("Possibility 1: You installed the CPU-only version of PyTorch.")
    print("Possibility 2: You do not have an NVIDIA GPU (AMD/Intel GPUs don't work with standard PyTorch).")
    print("Possibility 3: You have an NVIDIA GPU, but your drivers are outdated.")