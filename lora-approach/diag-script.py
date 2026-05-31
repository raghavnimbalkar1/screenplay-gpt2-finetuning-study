import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch

print("--- AMD ROCm Diagnostic ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Is 'CUDA' (ROCm) available?: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("WARNING: PyTorch cannot see the AMD GPU. Check your ROCm drivers.")