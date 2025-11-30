import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

print("GPU Info:")
print(f"  - Name: {torch.cuda.get_device_name(0)}")
print(f"  - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  - CUDA available: {torch.cuda.is_available()}")
