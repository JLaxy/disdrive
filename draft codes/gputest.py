import torch

# Check if PyTorch detects GPU
print("PyTorch version:", torch.__version__)
print("Is CUDA available?:", torch.cuda.is_available())

# Check GPU details
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("No GPU detected.")
