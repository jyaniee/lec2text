import torch

print("PyTorch version :", torch.__version__)
print("CUDA build      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU count       :", torch.cuda.device_count())