import platform
import torch

print('Python platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(256, 256, device='mps')
    print('MPS matrix multiply OK:', (x @ x).shape)
else:
    print('Using CPU; verify Apple Silicon PyTorch installation.')
