import os
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT
import torch
torch.set_grad_enabled(False)  # Disable gradients

# Patch torch._classes to avoid path issues
import torch._classes
def mock_getattr(*args, **kwargs):
    return None
torch._classes.__getattr__ = mock_getattr