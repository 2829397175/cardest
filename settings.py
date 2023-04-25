import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
print('Device', DEVICE)
MASK_SCHEME = 0
#MASK_SCHEME = 1