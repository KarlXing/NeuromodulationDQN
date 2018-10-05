import torch

def tanh_beta(x, beta):
    x = x * beta
    return torch.tanh(x)