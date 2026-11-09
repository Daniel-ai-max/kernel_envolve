import torch


def run(x, scale, bias, residual):
    return torch.relu(x * scale + bias) + residual
