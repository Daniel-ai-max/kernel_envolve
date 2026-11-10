import torch


def run(x):
    return torch.sum(x, dim=1)
