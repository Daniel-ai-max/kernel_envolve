import torch


def make_inputs(case, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    n, stride = case["n"], case["stride"]
    return tuple(
        torch.randn(n * stride, generator=generator, device="cuda", dtype=torch.float32)[::stride]
        for _ in range(2)
    )


def reference(x, y):
    return x + y


def baseline(x, y):
    return torch.add(x, y)
