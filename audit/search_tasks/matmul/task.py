import torch

# The same setting applies to the oracle, original baseline and all candidates.
torch.backends.cuda.matmul.allow_tf32 = False


def make_inputs(case, seed):
    generator = torch.Generator(device=case["device"]).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device=case["device"])

    a = rand(case["m"], case["k"])
    b = rand(case["k"], case["n"] * case["stride"])
    return a, b[:, :: case["stride"]]


def reference(a, b):
    return (a.double() @ b.double()).float()


def baseline(a, b):
    return a @ b
