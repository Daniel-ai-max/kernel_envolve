import torch

# The same setting applies to the oracle, original baseline and all candidates.
torch.backends.cuda.matmul.allow_tf32 = False


def make_inputs(case, seed):
    generator = torch.Generator(device=case["device"]).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device=case["device"])

    x = rand(case["rows"], case["columns"] * case["stride"])
    return (x[:, :: case["stride"]],)


def reference(x):
    return x.double().sum(dim=1).float()


def baseline(x):
    return x.sum(dim=1)
