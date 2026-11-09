import torch

# The same setting applies to the oracle, original baseline and all candidates.
torch.backends.cuda.matmul.allow_tf32 = False


def make_inputs(case, seed):
    generator = torch.Generator(device=case["device"]).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device=case["device"])

    shape = (case["rows"], case["columns"] * case["stride"])
    x = rand(*shape)[:, :: case["stride"]]
    residual = rand(*shape)[:, :: case["stride"]]
    scale, bias = rand(case["columns"]), rand(case["columns"])
    return x, scale, bias, residual


def reference(x, scale, bias, residual):
    return (torch.relu(x.double() * scale.double() + bias.double()) + residual.double()).float()


def baseline(x, scale, bias, residual):
    return torch.relu(x * scale + bias) + residual
