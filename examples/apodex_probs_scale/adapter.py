"""Production MoE probability scaling, including dx and router-probability gradients."""

import torch
from apodex_core.ops import fused_probs_scale, probs_scale
from apodex_core.ops.candidate.triton_probs_scale import set_fused_dprobs

from kernel_envolve.training import Workload, forward_backward


def create(case, seed, device, oracle):
    args = case["args"]
    dtype = getattr(torch, args["dtype"])
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        args["tokens"], args["hidden"], device=device, dtype=dtype, generator=generator
    ).requires_grad_()
    probs = torch.rand(
        args["tokens"], 1, device=device, dtype=torch.float32, generator=generator
    ).requires_grad_()
    padding = args.get("grad_padding", 0)
    go = torch.randn(
        args["tokens"], args["hidden"] + padding, device=device, dtype=dtype, generator=generator
    )[:, : args["hidden"]]
    # Match the default production contract; opting into another reduction is a new task.
    set_fused_dprobs(False)
    fn = probs_scale if oracle else fused_probs_scale
    return Workload(
        {"x": x, "probs": probs, "grad_output": go},
        lambda: forward_backward(fn, (x, probs), (go,)),
    )
