"""Detect persistent changes to numerical and execution settings between roles."""

import sys

import numpy as np


def state():
    values = {"numpy_errors": np.geterr()}
    torch = sys.modules.get("torch")
    if torch is not None:
        values["torch"] = {
            "default_dtype": str(torch.get_default_dtype()),
            "grad_enabled": torch.is_grad_enabled(),
            "threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
        if torch.cuda.is_initialized():
            values["cuda"] = {
                "device": torch.cuda.current_device(),
                "stream": torch.cuda.current_stream().cuda_stream,
            }
    return values


class RuntimeGuard:
    def __init__(self):
        self.expected = state()

    def check(self):
        current = state()
        for key, value in self.expected.items():
            if current.get(key) != value:
                raise RuntimeError(
                    f"Persistent runtime setting changed: {key}; expected {value}, got {current.get(key)}"
                )
        # A NumPy-only task may lazily import torch; track it from its first appearance.
        self.expected = current

    def call(self, function, *args):
        self.check()
        value = function(*args)
        self.check()
        return value
