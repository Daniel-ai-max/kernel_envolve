import torch
import triton
import triton.language as tl


@triton.jit
def _add(X, Y, OUT, N: tl.constexpr, SX: tl.constexpr, SY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    a = tl.load(X + i * SX, mask, 0)
    b = tl.load(Y + i * SY, mask, 0)
    tl.store(OUT + i, a + b, mask)


def run(x, y):
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    _add[(triton.cdiv(x.numel(), 256),)](
        x, y, output, x.numel(), x.stride(0), y.stride(0), BLOCK=256
    )
    return output
