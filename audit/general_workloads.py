"""Input contract shared by development factories, not the final numerical oracle.

Public domains: row_sum rows 0..128, width 0..32769; fused_relu rows 0..512,
width 0..8193; matmul M/N 0..512, K 0..1024. FP32/FP16/BF16, independent
nonoverlapping positive-stride views. Empty dimensions are correctness guards.
All data is finite. These are benchmark domains, not a production frequency model.
"""

import torch

DTYPES = {name: getattr(torch, name) for name in ("float32", "float16", "bfloat16")}
# Per-dtype checks supplement the evaluator's common outer envelope.
TOLERANCES = {
    "row_sum": {"float32": (1e-5, 2e-4), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
    "matmul": {"float32": (1e-4, 5e-4), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
    "fused_relu": {"float32": (1e-5, 1e-5), "float16": (0.003, 0.003), "bfloat16": (0.025, 0.025)},
}


def strides(shape, layout):
    column = layout.get("column", 1)
    pad = layout.get("pad", 0)
    if len(shape) == 1:
        return (column,)
    rows, columns = shape
    if layout.get("transpose", False):
        return column, max(1, rows * column + pad)
    return max(1, columns * column + pad), column


def view(values, layout):
    shape = tuple(values.shape)
    stride = strides(shape, layout)
    offset = layout.get("offset", 0)
    extent = 1 + sum(max(n - 1, 0) * s for n, s in zip(shape, stride))
    base = torch.full(
        (offset + extent + 7,), float("nan"), dtype=values.dtype, device=values.device
    )
    result = torch.as_strided(base, shape, stride, offset)
    result.copy_(values)
    return result


def make_inputs(name, case, seed):
    dtype = DTYPES[case["dtype"]]
    device = case["device"]
    generator = torch.Generator(device=device).manual_seed(seed)
    if name == "matmul":
        shapes = [(case["m"], case["k"]), (case["k"], case["n"])]
    else:
        shapes = [(case["rows"], case["columns"])]
        if name == "fused_relu":
            shapes += [(case["columns"],), (case["columns"],), shapes[0]]
    values = [torch.randn(shape, generator=generator, device=device).to(dtype) for shape in shapes]
    pattern = case.get("pattern", "normal")
    if pattern == "zeros":
        values = [torch.zeros_like(x) for x in values]
    elif pattern == "cancellation":
        # Exactly representable paired values; small remaining tails are observable.
        values = [(x.float().mul(8).round().div(8)).to(dtype) for x in values]
        if name == "matmul":
            a, b = values
            pairs = a.shape[1] // 2
            a[:, 1 : 2 * pairs : 2] = a[:, : 2 * pairs : 2]
            b[1 : 2 * pairs : 2] = -b[: 2 * pairs : 2]
        elif name == "row_sum":
            x = values[0]
            pairs = x.shape[1] // 2
            x[:, 1 : 2 * pairs : 2] = -x[:, : 2 * pairs : 2]
        else:
            x, scale, bias, residual = values
            scale.fill_(0.5)
            x.copy_(torch.arange(x.shape[1], device=device).remainder(5).to(dtype))
            epsilon = torch.finfo(dtype).eps
            bias.copy_(-x.new_tensor(0.5) * torch.arange(x.shape[1], device=device).remainder(5))
            bias.add_(
                torch.arange(x.shape[1], device=device).remainder(3).sub(1).to(dtype) * epsilon
            )
            residual.mul_(0.125)
    elif pattern != "normal":
        raise ValueError("Unknown numerical pattern")
    return tuple(view(x, layout) for x, layout in zip(values, case["layouts"], strict=True))


def reference(name, *inputs):
    arrays = [x.double() for x in inputs]
    if name == "row_sum":
        out = arrays[0].sum(dim=1)
    elif name == "matmul":
        out = arrays[0] @ arrays[1]
    elif name == "fused_relu":
        x, scale, bias, residual = arrays
        out = torch.relu(x * scale + bias) + residual
    else:
        raise ValueError("Unknown workload")
    return out.to(inputs[0].dtype)


def baseline(name, *inputs):
    if name == "row_sum":
        return inputs[0].sum(dim=1)
    if name == "matmul":
        return inputs[0] @ inputs[1]
    x, scale, bias, residual = inputs
    return torch.relu(x * scale + bias) + residual


def validate(name, inputs, output):
    dtype = str(inputs[0].dtype).split(".")[-1]
    rtol, atol = TOLERANCES[name][dtype]
    expected = reference(name, *inputs)
    assert output.shape == expected.shape and output.dtype == expected.dtype
    assert output.device == expected.device
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol, equal_nan=False)
    for x in inputs:
        # The factory owns the whole backing allocation, including sentinel padding.
        # A write outside the logical view is a real bug even if its output is correct.
        base = x._base
        assert base is not None and base.ndim == 1, "Input lost factory backing storage"
        used = torch.zeros(base.shape, dtype=torch.bool, device=x.device)
        torch.as_strided(used, x.shape, x.stride(), x.storage_offset()).fill_(True)
        assert torch.isnan(base[~used]).all(), "Input padding was overwritten"
