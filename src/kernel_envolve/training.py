"""Explicit training workloads: state isolation and observable forward/backward results."""

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Workload:
    """Declare ALL tensor inputs, parameters, buffers and persistent update state.

    run returns a nonempty tensor tree, including every relevant gradient.
    The evaluator resets declared state outside timing; one run may contain an
    entire microbatch accumulation sequence. Custom stream/Graph setup belongs
    in the adapter. validate can assert dispatch counters or application invariants.
    """

    tensors: dict
    run: Callable
    mutable: tuple = ()
    validate: Callable = field(default=lambda result: None)


def forward_backward(function, inputs, grad_outputs):
    """Observe the real custom backward, without accumulating hidden .grad state."""
    import torch

    outputs = function(*inputs)
    flat = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    differentiable = tuple(x for x in inputs if x.requires_grad)
    gradients = torch.autograd.grad(flat, differentiable, grad_outputs=grad_outputs)
    return {"output": outputs, "gradients": gradients}


def tensor_tree(value, leaf):
    import torch

    if isinstance(value, torch.Tensor):
        return leaf(value)
    if type(value) in (tuple, list) and value:
        return type(value)(tensor_tree(v, leaf) for v in value)
    if type(value) is dict and value and all(isinstance(k, str) for k in value):
        return {k: tensor_tree(v, leaf) for k, v in value.items()}
    raise ValueError("Expected a nonempty tensor tree (dict, tuple or list)")


def metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "requires_grad": tensor.requires_grad,
        "conjugate": tensor.is_conj(),
        "negative": tensor.is_neg(),
    }


def same_storage(a, b):
    return (
        a.device == b.device
        and a.untyped_storage().nbytes() > 0
        and b.untyped_storage().nbytes() > 0
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


def host(tensor):
    return tensor.detach().resolve_conj().resolve_neg().cpu().clone()


def equal(a, b):
    """Exact values, including signed zero and NaN payloads, for state preservation."""
    import torch

    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8),
            b.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8),
        )
    )


def byte_view(tensor):
    """Logical values in bitwise order, including signed zeros and NaN payloads."""
    import torch

    return tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)


class Session:
    def __init__(self, workload, device, device_guard_limit=0):
        import torch

        from .runtime import RuntimeGuard

        if not isinstance(workload, Workload) or not workload.tensors:
            raise ValueError("Adapter must return Workload with explicit tensors")
        if set(workload.mutable) - workload.tensors.keys():
            raise ValueError("Unknown mutable tensor")
        if len(set(workload.mutable)) != len(workload.mutable):
            raise ValueError("Duplicate mutable tensor")
        for name, x in workload.tensors.items():
            if not isinstance(name, str) or not isinstance(x, torch.Tensor):
                raise ValueError("Workload.tensors must map names to tensors")
            if x.layout != torch.strided or x.device != torch.device(device) or not x.is_leaf:
                raise ValueError("Declare strided leaf tensors on the selected device")
        self.workload = workload
        self.specs = {k: metadata(v) for k, v in workload.tensors.items()}
        self.bindings = {
            k: (id(v), v.untyped_storage().data_ptr()) for k, v in workload.tensors.items()
        }
        started = time.perf_counter()
        self.values = {k: host(v) for k, v in workload.tensors.items()}
        self.input_snapshot_seconds = time.perf_counter() - started
        self.input_check_seconds = 0.0
        self.input_check_calls = 0
        self.input_guard_fallbacks = {}
        self.device = device
        self.device_guard_limit = device_guard_limit
        self._device_values = {}
        self._guard_allocations = {}
        self.aliases = {
            a: [b for b, y in workload.tensors.items() if same_storage(x, y)]
            for a, x in workload.tensors.items()
        }
        # Partially writable shared storage needs a specialized reset, not an ambiguous copy order.
        for name in workload.mutable:
            if any(n != name for n in self.aliases[name]):
                raise ValueError("Mutable shared storage requires a dedicated workload adapter")
        self._cache_inputs()
        self.guard = RuntimeGuard()
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None

    @property
    def private_allocated_bytes(self):
        """Actual live CUDA allocator deltas, never logical tensor sizes."""
        return sum(self._guard_allocations.values())

    def _cache_inputs(self):
        import torch

        started = time.perf_counter()
        self._device_values.clear()
        self._guard_allocations.clear()
        cuda = self.device.startswith("cuda")
        budget = self.device_guard_limit
        if cuda and budget:
            budget = min(budget, torch.cuda.mem_get_info(self.device)[0] // 8)
        for name, saved in self.values.items():
            size = saved.numel() * saved.element_size()
            if name in self.workload.mutable or not self.device_guard_limit:
                continue
            if size > budget:
                self._guard_fallback("budget")
                continue
            before = torch.cuda.memory_allocated(self.device) if cuda else 0
            try:
                # Use the immutable CPU original, not a second read of live input.
                # A factory's asynchronous write cannot redefine the expected bytes.
                self._device_values[name] = byte_view(saved).to(self.device, copy=True)
            except torch.cuda.OutOfMemoryError:
                self._guard_fallback("allocation_oom")
                break  # Keep the original exact CPU path for all uncached inputs.
            allocated = (
                torch.cuda.memory_allocated(self.device) - before if cuda else 0
            )
            cost = allocated if cuda else size
            if cost > budget or (cuda and allocated < size):
                # Account for allocator rounding; ambiguous deltas are not deducted.
                del self._device_values[name]
                self._guard_fallback("allocator_budget")
                break
            self._guard_allocations[name] = allocated
            budget -= cost
        self.input_snapshot_seconds += time.perf_counter() - started

    def _guard_fallback(self, reason):
        self.input_guard_fallbacks[reason] = self.input_guard_fallbacks.get(reason, 0) + 1

    def input_guard_stats(self):
        return {
            "device_inputs": sorted(self._device_values),
            "host_inputs": sorted(
                self.workload.tensors.keys() - self._device_values.keys() - set(self.workload.mutable)
            ),
            "private_allocated_bytes": self.private_allocated_bytes,
            "snapshot_seconds": self.input_snapshot_seconds,
            "check_seconds": self.input_check_seconds,
            "check_calls": self.input_check_calls,
            "fallbacks": dict(self.input_guard_fallbacks),
        }

    def _unchanged(self, name, tensor):
        import torch

        if name in self._device_values:
            try:
                return torch.equal(byte_view(tensor), self._device_values[name])
            except torch.cuda.OutOfMemoryError:
                # Noncontiguous inputs can require a temporary contiguous tensor.
                # Release only this private cache and retain exact host comparison.
                del self._device_values[name]
                del self._guard_allocations[name]
                self._guard_fallback("comparison_oom")
        return equal(tensor, self.values[name])

    def reset(self):
        import torch

        with torch.no_grad():
            for name, tensor in self.workload.tensors.items():
                tensor.grad = None
                if name in self.workload.mutable:
                    tensor.copy_(self.values[name])
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, self.device)

    def check_state(self):
        started = time.perf_counter()
        self.input_check_calls += 1
        try:
            if self.workload.tensors.keys() != self.specs.keys():
                raise AssertionError("State names changed")
            for name, tensor in self.workload.tensors.items():
                if (
                    metadata(tensor) != self.specs[name]
                    or (id(tensor), tensor.untyped_storage().data_ptr()) != self.bindings[name]
                ):
                    raise AssertionError(f"Tensor binding/layout changed: {name}")
                if name not in self.workload.mutable and not self._unchanged(name, tensor):
                    raise AssertionError(f"Forbidden input/state mutation: {name}")
            self.guard.check()
        finally:
            self.input_check_seconds += time.perf_counter() - started

    def observe(self, output):
        self.check_state()
        self.workload.validate(output)
        self.check_state()
        return {
            "inputs": self.values,
            "input_metadata": self.specs,
            "input_aliases": self.aliases,
            "mutable": list(self.workload.mutable),
            "output": tensor_tree(output, host),
            "output_metadata": tensor_tree(output, metadata),
            "output_aliases": tensor_tree(
                output,
                lambda x: [name for name, y in self.workload.tensors.items() if same_storage(x, y)],
            ),
            "state": {k: host(self.workload.tensors[k]) for k in self.workload.mutable},
        }

    def refresh(self, fresh):
        """Change values in existing objects, preserving stride, storage, and aliases."""
        import torch

        other = Session(fresh, self.device)
        if (self.specs, self.aliases, self.workload.mutable) != (
            other.specs,
            other.aliases,
            other.workload.mutable,
        ):
            raise ValueError("Seed changed workload metadata or alias contract")
        with torch.no_grad():
            for name, tensor in self.workload.tensors.items():
                if not equal(tensor, fresh.tensors[name]):
                    # Refuse overlapping writes (e.g. expand) rather than silently densifying.
                    tensor.copy_(fresh.tensors[name])
        self.values = other.values
        self.input_snapshot_seconds += other.input_snapshot_seconds
        self._cache_inputs()
        self.cpu_rng, self.cuda_rng = other.cpu_rng, other.cuda_rng
