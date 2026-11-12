"""Small, in-process DSA dQ/dW screen; never formal qualification or promotion."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from kernel_envolve.research import paired_probe, run_stages

PACKAGE = "apodex_core.ops.candidate.dsa_tilelang.flash_fast"
QUERY = Path("apodex_core/ops/candidate/dsa_tilelang/flash_fast/residual_indexer_query.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_query(path):
    # A separate module name retains each callable and its JIT globals. Shared
    # helpers come from the explicit --core root; this screen changes ONE leaf.
    name = PACKAGE + ".research_query_" + digest(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].backward_query


def ratios(probe):
    trials = sorted({row["trial"] for row in probe["samples"]})
    return [
        statistics.median(
            row["duration"]
            for row in probe["samples"]
            if row["trial"] == trial and row["role"] == "candidate"
        )
        / statistics.median(
            row["duration"]
            for row in probe["samples"]
            if row["trial"] == trial and row["role"] == "baseline"
        )
        for trial in trials
    ]


def decide(aa, ab, *, guard, margin):
    """Conservative research screen, not a confidence interval or formal score."""
    if any(abs(value - 1) > margin for value in aa):
        return "inconclusive", "A/A instability; diagnose this measurement before expanding"
    if min(ab) > 1 + margin:
        return "reject", "Every paired block regresses beyond the declared margin"
    if guard and max(ab) <= 1 + margin:
        return "continue", "No material regression in this counterexample screen"
    if not guard and max(ab) < 1 - margin:
        return "continue", "Benefit signal; inspect complete-call costs before formal evaluation"
    return "inconclusive", "Mixed or small signal; no automatic expensive follow-up"


def make_case(length, topk, seed, full, torch):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    q = randn(length, 32, 128, dtype=torch.float16)
    key = randn(length, 128, dtype=torch.float16)
    weight = randn(length, 32)
    gradient = randn(length, topk, dtype=torch.float16)
    indices = torch.randint(length, (length, topk), device="cuda", generator=generator)
    if not full:
        slots = torch.arange(topk, device="cuda")[None, :]
        ends = torch.arange(length, device="cuda")[:, None] % (topk + 1)
        indices.masked_fill_(slots >= ends, -1)
        indices[1::7, 2::4] = -1
        indices[2::13, -1] = 0  # A far valid slot prevents invalid causal assumptions.
    mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (length, topk, 1),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    scale = torch.rand(length, device="cuda", generator=generator) + 0.1
    values = (q, key, weight, gradient, indices, mask, scale)
    originals = tuple(value.clone() for value in values)

    def bind(fn):
        # This API allocates dQ/dW and does not accumulate mutable output state.
        # Descriptor preparation, layout handling and output allocation stay timed.
        return lambda: fn(*values[:6], gradient_scale=scale, warp_m=1, warp_n=2)

    def check(a, b):
        expected, actual = a(), b()
        if len(expected) != 2 or len(actual) != 2:
            raise AssertionError("Missing dQ/dW output")
        for x, y in zip(expected, actual):
            if x.shape != y.shape or x.dtype != y.dtype or not torch.isfinite(y).all():
                raise AssertionError("Invalid query gradient")
            if not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
                raise AssertionError("Query dQ/dW differs bytewise from the selected control")
        if any(not torch.equal(x, y) for x, y in zip(values, originals)):
            raise AssertionError("Query mutated its inputs")

    return bind, check


def screen(candidates, stage_factory, emit):
    """Only dependent stages of ONE candidate stop; independent candidates continue.

    Exceptions deliberately abort the command. A CUDA failure needs a clean
    process; it must not be caught and treated like a performance rejection.
    """
    results = {}
    for name, candidate in candidates:
        results[name] = run_stages(
            stage_factory(candidate), emit=lambda event: emit({"candidate": name, **event})
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Frozen shared helper/source root")
    parser.add_argument(
        "--control", type=Path, help="Strong research control; defaults to core query"
    )
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=QUERY_PY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7981)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args(argv)
    if (
        args.length < 1
        or not 1 <= args.topk <= 2048
        or args.trials < 2
        or args.repeats < 1
        or not math.isfinite(args.margin)
        or not 0 < args.margin < 1
    ):
        parser.error("Invalid shape, sampling budget or margin")
    candidates = []
    for value in args.candidate:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or name in [n for n, _ in candidates]:
            parser.error("Candidates require distinct names and NAME=QUERY_PY")
        candidates.append((name, Path(path).resolve(strict=True)))
    args.core = args.core.resolve(strict=True)
    control = (args.control or args.core / QUERY).resolve(strict=True)
    sources = {str(p): digest(p) for p in [args.core / QUERY, control, *[p for _, p in candidates]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Immutable per-attempt output: a repeat must get a new output path.
    report = {
        "research_only": True,
        "eligible_for_best": False,
        "status": "running",
        "pid": os.getpid(),
        "sources": sources,
        "core": str(args.core),
        "control": str(control),
        "config": vars(args) | {"core": str(args.core)},
        "scope": "dQ/dW public wrapper; not complete auxiliary F+B or independent numerical gates",
    }
    report["config"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in report["config"].items()
    }
    started = time.perf_counter()
    with args.output.open("x") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    with args.output.with_suffix(args.output.suffix + ".jsonl").open("x") as events:

        def emit(event):
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)

        try:
            sys.path.insert(0, str(args.core))
            import apodex_core
            import torch

            if not Path(apodex_core.__file__).resolve().is_relative_to(args.core):
                raise ValueError("apodex_core was imported from a different source root")
            torch.use_deterministic_algorithms(True)
            report["gpu"] = torch.cuda.get_device_name(0)
            report["torch"] = torch.__version__
            baseline = load_query(control)
            original = load_query(args.core / QUERY)
            report["startup_seconds"] = time.perf_counter() - started
            cache = {}
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def measure(fn):
                torch.cuda.synchronize()
                begin.record()
                for _ in range(args.repeats):
                    fn()
                end.record()
                end.synchronize()
                return begin.elapsed_time(end) / args.repeats

            def stage_factory(path):
                candidate = load_query(path)

                def stage(full):
                    if full not in cache:
                        bind, check = make_case(args.length, args.topk, args.seed, full, torch)
                        a = bind(baseline)
                        check(bind(original), a)
                        aa = paired_probe(
                            a, a, measure=measure, check=lambda: check(a, a), trials=args.trials
                        )
                        cache[full] = bind, check, aa
                    bind, check, aa = cache[full]
                    a, b = bind(baseline), bind(candidate)
                    ab = paired_probe(
                        a, b, measure=measure, check=lambda: check(a, b), trials=args.trials
                    )
                    decision, reason = decide(
                        ratios(aa), ratios(ab), guard=full, margin=args.margin
                    )
                    return {"decision": decision, "reason": reason, "aa": aa, "ab": ab}

                return [
                    ("full_support_guard", lambda: stage(True)),
                    ("holey_support_benefit", lambda: stage(False)),
                ]

            report["candidates"] = screen(candidates, stage_factory, emit)
            report["case_builds"] = len(cache)
            report["status"] = "completed"
        except BaseException as exc:
            report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            report["seconds"] = time.perf_counter() - started
            report["sources_unchanged"] = all(
                Path(p).is_file() and digest(p) == h for p, h in sources.items()
            )
            if not report["sources_unchanged"]:
                report["status"] = "error"
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
    if not report["sources_unchanged"]:
        raise RuntimeError("Source changed during the research batch")


if __name__ == "__main__":
    main()
