"""Check the scope of apodex-core's numerical gate, without executing GPU kernels.

Run with the core checkout being audited and an output JSON path. Synthetic
identity operators deliberately violate API contracts; no production operator
is modified. PASS here means the observed gate behavior matches this audit.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core.resolve()))
    import torch
    from apodex_core.verification.method import gate

    def gen_input():
        return torch.randn(2, 4, dtype=torch.float32)

    probes = {
        "correct_identity": (lambda x: x.clone(), "PASS", True),
        "wrong_output_shape": (lambda x: x.unsqueeze(0).clone(), "PASS", False),
        "wrong_output_dtype": (lambda x: x.double(), "PASS", False),
        "forbidden_input_mutation": (lambda x: x.zero_().clone(), "PASS", False),
        "wrong_values": (lambda x: x + 100, "FAIL", False),
        "nonfinite_values": (lambda x: torch.full_like(x, float("nan")), "FAIL", False),
    }
    results = []
    for name, (candidate, expected_gate, expected_contract) in probes.items():
        # The independent check preserves the original input before candidate runs.
        torch.manual_seed(7)
        x = gate.vars_.shift_input(gen_input())
        original = x.clone()
        output = candidate(x)
        checks = {
            "shape": output.shape == original.shape,
            "dtype": output.dtype == original.dtype,
            "input_unchanged": torch.equal(x, original),
            "identity_values": torch.equal(output, original),
        }
        case = SimpleNamespace(
            candidate=candidate,
            reference=lambda x: x.clone(),
            golden=lambda x: x.double(),
            gen_input=gen_input,
        )
        v3, u, xi, v2, zmax, count = gate.run(
            case,
            torch.float32,
            gate._CFG["vars_gate"]["default_K"],
            gate._CFG["vars_gate"]["default_fail_K"],
            gate.bias.load_config(),
        )
        verdict = gate.combine(v2, v3)
        assert verdict == expected_gate, (name, verdict, expected_gate)
        assert all(checks.values()) == expected_contract, (name, checks)
        results.append(
            {
                "probe": name,
                "native_numerical_verdict": verdict,
                "bias_verdict": v2,
                "magnitude_verdict": v3,
                "replicates": count,
                "identity_contract_checks": checks,
            }
        )
    padded = torch.randn(2, 11, dtype=torch.float32)[:, :4]
    shifted = gate.vars_.shift_input(padded)
    layout_probe = {
        "generator_stride": list(padded.stride()),
        "gate_input_stride": list(shifted.stride()),
        "row_padding_preserved": padded.stride() == shifted.stride(),
    }
    assert not layout_probe["row_padding_preserved"], layout_probe
    payload = {
        "scope": "CPU execution of native numerical gate with synthetic identity operators; not GPU or production-operator validation",
        "core_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.core, text=True
        ).strip(),
        "torch_version": torch.__version__,
        "r_min": gate.R_MIN,
        "results": results,
        "shift_input_layout_probe": layout_probe,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for result in results:
        print(
            result["probe"],
            result["native_numerical_verdict"],
            "contract_ok=" + str(all(result["identity_contract_checks"].values())),
        )


if __name__ == "__main__":
    main()
