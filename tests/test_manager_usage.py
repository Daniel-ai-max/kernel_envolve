"""Allocation accounting preserves idle time, task lineage and import idempotence."""

import copy
import json

import pytest

from kernel_envolve.manager.cli import main
from kernel_envolve.manager.usage import allocations, import_usage, usage_view
from kernel_envolve.manager.views import detail, overview

pytest_plugins = ["test_manager_supersede"]


def receipt(**changes):
    return {
        "version": 1,
        "allocations": [
            {
                "id": "platform-job-1",
                "started_at": 1000,
                "observed_at": 4600,
                "ended": False,
                "gpus": 2,
                "tasks": {"old": 1, "third": 1},
                "source": "platform-observation.json",
                **changes,
            }
        ],
    }


def test_idle_time_and_migration_are_counted_once(replaced):
    import_usage(replaced, receipt())
    import_usage(replaced, receipt())
    assert len(allocations(replaced)) == 1
    assert usage_view(allocations(replaced))["gpu_hours"] == 2
    assert detail(replaced, "old")["allocation_usage"]["gpu_hours"] == 1
    replaced.supersede("old", "new")
    assert detail(replaced, "new")["allocation_usage"]["gpu_hours"] == 1
    assert overview(replaced, compact=True)["allocation_usage"]["gpu_hours"] == 2
    assert replaced.task("old")["gpu_seconds"] == 10  # Command timer is untouched.
    import_usage(replaced, receipt(observed_at=8200, ended=True))
    assert usage_view(allocations(replaced))["gpu_hours"] == 4
    assert usage_view(allocations(replaced))["ended"]
    with pytest.raises(ValueError):
        import_usage(replaced, receipt(observed_at=11800, ended=True))


@pytest.mark.parametrize(
    "changes",
    [
        {"gpus": 1},
        {"tasks": {"missing": 2}},
        {"tasks": {"old": -1, "third": 3}},
        {"observed_at": float("nan")},
        {"observed_at": float("inf")},
        {"observed_at": 500},
        {"gpus": True},
        {"ended": "false"},
        {"observed_at": 1e12},
        {"source": ""},
    ],
)
def test_invalid_receipts_do_not_mutate_usage(replaced, changes):
    with pytest.raises(ValueError):
        import_usage(replaced, receipt(**changes))
    assert allocations(replaced) == []


def test_updates_cannot_move_cost_or_decrease_it(replaced):
    import_usage(replaced, receipt())
    for changes in ({"observed_at": 4500}, {"tasks": {"new": 1, "third": 1}}, {"started_at": 900}):
        with pytest.raises(ValueError):
            import_usage(replaced, receipt(**changes))
    assert usage_view(allocations(replaced))["gpu_hours"] == 2


def test_import_is_atomic_and_missing_usage_is_not_zero(replaced):
    assert overview(replaced)["allocation_usage"] is None
    assert detail(replaced, "old")["allocation_usage"] is None
    batch = receipt()
    other = copy.deepcopy(batch["allocations"][0])
    other.update(id="job-2", tasks={"unknown": 2})
    batch["allocations"].append(other)
    with pytest.raises(ValueError):
        import_usage(replaced, batch)
    assert allocations(replaced) == []
    duplicate = receipt()
    duplicate["allocations"] *= 2
    with pytest.raises(ValueError):
        import_usage(replaced, duplicate)
    assert allocations(replaced) == []


def test_cli_import_preserves_source(replaced, tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps(receipt()))
    assert main(["--state", str(replaced.root), "import-usage", "--file", str(path)]) == 0
    assert overview(replaced)["allocation_usage"]["sources"] == ["platform-observation.json"]
