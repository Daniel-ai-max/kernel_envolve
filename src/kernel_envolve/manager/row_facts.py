"""One system-owned snapshot for a coherent narrated dashboard row."""

from ..storage import digest


def row_facts(task):
    delivery = task.get("delivery") or {}
    delivered = delivery.get("best") if delivery.get("status") == "verified" else None
    best, assessment = task.get("best") or delivered, task.get("verified_assessment")
    benefit = {"ref": "system:benefit", "kind": "none", "value": None}
    if best:
        benefit.update(
            kind="native_delivery" if delivered else "retained_best",
            value={
                k: best.get(k)
                for k in ("baseline_speedup", "incremental_speedup", "run", "candidate_sha256")
            },
        )
    elif assessment and not task.get("evidence_error"):
        benefit.update(
            kind="scoped_assessment",
            value={
                k: assessment.get(k)
                for k in ("source_sha256", "scope_label", "metrics", "recorded_at")
            },
        )
    candidates = []
    for field, kind in (
        ("research_result", "research"),
        ("verified_assessment", "assessment"),
        ("latest_validation", "validation"),
        ("recent_evaluation", "evaluation"),
    ):
        value = task.get(field)
        if value:
            candidates.append((value.get("recorded_at", 0) or 0, kind, value))
    latest = max(candidates, key=lambda x: (x[0], x[1] == "validation"), default=(0, "none", {}))
    latest_value = {
        k: latest[2].get(k)
        for k in (
            "status",
            "scope",
            "summary",
            "limitations",
            "source_sha256",
            "recorded_at",
            "mode",
            "execution",
            "correctness",
            "baseline_speedup",
            "task",
        )
        if k in latest[2]
    }
    if delivered:
        candidates.append(
            (delivered.get("recorded_at", 0), "delivery", delivered | {"task": delivery["task"]})
        )
        latest = max(candidates, key=lambda x: (x[0], x[1] == "validation"))
        latest_value = dict(latest[2]) if latest[1] == "delivery" else latest_value
    pr = task.get("implementation_pr") or delivery.get("implementation_pr")
    if delivered and pr and not pr.get("current_head_verified"):
        pr = None
    value = {
        "task": task["id"],
        "baseline": task["spec"]["baseline"]["revision"],
        "lifecycle": {"desired": task.get("desired"), "phase": task.get("phase")},
        "execution": {
            k: v
            for k, v in (task.get("recent_execution") or {}).items()
            if k
            in ("id", "task", "kind", "status", "mode", "execution", "correctness", "recorded_at")
        },
        "evidence_error": task.get("evidence_error"),
        "delivery": {k: delivery.get(k) for k in ("task", "status", "policy_phase")},
        "benefit": benefit,
        "optimization_pr": {"ref": "system:optimization_pr", "value": pr},
        "latest_result": {"ref": "system:latest_result", "kind": latest[1], "value": latest_value},
    }
    return value | {"revision": digest(value)}
