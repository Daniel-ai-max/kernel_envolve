# Continuous optimization manager

Implementation target, agreed 2026-11-10. This document records the full acceptance scope;
it is not a completion report. Implementation mapping, verification results and remaining hardware validation are recorded in [the comparison](super-agent-comparison.md).

## Responsibilities

- A durable optimization task outlives individual agent sessions and rounds. One writer
  and one workspace per task. Exit from a round does not complete optimization.
- Configure a fixed task list, a command-based agent adapter, and one or two sessions
  per GPU. Resume from notes, existing jobs, runs and best rather than restart research.
- Separate session concurrency from GPU concurrency. Serialize all managed GPU work,
  including correctness jobs, by physical GPU identity. Never release on heartbeat
  expiry alone. Recover or quarantine abandoned work before dispatching a replacement.
- Durable state transitions, retry/backoff, bounded failures, operator pause/resume,
  resource reassignment and cumulative wall/GPU budgets. No automatic resume of a
  user-paused task. No unbounded restart storm.
- Index both native and generic evaluator evidence. Preserve baseline/candidate/best/
  integrated distinctions. Never turn an agent's self-report into verified speedup.
- Baseline repository/version/entrypoint, implementation introduction date and source,
  freeze date, hardware and evaluation contract. Unknown introduction dates stay unknown.
- Overview, task details and resources views, with baseline columns, best history,
  per-case latency/regression, coverage, hypotheses, experiment outcomes, next action,
  session/PR links and resource usage. Empty/error/stale states are explicit.
- Existing evaluator API and standalone workflow remain compatible.

## Scope and boundaries

One coordinator per state directory on a POSIX GPU host; browser can use an SSH tunnel.
Agent commands run in dedicated workspaces and receive a checkpoint prompt on stdin.
They can adapt any coding-agent runtime through an executable; this project does not
reimplement a model harness. Host supervision is independent of an LLM response.
The agent command must route GPU commands through the managed wrapper. Like Super-agent,
this is a trusted-operator system, not an OS sandbox. Unmanaged GPU processes are detected
and block measurement; process isolation is not claimed to be a security boundary.
No messaging connectors, notification service, inferred task authorization, distributed
cluster scheduler or model-based router is needed for the agreed fixed-task workflow.

## Verification required

- SQLite transactions and idempotent registration; baseline identity cannot drift.
- Concurrent task launches and GPU requests; physical-device exclusivity and FIFO fairness.
- Worker/manager termination, startup crash windows, surviving child processes, stale
  process identities and unavailable hardware: do not overlap/replay blindly.
- Pause/resume, budget exhaustion, failed rounds, restart backoff and checkpoint continuity.
- Real generic evaluator receipts and native evidence fixtures, corrupt/tampered artifacts,
  changed contracts, missing runs and quick-only results. Immutable promotion history.
- HTTP control authentication/origin validation, asset packaging, responsive browser checks.
- Existing tests, lint, formatting, package build and installed-command smoke.
- Document a source-linked comparison against Super-agent commit
  `8c32a9dc5d72c033816c6a8110b37f40cf94fba4`, including material limitations and test evidence.

## UI contract

Dark-Dashboard for daily engineering supervision. Canvas #090e12, text #e9ebee,
muted #a2a5a7, focus #6aa7f4; success #73bf69, warning #ff9830, failure #f2495c.
Menlo/SF Mono for values and labels; system sans/PingFang SC for prose. Body 13px/1.5,
headings 15px/600; tabular figures. State text accompanies colors. No gradients,
glow, large rounded cards, decorative images, fake percentages or invented production data.
No logo/images required. System fonts; local assets only. All example data explicitly marked.
