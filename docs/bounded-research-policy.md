# Optional bounded research outcome policy

The default manager continues optimization within its configured budgets. A host
may opt a **single-device research lane** into two fixed natural-time windows:
4 hours from the allocation's original charged start, then at most one extension
to that same start plus 8 hours. Setup, draining and cleanup consume that window.
This policy never grants evaluator qualification or rewrites consumed wall/GPU
counters. Active agent seconds, executed GPU card-seconds and scheduler allocation
card-seconds are different quantities.

Register an executable `kind=research` task with its actual `host_command`. Pause
and drain it, then install the policy before starting the coordinator:

```sh
kernel-envolve manage --state /host/state policy-install lane \
  --started-at 1794811000 --min-gain 0.029126213592233 \
  --deadline-command '["/opt/venv/bin/python","/host/policy_deadline.py","/host/config.json"]'
kernel-envolve manage --state /host/state run
```

The timestamp is the allocation's authoritative original start, **not** the time
a model restarts. The example value must be replaced with that actual timestamp.
`min-gain` is median fixed-baseline latency reduction, independently required in
both full evaluation and confirmation. A speedup ratio threshold of 1.03 is
`1 - 1/1.03 = 0.029126213592233` latency reduction, not 0.03. Installation requires no evaluator,
adapter, test case list, delivery task, or per-round notes. It does not execute a
model itself. The coordinator automatically invokes the policy after evidence and
publication refresh. Once host admission succeeds, existing configure/resume,
schedule and supervisor paths launch the real model host.

Installation is one-time. Reopening the state, retrying an adapter, or reaching a
round boundary does not create a new clock. The ledger keeps all previous usage;
first/second cumulative limits are installation usage plus 4/8 hours. The absolute
clock and allocation admission remain independent caps. Use `policy-show TASK`
for the current immutable clocks and mutable transition state. `policy-tick TASK`
is also available for an owning host; it uses the same lock and logic as service.

## Host deadline adapter

The trusted host adapter synchronizes **all** node/allocation/model guards. It
receives JSON on stdin, has a 10-second call bound and must be idempotent by
`transition_id`. Launching a remote allocation, if necessary, must return pending
rather than blocking the coordinator. Do not install an adapter under the mutable
candidate checkout. The manager records `pending_transition` before invocation;
the host should read its authoritative manager state and require exact equality,
the expected task, original start and phase `starting` or `extending`.

Request schema `research-policy-deadline-v1` contains:

- `task`, `transition_id`, `started_at`, `initial_end`, `maximum_end`;
- `deadline_at`: original start + 4h, or + 8h on the sole extension;
- `extension_used`: false for initial admission, true for extension;
- cumulative `wall_budget_seconds` and `gpu_budget_seconds` (never reset usage).

Return the same `schema`, `task`, `transition_id` and `maximum_end`, plus:

- `status`: `pending`, or `ready` only after guard synchronization;
- `deadline_at`: admitted manager deadline, no later than the requested deadline;
- `allocation_deadline_at`: admitted allocation/node guard deadline, at least the
  manager deadline and no later than original start + 8h;
- `allocation_gpu_seconds_used`: cumulative charged allocation usage across all
  attempts, including setup/idle/draining; it must never decrease;
- `allocation_gpu_seconds_limit`: at most 28,800 card-seconds for this lane.

The ready receipt must fit `used + (allocation_deadline_at - now) <= limit`.
The host is responsible for authoritative platform accounting and enforcing the
receipt; the framework cannot derive platform billing from local GPU processes.
Guard synchronization failures or mismatched receipts leave the lane paused.

Reserve a bounded handoff margin: for example manager deadline = requested end
minus 120 seconds and node guard deadline = requested end minus 30 seconds. The
policy starts draining at the **admitted manager deadline**, so it can authorize
extension before the initial node guard releases resources. An extension is never
`now + 4h`; it retains the original +8h ceiling. The host must retain its absolute
outer watchdog even if the coordinator fails. The adapter does not restart an
already running round: all previous research jobs must finish before configure
and real model relaunch.

Policy phases `starting`, `initial`, `extending`, `extended`, `stopping` are not
allocation-release signals. Terminal phases are `succeeded`, `exhausted`, and
`suspended`. A blocked host (including exit 78 or three empty rounds) becomes
suspended. A manual pause, resume or configure suspends automatic policy control;
the policy never undoes a user's stop. An operator can inspect the original state
and take explicit action, but cannot reinstall the policy to restart its clock.

## Optional independent delivery task

When a candidate merits delivery, register a separate native task and bind it:

```sh
kernel-envolve manage --state /host/state policy-bind-delivery lane \
  --delivery-task lane-delivery --source-root core
```

Both tasks retain their own workspaces, evaluator contracts, jobs and best. Binding
requires the same frozen baseline revision, an owned native candidate root whose
file contents match the research source, all operator/numerical/integration scopes,
and independent confirmation configuration. No research result becomes a native
best. Full native evaluation, independent confirmation, promotion and `sync-pr`
use the existing official paths. Native delivery resource scheduling is explicit;
this policy does not allocate extra GPUs or transfer research budget to it.

Early success requires the **current** verified native best; deep full and
confirmation receipt/source audit; the minimum gain in both full and confirmation;
a fresh same-source PR binding; and research source files equal to the frozen
candidate. A scoped summary, external `passed` validation, successful command,
quick run or PR alone never qualifies. The policy pauses and drains research, then
rechecks the proof before recording success. Changed sources/evidence during drain
suspend completion. It stores only a delivery reference in policy metadata; it
never writes a research `best` or merges qualification between tasks.
