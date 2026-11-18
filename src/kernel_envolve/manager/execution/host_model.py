"""Small Codex host helpers; no model loop, scheduler, or credential storage."""

import json
import re
from urllib.parse import urlsplit


def codex_command(
    model,
    *,
    codex="codex",
    provider=None,
    effort=None,
    writable=(),
    session=None,
    context_window=None,
    auto_compact_tokens=None,
):
    """Select an explicit model/provider without changing the user's global config.

    The caller supplies credentials in the named environment variable and owns
    process lifetime, trace persistence, and the manager's existing continuation.
    """
    if not isinstance(model, str) or not model.strip() or model.startswith("-"):
        raise ValueError("An explicit model is required")
    cfg = {
        "model": model,
        "approval_policy": "never",
        "sandbox_workspace_write.network_access": True,
    }
    for key, value in (
        ("model_context_window", context_window),
        ("model_auto_compact_token_limit", auto_compact_tokens),
    ):
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError("Context limits must be positive integers")
            cfg[key] = value
    if auto_compact_tokens is not None and (
        context_window is None or auto_compact_tokens >= context_window
    ):
        raise ValueError("Compaction threshold must be below an explicit context window")
    if effort is not None:
        if effort not in ("minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning effort")
        cfg["model_reasoning_effort"] = effort
    if provider is not None:
        if set(provider) != {"base_url", "env_key"}:
            raise ValueError(
                "Provider requires only base_url and env_key; never inline credentials"
            )
        endpoint, env_key = provider["base_url"], provider["env_key"]
        if not isinstance(endpoint, str) or not isinstance(env_key, str):
            raise ValueError("Provider fields must be strings")
        parsed = urlsplit(endpoint)
        local = parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if (parsed.scheme != "https" and not local) or not parsed.hostname:
            raise ValueError("Provider requires HTTPS or a loopback endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Provider URL must not contain credentials or query parameters")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_key):
            raise ValueError("env_key must name an environment variable")
        cfg.update(
            {
                "model_provider": "kernel_host",
                "model_providers.kernel_host.name": "Kernel research provider",
                "model_providers.kernel_host.base_url": endpoint,
                "model_providers.kernel_host.env_key": env_key,
                "model_providers.kernel_host.requires_openai_auth": False,
                "model_providers.kernel_host.wire_api": "responses",
                "model_providers.kernel_host.supports_websockets": False,
                "model_providers.kernel_host.request_max_retries": 0,
                "model_providers.kernel_host.stream_max_retries": 0,
                "features.enable_request_compression": False,
                "features.unbounded_connection_retries": False,
                # Keep the provider credential out of agent-launched shell processes.
                "shell_environment_policy.filters." + env_key: "exclude",
                "shell_environment_policy.ignore_default_excludes": False,
                "allow_login_shell": False,
                "features.shell_snapshot": False,
                "web_search": "disabled",
            }
        )
    argv = [
        codex,
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "-s",
        "workspace-write",
        "-m",
        model,
    ]
    for path in writable:
        argv.extend(("--add-dir", str(path)))
    if session is not None:
        if not isinstance(session, str) or not session or session.startswith("-"):
            raise ValueError("Invalid resumed session")
        argv.extend(("resume", session))
    # Resume has its own config parser. Overrides before that subcommand can
    # leave resumed thread settings on the default provider, including compact.
    for key, value in cfg.items():
        argv.extend(("-c", key + "=" + json.dumps(value)))
    return argv + ["-"]


def configuration_failure(events, process_exit_code):
    """Ignored host settings are not proof of a correctly configured preflight.

    Only inspect CLI error items, never model prose or nested command output.
    Some CLI releases ignore newer provider capabilities even on a successful turn.
    """
    for event in events:
        item = event.get("item", {})
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "error"
            and str(item.get("message", "")).startswith("Codex is ignoring ")
            and "configuration setting" in str(item.get("message", ""))
        ):
            return {
                "category": "host_configuration_ignored",
                "retryable": False,
                "exit_code": 78,
                "process_exit_code": process_exit_code,
            }
    return None


def resume_state_failure(events, process_exit_code, stderr):
    """Trusted CLI startup failure: a thread ID is not a durable session checkpoint.

    Never inspect stderr after a model turn started, where tool output may appear.
    Call only for an attempt explicitly resuming the original session.
    """
    if not process_exit_code or any(e.get("type") == "turn.started" for e in events):
        return None
    if any(
        line.startswith("Error: thread/resume:")
        and "failed to read" in line
        and any(
            reason in line
            for reason in ("is empty", "empty session file", "No such file", "not found")
        )
        for line in stderr.splitlines()
    ):
        return {
            "category": "model_resume_state_unavailable",
            "retryable": False,
            "exit_code": 78,
            "process_exit_code": process_exit_code,
        }
    return None


def terminal_failure(events, process_exit_code):
    """Classify only trusted CLI terminal events, never tool output or model prose.

    Exit 78 is the existing manager protocol for non-retryable host failures.
    Keep the raw process exit separately; historical experiment progress remains
    valid even when the provider refuses the next model request.
    """
    if ignored := configuration_failure(events, process_exit_code):
        return ignored
    terminal = None
    for event in events:
        if event.get("type") in ("turn.failed", "turn.completed"):
            terminal = event
    if not terminal or terminal.get("type") != "turn.failed":
        return None
    error = terminal.get("error", {})
    if not isinstance(error, dict):
        return None
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", "")).lower()
    category = None
    if code in {
        "insufficient_quota",
        "spend_cap_exceeded",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "credit_balance_exhausted",
    } or any(
        text in message
        for text in (
            "you hit your spend cap set by the owner of your workspace",
            "exceeded your current quota",
            "insufficient_quota",
            "credit_balance_exhausted",
        )
    ):
        category = "model_quota_exhausted"
    elif code in {"invalid_api_key", "authentication_error", "permission_denied"} or any(
        text in message
        for text in (
            "incorrect api key provided",
            "invalid api key",
            "unexpected status 401",
            "unexpected status 403",
        )
    ):
        category = "model_access_denied"
    if category is None:
        return None  # Rate limits, network errors and unknown failures keep normal backoff.
    return {
        "category": category,
        "retryable": False,
        "exit_code": 78,
        "process_exit_code": process_exit_code,
    }
