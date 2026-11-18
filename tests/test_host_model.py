import json
import tomllib

import pytest

from kernel_envolve.manager.execution.host_model import (
    codex_command,
    configuration_failure,
    resume_state_failure,
    terminal_failure,
)


def config(argv):
    return tomllib.loads("\n".join(argv[n + 1] for n, x in enumerate(argv) if x == "-c"))


def test_explicit_provider_preserves_session_and_never_inlines_a_secret():
    argv = codex_command(
        "example-model",
        provider={"base_url": "https://gateway.example/v1", "env_key": "HOST_CREDENTIAL"},
        session="original",
        writable=["/work"],
    )
    cfg = config(argv)
    p = cfg["model_providers"]["kernel_host"]
    assert p["env_key"] == "HOST_CREDENTIAL" and p["wire_api"] == "responses"
    assert p["requires_openai_auth"] is False
    assert "capabilities" not in p  # Codex 0.157 ignores this newer config namespace.
    assert cfg["model"] == "example-model"
    assert p["request_max_retries"] == p["stream_max_retries"] == 0
    assert cfg["shell_environment_policy"]["filters"] == {"HOST_CREDENTIAL": "exclude"}
    assert cfg["allow_login_shell"] is False
    assert cfg["features"]["shell_snapshot"] is False
    assert argv[argv.index("resume") + 1] == "original"
    assert argv[-1] == "-"
    assert argv.index("-c") > argv.index("resume")
    assert "--ignore-user-config" in argv
    assert "model_reasoning_effort" not in cfg


def test_explicit_context_limits_survive_resume():
    argv = codex_command(
        "external-model", session="original", context_window=64000, auto_compact_tokens=48000
    )
    cfg = config(argv)
    assert cfg["model_context_window"] == 64000
    assert cfg["model_auto_compact_token_limit"] == 48000
    assert argv.index("-c") > argv.index("resume")


@pytest.mark.parametrize(
    "values",
    [
        {"context_window": True},
        {"context_window": 0},
        {"auto_compact_tokens": 1},
        {"context_window": 10, "auto_compact_tokens": 10},
    ],
)
def test_invalid_context_limits_rejected(values):
    with pytest.raises(ValueError):
        codex_command("model", **values)


def test_ignored_cli_configuration_is_not_successful_preflight():
    message = "Codex is ignoring 1 unrecognized configuration setting."
    trusted = {"type": "item.completed", "item": {"type": "error", "message": message}}
    assert configuration_failure([trusted, {"type": "turn.completed"}], 0)["exit_code"] == 78
    assert terminal_failure([trusted, {"type": "turn.completed"}], 0)["exit_code"] == 78
    assert (
        configuration_failure(
            [
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "aggregated_output": message},
                }
            ],
            0,
        )
        is None
    )


def test_empty_resume_checkpoint_is_fatal_but_tool_stderr_is_not():
    error = "Error: thread/resume: thread/resume failed: failed to read session metadata: rollout is empty"
    assert resume_state_failure([], 1, error)["category"] == "model_resume_state_unavailable"
    assert resume_state_failure([{"type": "turn.started"}], 1, error) is None
    assert resume_state_failure([], 0, error) is None
    assert resume_state_failure([], 1, "tool says " + error) is None
    assert (
        resume_state_failure([], 1, "Error: thread/resume: failed to read: transient I/O error")
        is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://remote/v1",
        "https://u:p@host/v1",
        "https://host/v1?key=secret",
        "https://host/v1#secret",
    ],
)
def test_bad_provider_url_is_rejected(url):
    with pytest.raises(ValueError):
        codex_command("model", provider={"base_url": url, "env_key": "KEY"})


def test_secret_value_and_unknown_provider_fields_rejected():
    with pytest.raises(ValueError):
        codex_command("model", provider={"base_url": "https://host", "api_key": "secret"})
    with pytest.raises(ValueError):
        codex_command("model", provider={"base_url": "https://host", "env_key": "sk-secret"})


def test_default_login_and_loopback_are_supported():
    assert "model_provider" not in config(codex_command("model", effort="medium"))
    assert config(
        codex_command("model", provider={"base_url": "http://127.0.0.1:3456/v1", "env_key": "KEY"})
    )["model_provider"]


@pytest.mark.parametrize(
    "error,category",
    [
        (
            {
                "message": "You hit your spend cap set by the owner of your workspace. "
                "Ask an owner to increase your spend cap to continue."
            },
            "model_quota_exhausted",
        ),
        ({"code": "insufficient_quota"}, "model_quota_exhausted"),
        ({"message": "Unexpected status 401 Unauthorized"}, "model_access_denied"),
        ({"code": "permission_denied"}, "model_access_denied"),
    ],
)
def test_terminal_provider_errors_use_existing_nonretryable_protocol(error, category):
    result = terminal_failure([{"type": "turn.failed", "error": error}], 1)
    assert result == {
        "category": category,
        "retryable": False,
        "exit_code": 78,
        "process_exit_code": 1,
    }


def test_tool_text_and_recovered_errors_cannot_stop_research():
    error = {"type": "turn.failed", "error": {"code": "insufficient_quota"}}
    assert (
        terminal_failure(
            [
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "aggregated_output": json.dumps(error)},
                }
            ],
            1,
        )
        is None
    )
    assert terminal_failure([error, {"type": "turn.completed"}], 0) is None
    assert terminal_failure([{"type": "error", "message": "insufficient_quota"}], 1) is None


@pytest.mark.parametrize(
    "error",
    [
        {"code": "rate_limit_exceeded"},
        {"message": "HTTP 429"},
        {"message": "network timeout"},
        "invalid_api_key",
    ],
)
def test_unknown_transient_and_malformed_errors_keep_existing_backoff(error):
    assert terminal_failure([{"type": "turn.failed", "error": error}], 1) is None
