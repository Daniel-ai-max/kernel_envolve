import json
import tomllib

import pytest

from kernel_envolve.manager.execution.host_model import codex_command, terminal_failure


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
    assert p["request_max_retries"] == p["stream_max_retries"] == 0
    assert cfg["shell_environment_policy"]["filters"] == {"HOST_CREDENTIAL": "exclude"}
    assert cfg["allow_login_shell"] is False
    assert cfg["features"]["shell_snapshot"] is False
    assert argv[-3:] == ["resume", "original", "-"]
    assert "--ignore-user-config" in argv
    assert "model_reasoning_effort" not in cfg


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
