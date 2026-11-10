"""Build a native Codex provider command; credentials are supplied through env names.

No HTTP proxy, SSE parser, model requests or resource allocation lives here.
Configuration reference: https://learn.chatgpt.com/docs/config-file/config-reference
"""

import json


def command(base_url, env_key, env_headers=None):
    cfg = {
        "model_provider": "comparison",
        "model_reasoning_effort": "medium",
        "model_catalog_json": "/opt/pilot/model-catalog.json",
        "model_providers.comparison.name": "Direct comparison provider",
        "model_providers.comparison.base_url": base_url,
        "model_providers.comparison.env_key": env_key,
        "model_providers.comparison.wire_api": "responses",
        "model_providers.comparison.requires_openai_auth": False,
        "model_providers.comparison.supports_websockets": False,
        "model_providers.comparison.request_max_retries": 0,
        "model_providers.comparison.stream_max_retries": 0,
        "web_search": "disabled",
    }
    for name in (
        "multi_agent",
        "multi_agent_v2",
        "apps",
        "plugins",
        "remote_plugin",
        "hooks",
        "enable_request_compression",
        "unbounded_connection_retries",
        "memories",
        "browser_use",
        "computer_use",
        "image_generation",
    ):
        cfg["features." + name] = False
    result = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        "gpt-6-astra",
    ]
    for key, value in cfg.items():
        result += ["-c", key + "=" + json.dumps(value)]
    if env_headers:
        # A TOML inline table uses '=', not JSON's ':'. Values are environment names.
        table = ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in env_headers.items())
        result += ["-c", "model_providers.comparison.env_http_headers={" + table + "}"]
    return result + ["-"]
