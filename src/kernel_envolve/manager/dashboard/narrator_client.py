"""Tool-free Responses adapter. Credentials stay in the host environment."""

import json
import os
import sys
import urllib.request
from urllib.parse import urlsplit


def response_format(tasks):
    """Constrain shape and references at generation; semantic checks remain local."""

    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def fixed(value):
        return {"type": "string", "enum": [value]}

    properties = {}
    for task in tasks:
        claim = obj(
            {
                "text": {
                    "type": "string",
                    "description": "1–2 complete short sentences; aim for 50–80 Chinese characters, not the maximum.",
                },
                "sources": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string", "enum": list(task["sources"])},
                },
            }
        )
        optional = {"anyOf": [claim, {"type": "null"}]}
        fields = {k: optional for k in ("direction", "finding", "failure", "next")}
        if task.get("row_facts"):
            fields = {k: {"type": "null"} for k in fields}
            fields["finding"] = obj(
                {
                    **claim["properties"],
                    "text": {
                        "type": "string",
                        "description": "1–2 complete short sentences; aim for 60–90 Chinese characters, not the maximum.",
                    },
                }
            )
            fields["row"] = obj(
                {
                    "revision": fixed(task["row_facts"]["revision"]),
                    "benefit": fixed("system:benefit"),
                    "optimization_pr": fixed("system:optimization_pr"),
                    "current_progress": fixed("finding"),
                    "latest_result": obj(
                        {"ref": fixed("system:latest_result"), **claim["properties"]}
                    ),
                }
            )
        properties[task["task"]] = {"anyOf": [obj(fields), {"type": "null"}]}
    return {
        "type": "json_schema",
        "name": "research_rows",
        "strict": True,
        "schema": obj(properties),
    }


def respond(request):
    endpoint = os.environ["NARRATOR_RESPONSES_URL"]
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
    ):
        raise ValueError("Narrator endpoint requires HTTPS (or loopback for local inference)")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Do not put narrator credentials in the endpoint URL")
    style = os.environ.get("NARRATOR_API_STYLE", "responses")
    if style not in ("responses", "chat_completions"):
        raise ValueError("Unsupported narrator API style")
    output_mode = os.environ.get("NARRATOR_OUTPUT_MODE", "json_schema")
    if output_mode not in ("json_schema", "prompt_json"):
        raise ValueError("Unsupported narrator output mode")
    key = os.environ["NARRATOR_API_KEY"]
    body = {
        "model": os.environ["NARRATOR_MODEL"],
        "instructions": request["instructions"],
        "input": "请根据以下研究材料返回 JSON 摘要：\n"
        + json.dumps(request["tasks"], ensure_ascii=False),
        "tools": [],
        "store": False,
        "stream": False,
        "max_output_tokens": 8192,
        "text": {"format": response_format(request["tasks"])},
    }
    if effort := os.environ.get("NARRATOR_REASONING_EFFORT"):
        if effort not in ("minimal", "low", "medium", "high"):
            raise ValueError("Unsupported narrator reasoning effort")
        body["reasoning"] = {"effort": effort}
    if style == "chat_completions":
        schema = body["text"]["format"]
        body = {
            "model": body["model"],
            "tools": [],
            "store": False,
            "stream": False,
            "max_tokens": body["max_output_tokens"],
            "messages": [
                {"role": "system", "content": body["instructions"]},
                {"role": "user", "content": body["input"]},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {k: v for k, v in schema.items() if k != "type"},
            },
        }
        if effort:
            body["reasoning_effort"] = effort
    if output_mode == "prompt_json":
        # Explicit compatibility choice, never an automatic fallback/retry.
        # The service still validates JSON shape, source references and facts.
        instruction = "\nReturn only JSON matching this schema, without Markdown: " + json.dumps(
            response_format(request["tasks"])["schema"], ensure_ascii=False
        )
        if style == "chat_completions":
            body.pop("response_format")
            body["messages"][0]["content"] += instruction
        else:
            body.pop("text")
            body["instructions"] += instruction
    req = urllib.request.Request(
        endpoint,
        json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    timeout = int(os.environ.get("NARRATOR_TIMEOUT_SECONDS", "35"))
    if not 1 <= timeout <= 300:
        raise ValueError("Narrator timeout must be between 1 and 300 seconds")
    with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
        raw = response.read(262145)
    if len(raw) > 262144:
        raise ValueError("Narrator response too large")
    value = json.loads(raw)
    if style == "chat_completions":
        choices = value.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Narrator response incomplete")
        message = choices[0].get("message", {})
        if message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
            raise ValueError("Narrator must only return text")
        if not isinstance(message.get("content"), str):
            raise ValueError("Narrator did not return text")
        return json.loads(message["content"])
    if value.get("status") != "completed":
        raise ValueError("Narrator response incomplete")
    chunks = []
    for item in value["output"]:
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message":
            raise ValueError("Narrator must only return text")
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                raise ValueError("Narrator did not return text")
            chunks.append(part["text"])
    return json.loads("".join(chunks))


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(524289)
        if len(raw) > 524288:
            raise ValueError("Narrator input too large")
        print(json.dumps(respond(json.loads(raw)), ensure_ascii=False))
    except Exception as exc:
        # Never echo endpoint responses, request content or credentials.
        import urllib.error

        code = f":{exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
        print(type(exc).__name__ + code, file=sys.stderr)
        sys.exit(1)
