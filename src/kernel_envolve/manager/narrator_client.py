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
                    "minLength": 1,
                    "maxLength": 120,
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
                        "minLength": 1,
                        "maxLength": 160,
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
    req = urllib.request.Request(
        endpoint,
        json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    with urllib.request.build_opener(NoRedirect).open(req, timeout=35) as response:
        raw = response.read(262145)
    if len(raw) > 262144:
        raise ValueError("Narrator response too large")
    value = json.loads(raw)
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
