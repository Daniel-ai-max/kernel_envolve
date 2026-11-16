"""Tool-free Responses adapter. Credentials stay in the host environment."""

import json
import os
import sys
import urllib.request
from urllib.parse import urlsplit


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
        "max_output_tokens": 4096,
        "text": {"format": {"type": "json_object"}},
    }
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
