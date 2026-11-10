"""Synthetic Codex transport/lifecycle probe. No real credentials, GPU or inference."""

import argparse
import hashlib
import http.server
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from direct_codex_command import command

IMAGE = "sha256:9a01c6b6c843cb744230328611ddc741d1a8633660fd5a5b51f331e1cf1743bf"
ROOT = Path(__file__).resolve().parent


def probe(folder, timeout_trial):
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    output = folder / "evidence"
    prompt = folder / "prompt.txt"
    prompt.write_text("direct-provider-static-probe: execute the fixed tool sequence.")
    key, routing = secrets.token_hex(16), secrets.token_hex(16)
    requests = []

    class Fake(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                self.respond()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def respond(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            auth = self.headers.get("Authorization") == "Bearer " + key
            header = self.headers.get("X-Static-Routing") == routing
            index = len(requests)
            requests.append({"model": req.get("model"), "auth_ok": auth, "header_ok": header})
            assert auth and header, "Native provider did not load credentials from env names"
            assert req.get("model") == "gpt-6-astra"
            if index == 0:
                assert "direct-provider-static-probe" in json.dumps(req)
            if timeout_trial and index > 0:
                time.sleep(10)
                self.send_error(504)
                return
            if index < 3:
                if index == 0:
                    script = "from pathlib import Path; Path('submission.py').write_text('def run(x): return x\\n')"
                    if timeout_trial:
                        script += (
                            "; import subprocess; p=subprocess.Popen(['python3','-c',"
                            "'import os,time; os.setsid(); time.sleep(60)'],"
                            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                            "Path('detached.pid').write_text(str(p.pid))"
                        )
                elif index == 1:
                    script = "from pathlib import Path; assert Path('submission.py').is_file()"
                else:
                    script = "print('third-tool-pass')"
                import shlex

                code = (
                    "text(await tools.exec_command("
                    + json.dumps({"cmd": "python3 -c " + shlex.quote(script)})
                    + "));"
                )
                item = {
                    "type": "custom_tool_call",
                    "id": f"call_{index}",
                    "call_id": f"call_{index}",
                    "name": "exec",
                    "input": code,
                    "status": "completed",
                }
                added = dict(item, input="", status="in_progress")
                middle = [
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": item["id"],
                        "output_index": 0,
                        "delta": code,
                    },
                    {
                        "type": "response.custom_tool_call_input.done",
                        "item_id": item["id"],
                        "output_index": 0,
                        "input": code,
                    },
                ]
            else:
                item = {
                    "type": "message",
                    "id": "done",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Static check completed.",
                            "annotations": [],
                        }
                    ],
                }
                added = dict(item, content=[], status="in_progress")
                middle = [
                    {
                        "type": "response.content_part.added",
                        "item_id": "done",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                    {
                        "type": "response.output_text.delta",
                        "item_id": "done",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "Static check completed.",
                    },
                ]
            response = {
                "id": f"response_{index}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6-astra",
                "output": [item],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            events = [
                {
                    "type": "response.created",
                    "response": dict(response, status="in_progress", output=[]),
                },
                {"type": "response.output_item.added", "item": added, "output_index": 0},
                *middle,
                {"type": "response.output_item.done", "item": item, "output_index": 0},
                {"type": "response.completed", "response": response},
            ]
            data = "".join(
                "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
                for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data) + 1))
            self.end_headers()
            self.wfile.write(data[:-1])
            self.wfile.flush()
            time.sleep(0.05)
            self.wfile.write(data[-1:])
            self.wfile.flush()
            time.sleep(5)  # Completion must not depend on HTTP EOF.

    fake = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    try:
        cmd = command(
            f"http://host.lima.internal:{fake.server_address[1]}/v1",
            "KE_STATIC_KEY",
            {"X-Static-Routing": "KE_STATIC_ROUTING"},
        )
        limit = 4 if timeout_trial else 30
        invocation = [
            sys.executable,
            str(ROOT / "run_bounded_trial.py"),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
            "--image",
            IMAGE,
            "--seconds",
            str(limit),
            "--network",
            "bridge",
            "--stdin-file",
            str(prompt),
            "--env-name",
            "KE_STATIC_KEY",
            "--env-name",
            "KE_STATIC_ROUTING",
            "--",
            *cmd,
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            invocation,
            env=os.environ | {"KE_STATIC_KEY": key, "KE_STATIC_ROUTING": routing},
            capture_output=True,
            text=True,
            timeout=limit + 40,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        receipt = json.loads((output / "trial.json").read_text())
        assert receipt["container_removed"] and receipt["timed_out"] == timeout_trial
        assert (
            receipt["submission"]["sha256"]
            == hashlib.sha256((output / "submission.py").read_bytes()).hexdigest()
        )
        if timeout_trial:
            assert (workspace / "detached.pid").is_file()
            assert len(requests) == 2
        else:
            assert len(requests) == 4
            assert "turn.completed" in (output / "stdout.log").read_text()
            assert "third-tool-pass" in (output / "stdout.log").read_text()
        missing = subprocess.run(
            ["docker", "container", "inspect", (output / "container-id").read_text().strip()],
            capture_output=True,
            text=True,
        )
        assert missing.returncode != 0 and "no such container" in missing.stderr.lower(), (
            missing.stdout,
            missing.stderr,
        )
        compile((output / "submission.py").read_text(), "submission.py", "exec")
        (workspace / "submission.py").write_text("changed after freeze\n")
        assert (
            receipt["submission"]["sha256"]
            == hashlib.sha256((output / "submission.py").read_bytes()).hexdigest()
        )
        for path in output.iterdir():
            if path.is_file():
                assert (
                    key.encode() not in path.read_bytes()
                    and routing.encode() not in path.read_bytes()
                )
        return {
            "timeout_trial": timeout_trial,
            "synthetic_requests": requests,
            "inference_calls": 0,
            "wall_seconds": time.perf_counter() - started,
            "native_env_auth": True,
            "stdin_delivered": True,
            "container_removed": True,
            "frozen_source_unchanged": True,
            "receipt_has_no_credential_values": True,
            "trial": receipt,
        }
    finally:
        fake.shutdown()
        fake.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    report = {
        "model_inference_calls": 0,
        "gpu_count": 0,
        "image": IMAGE,
        "cli_version": "0.157.0",
        "model_http_proxy": False,
        "official_configuration_reference": "https://learn.chatgpt.com/docs/config-file/config-reference",
        "cases": [probe(args.directory / "normal", False), probe(args.directory / "timeout", True)],
    }
    report["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["run_bounded_trial.py", "direct_codex_command.py", "probe_direct_provider.py"]
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model_inference_calls": 0, "passed_cases": len(report["cases"])}))
