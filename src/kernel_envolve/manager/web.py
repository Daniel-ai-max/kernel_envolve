"""Loopback-only UI. No command execution API, directory serving or cross-origin writes."""

import json
import secrets
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, unquote, urlsplit

from .service import clear_resource, resource_sessions
from .store import Store
from .views import detail, snapshot


def server(root, port=8787):
    initial = Store(root)
    initial.close()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, content, kind="application/json; charset=utf-8"):
            body = (
                json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
                if kind.startswith("application/json")
                else content
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = self.headers.get("Host", "")
            if host not in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            ):
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin == f"http://{host}"

        def do_GET(self):
            if not self.allowed():
                self.send(403, {"error": "Loopback origin required"})
                return
            path = unquote(urlsplit(self.path).path)
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/sync.js": ("sync.js", "text/javascript; charset=utf-8"),
                "/summary.js": ("summary.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path in assets:
                name, kind = assets[path]
                self.send(
                    200, files("kernel_envolve.manager").joinpath("static", name).read_bytes(), kind
                )
                return
            store = Store(root, readonly=True)
            try:
                if path == "/api/state":
                    selected = parse_qs(urlsplit(self.path).query).get("task", [None])[0]
                    self.send(200, snapshot(store, selected) | {"token": token})
                elif path.startswith("/api/tasks/"):
                    with store.transaction(readonly=True):
                        data = detail(store, path.removeprefix("/api/tasks/"))
                    self.send(200, data)
                else:
                    self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, OSError, TypeError, sqlite3.Error) as exc:
                self.send(400, {"error": str(exc)})
            finally:
                store.close()

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(
                self.headers.get("X-Manager-Token", ""), token
            ):
                self.send(403, {"error": "Same-origin control token required"})
                return
            store = Store(root)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < length <= 16384
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Expected a JSON object up to 16 KiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if self.path == "/api/control":
                    store.control(body["task"], body["action"], body.get("changes"))
                elif self.path == "/api/resource":
                    if body.get("clear"):
                        clear_resource(store, body["id"])
                    if "sessions" in body:
                        resource_sessions(store, body["id"], body["sessions"])
                else:
                    self.send(404, {"error": "Not found"})
                    return
                self.send(200, {"ok": True})
            except (ValueError, KeyError, OSError, TypeError, sqlite3.Error) as exc:
                self.send(400, {"error": str(exc)})
            finally:
                store.close()

    result = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    result.daemon_threads = True
    return result


def serve(root, port):
    with server(root, port) as http:
        print(f"Kernel Envolve: http://127.0.0.1:{http.server_port}", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
