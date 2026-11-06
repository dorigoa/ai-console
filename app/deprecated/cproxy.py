#!/usr/bin/env python3
"""Proxy di logging tra Claude Code e llama-server."""

import ssl, certifi
SSLCTX = ssl.create_default_context(cafile=certifi.where())

import http.server, socketserver, urllib.request, urllib.error
import json, os, sys, time, threading

UPSTREAM = os.environ.get("UPSTREAM", "https://ai.exocomet-boga.ts.net")
DUMPDIR  = os.environ.get("DUMPDIR", "/tmp/cc-dump")
PORT     = int(os.environ.get("PORT", "8099"))

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade", "host",
       "content-length", "accept-encoding"}

os.makedirs(DUMPDIR, exist_ok=True)
_lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _dump(self, body):
        ts = time.strftime("%H%M%S")
        name = self.path.strip("/").replace("/", "_") or "root"
        path = os.path.join(DUMPDIR, f"{ts}-{name}.json")
        with _lock, open(path, "wb") as f:
            try:
                f.write(json.dumps(json.loads(body), indent=2,
                                   ensure_ascii=False).encode())
            except Exception:
                f.write(body)
        print(f"[dump] {path} ({len(body)} B)", file=sys.stderr)

    def _proxy(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            if body:
                self._dump(body)

            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            hdrs["Accept-Encoding"] = "identity"
            req = urllib.request.Request(UPSTREAM + self.path, data=body or None,
                                         headers=hdrs, method=self.command)

            #with urllib.request.urlopen(req, timeout=900) as resp:
            with urllib.request.urlopen(req, timeout=900, context=SSLCTX) as resp:
                print(f"[upstream] {resp.status} {resp.getheader('Content-Type')}", file=sys.stderr)
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        except urllib.error.HTTPError as e:
            print(f"[upstream-error] {e.code}: {data[:800]!r}", file=sys.stderr)
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            print(f"[proxy-error] {type(e).__name__}: {e}", file=sys.stderr)
            msg = json.dumps({"error": str(e)}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception:
                pass

    do_GET = do_POST = do_PUT = do_DELETE = _proxy

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as srv:
        print(f"proxy :{PORT} → {UPSTREAM}  (dump in {DUMPDIR})", file=sys.stderr)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
