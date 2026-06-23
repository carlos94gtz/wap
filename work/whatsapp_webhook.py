#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "mi_token_de_verificacion")
PORT = int(os.getenv("PORT", "8000"))


class WhatsAppWebhookHandler(BaseHTTPRequestHandler):
    def _send_text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        mode = params.get("hub.mode", [""])[0]
        token = params.get("hub.verify_token", [""])[0]
        challenge = params.get("hub.challenge", [""])[0]

        if mode == "subscribe" and token == VERIFY_TOKEN:
            self._send_text(200, challenge)
            return

        self._send_text(403, "Forbidden")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {"raw": raw.decode("utf-8", errors="replace")}

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        self._send_text(200, "ok")

    def log_message(self, format: str, *args) -> None:
        # Keep the console output focused on webhook payloads.
        return


def main() -> int:
    server = HTTPServer(("0.0.0.0", PORT), WhatsAppWebhookHandler)
    print(f"Listening on http://localhost:{PORT}")
    print(f"Verify token: {VERIFY_TOKEN}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
