import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adapterguard.runtime_openai import _request_completion


def test_request_completion_uses_real_http_and_bearer_auth():
    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("Authorization")
            captured["payload"] = payload

            body = json.dumps(
                {
                    "model": "served/model",
                    "choices": [
                        {
                            "text": " alpha beta",
                            "logprobs": {"tokens": [" alpha", " beta"]},
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        response = _request_completion(
            endpoint=endpoint,
            model="requested/model",
            prompt="hello runtime",
            api_key="secret-token",
            max_tokens=2,
            timeout=2.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert captured["path"] == "/v1/completions"
    assert captured["authorization"] == "Bearer secret-token"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload == {
        "model": "requested/model",
        "prompt": "hello runtime",
        "temperature": 0,
        "max_tokens": 2,
        "logprobs": 5,
    }
    assert response.text == " alpha beta"
    assert response.tokens == [" alpha", " beta"]
    assert response.served_model == "served/model"
    assert response.latency_ms >= 0
