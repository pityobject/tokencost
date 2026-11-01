"""
_mock_server.py —— 假的 llama-server，只用于"没有模型时先验证测量脚本本身对不对"。

它不做任何推理，只是按 OpenAI 流式格式吐 48 个 token，每个间隔 20ms。
作用：在下载模型、装 CUDA 之前，就能把 tc_loadgen.py 跑通并检查指标算得对不对。
正式实验不需要这个文件。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" _mock_server.py          # 另开一个窗口
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_loadgen.py --concurrency 4 --requests 12
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

N_TOKENS = 48
PER_TOKEN_S = 0.02


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/health", "/health/"):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        max_tokens = int(req.get("max_tokens", N_TOKENS))
        n = min(max_tokens, N_TOKENS)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj):
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        # 首块：只带 role，不含 content（真实服务端就是这样，不要把它算成首字）
        send({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        time.sleep(0.05)   # 假装 prefill 用了 50ms

        for i in range(n):
            time.sleep(PER_TOKEN_S)
            send({"choices": [{"index": 0, "delta": {"content": "字"}, "finish_reason": None}]})

        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 256, "completion_tokens": n, "total_tokens": 256 + n}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = 8090
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"假服务已启动：http://127.0.0.1:{port}")
    print("预期：TTFT ≈ 50ms，TPOT ≈ 20ms，吞吐 ≈ 并发 × 50 token/s")
    srv.serve_forever()
