# -*- coding: utf-8 -*-
"""抓一次原始响应 JSON，确认 /v1/chat/completions 的结构。"""
import json
import sys

import httpx

import tc_config as cfg
import tc_server as srv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for key in [("7B", "q4_k_m"), ("1.5B", "q4_k_m")]:
    print("=" * 78)
    print(f"{key[0]} {key[1]}")
    print("=" * 78)
    proc = None
    try:
        proc, _ = srv.start(key, 2048, 1, tag=f"apidiag_{key[0]}")
        r = httpx.post(f"{cfg.BASE_URL}/v1/chat/completions", json={
            "model": "local",
            "messages": [{"role": "user", "content": "计算 723 + 450，只输出最后的数字，不要任何解释。"}],
            "max_tokens": 48,
            "temperature": 0.0,
            "stream": False,
        }, timeout=120)
        j = r.json()
        print(json.dumps(j, ensure_ascii=False, indent=2)[:2000])
        print("\n--- 关键字段类型 ---")
        msg = j["choices"][0]["message"]
        for k, v in msg.items():
            print(f"  message.{k}: type={type(v).__name__}  repr={repr(v)[:80]}")
    except Exception as e:
        print("失败:", type(e).__name__, e)
    finally:
        if proc is not None:
            srv.stop(proc)
    print()
