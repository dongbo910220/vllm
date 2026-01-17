# SPDX-License-Identifier: Apache-2.0

"""
PD Proxy (P2P NCCL) - Network jitter simulator

目标：
  - 收到请求后，立即把请求转发给 Decode
  - Prefill 请求在后台发送，但会先 sleep 一段“随机抖动 + 小概率尖峰”的延迟

用途：
  - 模拟跨机部署时网络/排队抖动，让 Decode 更稳定地落入 “KV 未到” 窗口，
    从而凸显 PUT/PUT_ASYNC 异步不阻塞调度的收益。

注意：
  - 这个脚本的延迟并不直接作用于 NCCL 传输本身，而是作用于
    “发起 Prefill 请求”。但它能等价地扩大
    “Decode 先到 / Prefill(KV) 后到” 的时间差窗口。
"""

# ========== PyCharm调试器兼容性修复 ==========
import os
import sys

if "pydevd" in sys.modules or any("pydev" in arg for arg in sys.argv):
    os.environ["PYDEVD_USE_CYTHON"] = "NO"
    os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"
    print("[proxy] PyCharm debugger detected: disable asyncio patching")
# ==============================================

import argparse
import asyncio
import inspect
import random
import uuid
from contextlib import suppress
from typing import Any

import httpx
import uvicorn.server as _uvicorn_server
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.utils.network_utils import get_ip

# PyCharm 会 monkey-patch asyncio.run，不带 loop_factory 参数，
# 而 uvicorn 的 Server.run 会传 loop_factory，导致 TypeError。
_run_sig = inspect.signature(asyncio.run)
if "loop_factory" not in _run_sig.parameters:

    def _asyncio_run_compat(main, *, debug=None, loop_factory=None):
        return asyncio.run(main, debug=debug)

    _uvicorn_server.asyncio_run = _asyncio_run_compat  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD proxy with prefill jitter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    parser.add_argument(
        "--kv-ip",
        default="auto",
        help="IP embedded into request_id for P2pNcclConnector (prefill & decode).",
    )
    parser.add_argument(
        "--prefill-kv-port",
        type=int,
        default=14579,
        help="ZMQ port bound by prefill P2pNcclConnector.",
    )
    parser.add_argument(
        "--decode-kv-port",
        type=int,
        default=14680,
        help="ZMQ port bound by decode P2pNcclConnector.",
    )

    # Jitter model: small uniform + rare spike.
    parser.add_argument(
        "--prefill-jitter-max-ms",
        type=float,
        default=200.0,
        help="Uniform jitter in [0, max_ms] added before sending prefill.",
    )
    parser.add_argument(
        "--prefill-spike-prob",
        type=float,
        default=0.02,
        help="Probability of adding an extra spike delay.",
    )
    parser.add_argument(
        "--prefill-spike-min-ms",
        type=float,
        default=1000.0,
        help="Spike delay lower bound (ms).",
    )
    parser.add_argument(
        "--prefill-spike-max-ms",
        type=float,
        default=2000.0,
        help="Spike delay upper bound (ms).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for deterministic jitter (optional).",
    )
    parser.add_argument(
        "--log-delay",
        action="store_true",
        help="Log the sampled prefill delay for each request.",
    )

    parser.add_argument("--request-timeout", type=float, default=36000.0)
    return parser.parse_args()


args = parse_args()

KV_IP = args.kv_ip if args.kv_ip != "auto" else get_ip()
PREFILL_KV_ADDR = f"{KV_IP}:{args.prefill_kv_port}"
DECODE_KV_ADDR = f"{KV_IP}:{args.decode_kv_port}"

_rng = random.Random(args.seed)

app = FastAPI()
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup_http_client() -> None:  # pragma: no cover
    global _http_client
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(
        max_connections=2048,
        max_keepalive_connections=2048,
    )
    _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)


@app.on_event("shutdown")
async def _shutdown_http_client() -> None:  # pragma: no cover
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items()}


def _prepare_prefill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = _clone_payload(payload)
    data["max_tokens"] = 1
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 1
    data["stream"] = False
    data.pop("stream_options", None)
    data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return data


def _build_headers(request: Request, request_id: str) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    auth_header = request.headers.get("authorization")
    if auth_header:
        headers["Authorization"] = auth_header
    return headers


def _make_request_id() -> str:
    suffix = uuid.uuid4().hex
    return (
        f"cmpl-___prefill_addr_{PREFILL_KV_ADDR}"
        f"___decode_addr_{DECODE_KV_ADDR}_{suffix}"
    )


def _sample_prefill_delay_s() -> float:
    jitter = 0.0
    if args.prefill_jitter_max_ms > 0:
        jitter += _rng.uniform(0.0, args.prefill_jitter_max_ms) / 1000.0
    if args.prefill_spike_prob > 0 and _rng.random() < args.prefill_spike_prob:
        lo = max(0.0, args.prefill_spike_min_ms)
        hi = max(lo, args.prefill_spike_max_ms)
        jitter += _rng.uniform(lo, hi) / 1000.0
    return jitter


async def _forward_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    return await client.post(url, json=payload, headers=headers)


async def _handle(api_path: str, request: Request):
    try:
        original_request = await request.json()
    except Exception as exc:  # pragma: no cover
        return JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)

    request_id = _make_request_id()
    headers = _build_headers(request, request_id)

    client = _http_client
    if client is None:  # pragma: no cover
        return JSONResponse({"error": "HTTP client not initialized"}, status_code=500)

    prefill_payload = _prepare_prefill_payload(original_request)
    decode_payload = _clone_payload(original_request)

    async def _run_prefill() -> httpx.Response:
        delay_s = _sample_prefill_delay_s()
        if args.log_delay:
            print(
                f"[proxy] req_id={request_id} prefill_delay={delay_s:.3f}s",
                flush=True,
            )
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return await _forward_request(
            client, f"{args.prefill_url}{api_path}", prefill_payload, headers
        )

    prefill_task = asyncio.create_task(_run_prefill())

    decode_cm = client.stream(
        "POST",
        f"{args.decode_url}{api_path}",
        json=decode_payload,
        headers=headers,
    )
    decode_resp = await decode_cm.__aenter__()
    if decode_resp.status_code != 200:
        detail = await decode_resp.aread()
        await decode_cm.__aexit__(None, None, None)
        return JSONResponse(
            {
                "error": "Decode request failed",
                "status_code": decode_resp.status_code,
                "detail": detail.decode(errors="ignore"),
            },
            status_code=502,
        )

    media_type = decode_resp.headers.get("content-type", "application/json")

    async def _iter_bytes():
        try:
            async for chunk in decode_resp.aiter_bytes():
                yield chunk
        finally:
            await decode_cm.__aexit__(None, None, None)
            try:
                prefill_resp = await prefill_task
                if prefill_resp.status_code != 200:
                    print(
                        f"[proxy] prefill failed: {prefill_resp.status_code} "
                        f"{prefill_resp.text}",
                        flush=True,
                    )
            except Exception as exc:  # pragma: no cover
                print(f"[proxy] prefill exception: {exc}", flush=True)

    stream_response = StreamingResponse(_iter_bytes(), media_type=media_type)
    with suppress(Exception):
        stream_response.headers["x-request-id"] = request_id
    return stream_response


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/v1/chat/completions", request)


def main() -> None:
    import uvicorn

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
