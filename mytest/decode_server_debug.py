#!/usr/bin/env python3
"""
Decode Server - Direct Launch for PyCharm Debugging

直接导入 vLLM 模块启动，避免 subprocess 导致的调试器分离问题
"""

# ========== 关键修复：必须在任何其他导入之前处理 sys.path ==========
import sys
from pathlib import Path

# 获取脚本目录
SCRIPT_DIR = Path(__file__).resolve().parent

# 从 sys.path 中移除当前目录，避免导入源码目录的 vllm/
paths_to_remove = [str(SCRIPT_DIR), '', '.']
for path in paths_to_remove:
    while path in sys.path:
        sys.path.remove(path)

# 导入其他模块
import os
import json
import asyncio

# 切换工作目录到 /tmp，避免 Python 自动添加当前目录
os.chdir('/tmp')
# ==========================================================

# ========== 强制禁用 uvloop ==========
os.environ['VLLM_USE_UVLOOP'] = '0'
print(f"🔧 设置 VLLM_USE_UVLOOP=0")

# ========== 配置 ==========
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
SHARED_STORAGE_DIR = BASE_DIR / "shared_storage"

MODEL_NAME = os.environ.get(
    "PD_MODEL_NAME",
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
)
DECODE_PORT = int(os.environ.get("DECODE_PORT", "8200"))
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
KV_PORT = int(os.environ.get("PD_KV_PORT", "14580"))
MAX_MODEL_LEN = os.environ.get("PD_MAX_MODEL_LEN", "256")
GPU_MEM_UTIL = os.environ.get("PD_GPU_MEMORY_UTIL", "0.5")

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CONNECTOR_LOOKUP = {
    "nixl": "nixl",
    "shared": "shared_storage",
    "p2p": "p2p_nccl",
}
CONNECTOR_TYPE = CONNECTOR_LOOKUP.get(
    os.environ.get("PD_CONNECTOR", "shared_storage").strip().lower(),
    "shared_storage",
)
SEND_TYPE = os.environ.get("PD_SEND_TYPE", "PUT_ASYNC").strip().upper()

def build_kv_config() -> str:
    """构建Decode的KV transfer配置"""
    if CONNECTOR_TYPE == "nixl":
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
        }
    elif CONNECTOR_TYPE == "shared_storage":
        SHARED_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "kv_connector": "SharedStorageConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(SHARED_STORAGE_DIR)
            },
        }
    else:  # p2p_nccl
        config = {
            "kv_connector": "P2pNcclConnector",
            "kv_role": "kv_consumer",
            "kv_rank": 1,
            "kv_parallel_size": 2,
            # 与 prefill 使用不同 kv_port，默认 14580，可通过 PD_KV_PORT 覆盖
            "kv_port": KV_PORT,
            "kv_connector_extra_config": {
                "send_type": SEND_TYPE,
            },
        }
    return json.dumps(config)

def main():
    print("=" * 80)
    print("Decode Server (KV Consumer) - Direct Launch")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Port: {DECODE_PORT}")
    print(f"GPU: {GPU_ID}")
    print(f"Connector: {CONNECTOR_TYPE}")
    print(f"KV Config: {build_kv_config()}")
    print("=" * 80)
    print()

    # 设置 sys.argv 模拟命令行参数
    sys.argv = [
        'decode_server_debug.py',
        '--model', MODEL_NAME,
        '--port', str(DECODE_PORT),
        '--max-model-len', str(MAX_MODEL_LEN),
        '--gpu-memory-utilization', str(GPU_MEM_UTIL),
        '--trust-remote-code',
        '--enforce-eager',  # ✅ 禁用 CUDA Graph，允许打断点
        '--enable-request-id-headers',
        '--kv-transfer-config', build_kv_config(),
    ]

    print(f"🚀 启动参数: {' '.join(sys.argv[1:])}")
    print()

    # 直接导入并运行 vLLM API server
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup, FlexibleArgumentParser, make_arg_parser,
        validate_parsed_serve_args, run_server
    )

    print("🔧 启动 vLLM Decode 服务器 (使用 asyncio)")

    # 初始化环境
    cli_env_setup()

    # 解析参数
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # 使用标准 asyncio.run 而不是 uvloop.run
    asyncio.run(run_server(args))

if __name__ == "__main__":
    main()
