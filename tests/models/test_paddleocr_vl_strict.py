# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Strict consistency (greedy decoding) check for a multimodal model (PaddleOCR‑VL).

This upgrades the prior text-only strict test to align with vLLM's "Strict
Consistency" definition used by multimodal tests:
  - greedy decoding (temperature=0)
  - text AND per-token logprobs comparison (top-k)
  - unified dtype across HF and vLLM (float16)

We build the chat prompt via the HF processor (with <image> markers) and pass
the same prompt to both HF and vLLM, while feeding images via the respective
runner inputs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoProcessor

from tests.conftest import HfRunner, VllmRunner
from tests.models.utils import check_logprobs_close, check_outputs_equal
from vllm.assets.image import ImageAsset


@pytest.fixture(scope="function", autouse=True)
def _allow_pickle(monkeypatch):
    """LLM.apply_model requires pickling a function in some runners."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _build_prompt(proc: AutoProcessor, images: Sequence[Image.Image], text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            ),
        }
    ]
    return proc.apply_chat_template(messages, add_generation_prompt=True)


def _load_asset_image(name: str) -> Image.Image:
    # Try jpg first, then png (some assets are PNG)
    try:
        return Image.open(ImageAsset(name).get_path("jpg"))
    except Exception:
        return Image.open(ImageAsset(name).get_path("png"))


def _strip_eos(text: str) -> str:
    return text.rstrip().removesuffix("</s>").rstrip()


@pytest.mark.core_model
@pytest.mark.parametrize(
    "asset_name",
    [
        # Use a broader set of public JPG assets for coverage
        "stop_sign",
        "cherry_blossom",
        "hato",
        "handelsblatt-preview",
        "paper-11",
    ],
)
def test_paddleocr_vl_strict_single_image(tmp_path: Path, asset_name: str):
    model_id = os.getenv("PADDLEOCRVL_MODEL", "PaddlePaddle/PaddleOCR-VL")
    # Use vLLM public image assets to align with other multimodal tests.
    img = _load_asset_image(asset_name)
    prompt_text = os.getenv("PADDLEOCRVL_PROMPT", "请识别图片中的关键信息。")
    max_new = int(os.getenv("PADDLEOCRVL_MAX_NEW", "64"))

    # Build prompt with HF processor chat template
    hf_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    prompt = _build_prompt(hf_proc, [img], prompt_text)

    # Unify dtype=float16 for both HF and vLLM, compare text + logprobs.
    num_logprobs = 10

    with VllmRunner(
        model_name=model_id,
        dtype="float16",
        enforce_eager=True,
        disable_log_stats=True,
        max_model_len=4096,
        gpu_memory_utilization=float(os.getenv("GPU_UTIL", "0.8")),
    ) as vllm_model:
        v_outputs = vllm_model.generate_greedy_logprobs(
            [prompt],
            max_tokens=max_new,
            num_logprobs=num_logprobs,
            images=[[img]],  # list-of-list for potential multi-image compatibility
        )

    with HfRunner(model_name=model_id, dtype="float16") as hf_model:
        h_outputs = hf_model.generate_greedy_logprobs_limit(
            [prompt],
            max_tokens=max_new,
            num_logprobs=num_logprobs,
            images=[[img]],
        )

    # 1) strict token/text equality; 2) logprobs close (top-k contains mutual tokens)
    check_outputs_equal(
        outputs_0_lst=[(o[0], _strip_eos(o[1])) for o in h_outputs],
        outputs_1_lst=[(o[0], _strip_eos(o[1])) for o in v_outputs],
        name_0="hf",
        name_1="vllm",
    )
    check_logprobs_close(
        outputs_0_lst=h_outputs,
        outputs_1_lst=v_outputs,
        name_0="hf",
        name_1="vllm",
        always_check_logprobs=True,
    )

    # Done
