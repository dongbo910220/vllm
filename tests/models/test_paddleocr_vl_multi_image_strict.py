# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Strict greedy + logprobs consistency for PaddleOCR‑VL with multiple images.

Requirements aligned with vLLM VLM tests:
  - Greedy decoding (temperature=0)
  - Unify dtype across HF and vLLM (float16)
  - Compare text AND per-token logprobs (top-k)

We construct the chat prompt with multiple image placeholders via HF processor
and pass the images through the respective runners.
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
    try:
        return Image.open(ImageAsset(name).get_path("jpg"))
    except Exception:
        return Image.open(ImageAsset(name).get_path("png"))


def _strip_eos(text: str) -> str:
    return text.rstrip().removesuffix("</s>").rstrip()


@pytest.mark.core_model
@pytest.mark.parametrize(
    "asset_names",
    [
        ("stop_sign", "cherry_blossom"),
        ("hato", "stop_sign"),
        ("RGBA_comp", "237-400x300", "231-200x300"),  # 3-image combo
    ],
)
def test_paddleocr_vl_strict_multi_image(tmp_path: Path, asset_names: tuple[str, ...]):
    model_id = os.getenv("PADDLEOCRVL_MODEL", "PaddlePaddle/PaddleOCR-VL")
    imgs = [_load_asset_image(n) for n in asset_names]
    prompt_text = os.getenv("PADDLEOCRVL_PROMPT", "请分别描述这些图片中的关键信息。")
    max_new = int(os.getenv("PADDLEOCRVL_MAX_NEW", "64"))

    hf_proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    prompt = _build_prompt(hf_proc, imgs, prompt_text)

    num_logprobs = 10

    with VllmRunner(
        model_name=model_id,
        dtype="float16",
        enforce_eager=True,
        disable_log_stats=True,
        max_model_len=8192,
        gpu_memory_utilization=float(os.getenv("GPU_UTIL", "0.8")),
    ) as vllm_model:
        v_outputs = vllm_model.generate_greedy_logprobs(
            [prompt],
            max_tokens=max_new,
            num_logprobs=num_logprobs,
            images=[imgs],  # multi-image for single prompt
        )

    with HfRunner(model_name=model_id, dtype="float16") as hf_model:
        h_outputs = hf_model.generate_greedy_logprobs_limit(
            [prompt],
            max_tokens=max_new,
            num_logprobs=num_logprobs,
            images=[imgs],
        )

    # Optional strict text equality: enable by setting PADDLEOCRVL_STRICT_TEXT=1
    if os.getenv("PADDLEOCRVL_STRICT_TEXT", "0") == "1":
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
