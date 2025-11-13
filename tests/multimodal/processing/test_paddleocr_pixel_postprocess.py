# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.processing import InputProcessingContext


class _DummyTokenizer:
    pass


class _DummyHFConfig:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type


class _DummyModelConfig:
    def __init__(self, dtype: torch.dtype, model_type: str) -> None:
        self.dtype = dtype
        self.hf_config = _DummyHFConfig(model_type)


def _ctx_for(
    model_type: str, dtype: torch.dtype = torch.bfloat16
) -> InputProcessingContext:
    mc = _DummyModelConfig(dtype=dtype, model_type=model_type)
    # Tokenizer is unused in _postprocess_output
    return InputProcessingContext(model_config=mc, tokenizer=_DummyTokenizer())


def test_postprocess_preserves_pixel_values_for_paddleocr_vl() -> None:
    ctx = _ctx_for("paddleocr_vl", dtype=torch.bfloat16)

    pv = torch.randn(1, 3, 384, 384, dtype=torch.float32)
    other = torch.tensor([1.0], dtype=torch.float32)

    out = ctx._postprocess_output({"pixel_values": pv, "other": other})

    assert isinstance(out, dict)
    # pixel_values should remain float32
    assert isinstance(out["pixel_values"], torch.Tensor)
    assert out["pixel_values"].dtype == torch.float32
    # other tensors are cast to model dtype
    assert isinstance(out["other"], torch.Tensor)
    assert out["other"].dtype == torch.bfloat16


def test_postprocess_casts_pixel_values_for_non_paddleocr() -> None:
    ctx = _ctx_for("llava", dtype=torch.bfloat16)

    pv = torch.randn(1, 3, 384, 384, dtype=torch.float32)
    out = ctx._postprocess_output({"pixel_values": pv})

    assert isinstance(out, dict)
    assert isinstance(out["pixel_values"], torch.Tensor)
    # For non-PaddleOCR-VL models, pixel_values are cast to model dtype
    assert out["pixel_values"].dtype == torch.bfloat16
