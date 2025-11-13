# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLMultiModalProcessor,
)


class _DummyInfo:
    """Minimal info stub to instantiate the processor without HF access."""

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_allowed_mm_limits(self):
        return {"image": 999}


def _make_proc() -> PaddleOCRVLMultiModalProcessor:  # type: ignore[override]
    return PaddleOCRVLMultiModalProcessor(info=_DummyInfo(), dummy_inputs=object())


def test_presplit_L_must_match_grid():
    proc = _make_proc()
    # Two images, each with grid t*h*w = 4
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    # Mismatch: L=3 while t*h*w=4
    pixel = torch.randn(2, 3, 3, 14, 14)  # [B=2, L=3, 3, p, p]
    with pytest.raises(ValueError, match="does not match grid_thw product"):
        proc._parse_and_validate_image_input(
            pixel_values=pixel,
            image_grid_thw=grid,
        )


def test_batch_rows_must_match():
    proc = _make_proc()
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)  # rows=1
    pixel = torch.randn(2, 4, 3, 14, 14)  # B=2, L=4
    with pytest.raises(ValueError, match="batch size and image_grid_thw rows"):
        proc._parse_and_validate_image_input(
            pixel_values=pixel,
            image_grid_thw=grid,
        )


def test_flattened_length_must_match():
    proc = _make_proc()
    # grid sums to 5, but we provide 6 flat patches
    grid = torch.tensor([[1, 2, 2], [1, 1, 1]], dtype=torch.long)
    flat = torch.randn(6, 3, 14, 14)
    with pytest.raises(ValueError, match="Flattened pixel_values length"):
        proc._parse_and_validate_image_input(
            pixel_values=flat,
            image_grid_thw=grid,
        )


def test_accepts_4d_and_5d_shapes():
    proc = _make_proc()
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    # 4D path: [B, 3, H, W] should pass validation
    pixel_4d = torch.randn(1, 3, 224, 224)
    out = proc._parse_and_validate_image_input(
        pixel_values=pixel_4d,
        image_grid_thw=grid,
    )
    assert out is not None and out["type"] == "pixel_values"

    # 5D path with correct L=t*h*w (=4) should pass
    pixel_5d = torch.randn(1, 4, 3, 14, 14)
    out2 = proc._parse_and_validate_image_input(
        pixel_values=pixel_5d,
        image_grid_thw=grid,
    )
    assert out2 is not None and out2["type"] == "pixel_values"
