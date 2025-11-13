# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton

from .base import RotaryEmbedding
from .common import apply_rotary_emb_dispatch, rotate_gptj, rotate_neox
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding, yarn_get_mscale


@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    # instead of (3, bsz, seq_len, head_dim), also supports interleaved rotary
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)

    # Updated stride calculation for half head_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # Since cos and sin are now half-size,
    # we use the same cos_row and sin_row for both halves
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2VL mrope kernel.

    Args:
        q: [num_tokens, num_heads * head_size]
        k: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
    """
    n_row, n_q_head_head_dim = q.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # ensure tensors passed into the kernel are contiguous.
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_mrope_forward[(n_row,)](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
    )
    return q, k


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """HF-style interleaved mRoPE reordering.

    Inputs:
      - x: [3, S, half] where the first dim corresponds to (T, H, W).
      - mrope_section: [t_len, h_len, w_len] measured on the half-dim (pairs).

    Output:
      - [S, half] with features laid out in an interleaved order T, H, W
        repeated until each section length is exhausted, then continuing
        with the remaining axes. This mirrors HF's chunk-wise selection
        (i % 3) semantics over the 3-way split of the rotary half-dimension.
    """
    assert x.ndim == 3 and x.shape[0] == 3, "expected [3, S, half]"
    # sequence length for output allocation
    s = int(x.shape[1])
    half = x.shape[2]
    t_len, h_len, w_len = map(int, mrope_section)
    # Guard against malformed configs
    total = t_len + h_len + w_len
    if total != half:
        # Clamp to available size (keeps behavior stable if round-off exists)
        t_len = min(t_len, half)
        h_len = min(h_len, max(0, half - t_len))
        w_len = min(w_len, max(0, half - t_len - h_len))
        total = t_len + h_len + w_len
    out = x.new_empty((s, half))
    # Per-axis cursors and remaining counts
    idx = [0, 0, 0]
    rem = [t_len, h_len, w_len]
    pos = 0
    # Interleave in T,H,W order repeatedly
    while pos < total:
        for axis in (0, 1, 2):
            if rem[axis] > 0 and pos < total:
                out[:, pos] = x[axis, :, idx[axis]]
                idx[axis] += 1
                rem[axis] -= 1
                pos += 1
    # If any slack due to clamping, pad tail with T-axis remainder (stable)
    if pos < half:
        tail = min(half - pos, max(0, x.shape[-1] - idx[0]))
        if tail > 0:
            out[:, pos : pos + tail] = x[0, :, idx[0] : idx[0] + tail]
            pos += tail
    # If still short (shouldn't happen), zero-fill
    if pos < half:
        out[:, pos:] = 0
    return out


def apply_hf_chunked_rope_half(
    x: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """HF-style mapping directly on the HALF dimension.

    This constructs the HALF rotary tables by selecting the first ``t``,
    ``h``, and ``w`` features from the T/H/W axis-wise HALF tables and
    concatenating them in order. This avoids compressing from FULL where
    chunk boundaries may break pair structure.

    Args:
      x: [3, S, half] with axes (T, H, W) over the rotary half-dimension.
      mrope_section: [t, h, w] measured on the half-dimension.

    Returns:
      Tensor of shape [S, half].
    """
    assert x.ndim == 3 and x.shape[0] == 3, "expected [3, S, half]"
    half = int(x.shape[-1])
    if len(mrope_section) == 3:
        t, h, w = [int(v) for v in mrope_section]
    else:
        t, h, w = [int(v) for v in mrope_section]
    # best-effort clamp if minor drift exists
    t = max(0, min(t, half))
    h = max(0, min(h, half - t))
    w = max(0, min(w, half - t - h))
    out = torch.cat([x[0, :, :t], x[1, :, :h], x[2, :, :w]], dim=-1)
    # pad if needed (should rarely happen)
    if out.shape[-1] != half:
        if out.shape[-1] > half:
            out = out[..., :half]
        else:
            out = torch.nn.functional.pad(out, (0, half - out.shape[-1]))
    return out.contiguous()


def apply_paddle_hw_then_t_half(
    x: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """PaddleOCR-VL style mapping from HALF tables.

    Emulates HF's apply_multimodal_rotary_pos_emb reordering where the
    front consists of interleaved H/W pairs and the tail consists of T.
    We expand to FULL via repeat_interleave, apply mapping, then ::2 compress.

    Args:
      x: [3, S, half] with axes (T, H, W).
      mrope_section: [t, h, w] measured on the half-dimension.
    Returns:
      [S, half] table arranged as [HW pairs..., T tail].
    """
    assert x.ndim == 3 and x.shape[0] == 3, "expected [3, S, half]"
    x_full = x.repeat_interleave(2, dim=-1)
    return apply_paddle_hw_then_t_half_from_full(x_full, mrope_section)


def apply_paddle_hw_then_t_half_from_full(
    x_full: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """PaddleOCR-VL style mapping from FULL tables to HALF tables.

    - Take front 2*(h+w) along FULL; even positions -> H axis, odd -> W axis;
      interleave back to FULL order for HW.
    - Take tail 2*t from T axis.
    - Concatenate [HW_full, T_full] and compress to HALF with ::2.
    """
    assert x_full.ndim == 3 and x_full.shape[0] == 3, "expected [3, S, full]"
    # mrope_section follows HF convention [t, h, w] (measured on half-dim).
    if len(mrope_section) == 3:
        t, h, w = [int(v) for v in mrope_section]
    else:
        t, h, w = [int(v) for v in mrope_section]
    full = int(x_full.shape[-1])
    half = t + h + w
    # Build H and W FULL segments explicitly and interleave them:
    h_full_seg = x_full[1, :, : 2 * h]
    w_full_seg = x_full[2, :, : 2 * w]
    # Interleave H and W features along the last dimension: H0, W0, H1, W1, ...
    # When h != w, append the remaining tail from the longer axis.
    min_hw = min(h_full_seg.shape[-1], w_full_seg.shape[-1]) // 2
    if min_hw > 0:
        h_pairs = h_full_seg.reshape(h_full_seg.shape[0], -1, 2)
        w_pairs = w_full_seg.reshape(w_full_seg.shape[0], -1, 2)
        to_stack = []
        for i in range(min_hw):
            to_stack.append(h_pairs[:, i, :])
            to_stack.append(w_pairs[:, i, :])
        hw_full_tensor = torch.cat(to_stack, dim=-1)
    else:
        hw_full_tensor = torch.cat([h_full_seg, w_full_seg], dim=-1)
    # Append any leftover FULL dims from H or W if lengths differ
    if h_full_seg.shape[-1] > 2 * min_hw:
        hw_full_tensor = torch.cat(
            [hw_full_tensor, h_full_seg[:, 2 * min_hw :]], dim=-1
        )
    if w_full_seg.shape[-1] > 2 * min_hw:
        hw_full_tensor = torch.cat(
            [hw_full_tensor, w_full_seg[:, 2 * min_hw :]], dim=-1
        )
    # T tail from axis 0 (take the last 2*t dims)
    t_full_len = min(2 * t, max(0, full - hw_full_tensor.shape[-1]))
    t_full_tensor = x_full[0, :, -t_full_len:] if t_full_len > 0 else x_full[0, :, :0]
    mapped_full = torch.cat([hw_full_tensor, t_full_tensor], dim=-1)
    # Compress back to HALF by taking one value per pair. Default to even index;
    # allow switching to odd via env for diagnostics.
    compress = os.getenv("PADDLEOCRVL_MROPE_COMPRESS", "even").lower()
    if compress == "odd":
        mapped_half = mapped_full[:, 1::2].contiguous()
    else:
        mapped_half = mapped_full[:, ::2].contiguous()
    # Best-effort sanity check
    if mapped_half.shape[-1] != half:
        # Clamp/pad to expected length if minor shape drift occurs
        if mapped_half.shape[-1] > half:
            mapped_half = mapped_half[..., :half]
        else:
            pad = half - mapped_half.shape[-1]
            mapped_half = torch.nn.functional.pad(mapped_half, (0, pad))
    return mapped_half


def apply_paddle_hw_then_t_full(
    x_full: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """PaddleOCR‑VL style mapping that returns FULL tables.

    Equivalent to apply_paddle_hw_then_t_half_from_full but without the
    final FULL→HALF compression.

    Args:
      x_full: [3, S, full] per-axis FULL tables (T/H/W in dim 0)
      mrope_section: [t, h, w] measured on HALF
    Returns:
      [S, full] mapped FULL table.
    """
    assert x_full.ndim == 3 and x_full.shape[0] == 3, "expected [3, S, full]"
    if len(mrope_section) == 3:
        t, h, w = [int(v) for v in mrope_section]
    else:
        t, h, w = [int(v) for v in mrope_section]
    full = int(x_full.shape[-1])
    # Build HW FULL front by interleaving pairs from H and W axes
    h_full_seg = x_full[1, :, : 2 * h]
    w_full_seg = x_full[2, :, : 2 * w]
    min_hw = min(h_full_seg.shape[-1], w_full_seg.shape[-1]) // 2
    if min_hw > 0:
        h_pairs = h_full_seg.reshape(h_full_seg.shape[0], -1, 2)
        w_pairs = w_full_seg.reshape(w_full_seg.shape[0], -1, 2)
        to_stack = []
        for i in range(min_hw):
            to_stack.append(h_pairs[:, i, :])
            to_stack.append(w_pairs[:, i, :])
        hw_full_tensor = torch.cat(to_stack, dim=-1)
    else:
        hw_full_tensor = torch.cat([h_full_seg, w_full_seg], dim=-1)
    if h_full_seg.shape[-1] > 2 * min_hw:
        hw_full_tensor = torch.cat(
            [hw_full_tensor, h_full_seg[:, 2 * min_hw :]], dim=-1
        )
    if w_full_seg.shape[-1] > 2 * min_hw:
        hw_full_tensor = torch.cat(
            [hw_full_tensor, w_full_seg[:, 2 * min_hw :]], dim=-1
        )
    # T tail (respect available remaining size)
    t_full_len = min(2 * t, max(0, full - hw_full_tensor.shape[-1]))
    t_full_tensor = x_full[0, :, -t_full_len:] if t_full_len > 0 else x_full[0, :, :0]
    return torch.cat([hw_full_tensor, t_full_tensor], dim=-1)


def _compress_full_to_half(x_full: torch.Tensor, method: str = "even") -> torch.Tensor:
    """Compress FULL rotary table [S, full] to HALF [S, half].

    Methods:
      - "even": take even indices (x[:, ::2]) matching pair-wise layout.
      - "first": take the first half slice (x[:, : x.shape[-1] // 2]).
    """
    if method == "first":
        return x_full[:, : x_full.shape[-1] // 2].contiguous()
    # default: even
    return x_full[:, ::2].contiguous()


def apply_hf_chunked_rope_half_from_full(
    x_full: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """HF-style chunked mapping starting from the FULL rotary dimension.

    Args:
      x_full: [3, S, full] where full == 2 * half (i.e., head_dim)
      mrope_section: [t, h, w] measured on the half-dimension.

    Returns:
      [S, half] half-dimension table after HF chunked selection and ::2 compress.
    """
    assert x_full.ndim == 3 and x_full.shape[0] == 3, "expected [3, S, full]"
    # s = int(x_full.shape[1])  # sequence length (unused)
    full = int(x_full.shape[2])
    # mrope_section follows HF convention [t, h, w].
    t, h, w = [int(v) for v in mrope_section]
    half = t + h + w
    assert 2 * half == full, "full must equal 2*(t+h+w)"

    sections_full = [2 * t, 2 * h, 2 * w]
    c_t, c_h, c_w = x_full.split(sections_full, dim=-1)
    mapped_full = torch.cat([c_t[0], c_h[1], c_w[2]], dim=-1)

    mapped_half = mapped_full[:, ::2].contiguous()
    return mapped_half


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        # YaRN parameters.
        *,
        scaling_factor: float | None = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        if self.scaling_factor is not None:
            # Get n-d magnitude scaling corrected for interpolation
            self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        else:
            self.mscale = 1.0

        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_inv_freq(base)
        return YaRNScalingRotaryEmbedding._compute_inv_freq(self, base)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_cos_sin_cache()
        return YaRNScalingRotaryEmbedding._compute_cos_sin_cache(self)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos_base, sin_base = cos_sin.chunk(2, dim=-1)
        full_tables: tuple[torch.Tensor, torch.Tensor] | None = None
        if positions.ndim == 2:
            assert self.mrope_section
            # Build FULL-dim per-axis base and map in FULL, then compress to HALF
            # (matches HF RotaryEmbedding + apply_multimodal_rotary_pos_emb semantics).
            # Interleaved uses the existing helper over HALF (unchanged).
            if self.mrope_interleaved:
                cos_half3_fp32 = cos_base.to(torch.float32)
                sin_half3_fp32 = sin_base.to(torch.float32)
                cos = apply_interleaved_rope(cos_half3_fp32, self.mrope_section)
                sin = apply_interleaved_rope(sin_half3_fp32, self.mrope_section)
            else:
                mapping = os.getenv("PADDLEOCRVL_MROPE_MAP", "hf").lower()
                if mapping == "paddle":
                    # PaddleOCR‑VL 真实分段映射：HW 前缀交错 + T 尾段。
                    # 默认：按旧路径在 FULL 维度上映射，避免 E2E 回归。
                    # 可选：开启 PADDLEOCRVL_USE_HALF_DUP=1 使用
                    # HALF→FULL 复制路径（A/B 对照）。
                    cos_half3_fp32 = cos_base.to(torch.float32)
                    sin_half3_fp32 = sin_base.to(torch.float32)
                    use_half_dup = os.getenv(
                        "PADDLEOCRVL_USE_HALF_DUP", "0"
                    ).lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    )
                    # HALF 映射（供 half-dispatch/调试使用，不影响 FULL 旋转路径）
                    cos = apply_paddle_hw_then_t_half(
                        cos_half3_fp32, self.mrope_section
                    )
                    sin = apply_paddle_hw_then_t_half(
                        sin_half3_fp32, self.mrope_section
                    )
                    # FULL 轴表
                    cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)
                    sin_full3 = torch.cat([sin_half3_fp32, sin_half3_fp32], dim=-1)
                    if use_half_dup:
                        # HALF→FULL 复制（开关下用于对照试验）
                        cos_full_mapped = torch.cat([cos, cos], dim=-1)
                        sin_full_mapped = torch.cat([sin, sin], dim=-1)
                    else:
                        # 旧路径：直接在 FULL 上做 Paddle 风格映射
                        cos_full_mapped = apply_paddle_hw_then_t_full(
                            cos_full3, self.mrope_section
                        )
                        sin_full_mapped = apply_paddle_hw_then_t_full(
                            sin_full3, self.mrope_section
                        )
                    if getattr(self, "mscale", 1.0) != 1.0:
                        cos_full_mapped = cos_full_mapped * float(self.mscale)
                        sin_full_mapped = sin_full_mapped * float(self.mscale)
                    # Optional: apply in-segment reorder fix (same knobs as native)
                    try:
                        if self.mrope_section:
                            t_len, h_len, w_len = (
                                int(self.mrope_section[0]),
                                int(self.mrope_section[1]),
                                int(self.mrope_section[2]),
                            )
                            front = 2 * (h_len + w_len)
                            tail = 2 * t_len
                            if cos_full_mapped.shape[-1] >= (
                                front + tail
                            ) and sin_full_mapped.shape[-1] >= (front + tail):
                                swap_front = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                swap_tail = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_TAIL", "0"
                                ) not in ("", "0", "false", "False")
                                flip_front = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                flip_tail = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_TAIL", "0"
                                ) not in ("", "0", "false", "False")

                                if swap_front and front > 0:
                                    even = cos_full_mapped[:, 0:front:2]
                                    odd = cos_full_mapped[:, 1:front:2]
                                    cos_front_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [cos_front_swapped, cos_full_mapped[:, front:]],
                                        dim=-1,
                                    )
                                    even_s = sin_full_mapped[:, 0:front:2]
                                    odd_s = sin_full_mapped[:, 1:front:2]
                                    sin_front_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [sin_front_swapped, sin_full_mapped[:, front:]],
                                        dim=-1,
                                    )
                                if swap_tail and tail > 0:
                                    start = front
                                    end = front + tail
                                    even = cos_full_mapped[:, start:end:2]
                                    odd = cos_full_mapped[:, start + 1 : end : 2]
                                    cos_tail_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [
                                            cos_full_mapped[:, :start],
                                            cos_tail_swapped,
                                            cos_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )
                                    even_s = sin_full_mapped[:, start:end:2]
                                    odd_s = sin_full_mapped[:, start + 1 : end : 2]
                                    sin_tail_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [
                                            sin_full_mapped[:, :start],
                                            sin_tail_swapped,
                                            sin_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )
                                if flip_front and front > 0:
                                    sin_full_mapped[:, :front] = -sin_full_mapped[
                                        :, :front
                                    ]
                                if flip_tail and tail > 0:
                                    sin_full_mapped[
                                        :, front : front + tail
                                    ] = -sin_full_mapped[:, front : front + tail]
                                if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                                    "",
                                    "0",
                                    "false",
                                    "False",
                                ):
                                    try:
                                        run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                                        out_dir = os.path.join(
                                            "test_result", "text_stages"
                                        )
                                        os.makedirs(out_dir, exist_ok=True)
                                        with open(
                                            os.path.join(
                                                out_dir,
                                                f"vllm_{run_id}_segment_reorder_applied.txt",
                                            ),
                                            "w",
                                            encoding="utf-8",
                                        ) as f:
                                            f.write(
                                                f"swap_front={swap_front}, "
                                                f"swap_tail={swap_tail}\n"
                                            )
                                            f.write(
                                                f"flip_front={flip_front}, "
                                                f"flip_tail={flip_tail}\n"
                                            )
                                            tmpl = "front={f}, tail={t}, full={u}\n"
                                            msg = tmpl.format(
                                                f=front,
                                                t=tail,
                                                u=cos_full_mapped.shape[-1],
                                            )
                                            f.write(msg)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    # Optional: PaddleOCR-VL only — apply in-segment reorder fix.
                    # Env knobs (default off):
                    #   PADDLEOCRVL_SWAP_PAIRS_FRONT=1 → swap even/odd within
                    #     each pair across the HW front (2*(h+w)).
                    #   PADDLEOCRVL_SWAP_PAIRS_TAIL=1 → swap even/odd within
                    #     each pair across the T tail (2*t).
                    #   PADDLEOCRVL_SIN_FLIP_FRONT=1 → flip sign of sin on the
                    #     HW front only.
                    #   PADDLEOCRVL_SIN_FLIP_TAIL=1 → flip sign of sin on the
                    #     T tail only.
                    try:
                        if self.mrope_section:
                            t_len, h_len, w_len = (
                                int(self.mrope_section[0]),
                                int(self.mrope_section[1]),
                                int(self.mrope_section[2]),
                            )
                            front = 2 * (h_len + w_len)
                            tail = 2 * t_len
                            if cos_full_mapped.shape[-1] >= (
                                front + tail
                            ) and sin_full_mapped.shape[-1] >= (front + tail):
                                swap_front = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                swap_tail = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_TAIL", "0"
                                ) not in ("", "0", "false", "False")
                                flip_front = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                flip_tail = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_TAIL", "0"
                                ) not in ("", "0", "false", "False")

                                if swap_front and front > 0:
                                    even = cos_full_mapped[:, 0:front:2]
                                    odd = cos_full_mapped[:, 1:front:2]
                                    cos_front_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [cos_front_swapped, cos_full_mapped[:, front:]],
                                        dim=-1,
                                    )
                                    even_s = sin_full_mapped[:, 0:front:2]
                                    odd_s = sin_full_mapped[:, 1:front:2]
                                    sin_front_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [sin_front_swapped, sin_full_mapped[:, front:]],
                                        dim=-1,
                                    )

                                if swap_tail and tail > 0:
                                    start = front
                                    end = front + tail
                                    even = cos_full_mapped[:, start:end:2]
                                    odd = cos_full_mapped[:, start + 1 : end : 2]
                                    cos_tail_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [
                                            cos_full_mapped[:, :start],
                                            cos_tail_swapped,
                                            cos_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )

                                    even_s = sin_full_mapped[:, start:end:2]
                                    odd_s = sin_full_mapped[:, start + 1 : end : 2]
                                    sin_tail_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [
                                            sin_full_mapped[:, :start],
                                            sin_tail_swapped,
                                            sin_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )

                                if flip_front and front > 0:
                                    sin_full_mapped[:, :front] = -sin_full_mapped[
                                        :, :front
                                    ]
                                if flip_tail and tail > 0:
                                    sin_full_mapped[
                                        :, front : front + tail
                                    ] = -sin_full_mapped[:, front : front + tail]

                                if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                                    "",
                                    "0",
                                    "false",
                                    "False",
                                ):
                                    try:
                                        run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                                        out_dir = os.path.join(
                                            "test_result", "text_stages"
                                        )
                                        os.makedirs(out_dir, exist_ok=True)
                                        with open(
                                            os.path.join(
                                                out_dir,
                                                f"vllm_{run_id}_segment_reorder_applied.txt",
                                            ),
                                            "w",
                                            encoding="utf-8",
                                        ) as f:
                                            f.write(
                                                f"swap_front={swap_front}, "
                                                f"swap_tail={swap_tail}\n"
                                            )
                                            f.write(
                                                f"flip_front={flip_front}, "
                                                f"flip_tail={flip_tail}\n"
                                            )
                                            tmpl = "front={f}, tail={t}, full={u}\n"
                                            msg = tmpl.format(
                                                f=front,
                                                t=tail,
                                                u=cos_full_mapped.shape[-1],
                                            )
                                            f.write(msg)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    full_tables = (cos_full_mapped, sin_full_mapped)
            else:
                # HF Chunked 映射（与 HF 完全一致）：
                # 先将每轴 HALF 表复制为 FULL 表，再按 [2t,2h,2w] 在 FULL 上分段，
                # 对角选择 (T/H/W) 后拼接成最终 FULL 表用于旋转。
                cos_half3_fp32 = cos_base.to(torch.float32)
                sin_half3_fp32 = sin_base.to(torch.float32)

                # 供 HALF-dispatch/FALLBACK 使用：直接在 HALF 上做分段对角映射
                cos = apply_hf_chunked_rope_half(cos_half3_fp32, self.mrope_section)
                sin = apply_hf_chunked_rope_half(sin_half3_fp32, self.mrope_section)

                # FULL 轴向表（Neox pairing：cat(half, half)）
                cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)
                sin_full3 = torch.cat([sin_half3_fp32, sin_half3_fp32], dim=-1)

                t, h, w = map(int, self.mrope_section)
                sections_full = [2 * t, 2 * h, 2 * w]
                c_t, c_h, c_w = cos_full3.split(sections_full, dim=-1)
                s_t, s_h, s_w = sin_full3.split(sections_full, dim=-1)
                cos_full_mapped = torch.cat([c_t[0], c_h[1], c_w[2]], dim=-1)
                sin_full_mapped = torch.cat([s_t[0], s_h[1], s_w[2]], dim=-1)
                    # Apply magnitude scaling if configured (YaRN-style)
                    if getattr(self, "mscale", 1.0) != 1.0:
                        cos_full_mapped = cos_full_mapped * float(self.mscale)
                        sin_full_mapped = sin_full_mapped * float(self.mscale)
                    # Optional: apply the same in-segment reorder fix knobs here
                    # as well (useful for HF mapping A/B during调试).
                    try:
                        if self.mrope_section:
                            t_len, h_len, w_len = (
                                int(self.mrope_section[0]),
                                int(self.mrope_section[1]),
                                int(self.mrope_section[2]),
                            )
                            front = 2 * (h_len + w_len)
                            tail = 2 * t_len
                            if cos_full_mapped.shape[-1] >= (
                                front + tail
                            ) and sin_full_mapped.shape[-1] >= (front + tail):
                                swap_front = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                swap_tail = os.getenv(
                                    "PADDLEOCRVL_SWAP_PAIRS_TAIL", "0"
                                ) not in ("", "0", "false", "False")
                                flip_front = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_FRONT", "0"
                                ) not in ("", "0", "false", "False")
                                flip_tail = os.getenv(
                                    "PADDLEOCRVL_SIN_FLIP_TAIL", "0"
                                ) not in ("", "0", "false", "False")

                                if swap_front and front > 0:
                                    even = cos_full_mapped[:, 0:front:2]
                                    odd = cos_full_mapped[:, 1:front:2]
                                    cos_front_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [cos_front_swapped, cos_full_mapped[:, front:]],
                                        dim=-1,
                                    )
                                    even_s = sin_full_mapped[:, 0:front:2]
                                    odd_s = sin_full_mapped[:, 1:front:2]
                                    sin_front_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [sin_front_swapped, sin_full_mapped[:, front:]],
                                        dim=-1,
                                    )
                                if swap_tail and tail > 0:
                                    start = front
                                    end = front + tail
                                    even = cos_full_mapped[:, start:end:2]
                                    odd = cos_full_mapped[:, start + 1 : end : 2]
                                    cos_tail_swapped = torch.stack(
                                        (odd, even), dim=-1
                                    ).reshape(cos_full_mapped.shape[0], -1)
                                    cos_full_mapped = torch.cat(
                                        [
                                            cos_full_mapped[:, :start],
                                            cos_tail_swapped,
                                            cos_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )
                                    even_s = sin_full_mapped[:, start:end:2]
                                    odd_s = sin_full_mapped[:, start + 1 : end : 2]
                                    sin_tail_swapped = torch.stack(
                                        (odd_s, even_s), dim=-1
                                    ).reshape(sin_full_mapped.shape[0], -1)
                                    sin_full_mapped = torch.cat(
                                        [
                                            sin_full_mapped[:, :start],
                                            sin_tail_swapped,
                                            sin_full_mapped[:, end:],
                                        ],
                                        dim=-1,
                                    )
                                if flip_front and front > 0:
                                    sin_full_mapped[:, :front] = -sin_full_mapped[
                                        :, :front
                                    ]
                                if flip_tail and tail > 0:
                                    sin_full_mapped[
                                        :, front : front + tail
                                    ] = -sin_full_mapped[:, front : front + tail]
                                if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                                    "",
                                    "0",
                                    "false",
                                    "False",
                                ):
                                    try:
                                        run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                                        out_dir = os.path.join(
                                            "test_result", "text_stages"
                                        )
                                        os.makedirs(out_dir, exist_ok=True)
                                        with open(
                                            os.path.join(
                                                out_dir,
                                                f"vllm_{run_id}_segment_reorder_applied.txt",
                                            ),
                                            "w",
                                            encoding="utf-8",
                                        ) as f:
                                            f.write(
                                                f"swap_front={swap_front}, "
                                                f"swap_tail={swap_tail}\n"
                                            )
                                            f.write(
                                                f"flip_front={flip_front}, "
                                                f"flip_tail={flip_tail}\n"
                                            )
                                            tmpl = "front={f}, tail={t}, full={u}\n"
                                            msg = tmpl.format(
                                                f=front,
                                                t=tail,
                                                u=cos_full_mapped.shape[-1],
                                            )
                                            f.write(msg)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    # 可选：用 HF dump 的 FULL 表强行替换对应 token（仅调试定位）
                    if os.getenv("PADDLEOCRVL_FORCE_HF_FULL", "0") not in (
                        "",
                        "0",
                        "false",
                        "False",
                    ):
                        try:
                            # 选择同一个代表性 token
                            pos_cpu = positions.detach().cpu()
                            env_last = os.getenv("PADDLEOCRVL_LAST_IDX")
                            if env_last is not None and env_last.strip().isdigit():
                                last_idx = max(
                                    0, min(int(env_last), pos_cpu.shape[-1] - 1)
                                )
                            else:
                                used_mask = (pos_cpu != 0).any(dim=0)
                                axis_diff_mask = (
                                    (pos_cpu[0] != pos_cpu[1])
                                    | (pos_cpu[0] != pos_cpu[2])
                                    | (pos_cpu[1] != pos_cpu[2])
                                )
                                sel = (
                                    (used_mask & axis_diff_mask)
                                    .nonzero(as_tuple=False)
                                    .flatten()
                                )
                                if sel.numel() == 0:
                                    sel = used_mask.nonzero(as_tuple=False).flatten()
                                last_idx = (
                                    int(sel[-1].item())
                                    if sel.numel() > 0
                                    else (pos_cpu.shape[-1] - 1)
                                )

                            import json as _json

                            base = os.getenv("PADDLEOCRVL_HF_FULL_DIR", "test_result")
                            meta_path = os.path.join(base, "hf_mrope_mapping_meta.json")
                            with open(meta_path, encoding="utf-8") as _f:
                                meta = _json.loads(_f.read())
                            sections_full = meta.get("sections_full", [0, 0, 0])
                            sections_full = [
                                int(sections_full[0]),
                                int(sections_full[1]),
                                int(sections_full[2]),
                            ]
                            # 优先直接使用 HF 已映射 FULL（匹配代表性 token 索引的规则）
                            cos_full_path = os.path.join(
                                base, "hf_layer0_cos_full_last.pt"
                            )
                            sin_full_path = os.path.join(
                                base, "hf_layer0_sin_full_last.pt"
                            )
                            if os.path.exists(cos_full_path) and os.path.exists(
                                sin_full_path
                            ):
                                cos_hf_full = torch.load(
                                    cos_full_path, map_location="cpu"
                                ).to(cos_full_mapped.dtype)
                                sin_hf_full = torch.load(
                                    sin_full_path, map_location="cpu"
                                ).to(sin_full_mapped.dtype)
                            else:
                                # 退化：用轴向 FULL 自行映射
                                cos_axes = torch.load(
                                    os.path.join(base, "hf_cos_full_axes_last.pt"),
                                    map_location="cpu",
                                )
                                sin_axes = torch.load(
                                    os.path.join(base, "hf_sin_full_axes_last.pt"),
                                    map_location="cpu",
                                )

                                def _map_axes(
                                    x: torch.Tensor, s: list[int]
                                ) -> torch.Tensor:
                                    tt, hh, ww = int(s[0]), int(s[1]), int(s[2])
                                    return torch.cat(
                                        [x[0, :tt], x[1, :hh], x[2, :ww]], dim=-1
                                    )

                                cos_hf_full = _map_axes(cos_axes, sections_full).to(
                                    cos_full_mapped.dtype
                                )
                                sin_hf_full = _map_axes(sin_axes, sections_full).to(
                                    sin_full_mapped.dtype
                                )
                            # 用 HF 替换当前 token 的 FULL 表
                            cos_full_mapped[last_idx] = cos_hf_full.to(
                                device=cos_full_mapped.device
                            )
                            sin_full_mapped[last_idx] = sin_hf_full.to(
                                device=sin_full_mapped.device
                            )
                        except Exception:
                            pass

                    # Use FULL tables for rotary application (HF-equivalent)
                    full_tables = (cos_full_mapped, sin_full_mapped)
                    # Build HALF tables directly from HALF axes (no compression)
                    cos = apply_hf_chunked_rope_half(
                        cos_base.to(torch.float32), self.mrope_section
                    )
                    sin = apply_hf_chunked_rope_half(
                        sin_base.to(torch.float32), self.mrope_section
                    )

            # Prepare mapped tables for compute (cast), keep FP32 for debugging
            cos_use = cos.to(dtype=query.dtype) if cos.dtype != query.dtype else cos
            sin_use = sin.to(dtype=query.dtype) if sin.dtype != query.dtype else sin
        else:
            # Text-only: use base half tables directly
            cos_use = cos_base
            sin_use = sin_base

            # Optional debug dump of cos/sin for a representative token
            # (once per run)
            if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") != "0" and not getattr(
                self, "_dumped_mrope", False
            ):
                try:
                    run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                    out_dir = os.path.join("test_result", "text_stages")
                    os.makedirs(out_dir, exist_ok=True)
                    pos_cpu = positions.detach().cpu()
                    # 允许手动覆盖 last_idx。
                    # 否则选择“最后一个非 padding 且 T/H/W 不全相等”的 token。
                    env_last = os.getenv("PADDLEOCRVL_LAST_IDX")
                    if env_last is not None and env_last.strip().isdigit():
                        last_idx = max(0, min(int(env_last), pos_cpu.shape[-1] - 1))
                    else:
                        used_mask = (pos_cpu != 0).any(dim=0)
                        axis_diff_mask = (
                            (pos_cpu[0] != pos_cpu[1])
                            | (pos_cpu[0] != pos_cpu[2])
                            | (pos_cpu[1] != pos_cpu[2])
                        )
                        sel = (
                            (used_mask & axis_diff_mask)
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        if sel.numel() == 0:
                            sel = used_mask.nonzero(as_tuple=False).flatten()
                        last_idx = (
                            int(sel[-1].item())
                            if sel.numel() > 0
                            else (pos_cpu.shape[-1] - 1)
                        )
                    torch.save(
                        pos_cpu, os.path.join(out_dir, f"vllm_{run_id}_positions.pt")
                    )
                    torch.save(
                        cos[last_idx].detach().float().cpu(),
                        os.path.join(out_dir, f"vllm_{run_id}_cos_half_last.pt"),
                    )
                    torch.save(
                        sin[last_idx].detach().float().cpu(),
                        os.path.join(out_dir, f"vllm_{run_id}_sin_half_last.pt"),
                    )
                    # Save the actually used FULL tables (from full_tables)
                    # to ensure parity checks reflect runtime mapping.
                    try:
                        if full_tables is None:
                            raise RuntimeError("no_full_tables_available")
                        cos_full_rt, sin_full_rt = full_tables
                        last = last_idx
                        for stem in (
                            "cos_full_last.pt",
                            "cos_full_chunked_last.pt",
                        ):
                            torch.save(
                                cos_full_rt[last].detach().float().cpu(),
                                os.path.join(out_dir, f"vllm_{run_id}_{stem}"),
                            )
                        for stem in (
                            "sin_full_last.pt",
                            "sin_full_chunked_last.pt",
                        ):
                            torch.save(
                                sin_full_rt[last].detach().float().cpu(),
                                os.path.join(out_dir, f"vllm_{run_id}_{stem}"),
                            )
                    except Exception:
                        pass
                    self._dumped_mrope = True
                except Exception:
                    pass

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        if full_tables is not None:
            # Apply FULL-dim rotary to mirror HF exactly
            cos_full, sin_full = full_tables
            cos_b = cos_full.to(dtype=query.dtype).unsqueeze(-2)
            sin_b = sin_full.to(dtype=query.dtype).unsqueeze(-2)
            # Optional introspection: confirm FULL path and dtype alignment
            if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                "",
                "0",
                "false",
                "False",
            ):
                try:
                    run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                    out_dir = os.path.join("test_result", "text_stages")
                    os.makedirs(out_dir, exist_ok=True)
                    with open(
                        os.path.join(out_dir, f"vllm_{run_id}_mrope_path.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write("path=FULL\n")
                        f.write(f"cos_full_shape={tuple(cos_full.shape)}\n")
                        f.write(f"sin_full_shape={tuple(sin_full.shape)}\n")
                        f.write(f"query_dtype={str(query.dtype)}\n")
                        f.write(f"cos_b_dtype={str(cos_b.dtype)}\n")
                except Exception:
                    pass
            # Extra debug for multi-image mRoPE parity (env-gated)
            try:
                if os.getenv("DEBUG_MROPE_MULTI_IMAGE", "0") not in (
                    "",
                    "0",
                    "false",
                    "False",
                ) and not getattr(self, "_dumped_mrope_mm", False):
                    # Dump inputs/rope tables once (treat as layer 0 snapshot)
                    out_dir = os.path.join("test_result", "mrope_debug")
                    os.makedirs(out_dir, exist_ok=True)
                    # positions as provided (T/H/W)
                    torch.save(
                        positions.detach().cpu(),
                        os.path.join(out_dir, "position_ids_vllm.pt"),
                    )
                    # FULL tables currently selected
                    torch.save(
                        cos_full.detach().float().cpu(),
                        os.path.join(out_dir, "cos_full_vllm.pt"),
                    )
                    torch.save(
                        sin_full.detach().float().cpu(),
                        os.path.join(out_dir, "sin_full_vllm.pt"),
                    )
                    # Also persist simple FULL chunks based on mrope_section
                    try:
                        assert self.mrope_section is not None
                        t_len, h_len, w_len = (
                            int(self.mrope_section[0]),
                            int(self.mrope_section[1]),
                            int(self.mrope_section[2]),
                        )
                        front = 2 * (h_len + w_len)
                        tail = 2 * t_len
                        cos_front = cos_full[:, :front].detach().float().cpu()
                        cos_tail = cos_full[:, -tail:] if tail > 0 else cos_full[:, :0]
                        sin_front = sin_full[:, :front].detach().float().cpu()
                        sin_tail = sin_full[:, -tail:] if tail > 0 else sin_full[:, :0]
                        cos_front_path = os.path.join(out_dir, "cos_full_front_vllm.pt")
                        cos_tail_path = os.path.join(out_dir, "cos_full_tail_vllm.pt")
                        sin_front_path = os.path.join(out_dir, "sin_full_front_vllm.pt")
                        sin_tail_path = os.path.join(out_dir, "sin_full_tail_vllm.pt")
                        torch.save(cos_front, cos_front_path)
                        torch.save(cos_tail.detach().float().cpu(), cos_tail_path)
                        torch.save(sin_front, sin_front_path)
                        torch.save(sin_tail.detach().float().cpu(), sin_tail_path)
                    except Exception:
                        pass
                    self._dumped_mrope_mm = True
            except Exception:
                pass
            # Optional strict dtype check (off by default)
            if os.getenv("PADDLEOCRVL_ASSERT_DTYPE", "0") not in (
                "",
                "0",
                "false",
                "False",
            ):
                assert query.dtype == cos_b.dtype, "mRoPE FULL path dtype mismatch"
            # 允许调试覆盖旋转风格：neox/gptj
            _rotate_style = os.getenv("PADDLEOCRVL_ROTATE_STYLE", "").lower()
            use_neox = (
                self.is_neox_style if _rotate_style == "" else (_rotate_style == "neox")
            )
            rot_fn = rotate_neox if use_neox else rotate_gptj
            # Optional mixed rotate style: use GPT-J on [T|H] front, NeoX on [W] tail
            if (
                os.getenv("PADDLEOCRVL_MIX_GPTJ_TH", "0")
                not in ("", "0", "false", "False")
                and self.mrope_section
            ):
                t, h, _w = (
                    int(self.mrope_section[0]),
                    int(self.mrope_section[1]),
                    int(self.mrope_section[2]),
                )
                front = 2 * (t + h)
                r_neox_q = rotate_neox(query_rot)
                r_gptj_q = rotate_gptj(query_rot)
                r_mix_q = torch.cat(
                    (r_gptj_q[..., :front], r_neox_q[..., front:]), dim=-1
                )
                query_rot = query_rot * cos_b + r_mix_q * sin_b
                r_neox_k = rotate_neox(key_rot)
                r_gptj_k = rotate_gptj(key_rot)
                r_mix_k = torch.cat(
                    (r_gptj_k[..., :front], r_neox_k[..., front:]), dim=-1
                )
                key_rot = key_rot * cos_b + r_mix_k * sin_b
            else:
                query_rot = query_rot * cos_b + rot_fn(query_rot) * sin_b
                key_rot = key_rot * cos_b + rot_fn(key_rot) * sin_b
            # 可选自校验：手工计算与实际结果的一致性（最后一个代表性 token）
            if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                "",
                "0",
                "false",
                "False",
            ):
                try:
                    import torch.nn.functional as _F

                    run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                    out_dir = os.path.join("test_result", "text_stages")
                    os.makedirs(out_dir, exist_ok=True)
                    # 选取代表性 token
                    pos_cpu = positions.detach().cpu()
                    env_last = os.getenv("PADDLEOCRVL_LAST_IDX")
                    if env_last is not None and env_last.strip().isdigit():
                        last_idx = max(0, min(int(env_last), pos_cpu.shape[-1] - 1))
                    else:
                        used_mask = (pos_cpu != 0).any(dim=0)
                        axis_diff_mask = (
                            (pos_cpu[0] != pos_cpu[1])
                            | (pos_cpu[0] != pos_cpu[2])
                            | (pos_cpu[1] != pos_cpu[2])
                        )
                        sel = (
                            (used_mask & axis_diff_mask)
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        if sel.numel() == 0:
                            sel = used_mask.nonzero(as_tuple=False).flatten()
                        last_idx = (
                            int(sel[-1].item())
                            if sel.numel() > 0
                            else (pos_cpu.shape[-1] - 1)
                        )
                    # 手工计算
                    q_pre = query.view(num_tokens, -1, self.head_size)[
                        last_idx, :, : self.rotary_dim
                    ]
                    cos_last = cos_full[last_idx].to(dtype=query.dtype)
                    sin_last = sin_full[last_idx].to(dtype=query.dtype)
                    q_manual = q_pre * cos_last.unsqueeze(0) + rot_fn(
                        q_pre
                    ) * sin_last.unsqueeze(0)
                    q_actual = query_rot[last_idx]
                    cos_sim = _F.cosine_similarity(
                        q_manual.reshape(1, -1).float(), q_actual.reshape(1, -1).float()
                    ).item()
                    with open(
                        os.path.join(
                            out_dir, f"vllm_{run_id}_manual_vs_actual_q_cos.txt"
                        ),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"cos={cos_sim}\n")

                    # Additional per-axis segment diagnostics (T/H/W)
                    try:
                        if self.mrope_section:
                            t, h, w = (
                                int(self.mrope_section[0]),
                                int(self.mrope_section[1]),
                                int(self.mrope_section[2]),
                            )
                            # FULL split sizes in pairs; ensure boundaries are even.
                            sections_full = [2 * t, 2 * h, 2 * w]

                            # Slice [T|H|W] on FULL dim for both manual/actual.
                            def _seg(x: torch.Tensor) -> list[torch.Tensor]:
                                parts = x.split(sections_full, dim=-1)
                                return [parts[0], parts[1], parts[2]]

                            q_man_last = q_manual.reshape(-1, self.rotary_dim)[0]
                            q_act_last = q_actual.reshape(-1, self.rotary_dim)
                            # Broadcast heads -> flatten
                            q_act_last = q_act_last.flatten()
                            man_T, man_H, man_W = _seg(q_man_last)
                            act_T, act_H, act_W = _seg(q_act_last)
                            seg_cos = {
                                "T": float(
                                    _F.cosine_similarity(
                                        man_T.float().unsqueeze(0),
                                        act_T.float().unsqueeze(0),
                                        dim=1,
                                    ).item()
                                ),
                                "H": float(
                                    _F.cosine_similarity(
                                        man_H.float().unsqueeze(0),
                                        act_H.float().unsqueeze(0),
                                        dim=1,
                                    ).item()
                                ),
                                "W": float(
                                    _F.cosine_similarity(
                                        man_W.float().unsqueeze(0),
                                        act_W.float().unsqueeze(0),
                                        dim=1,
                                    ).item()
                                ),
                            }
                            with open(
                                os.path.join(
                                    out_dir,
                                    f"vllm_{run_id}_manual_vs_actual_q_segments.json",
                                ),
                                "w",
                                encoding="utf-8",
                            ) as jf:
                                import json as _json

                                _json.dump(seg_cos, jf, indent=2)
                    except Exception:
                        pass
                except Exception:
                    pass
        else:
            # Use HALF-dim dispatch
            query_rot = apply_rotary_emb_dispatch(
                query_rot, cos_use, sin_use, self.is_neox_style
            )
            key_rot = apply_rotary_emb_dispatch(
                key_rot, cos_use, sin_use, self.is_neox_style
            )

        # Post-RoPE tensors
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        # Dump post-RoPE Q/K once when enabled
        try:
            if os.getenv("DEBUG_MROPE_MULTI_IMAGE", "0") not in (
                "",
                "0",
                "false",
                "False",
            ) and not getattr(self, "_dumped_post_rope_mm", False):
                out_dir = os.path.join("test_result", "mrope_debug")
                os.makedirs(out_dir, exist_ok=True)
                import torch.nn.functional as _F

                torch.save(
                    query.detach().float().cpu(),
                    os.path.join(out_dir, "q_post_rope_vllm.pt"),
                )
                torch.save(
                    key.detach().float().cpu(),
                    os.path.join(out_dir, "k_post_rope_vllm.pt"),
                )
                try:
                    cs = float(
                        _F.cosine_similarity(
                            query.flatten().float(), key.flatten().float(), dim=0
                        ).item()
                    )
                except Exception:
                    cs = float("nan")
                with open(
                    os.path.join(out_dir, "post_rope_qk_cosine_vllm.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(f"cosine_similarity={cs}\n")
                self._dumped_post_rope_mm = True
        except Exception:
            pass
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos_base, sin_base = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        full_tables_cuda: tuple[torch.Tensor, torch.Tensor] | None = None
        if positions.ndim == 2:
            assert self.mrope_section
            if self.mrope_interleaved:
                cos_half3_fp32 = cos_base.to(torch.float32)
                sin_half3_fp32 = sin_base.to(torch.float32)
                cos_h = apply_interleaved_rope(cos_half3_fp32, self.mrope_section)
                sin_h = apply_interleaved_rope(sin_half3_fp32, self.mrope_section)
            else:
                mapping = os.getenv("PADDLEOCRVL_MROPE_MAP", "hf").lower()
                if mapping == "paddle":
                    cos_half3_fp32 = cos_base.to(torch.float32)
                    sin_half3_fp32 = sin_base.to(torch.float32)
                    use_half_dup = os.getenv(
                        "PADDLEOCRVL_USE_HALF_DUP", "0"
                    ).lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    )
                    cos_h = apply_paddle_hw_then_t_half(
                        cos_half3_fp32, self.mrope_section
                    )
                    sin_h = apply_paddle_hw_then_t_half(
                        sin_half3_fp32, self.mrope_section
                    )
                    cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)
                    sin_full3 = torch.cat([sin_half3_fp32, sin_half3_fp32], dim=-1)
                    if use_half_dup:
                        cos_full_mapped = torch.cat([cos_h, cos_h], dim=-1)
                        sin_full_mapped = torch.cat([sin_h, sin_h], dim=-1)
                    else:
                        cos_full_mapped = apply_paddle_hw_then_t_full(
                            cos_full3, self.mrope_section
                        )
                        sin_full_mapped = apply_paddle_hw_then_t_full(
                            sin_full3, self.mrope_section
                        )
                else:
                    # HF Chunked（严格一致）：在 FULL 维度上按 [2t,2h,2w] 分段并对角选择。
                    cos_half3_fp32 = cos_base.to(torch.float32)
                    sin_half3_fp32 = sin_base.to(torch.float32)

                    # HALF‑dispatch 备用表（HALF 上的分段对角映射）
                    cos_h = apply_hf_chunked_rope_half(
                        cos_half3_fp32, self.mrope_section
                    )
                    sin_h = apply_hf_chunked_rope_half(
                        sin_half3_fp32, self.mrope_section
                    )

                    # FULL 轴向表（Neox pairing）
                    cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)
                    sin_full3 = torch.cat([sin_half3_fp32, sin_half3_fp32], dim=-1)
                    t, h, w = map(int, self.mrope_section)
                    sections_full = [2 * t, 2 * h, 2 * w]
                    c_t, c_h, c_w = cos_full3.split(sections_full, dim=-1)
                    s_t, s_h, s_w = sin_full3.split(sections_full, dim=-1)
                    cos_full_mapped = torch.cat([c_t[0], c_h[1], c_w[2]], dim=-1)
                    sin_full_mapped = torch.cat([s_t[0], s_h[1], s_w[2]], dim=-1)
                    if getattr(self, "mscale", 1.0) != 1.0:
                        cos_full_mapped = cos_full_mapped * float(self.mscale)
                        sin_full_mapped = sin_full_mapped * float(self.mscale)
                    # 可选：HF FULL 表替换（CUDA 分支）
                    if os.getenv("PADDLEOCRVL_FORCE_HF_FULL", "0") not in (
                        "",
                        "0",
                        "false",
                        "False",
                    ):
                        try:
                            import json as _json

                            base = os.getenv("PADDLEOCRVL_HF_FULL_DIR", "test_result")
                            meta_path = os.path.join(base, "hf_mrope_mapping_meta.json")
                            with open(meta_path, encoding="utf-8") as _f:
                                meta = _json.loads(_f.read())
                            # sections_full 仅用于轴向回退映射，不依赖 t/h/w 符号
                            sfull = meta.get("sections_full", [0, 0, 0])
                            sfull = [int(sfull[0]), int(sfull[1]), int(sfull[2])]
                            cos_full_path = os.path.join(
                                base, "hf_layer0_cos_full_last.pt"
                            )
                            sin_full_path = os.path.join(
                                base, "hf_layer0_sin_full_last.pt"
                            )
                            if os.path.exists(cos_full_path) and os.path.exists(
                                sin_full_path
                            ):
                                cos_hf = torch.load(
                                    cos_full_path, map_location="cpu"
                                ).to(
                                    dtype=cos_full_mapped.dtype,
                                    device=cos_full_mapped.device,
                                )
                                sin_hf = torch.load(
                                    sin_full_path, map_location="cpu"
                                ).to(
                                    dtype=sin_full_mapped.dtype,
                                    device=sin_full_mapped.device,
                                )
                            else:
                                cos_axes = torch.load(
                                    os.path.join(base, "hf_cos_full_axes_last.pt"),
                                    map_location="cpu",
                                )
                                sin_axes = torch.load(
                                    os.path.join(base, "hf_sin_full_axes_last.pt"),
                                    map_location="cpu",
                                )

                                def _map_axes(
                                    x: torch.Tensor, s: list[int]
                                ) -> torch.Tensor:
                                    tt, hh, ww = int(s[0]), int(s[1]), int(s[2])
                                    return torch.cat(
                                        [x[0, :tt], x[1, :hh], x[2, :ww]], dim=-1
                                    )

                                cos_hf = _map_axes(cos_axes, sfull).to(
                                    dtype=cos_full_mapped.dtype,
                                    device=cos_full_mapped.device,
                                )
                                sin_hf = _map_axes(sin_axes, sfull).to(
                                    dtype=sin_full_mapped.dtype,
                                    device=sin_full_mapped.device,
                                )
                            # last_idx 选择
                            pos_cpu = positions.detach().cpu()
                            env_last = os.getenv("PADDLEOCRVL_LAST_IDX")
                            if env_last is not None and env_last.strip().isdigit():
                                last_idx = max(
                                    0, min(int(env_last), pos_cpu.shape[-1] - 1)
                                )
                            else:
                                used_mask = (pos_cpu != 0).any(dim=0)
                                axis_diff_mask = (
                                    (pos_cpu[0] != pos_cpu[1])
                                    | (pos_cpu[0] != pos_cpu[2])
                                    | (pos_cpu[1] != pos_cpu[2])
                                )
                                sel = (
                                    (used_mask & axis_diff_mask)
                                    .nonzero(as_tuple=False)
                                    .flatten()
                                )
                                if sel.numel() == 0:
                                    sel = used_mask.nonzero(as_tuple=False).flatten()
                                last_idx = (
                                    int(sel[-1].item())
                                    if sel.numel() > 0
                                    else (pos_cpu.shape[-1] - 1)
                                )
                            cos_full_mapped[last_idx] = cos_hf
                            sin_full_mapped[last_idx] = sin_hf
                        except Exception:
                            pass
                    full_tables_cuda = (cos_full_mapped, sin_full_mapped)
                    # Extra debug (CUDA path): dump position ids and FULL tables once
                    try:
                        if os.getenv("DEBUG_MROPE_MULTI_IMAGE", "0") not in (
                            "",
                            "0",
                            "false",
                            "False",
                        ) and not getattr(self, "_dumped_mrope_mm_cuda", False):
                            out_dir = os.path.join("test_result", "mrope_debug")
                            os.makedirs(out_dir, exist_ok=True)
                            torch.save(
                                positions.detach().cpu(),
                                os.path.join(out_dir, "position_ids_vllm.pt"),
                            )
                            torch.save(
                                cos_full_mapped.detach().float().cpu(),
                                os.path.join(out_dir, "cos_full_vllm.pt"),
                            )
                            torch.save(
                                sin_full_mapped.detach().float().cpu(),
                                os.path.join(out_dir, "sin_full_vllm.pt"),
                            )
                            self._dumped_mrope_mm_cuda = True
                    except Exception:
                        pass
                    # HALF mapping directly from HALF axes
                    cos_h = apply_hf_chunked_rope_half(
                        cos_base.to(torch.float32), self.mrope_section
                    )
                    sin_h = apply_hf_chunked_rope_half(
                        sin_base.to(torch.float32), self.mrope_section
                    )

            # Prepare mapped tables for compute (cast), keep FP32 for debugging
            cos_h_use = (
                cos_h.to(dtype=query.dtype) if cos_h.dtype != query.dtype else cos_h
            )
            sin_h_use = (
                sin_h.to(dtype=query.dtype) if sin_h.dtype != query.dtype else sin_h
            )

            # Apply rotary; prefer FULL-dim path for exact HF semantics
            query = query.view(num_tokens, -1, self.head_size)
            key = key.view(num_tokens, -1, self.head_size)
            query_rot = query[..., : self.rotary_dim]
            query_pass = query[..., self.rotary_dim :]
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            if full_tables_cuda is not None:
                cos_full, sin_full = full_tables_cuda
                cos_b = cos_full.to(dtype=query.dtype).unsqueeze(-2)
                sin_b = sin_full.to(dtype=query.dtype).unsqueeze(-2)
                # Optional introspection: confirm CUDA FULL path was taken
                if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                    "",
                    "0",
                    "false",
                    "False",
                ):
                    try:
                        run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                        out_dir = os.path.join("test_result", "text_stages")
                        os.makedirs(out_dir, exist_ok=True)
                        with open(
                            os.path.join(out_dir, f"vllm_{run_id}_mrope_path_cuda.txt"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write("path=FULL_CUDA\n")
                            f.write(f"cos_full_shape={tuple(cos_full.shape)}\n")
                            f.write(f"query_dtype={str(query.dtype)}\n")
                            f.write(f"cos_b_dtype={str(cos_b.dtype)}\n")
                    except Exception:
                        pass
                _rotate_style = os.getenv("PADDLEOCRVL_ROTATE_STYLE", "").lower()
                use_neox = (
                    self.is_neox_style
                    if _rotate_style == ""
                    else (_rotate_style == "neox")
                )
                rot_fn = rotate_neox if use_neox else rotate_gptj
                if (
                    os.getenv("PADDLEOCRVL_MIX_GPTJ_TH", "0")
                    not in ("", "0", "false", "False")
                    and self.mrope_section
                ):
                    t, h, _w = (
                        int(self.mrope_section[0]),
                        int(self.mrope_section[1]),
                        int(self.mrope_section[2]),
                    )
                    front = 2 * (t + h)
                    r_neox_q = rotate_neox(query_rot)
                    r_gptj_q = rotate_gptj(query_rot)
                    r_mix_q = torch.cat(
                        (r_gptj_q[..., :front], r_neox_q[..., front:]), dim=-1
                    )
                    query_rot = query_rot * cos_b + r_mix_q * sin_b
                    r_neox_k = rotate_neox(key_rot)
                    r_gptj_k = rotate_gptj(key_rot)
                    r_mix_k = torch.cat(
                        (r_gptj_k[..., :front], r_neox_k[..., front:]), dim=-1
                    )
                    key_rot = key_rot * cos_b + r_mix_k * sin_b
                else:
                    query_rot = query_rot * cos_b + rot_fn(query_rot) * sin_b
                    key_rot = key_rot * cos_b + rot_fn(key_rot) * sin_b
                # 可选自校验：CUDA 路径手工 vs 实际
                if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") not in (
                    "",
                    "0",
                    "false",
                    "False",
                ):
                    try:
                        import torch.nn.functional as _F

                        run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                        out_dir = os.path.join("test_result", "text_stages")
                        os.makedirs(out_dir, exist_ok=True)
                        pos_cpu = positions.detach().cpu()
                        env_last = os.getenv("PADDLEOCRVL_LAST_IDX")
                        if env_last is not None and env_last.strip().isdigit():
                            last_idx = max(0, min(int(env_last), pos_cpu.shape[-1] - 1))
                        else:
                            used_mask = (pos_cpu != 0).any(dim=0)
                            axis_diff_mask = (
                                (pos_cpu[0] != pos_cpu[1])
                                | (pos_cpu[0] != pos_cpu[2])
                                | (pos_cpu[1] != pos_cpu[2])
                            )
                            sel = (
                                (used_mask & axis_diff_mask)
                                .nonzero(as_tuple=False)
                                .flatten()
                            )
                            if sel.numel() == 0:
                                sel = used_mask.nonzero(as_tuple=False).flatten()
                            last_idx = (
                                int(sel[-1].item())
                                if sel.numel() > 0
                                else (pos_cpu.shape[-1] - 1)
                            )
                        q_pre = query.view(num_tokens, -1, self.head_size)[
                            last_idx, :, : self.rotary_dim
                        ]
                        cos_last = cos_full.to(dtype=query.dtype)[last_idx]
                        sin_last = sin_full.to(dtype=query.dtype)[last_idx]
                        q_manual = q_pre * cos_last.unsqueeze(0) + rot_fn(
                            q_pre
                        ) * sin_last.unsqueeze(0)
                        q_actual = query_rot[last_idx]
                        cos_sim = _F.cosine_similarity(
                            q_manual.reshape(1, -1).float(),
                            q_actual.reshape(1, -1).float(),
                        ).item()
                        with open(
                            os.path.join(
                                out_dir, f"vllm_{run_id}_manual_vs_actual_q_cos.txt"
                            ),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(f"cos={cos_sim}\n")
                        # Per-axis segment diagnostics (T/H/W) on CUDA path
                        try:
                            if self.mrope_section:
                                t, h, w = (
                                    int(self.mrope_section[0]),
                                    int(self.mrope_section[1]),
                                    int(self.mrope_section[2]),
                                )
                                # Flatten all heads, then slice per-head segments
                                q_man_flat = q_manual.reshape(-1)
                                q_act_flat = q_actual.reshape(-1)
                                hd_full = int(self.rotary_dim)
                                n_elems = int(q_man_flat.numel())
                                assert n_elems == int(q_act_flat.numel())
                                assert n_elems % hd_full == 0
                                n_heads_flat = n_elems // hd_full
                                import torch as _tc

                                idx = _tc.arange(n_elems, device=q_man_flat.device)
                                pos_in_head = idx % hd_full
                                t_end = 2 * t
                                h_end = t_end + 2 * h
                                mask_T = pos_in_head < t_end
                                mask_H = (pos_in_head >= t_end) & (pos_in_head < h_end)
                                mask_W = pos_in_head >= h_end
                                man_T = q_man_flat[mask_T]
                                man_H = q_man_flat[mask_H]
                                man_W = q_man_flat[mask_W]
                                act_T = q_act_flat[mask_T]
                                act_H = q_act_flat[mask_H]
                                act_W = q_act_flat[mask_W]
                                seg_cos = {
                                    "T": float(
                                        _F.cosine_similarity(
                                            man_T.float().unsqueeze(0),
                                            act_T.float().unsqueeze(0),
                                            dim=1,
                                        ).item()
                                    ),
                                    "H": float(
                                        _F.cosine_similarity(
                                            man_H.float().unsqueeze(0),
                                            act_H.float().unsqueeze(0),
                                            dim=1,
                                        ).item()
                                    ),
                                    "W": float(
                                        _F.cosine_similarity(
                                            man_W.float().unsqueeze(0),
                                            act_W.float().unsqueeze(0),
                                            dim=1,
                                        ).item()
                                    ),
                                }
                                with open(
                                    os.path.join(
                                        out_dir,
                                        f"vllm_{run_id}_manual_vs_actual_q_segments.json",
                                    ),
                                    "w",
                                    encoding="utf-8",
                                ) as jf:
                                    import json as _json

                                    _json.dump(seg_cos, jf, indent=2)
                        except Exception:
                            pass
                    except Exception:
                        pass
            else:
                query_rot = apply_rotary_emb_dispatch(
                    query_rot, cos_h_use, sin_h_use, self.is_neox_style
                )
                key_rot = apply_rotary_emb_dispatch(
                    key_rot, cos_h_use, sin_h_use, self.is_neox_style
                )
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

            # Optional debug dump on CUDA path
            if os.getenv("PADDLEOCRVL_DEBUG_MROPE", "0") != "0" and not getattr(
                self, "_dumped_mrope", False
            ):
                try:
                    run_id = os.getenv("PADDLEOCRVL_RUN_ID", "run")
                    out_dir = os.path.join("test_result", "text_stages")
                    os.makedirs(out_dir, exist_ok=True)
                    pos_cpu = positions.detach().cpu()
                    # Heuristic: only dump when we detect non-zero positions
                    used_mask = (pos_cpu != 0).any(dim=0)
                    used_len = int(used_mask.sum().item())
                    if used_len == 0:
                        # Allow forced dump for diagnostics when positions are all-zero
                        # (e.g., on dummy profiling passes).
                        if os.getenv("PADDLEOCRVL_FORCE_DUMP", "0") != "0":
                            used_len = pos_cpu.shape[-1]
                        else:
                            # Skip dump so a later real forward
                            # can produce meaningful dumps.
                            raise RuntimeError("padding_only_positions")
                    torch.save(
                        pos_cpu,
                        os.path.join(out_dir, f"vllm_{run_id}_positions.pt"),
                    )
                    # Identify a representative non-padded token index where
                    # T/H/W positions are not all identical, if possible.
                    last_idx = max(0, used_len - 1)
                    try:
                        for col in range(used_len - 1, -1, -1):
                            col_vals = pos_cpu[:, col]
                            if not torch.equal(
                                col_vals[0], col_vals[1]
                            ) or not torch.equal(col_vals[0], col_vals[2]):
                                last_idx = col
                                break
                    except Exception:
                        pass
                    # Save raw axis-wise HALF tables (pre-mapping) for last token
                    try:
                        torch.save(
                            cos_half3_fp32[:, last_idx].detach().float().cpu(),
                            os.path.join(
                                out_dir, f"vllm_{run_id}_cos_full_axes_last.pt"
                            ),
                        )
                        torch.save(
                            sin_half3_fp32[:, last_idx].detach().float().cpu(),
                            os.path.join(
                                out_dir, f"vllm_{run_id}_sin_full_axes_last.pt"
                            ),
                        )
                    except Exception:
                        pass
                    # Manual vs actual (FULL) diagnostics on CUDA path
                    try:
                        import torch.nn.functional as _F

                        # Build manual q using FULL tables at last_idx
                        q_pre = query.view(num_tokens, -1, self.head_size)[
                            last_idx, :, : self.rotary_dim
                        ]
                        # Prefer runtime FULL mapping if available; otherwise
                        # rebuild from base FULL-axes splits for the current token.
                        has_rt = (
                            "cos_full_mapped" in locals()
                            and "sin_full_mapped" in locals()
                        )
                        if has_rt:
                            cos_full_rt = cos_full_mapped
                            sin_full_rt = sin_full_mapped
                        else:
                            # Rebuild mapped FULL using FULL-axes tables
                            # (cos_full3/sin_full3 exist above when dumping FULL).
                            t, h, w = (
                                int(self.mrope_section[0]),
                                int(self.mrope_section[1]),
                                int(self.mrope_section[2]),
                            )
                            sections_full = [2 * t, 2 * h, 2 * w]
                            cos_chunks = cos_full3.split(sections_full, dim=-1)
                            sin_chunks = sin_full3.split(sections_full, dim=-1)
                            cos_full_rt = torch.cat(
                                [cos_chunks[0][0], cos_chunks[1][1], cos_chunks[2][2]],
                                dim=-1,
                            )
                            sin_full_rt = torch.cat(
                                [sin_chunks[0][0], sin_chunks[1][1], sin_chunks[2][2]],
                                dim=-1,
                            )
                        cos_last = cos_full_rt[last_idx].to(dtype=query.dtype)
                        sin_last = sin_full_rt[last_idx].to(dtype=query.dtype)
                        q_manual = q_pre * cos_last.unsqueeze(0) + rot_fn(
                            q_pre
                        ) * sin_last.unsqueeze(0)
                        q_actual = query_rot[last_idx]
                        cos_sim = _F.cosine_similarity(
                            q_manual.reshape(1, -1).float(),
                            q_actual.reshape(1, -1).float(),
                        ).item()
                        with open(
                            os.path.join(
                                out_dir, f"vllm_{run_id}_manual_vs_actual_q_cos.txt"
                            ),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(f"cos={cos_sim}\n")
                        # Per-axis segments
                        try:
                            if self.mrope_section:
                                t, h, w = (
                                    int(self.mrope_section[0]),
                                    int(self.mrope_section[1]),
                                    int(self.mrope_section[2]),
                                )
                                hd_full = int(self.rotary_dim)
                                q_man_flat = q_manual.reshape(-1)
                                q_act_flat = q_actual.reshape(-1)
                                n_elems = int(q_man_flat.numel())
                                if (
                                    n_elems == int(q_act_flat.numel())
                                    and n_elems % hd_full == 0
                                ):
                                    import torch as _tc

                                    idx = _tc.arange(n_elems, device=q_man_flat.device)
                                    pos_in_head = idx % hd_full
                                    t_end = 2 * t
                                    h_end = t_end + 2 * h
                                    mask_T = pos_in_head < t_end
                                    mask_H = (pos_in_head >= t_end) & (
                                        pos_in_head < h_end
                                    )
                                    mask_W = pos_in_head >= h_end
                                    man_T = q_man_flat[mask_T]
                                    man_H = q_man_flat[mask_H]
                                    man_W = q_man_flat[mask_W]
                                    act_T = q_act_flat[mask_T]
                                    act_H = q_act_flat[mask_H]
                                    act_W = q_act_flat[mask_W]
                                    seg_cos = {
                                        "T": float(
                                            _F.cosine_similarity(
                                                man_T.float().unsqueeze(0),
                                                act_T.float().unsqueeze(0),
                                                dim=1,
                                            ).item()
                                        ),
                                        "H": float(
                                            _F.cosine_similarity(
                                                man_H.float().unsqueeze(0),
                                                act_H.float().unsqueeze(0),
                                                dim=1,
                                            ).item()
                                        ),
                                        "W": float(
                                            _F.cosine_similarity(
                                                man_W.float().unsqueeze(0),
                                                act_W.float().unsqueeze(0),
                                                dim=1,
                                            ).item()
                                        ),
                                    }
                                    import json as _json

                                    with open(
                                        os.path.join(
                                            out_dir,
                                            f"vllm_{run_id}_manual_vs_actual_q_segments.json",
                                        ),
                                        "w",
                                        encoding="utf-8",
                                    ) as jf:
                                        _json.dump(seg_cos, jf, indent=2)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    torch.save(
                        (cos_h[last_idx] if last_idx >= 0 else cos_h[-1])
                        .detach()
                        .float()
                        .cpu(),
                        os.path.join(out_dir, f"vllm_{run_id}_cos_half_last.pt"),
                    )
                    torch.save(
                        (sin_h[last_idx] if last_idx >= 0 else sin_h[-1])
                        .detach()
                        .float()
                        .cpu(),
                        os.path.join(out_dir, f"vllm_{run_id}_sin_half_last.pt"),
                    )
                    # Dump the actually used FULL mapping
                    # (cos_full_mapped/sin_full_mapped) so downstream
                    # comparators match the runtime path exactly.
                    try:
                        cos_last_full = cos_full_mapped[last_idx].detach().float().cpu()
                        sin_last_full = sin_full_mapped[last_idx].detach().float().cpu()
                        # Save under multiple compatible names
                        for stem in (
                            "cos_full_last.pt",
                            "cos_full_chunked_last.pt",
                            "cos_full_six_chunks_last.pt",
                        ):
                            torch.save(
                                cos_last_full,
                                os.path.join(out_dir, f"vllm_{run_id}_{stem}"),
                            )
                        for stem in (
                            "sin_full_last.pt",
                            "sin_full_chunked_last.pt",
                            "sin_full_six_chunks_last.pt",
                        ):
                            torch.save(
                                sin_last_full,
                                os.path.join(out_dir, f"vllm_{run_id}_{stem}"),
                            )
                    except Exception:
                        pass
                    # Also dump a "direct concat" half table [T|H|W] for diagnosis
                    try:
                        t, h, w = map(int, self.mrope_section)
                        # Derive direct-concat from FULL tables for consistency
                        # with the main mapping path, then compress to HALF.
                        cos_half3_dbg = cos_full3[..., ::2]
                        sin_half3_dbg = sin_full3[..., ::2]
                        cos_direct = torch.cat(
                            [
                                cos_half3_dbg[0, last_idx, :t],
                                cos_half3_dbg[1, last_idx, :h],
                                cos_half3_dbg[2, last_idx, :w],
                            ],
                            dim=-1,
                        )
                        sin_direct = torch.cat(
                            [
                                sin_half3_dbg[0, last_idx, :t],
                                sin_half3_dbg[1, last_idx, :h],
                                sin_half3_dbg[2, last_idx, :w],
                            ],
                            dim=-1,
                        )
                        torch.save(
                            cos_direct.detach().float().cpu(),
                            os.path.join(
                                out_dir, f"vllm_{run_id}_cos_half_last_direct.pt"
                            ),
                        )
                        torch.save(
                            sin_direct.detach().float().cpu(),
                            os.path.join(
                                out_dir, f"vllm_{run_id}_sin_half_last_direct.pt"
                            ),
                        )
                    except Exception:
                        pass
                    self._dumped_mrope = True
                except Exception:
                    pass

            # CUDA path post-rope dumps
            try:
                if os.getenv("DEBUG_MROPE_MULTI_IMAGE", "0") not in (
                    "",
                    "0",
                    "false",
                    "False",
                ) and not getattr(self, "_dumped_post_rope_mm_cuda", False):
                    out_dir = os.path.join("test_result", "mrope_debug")
                    os.makedirs(out_dir, exist_ok=True)
                    import torch.nn.functional as _F

                    torch.save(
                        query.detach().float().cpu(),
                        os.path.join(out_dir, "q_post_rope_vllm.pt"),
                    )
                    torch.save(
                        key.detach().float().cpu(),
                        os.path.join(out_dir, "k_post_rope_vllm.pt"),
                    )
                    try:
                        cs = float(
                            _F.cosine_similarity(
                                query.flatten().float(), key.flatten().float(), dim=0
                            ).item()
                        )
                    except Exception:
                        cs = float("nan")
                    with open(
                        os.path.join(out_dir, "post_rope_qk_cosine_vllm.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"cosine_similarity={cs}\n")
                    self._dumped_post_rope_mm_cuda = True
            except Exception:
                pass
            return query, key

        # Text-only path: use base half tables for rotary application
        cos = cos_base
        sin = sin_base
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(
                    context_len + mrope_position_delta, seq_len + mrope_position_delta
                )
            )
            for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ):
        values = np.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values
