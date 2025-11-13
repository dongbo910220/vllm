# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PaddleOCR-VL dedicated vision tower implementation.

This module implements a fully dedicated vision encoder for PaddleOCR-VL that:
1. Handles HF's pre-split patch format ([B, L, 3, pH, pW])
2. Maintains HF implementation parity for all vision components
3. Isolates PaddleOCR-VL from standard SigLIP to avoid cross-model interference

Architecture follows HF's modeling_paddleocr_vl.py implementation closely.

Debugging notes:
- When the environment variable PADDLEOCRVL_DEBUG_DUMP is set to "1" or "2",
  the encoder emits stage-wise dumps under test_result/paddleocrvl_stages/.
- You can narrow which stages to dump with PADDLEOCRVL_DUMP_STAGES (comma-
  separated names or "*" for all), and cap per-layer dumps with
  PADDLEOCRVL_DUMP_MAX_LAYERS (default 2).
- With value "1", only stats (shape/mean/std) are saved. With value "2", the
  full tensors are also persisted in the dump dict for offline cosine checks.
"""

import os
import time
from contextlib import suppress

import torch
import torch.nn as nn
from transformers import SiglipVisionConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.siglip import SiglipAttention, SiglipMLP

# ----------------------------------------------------------------------------
# Debug dumping utilities (stage-wise)
# ----------------------------------------------------------------------------
_DUMP_FLAG = os.getenv("PADDLEOCRVL_DEBUG_DUMP", "0")
_DEBUG_VISION = os.getenv("PADDLEOCRVL_DEBUG_VISION", "0") == "1"
_STAGE_FILTER = {
    s.strip() for s in os.getenv("PADDLEOCRVL_DUMP_STAGES", "*").split(",")
}
_MAX_LAYERS = int(os.getenv("PADDLEOCRVL_DUMP_MAX_LAYERS", "2") or 0)
_TOK_SLICE = int(os.getenv("PADDLEOCRVL_DUMP_TOKENS", "64") or 64)
_DIM_SLICE = int(os.getenv("PADDLEOCRVL_DUMP_DIMS", "64") or 64)
_RUN_ID = os.getenv("PADDLEOCRVL_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")


def _stage_enabled(name: str) -> bool:
    if _DUMP_FLAG not in {"1", "2"}:
        return False
    if "*" in _STAGE_FILTER:
        return True
    return name in _STAGE_FILTER


def _save_stage(name: str, tensor: torch.Tensor | None) -> None:
    if not _stage_enabled(name):
        return
    try:
        out_dir = os.path.join("test_result", "paddleocrvl_stages")
        os.makedirs(out_dir, exist_ok=True)
        payload: dict[str, object] = {
            "stage": name,
            "shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else [],
        }
        if isinstance(tensor, torch.Tensor):
            t = tensor.detach().to(device="cpu", dtype=torch.float32)
            with torch.no_grad():
                payload["mean"] = float(t.mean())
                payload["std"] = float(t.std())
            if _DUMP_FLAG == "2":
                payload["tensor"] = t
        path = os.path.join(out_dir, f"vllm_{_RUN_ID}_{name}.pt")
        torch.save(payload, path)
    except Exception:
        pass


def _maybe_slice_tokens_dims(x: torch.Tensor) -> torch.Tensor:
    return x


# ============================================================================
# Custom Embeddings: Supports HF's pre-split format
# ============================================================================
class PaddleOCRVLSiglipEmbeddings(nn.Module):
    """
    Vision embeddings strictly matching HF pre-split format.

    Supported input only: [B, num_patches, 3, patch_size, patch_size]

    Reference: HF modeling_paddleocr_vl.py:1137-1201
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Two tables to mirror HF exactly:
        # - position_embedding: base square grid (used for interpolation)
        # - packing_position_embedding: large table for packed pre-split ids
        # Reference: HF modeling_paddleocr_vl.py around 1050–1200
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

    @staticmethod
    def flatten_list(nested_list):
        """Flatten nested list structure."""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int, interpolate: bool
    ) -> torch.Tensor:
        num_patches = int(height) * int(width)
        num_positions = self.position_embedding.weight.shape[0]

        if not interpolate or num_patches == num_positions:
            return self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        h0_float = num_positions**0.5
        h0 = int(h0_float)
        if h0 * h0 != num_positions:
            # Should not happen since base table is square; fallback safe slice.
            return self.position_embedding.weight[:num_patches].unsqueeze(0)

        w0 = h0

        pos_embed = self.position_embedding.weight.unsqueeze(0)
        pos_embed = pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass strictly supporting HF pre-split inputs.

        Args:
            pixel_values: [B, L, 3, pH, pW]
            position_ids: Required for pre-split unless interpolate_pos_encoding
                           is explicitly enabled.
            image_grid_thw: Optional grid info for dynamic resolution.
            interpolate_pos_encoding: Whether to interpolate position embeddings.

        Returns:
            embeddings: [B, num_patches, embed_dim]
        """
        # HF pre-split format: [B, L, 3, pH, pW]
        if pixel_values.dim() != 5:
            raise NotImplementedError(
                "PaddleOCR-VL vision expects 5D pre-split inputs, got "
                f"{tuple(pixel_values.shape)}",
            )
        return self._forward_hf_presplit(
            pixel_values, position_ids, image_grid_thw, interpolate_pos_encoding
        )

    def _forward_hf_presplit(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor | None,
        image_grid_thw: list | None,
        interpolate_pos_encoding: bool,
    ) -> torch.Tensor:
        batch_size, sequence_len, channel, height, width = pixel_values.shape

        pixel_values = pixel_values.view(
            batch_size * sequence_len, channel, height, width
        )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # Direct, unsliced save for conv patch embeddings (diagnostic)
        try:
            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
            os.makedirs(_debug_dir, exist_ok=True)
            flat = (
                patch_embeds.flatten(-2)
                .squeeze(-1)
                .view(batch_size, sequence_len, -1)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            B, L, D = int(flat.shape[0]), int(flat.shape[1]), int(flat.shape[2])
            torch.save(
                {
                    "emb_patch_conv": flat.reshape(B * L, D),
                    "shape_info": {"B": B, "L": L, "D": D},
                },
                os.path.join(_debug_dir, "emb_patch_conv_direct.pt"),
            )
        except Exception:
            pass

        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.view(batch_size, sequence_len, -1)
        if _stage_enabled("emb_pre_pos"):
            _save_stage(
                "emb_pre_pos",
                _maybe_slice_tokens_dims(embeddings.reshape(-1, embeddings.shape[-1])),
            )

        if interpolate_pos_encoding and image_grid_thw is not None:
            # Support dynamic resolution for batch_size >= 1.
            if batch_size == 1:
                start = 0
                embeddings = embeddings.squeeze(0)
                tmp_embeddings = []

                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings[start:end, :]

                    # Interpolate position encoding for this grid
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .repeat(t, 1)
                    )
                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end

                embeddings = torch.cat(tmp_embeddings, dim=0).unsqueeze(0)
            else:
                # Heuristic: use the first grid entry as reference for all
                # packed batch items (common case for identical image sizes).
                ref = (
                    image_grid_thw[0]
                    if isinstance(image_grid_thw, (list, tuple))
                    else image_grid_thw
                )
                if isinstance(ref, (list, tuple)):
                    t, h, w = int(ref[0]), int(ref[1]), int(ref[2])
                else:
                    # torch.Tensor with shape [3]
                    t, h, w = int(ref[0].item()), int(ref[1].item()), int(ref[2].item())

                # Compute per-sample position embeddings of shape [1, L, D]
                pos = self.interpolate_pos_encoding(
                    embeddings[:, : t * h * w, :], h, w, True
                ).repeat(1, t, 1)  # [1, L, D]
                # Broadcast to batch and add
                embeddings = embeddings + pos.repeat(batch_size, 1, 1)
        else:
            # Pre-split path requires explicit position_ids per HF.
            if position_ids is None:
                raise AssertionError(
                    "position_ids must be provided for pre-split pixel_values"
                )
            # Use the large packing table (size 32768) mapped from HF's
            # packing_position_embedding. Broadcasting over batch is fine.
            pos_embed = self.packing_position_embedding(position_ids)
            embeddings = embeddings + pos_embed.to(dtype=embeddings.dtype)
            # Unconditional diagnostic: save full, unsliced position_ids for
            # long-sequence parity checks vs HF. Writes only to test_result/.
            try:
                _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                os.makedirs(_debug_dir, exist_ok=True)
                torch.save(
                    {
                        "position_ids": position_ids.detach().to(
                            device="cpu", dtype=torch.int64
                        )
                    },
                    os.path.join(_debug_dir, "position_ids_direct.pt"),
                )
            except Exception:
                pass
            if _stage_enabled("emb_pos_ids"):
                _save_stage("emb_pos_ids", position_ids.detach().to("cpu", torch.int64))

        if _stage_enabled("emb_post_pos"):
            _save_stage(
                "emb_post_pos",
                _maybe_slice_tokens_dims(embeddings.reshape(-1, embeddings.shape[-1])),
            )
        # Direct, unsliced save for embeddings after pos add (diagnostic)
        try:
            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
            os.makedirs(_debug_dir, exist_ok=True)
            B, L, D = (
                int(embeddings.shape[0]),
                int(embeddings.shape[1]),
                int(embeddings.shape[2]),
            )
            torch.save(
                {
                    "emb_post_pos": embeddings.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(B * L, D),
                    "shape_info": {"B": B, "L": L, "D": D},
                },
                os.path.join(_debug_dir, "emb_post_pos_direct.pt"),
            )
        except Exception:
            pass
        return embeddings

    # NOTE: Standard 4D [B,3,H,W] path is intentionally unsupported to
    # match HF exactly. Only pre-split 5D inputs are allowed.


class PaddleOCRVLSiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self._layer_idx = -1 if layer_idx is None else int(layer_idx)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        # Optional: dump raw QKV weights/bias for layer 0 to debug mapping
        if os.getenv("PADDLEOCRVL_DUMP_QKV", "0") != "0" and self._layer_idx == 0:
            with suppress(Exception):
                qkv = self.self_attn.qkv_proj.weight  # type: ignore[attr-defined]
                qkvb = self.self_attn.qkv_proj.bias  # type: ignore[attr-defined]
                Hh = self.self_attn.num_heads_per_partition  # type: ignore[attr-defined]
                Dh = self.self_attn.head_dim  # type: ignore[attr-defined]
                offset_q = 0
                offset_k = Hh * Dh
                offset_v = (Hh + Hh) * Dh
                qw = qkv.narrow(0, offset_q, Hh * Dh).detach().to("cpu", torch.float32)
                kw = qkv.narrow(0, offset_k, Hh * Dh).detach().to("cpu", torch.float32)
                vw = qkv.narrow(0, offset_v, Hh * Dh).detach().to("cpu", torch.float32)
                qb = qkvb.narrow(0, offset_q, Hh * Dh).detach().to("cpu", torch.float32)
                kb = qkvb.narrow(0, offset_k, Hh * Dh).detach().to("cpu", torch.float32)
                vb = qkvb.narrow(0, offset_v, Hh * Dh).detach().to("cpu", torch.float32)
                out_dir = os.path.join("test_result", "paddleocrvl_stages")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    {"q_w": qw, "k_w": kw, "v_w": vw, "q_b": qb, "k_b": kb, "v_b": vb},
                    os.path.join(out_dir, "layer0_qkv_weights_direct.pt"),
                )

        hidden_states = self.layer_norm1(hidden_states)
        # Direct, unsliced save for LN1 output at layer 0 (diagnostic)
        try:
            if self._layer_idx == 0:
                _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                os.makedirs(_debug_dir, exist_ok=True)
                B, S, D = (
                    int(hidden_states.shape[0]),
                    int(hidden_states.shape[1]),
                    int(hidden_states.shape[2]),
                )
                torch.save(
                    {
                        "ln1": hidden_states.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .reshape(B * S, D),
                        "shape_info": {"B": B, "S": S, "D": D},
                        "eps": float(getattr(self.layer_norm1, "eps", 0.0)),
                    },
                    os.path.join(_debug_dir, "layer0_ln1_direct.pt"),
                )
        except Exception:
            pass
        if (
            _MAX_LAYERS > 0
            and 0 <= self._layer_idx < _MAX_LAYERS
            and _stage_enabled(f"layer{self._layer_idx}_ln1")
        ):
            _save_stage(
                f"layer{self._layer_idx}_ln1",
                _maybe_slice_tokens_dims(
                    hidden_states.reshape(-1, hidden_states.shape[-1])
                ),
            )
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
        )
        # Match HF: capture attention module output before residual add.
        if (
            _MAX_LAYERS > 0
            and 0 <= self._layer_idx < _MAX_LAYERS
            and _stage_enabled(f"layer{self._layer_idx}_attn_out")
        ):
            _save_stage(
                f"layer{self._layer_idx}_attn_out",
                _maybe_slice_tokens_dims(attn_out.reshape(-1, attn_out.shape[-1])),
            )
        # Direct, unsliced save for layer 0 attn_out (after out_proj, before residual)
        try:
            if self._layer_idx == 0:
                _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                os.makedirs(_debug_dir, exist_ok=True)
                B, S, D = (
                    int(attn_out.shape[0]),
                    int(attn_out.shape[1]),
                    int(attn_out.shape[2]),
                )
                torch.save(
                    {
                        "attn_out": attn_out.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .reshape(B * S, D),
                        "shape_info": {"B": B, "S": S, "D": D},
                    },
                    os.path.join(_debug_dir, "attn_out_direct.pt"),
                )
        except Exception:
            pass
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # Stage dump: layerN_ln2 (post second LayerNorm, pre-MLP)
        if (
            _MAX_LAYERS > 0
            and 0 <= self._layer_idx < _MAX_LAYERS
            and _stage_enabled(f"layer{self._layer_idx}_ln2")
        ):
            _save_stage(
                f"layer{self._layer_idx}_ln2",
                _maybe_slice_tokens_dims(
                    hidden_states.reshape(-1, hidden_states.shape[-1])
                ),
            )
        mlp_out = self.mlp(hidden_states)
        # Match HF: capture MLP output before residual add.
        if (
            _MAX_LAYERS > 0
            and 0 <= self._layer_idx < _MAX_LAYERS
            and _stage_enabled(f"layer{self._layer_idx}_mlp_out")
        ):
            _save_stage(
                f"layer{self._layer_idx}_mlp_out",
                _maybe_slice_tokens_dims(mlp_out.reshape(-1, mlp_out.shape[-1])),
            )
        hidden_states = mlp_out + residual

        return hidden_states


# ============================================================================
# Encoder: Stack of encoder layers
# ============================================================================
class PaddleOCRVLSiglipEncoder(nn.Module):
    """
    Transformer encoder for PaddleOCR-VL vision tower.

    Note: Unlike standard SigLIP, HF's PaddleOCR-VL encoder has RoPE support,
    but for now we implement the simpler version without RoPE to match
    current vLLM usage patterns. RoPE can be added if needed.

    Reference: HF modeling_paddleocr_vl.py:1472-1654
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        num_layers = (
            num_hidden_layers_override
            if num_hidden_layers_override is not None
            else config.num_hidden_layers
        )

        self.layers = nn.ModuleList(
            [
                PaddleOCRVLSiglipEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )

        # Install a lightweight 2D-RoPE shim that applies rotary on
        # queries/keys inside attention. Enabled by default for
        # PaddleOCR-VL to mirror HF behavior. You can disable it for
        # diagnostics by setting PADDLEOCRVL_USE_LOCAL_ROPE=0.
        self._rope_layers: list[SiglipAttention] | None = None
        if os.getenv("PADDLEOCRVL_USE_LOCAL_ROPE", "1") != "0":
            self._install_rope_attention_shim(quant_config)

    def _install_rope_attention_shim(
        self, quant_config: QuantizationConfig | None
    ) -> None:
        class _RoPE2DAttn(SiglipAttention):  # type: ignore[misc]
            def __init__(
                self,
                config: SiglipVisionConfig,
                *,
                quant_config: QuantizationConfig | None,
                prefix: str,
            ) -> None:
                super().__init__(config)
                self.num_heads_per_partition = config.num_attention_heads
                self.head_dim = config.hidden_size // config.num_attention_heads
                self._rope_enabled = False
                self._rope_cos: torch.Tensor | None = None
                self._rope_sin: torch.Tensor | None = None
                self._debug_layer_idx: int = -1

            def set_rope(
                self, cos: torch.Tensor | None, sin: torch.Tensor | None
            ) -> None:
                self._rope_enabled = cos is not None and sin is not None
                self._rope_cos = cos
                self._rope_sin = sin

            @staticmethod
            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)

            def _apply_rope(
                self, q: torch.Tensor, k: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Apply 2D RoPE using HF-style rotate_half.

                - Merge Q/K along a pseudo-batch axis to share identical
                  broadcast tables, then split back (mirrors Qwen2‑VL/HF).
                - Split head dim Dh into two halves: H and W components.
                - For each half, apply rotary with rotate_half, i.e.
                  x_rot = x * cos + rotate_half(x) * sin.
                Expects prebuilt cos/sin of shape [S, Dh].
                """
                # Optional debug marker
                if _DEBUG_VISION:
                    print("DEBUG: Using new _apply_rope (no H/W split)")
                assert self._rope_cos is not None and self._rope_sin is not None
                base_cos = self._rope_cos
                base_sin = self._rope_sin
                # Broadcast to [1, S, 1, Dh] for inputs shaped [B, S, Hh, Dh]
                cos = base_cos.view(1, base_cos.size(0), 1, base_cos.size(1))
                sin = base_sin.view(1, base_sin.size(0), 1, base_sin.size(1))

                # Merge Q and K along batch axis for identical rotary handling
                qk = torch.cat([q, k], dim=0)  # [2B, S, Hh, Dh]

                # Direct, unsliced save of pre-RoPE Q/K for layer 0 (diagnostic)
                if _DEBUG_VISION:
                    try:
                        if self._debug_layer_idx == 0:
                            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                            os.makedirs(_debug_dir, exist_ok=True)
                            S, Hh, Dh = (
                                int(q.shape[1]),
                                int(q.shape[2]),
                                int(q.shape[3]),
                            )
                            torch.save(
                                {
                                    "q_pre_rope": q.detach()
                                    .to(device="cpu", dtype=torch.float32)
                                    .reshape(S * Hh, Dh),
                                    "k_pre_rope": k.detach()
                                    .to(device="cpu", dtype=torch.float32)
                                    .reshape(S * Hh, Dh),
                                    "shape_info": {"S": S, "Hh": Hh, "Dh": Dh},
                                },
                                os.path.join(_debug_dir, "pre_rope_qk_direct_save.pt"),
                            )
                    except Exception:
                        pass

                # Optional dtype upcast for numerical stability: perform RoPE
                # math in float32, then cast back to the original dtype. This
                # helps avoid outlier values observed in bf16/fp16.
                orig_dtype = qk.dtype
                qk_fp32 = qk.to(torch.float32)
                cos_fp32 = cos.to(torch.float32)
                sin_fp32 = sin.to(torch.float32)

                # Diagnostics: print dtypes/ranges only for first debug layer
                if _DEBUG_VISION and self._debug_layer_idx == 0:
                    with suppress(Exception):
                        q_min = float(qk_fp32.min())
                        q_max = float(qk_fp32.max())
                        c_min = float(cos_fp32.min())
                        c_max = float(cos_fp32.max())
                        s_min = float(sin_fp32.min())
                        s_max = float(sin_fp32.max())
                        msg = (
                            "🔴 RoPE dtype diag (layer0): "
                            f"qk={orig_dtype}, cos={cos_fp32.dtype}, "
                            f"sin={sin_fp32.dtype}; "
                            f"qk_range=[{q_min:.4f},{q_max:.4f}], "
                            f"cos_range=[{c_min:.4f},{c_max:.4f}], "
                            f"sin_range=[{s_min:.4f},{s_max:.4f}]"
                        )
                        print(msg)

                # Apply HF-style rotate_half across the entire Dh dimension.
                # cos/sin tables already encode H/W frequency halves; splitting
                # breaks the intended cross-half coupling. Apply directly.
                qk_r_fp32 = (qk_fp32 * cos_fp32) + (
                    self._rotate_half(qk_fp32) * sin_fp32
                )

                # Cast back to the original dtype
                qk_r = qk_r_fp32.to(orig_dtype)

                # Split back to Q and K
                B = q.shape[0]
                q_embed, k_embed = qk_r[:B], qk_r[B:]

                # Direct, unsliced post-RoPE saves for layer 0 to enable
                # parity checks against HF without stage slicing bias.
                if _DEBUG_VISION and self._debug_layer_idx == 0:
                    with suppress(Exception):
                        _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                        os.makedirs(_debug_dir, exist_ok=True)
                        S, Hh, Dh = (
                            int(q_embed.shape[1]),
                            int(q_embed.shape[2]),
                            int(q_embed.shape[3]),
                        )
                        # Save FP32 post-RoPE before downcast as well (diagnostic)
                        torch.save(
                            {
                                "q_post_rope_fp32": qk_r_fp32[: q_embed.shape[0]]
                                .detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(S * Hh, Dh),
                                "k_post_rope_fp32": qk_r_fp32[q_embed.shape[0] :]
                                .detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(S * Hh, Dh),
                                "shape_info": {"S": S, "Hh": Hh, "Dh": Dh},
                            },
                            os.path.join(
                                _debug_dir, "post_rope_qk_direct_save_fp32.pt"
                            ),
                        )
                        torch.save(
                            {
                                "q_post_rope": q_embed.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(S * Hh, Dh),
                                "k_post_rope": k_embed.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(S * Hh, Dh),
                                "shape_info": {"S": S, "Hh": Hh, "Dh": Dh},
                            },
                            os.path.join(_debug_dir, "post_rope_qk_direct_save.pt"),
                        )
                        if _DEBUG_VISION:
                            print(
                                "🔴 Saved post-RoPE Q/K to test_result/"
                                "paddleocrvl_stages/post_rope_qk_direct_save.pt"
                            )
                return q_embed, k_embed

            def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.chunk(3, dim=-1)
                need_unsqueeze = q.ndim == 2
                if need_unsqueeze:
                    q = q.unsqueeze(0)
                    k = k.unsqueeze(0)
                    v = v.unsqueeze(0)
                if (
                    _MAX_LAYERS > 0
                    and 0 <= self._debug_layer_idx < _MAX_LAYERS
                    and _stage_enabled(f"layer{self._debug_layer_idx}_qk_pre_rope")
                ):
                    B, S, D = q.shape
                    Hh = self.num_heads_per_partition
                    Dh = self.head_dim
                    qv = q.view(B, S, Hh, Dh)
                    kv = k.view(B, S, Hh, Dh)
                    _save_stage(
                        f"layer{self._debug_layer_idx}_q_pre_rope",
                        _maybe_slice_tokens_dims(qv.reshape(S * Hh, Dh)),
                    )
                    _save_stage(
                        f"layer{self._debug_layer_idx}_k_pre_rope",
                        _maybe_slice_tokens_dims(kv.reshape(S * Hh, Dh)),
                    )
                    # Optional FP32 pre-RoPE Q/K direct save for parity check
                    if (
                        _DEBUG_VISION
                        and os.getenv("PADDLEOCRVL_FP32_PRE_QK_SAVE", "0") != "0"
                        and self._debug_layer_idx == 0
                    ):
                        try:
                            # hidden_states: [B, S, D]
                            hs32 = hidden_states.to(torch.float32)
                            w = self.qkv_proj.weight  # type: ignore[attr-defined]
                            b = self.qkv_proj.bias  # type: ignore[attr-defined]
                            # Slice q/k shards from fused weight/bias
                            offset_q = 0
                            offset_k = Hh * Dh
                            qw = w.narrow(0, offset_q, Hh * Dh).to(torch.float32)
                            kw = w.narrow(0, offset_k, Hh * Dh).to(torch.float32)
                            qb = b.narrow(0, offset_q, Hh * Dh).to(torch.float32)
                            kb = b.narrow(0, offset_k, Hh * Dh).to(torch.float32)
                            q_pre = (
                                hs32.reshape(B * S, D) @ qw.t()
                            ) + qb  # [B*S, Hh*Dh]
                            k_pre = (hs32.reshape(B * S, D) @ kw.t()) + kb
                            q_pre = q_pre.view(B, S, Hh, Dh).reshape(S * Hh, Dh)
                            k_pre = k_pre.view(B, S, Hh, Dh).reshape(S * Hh, Dh)
                            _debug_dir = os.path.join(
                                "test_result", "paddleocrvl_stages"
                            )
                            os.makedirs(_debug_dir, exist_ok=True)
                            torch.save(
                                {
                                    "q_pre_rope_fp32": q_pre.detach().cpu(),
                                    "k_pre_rope_fp32": k_pre.detach().cpu(),
                                    "shape_info": {
                                        "S": int(S),
                                        "Hh": int(Hh),
                                        "Dh": int(Dh),
                                    },
                                },
                                os.path.join(_debug_dir, "pre_rope_qk_fp32_direct.pt"),
                            )
                        except Exception:
                            pass
                # Optionally disable rotary even if tables were set, to aid
                # diagnostics. Set PADDLEOCRVL_DISABLE_ROPE=1 to skip rotation
                # while preserving the custom attention path and dumps.
                rope_disabled = os.getenv("PADDLEOCRVL_DISABLE_ROPE", "0") == "1"
                if self._rope_enabled and not rope_disabled:
                    B, S, _ = q.shape
                    Hh = self.num_heads_per_partition
                    Dh = self.head_dim
                    qv = q.view(B, S, Hh, Dh)
                    kv = k.view(B, S, Hh, Dh)
                    qv, kv = self._apply_rope(qv, kv)
                    q = qv.reshape(B, S, Hh * Dh)
                    k = kv.reshape(B, S, Hh * Dh)
                    if (
                        _MAX_LAYERS > 0
                        and 0 <= self._debug_layer_idx < _MAX_LAYERS
                        and _stage_enabled(f"layer{self._debug_layer_idx}_qk_post_rope")
                    ):
                        _save_stage(
                            f"layer{self._debug_layer_idx}_q_post_rope",
                            _maybe_slice_tokens_dims(qv.reshape(S * Hh, Dh)),
                        )
                        _save_stage(
                            f"layer{self._debug_layer_idx}_k_post_rope",
                            _maybe_slice_tokens_dims(kv.reshape(S * Hh, Dh)),
                        )
                # Diagnostic: save V pre-attention for layer 0
                if _DEBUG_VISION and self._debug_layer_idx == 0:
                    with suppress(Exception):
                        _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                        os.makedirs(_debug_dir, exist_ok=True)
                        B, S, _ = q.shape
                        Hh = self.num_heads_per_partition
                        Dh = self.head_dim
                        vh = v.view(B, S, Hh, Dh)
                        torch.save(
                            {
                                "v_pre": vh.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(S * Hh, Dh)
                            },
                            os.path.join(_debug_dir, "v_pre_direct.pt"),
                        )

                # For HF parity: optionally force Torch SDPA (disable Flash-Attn)
                # to reduce numerical drift vs HF. Enabled by default; set
                # PADDLEOCRVL_FORCE_SDPA=0 to use the selected backend.
                use_forced_sdpa = os.getenv("PADDLEOCRVL_FORCE_SDPA", "1") != "0"
                if use_forced_sdpa:
                    import torch.nn.functional as F

                    B, S, _ = q.shape
                    Hh = self.num_heads_per_partition
                    Dh = self.head_dim
                    qh = q.view(B, S, Hh, Dh).transpose(1, 2)  # [B, H, S, Dh]
                    kh = k.view(B, S, Hh, Dh).transpose(1, 2)
                    vh = v.view(B, S, Hh, Dh).transpose(1, 2)
                    # Optional manual attention kernel to mimic HF operations
                    if os.getenv("PADDLEOCRVL_FORCE_MANUAL_ATTN", "0") == "1":
                        scale = 1.0 / (Dh**0.5)
                        if os.getenv("PADDLEOCRVL_ATTENTION_FP32", "1") != "0":
                            qh = qh.to(torch.float32)
                            kh = kh.to(torch.float32)
                            vh = vh.to(torch.float32)
                        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
                        probs = torch.softmax(scores, dim=-1)
                        attn_out = torch.matmul(probs, vh)
                        attn_out = attn_out.to(q.dtype)
                    else:
                        # Align with HF/Qwen2‑VL: explicit SDPA with dropout=0.0.
                        # Optionally upcast to FP32 for numeric parity.
                        if os.getenv("PADDLEOCRVL_ATTENTION_FP32", "1") != "0":
                            qh32, kh32, vh32 = (
                                qh.to(torch.float32),
                                kh.to(torch.float32),
                                vh.to(torch.float32),
                            )
                            attn_out = F.scaled_dot_product_attention(
                                qh32, kh32, vh32, dropout_p=0.0
                            )
                            attn_out = attn_out.to(qh.dtype)
                        else:
                            attn_out = F.scaled_dot_product_attention(
                                qh, kh, vh, dropout_p=0.0
                            )  # [B, H, S, Dh]
                    out = attn_out.transpose(1, 2).reshape(B, S, Hh * Dh)
                else:
                    out = self.attn(q, k, v)

                if (
                    _MAX_LAYERS > 0
                    and 0 <= self._debug_layer_idx < _MAX_LAYERS
                    and _stage_enabled(f"layer{self._debug_layer_idx}_attn_pre_outproj")
                ):
                    # Save attention output before the final linear projection
                    B, S, _ = out.shape
                    _save_stage(
                        f"layer{self._debug_layer_idx}_attn_pre_outproj",
                        _maybe_slice_tokens_dims(out.reshape(B * S, -1)),
                    )
                # Direct, unsliced diagnostic save for layer 0 pre-out-proj
                if self._debug_layer_idx == 0:
                    with suppress(Exception):
                        _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
                        os.makedirs(_debug_dir, exist_ok=True)
                        B, S, D = (
                            int(out.shape[0]),
                            int(out.shape[1]),
                            int(out.shape[2]),
                        )
                        torch.save(
                            {
                                "attn_pre_outproj": out.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .reshape(B * S, D),
                                "shape_info": {"B": B, "S": S, "D": D},
                            },
                            os.path.join(_debug_dir, "attn_pre_outproj_direct.pt"),
                        )
                if need_unsqueeze:
                    out = out.squeeze(0)
                out, _ = self.out_proj(out)
                return out, None

        # Swap attentions
        rope_layers: list[SiglipAttention] = []
        for idx, layer in enumerate(self.layers):
            attn: SiglipAttention = layer.self_attn  # type: ignore[assignment]
            rope_attn = _RoPE2DAttn(
                self.config, quant_config=quant_config, prefix=f"rope_attn_{idx}"
            )
            rope_attn.load_state_dict(attn.state_dict(), strict=True)
            layer.self_attn = rope_attn  # type: ignore[assignment]
            # Attach layer index for stage dumps
            if hasattr(layer.self_attn, "_debug_layer_idx"):
                layer.self_attn._debug_layer_idx = idx  # type: ignore[attr-defined]
            rope_layers.append(layer.self_attn)
        self._rope_layers = rope_layers

    def _set_rope_on_layers(
        self, cos: torch.Tensor | None, sin: torch.Tensor | None
    ) -> None:
        if not self._rope_layers:
            return
        for attn in self._rope_layers:
            if hasattr(attn, "set_rope"):
                attn.set_rope(cos, sin)  # type: ignore[attr-defined]

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            inputs_embeds: [B, seq_len, embed_dim]
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states

        Returns:
            last_hidden_state: [B, seq_len, embed_dim]
        """
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )

        return hidden_states


# ============================================================================
# Vision Transformer: Complete vision encoder
# ============================================================================
class PaddleOCRVLVisionTransformer(nn.Module):
    """
    Complete vision transformer for PaddleOCR-VL.

    Integrates:
    - Custom embeddings (supports HF pre-split)
    - Encoder layers
    - Post LayerNorm

    Reference: HF modeling_paddleocr_vl.py:1706-1856
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Custom embeddings supporting HF pre-split
        self.embeddings = PaddleOCRVLSiglipEmbeddings(config)

        # Encoder
        self.encoder = PaddleOCRVLSiglipEncoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        # Post LayerNorm (HF adds this, standard SigLIP doesn't)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: [B, 3, H, W] or [B, L, 3, pH, pW]
            position_ids: Optional position IDs for HF format
            image_grid_thw: Optional grid info
            interpolate_pos_encoding: Whether to interpolate positions

        Returns:
            vision_features: [B, num_patches, embed_dim]
        """
        # If pre-split input and grid provided, derive HF-style packing position ids.
        if (
            position_ids is None
            and image_grid_thw is not None
            and pixel_values.dim() == 5
        ):
            # image_grid_thw can be torch.Tensor[list[list[int]]]; normalize to list
            pos_chunks: list[torch.Tensor] = []
            for item in image_grid_thw:  # type: ignore[assignment]
                if isinstance(item, (list, tuple)):
                    t, h, w = int(item[0]), int(item[1]), int(item[2])
                else:
                    t = int(item[0].item())
                    h = int(item[1].item())
                    w = int(item[2].item())
                numel = t * h * w
                # Positions within each image: 0..(t*h*w-1) mapped to 0..(h*w-1)
                pos = torch.arange(
                    numel, device=pixel_values.device, dtype=torch.long
                ) % (h * w)
                pos_chunks.append(pos)
            if pos_chunks:
                position_ids = torch.cat(pos_chunks, dim=0)

        # Embeddings
        try:
            # Save pixel_values (diagnostic)
            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
            os.makedirs(_debug_dir, exist_ok=True)
            pv = pixel_values.detach().to(device="cpu", dtype=torch.float32)
            torch.save(
                {"pixel_values": pv, "shape": list(pv.shape)},
                os.path.join(_debug_dir, "pixel_values_direct.pt"),
            )
        except Exception:
            pass
        hidden_states = self.embeddings(
            pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        # Encoder (with optional local 2D-RoPE)
        # Enable local RoPE by default to match HF rotary usage.
        use_local_rope = os.getenv("PADDLEOCRVL_USE_LOCAL_ROPE", "1") != "0"
        if use_local_rope and image_grid_thw is not None:
            cos, sin = self._compute_2d_rope_cos_sin(image_grid_thw)
            self.encoder._set_rope_on_layers(cos, sin)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=False,
        )

        if use_local_rope and image_grid_thw is not None:
            self.encoder._set_rope_on_layers(None, None)

        # Post LayerNorm (HF-specific)
        # Direct, unsliced diagnostic save for encoder output pre-LN
        try:
            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
            os.makedirs(_debug_dir, exist_ok=True)
            B, S, D = (
                int(last_hidden_state.shape[0]),
                int(last_hidden_state.shape[1]),
                int(last_hidden_state.shape[2]),
            )
            torch.save(
                {
                    "vision_pre_post_ln": last_hidden_state.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(B * S, D),
                    "shape_info": {"B": B, "S": S, "D": D},
                },
                os.path.join(_debug_dir, "vision_pre_post_ln_direct.pt"),
            )
        except Exception:
            pass
        last_hidden_state = self.post_layernorm(last_hidden_state)
        if _stage_enabled("vision_post_ln"):
            _save_stage(
                "vision_post_ln",
                _maybe_slice_tokens_dims(
                    last_hidden_state.reshape(-1, last_hidden_state.shape[-1])
                ),
            )
        # Direct, unsliced diagnostic save for post-LN features
        try:
            _debug_dir = os.path.join("test_result", "paddleocrvl_stages")
            os.makedirs(_debug_dir, exist_ok=True)
            B, S, D = (
                int(last_hidden_state.shape[0]),
                int(last_hidden_state.shape[1]),
                int(last_hidden_state.shape[2]),
            )
            torch.save(
                {
                    "vision_post_ln": last_hidden_state.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(B * S, D),
                    "shape_info": {"B": B, "S": S, "D": D},
                },
                os.path.join(_debug_dir, "vision_post_ln_direct.pt"),
            )
        except Exception:
            pass

        return last_hidden_state

    def _compute_2d_rope_cos_sin(
        self, image_grid_thw: list | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for 2D rotary using row-major H×W indices.

        We construct per-token (h, w) indices across all images, build base
        frequencies for each axis, then concatenate and repeat to Dh to match
        HF's SigLIP rotary formulation.
        """
        device = next(self.parameters()).device
        # Diagnostics (gated): print input and env slicing settings
        import os as _os
        if _os.getenv("PADDLEOCRVL_DEBUG_MROPE"):
            try:
                print("🔴 DEBUG _compute_2d_rope_cos_sin:")
                print(f"  image_grid_thw: {image_grid_thw}")
                print(f"  _TOK_SLICE={_TOK_SLICE}, _DIM_SLICE={_DIM_SLICE}")
            except Exception:
                pass
        if not image_grid_thw:
            empty = torch.empty(0, device=device)
            return empty, empty

        # Build row/col indices
        h_all: list[torch.Tensor] = []
        w_all: list[torch.Tensor] = []
        for t, h, w in image_grid_thw:
            t = int(t)
            h = int(h)
            w = int(w)
            rows = torch.arange(h, device=device)
            cols = torch.arange(w, device=device)
            h_grid = rows.view(h, 1).expand(h, w).reshape(-1)
            w_grid = cols.view(1, w).expand(h, w).reshape(-1)
            if t > 1:
                h_grid = h_grid.repeat(t)
                w_grid = w_grid.repeat(t)
            h_all.append(h_grid)
            w_all.append(w_grid)
        h_full = torch.cat(h_all, dim=0)
        w_full = torch.cat(w_all, dim=0)

        # Mirror HF construction closely:
        # pids = stack([h_ids, w_ids]) -> gather base -> flatten(H/W) -> repeat
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        half = head_dim // 2  # Dh/2
        theta = 10000.0
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, half, 2, dtype=torch.float32, device=device) / half)
        )  # [quarter]
        max_hw = int(max(h_full.max().item(), w_full.max().item())) + 1
        seq = torch.arange(max_hw, dtype=inv_freq.dtype, device=device)
        base = torch.outer(seq, inv_freq)  # [max_hw, quarter]
        # Stack and flatten H/W along dim=1 to interleave like HF
        gather_h = base[h_full]  # [S, quarter]
        gather_w = base[w_full]  # [S, quarter]
        rope_hw = torch.stack([gather_h, gather_w], dim=1).flatten(1)  # [S, half]
        rope = rope_hw.repeat(1, 2)  # [S, Dh]
        dtype = next(self.parameters()).dtype
        cos = torch.cos(rope).to(dtype=dtype)
        sin = torch.sin(rope).to(dtype=dtype)
        # Optional debug dumps for indices and tables
        if _stage_enabled("rope_indices"):
            _save_stage("rope_h_indices", _maybe_slice_tokens_dims(h_full.view(-1, 1)))
            _save_stage("rope_w_indices", _maybe_slice_tokens_dims(w_full.view(-1, 1)))
        if _stage_enabled("rope_tables"):
            _save_stage("rope_cos", _maybe_slice_tokens_dims(cos))
            _save_stage("rope_sin", _maybe_slice_tokens_dims(sin))
        # Optional stats (gated)
        if _os.getenv("PADDLEOCRVL_DEBUG_MROPE"):
            try:
                msg = (
                    f"  Generated cos shape: {tuple(cos.shape)}, "
                    f"sin shape: {tuple(sin.shape)}"
                )
                print(msg)
                with torch.no_grad():
                    c_mean, c_std = float(cos.mean()), float(cos.std())
                    s_mean, s_std = float(sin.mean()), float(sin.std())
                print(
                    "  cos stats: mean="
                    f"{c_mean:.6f}, std={c_std:.6f}; "
                    "sin stats: mean="
                    f"{s_mean:.6f}, std={s_std:.6f}"
                )
            except Exception:
                pass

        # Optional direct save (gated)
        if _os.getenv("PADDLEOCRVL_DEBUG_MROPE"):
            try:
                _out_dir = _os.path.join("test_result", "paddleocrvl_stages")
                _os.makedirs(_out_dir, exist_ok=True)
                torch.save(
                    {
                        "cos": cos.detach().to(device="cpu", dtype=torch.float32),
                        "sin": sin.detach().to(device="cpu", dtype=torch.float32),
                        "shape": [int(cos.shape[0]), int(cos.shape[1])],
                    },
                    _os.path.join(_out_dir, "rope_direct_save.pt"),
                )
                print(
                    "🔴 Saved RoPE tables to test_result/"
                    "paddleocrvl_stages/rope_direct_save.pt"
                )
            except Exception:
                pass
        return cos, sin

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set:
        """
        Load weights into the vision transformer.

        Handles Q/K/V fusion for attention layers, matching HF's separate
        q_proj, k_proj, v_proj to vLLM's fused qkv_proj.

        Reference: Similar to SigLIP's load_weights in siglip.py:782-817
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.encoder.layers)

        # First pass: collect per-layer q/k/v weights and biases
        # to construct fused tensors for robust loading.
        qkv_tensors: dict[int, dict[str, torch.Tensor]] = {}
        others: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            if name.startswith("encoder.layers"):
                # Skip layers beyond override
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue
            if (
                ".self_attn.q_proj." in name
                or ".self_attn.k_proj." in name
                or ".self_attn.v_proj." in name
            ):
                # Extract layer index
                try:
                    layer_idx = int(name.split(".")[2])
                except Exception:
                    others.append((name, loaded_weight))
                    continue
                entry = qkv_tensors.setdefault(layer_idx, {})
                if ".q_proj.weight" in name:
                    entry["q_w"] = loaded_weight.to(torch.float32)
                elif ".q_proj.bias" in name:
                    entry["q_b"] = loaded_weight.to(torch.float32)
                elif ".k_proj.weight" in name:
                    entry["k_w"] = loaded_weight.to(torch.float32)
                elif ".k_proj.bias" in name:
                    entry["k_b"] = loaded_weight.to(torch.float32)
                elif ".v_proj.weight" in name:
                    entry["v_w"] = loaded_weight.to(torch.float32)
                elif ".v_proj.bias" in name:
                    entry["v_b"] = loaded_weight.to(torch.float32)
                else:
                    others.append((name, loaded_weight))
            else:
                others.append((name, loaded_weight))

        # Load fused qkv where all pieces are present
        for layer_idx, pieces in qkv_tensors.items():
            required = ["q_w", "k_w", "v_w", "q_b", "k_b", "v_b"]
            if not all(k in pieces for k in required):
                # Fallback to incremental loading via default path if incomplete
                continue
            # Construct fused names for this layer
            fused_w_name = f"encoder.layers.{layer_idx}.self_attn.qkv_proj.weight"
            fused_b_name = f"encoder.layers.{layer_idx}.self_attn.qkv_proj.bias"
            if fused_w_name in params_dict:
                q_w = pieces["q_w"].to(params_dict[fused_w_name].dtype)
                k_w = pieces["k_w"].to(params_dict[fused_w_name].dtype)
                v_w = pieces["v_w"].to(params_dict[fused_w_name].dtype)
                fused_w = torch.cat([q_w, k_w, v_w], dim=0)
                param_w = params_dict[fused_w_name]
                # Pass None shard id to trigger fused path in loader
                loader_w = getattr(param_w, "weight_loader", default_weight_loader)
                loader_w(param_w, fused_w)
                loaded_params.add(fused_w_name)
            if fused_b_name in params_dict:
                q_b = pieces["q_b"].to(params_dict[fused_b_name].dtype)
                k_b = pieces["k_b"].to(params_dict[fused_b_name].dtype)
                v_b = pieces["v_b"].to(params_dict[fused_b_name].dtype)
                fused_b = torch.cat([q_b, k_b, v_b], dim=0)
                param_b = params_dict[fused_b_name]
                loader_b = getattr(param_b, "weight_loader", default_weight_loader)
                loader_b(param_b, fused_b)
                loaded_params.add(fused_b_name)

        # Second pass: load remaining weights (including out_proj, mlp, ln, etc.)
        for name, loaded_weight in others:
            # post_layernorm is optional - skip if not present
            if name.startswith("post_layernorm") and self.post_layernorm is None:
                continue
            if name in loaded_params:
                continue
            # Load normally
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


# ============================================================================
# Vision Model: Top-level vision module
# ============================================================================
class PaddleOCRVLVisionModel(nn.Module):
    """
    Top-level vision model wrapping the vision transformer.

    This provides a clean interface for the main PaddleOCR-VL model.
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_model = PaddleOCRVLVisionTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """Forward to vision_model."""
        return self.vision_model(
            pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set:
        """
        Load weights into the vision model.

        Delegates to the underlying vision_model transformer.

        Reference: Similar to SiglipVisionModel.load_weights in siglip.py:868-926
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in vision_model
            if (
                name.startswith("vision_model.post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # Omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            # Handle Q/K/V fusion
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Standard weight loading
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
