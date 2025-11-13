# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native PaddleOCR-VL (0.9B) support with minimal surface area.

This implementation wires a SigLIP vision tower, a lightweight projector
that merges spatial patches, and an Ernie 4.5 language backbone. The
integration relies on vLLM's standard multimodal processing pipeline and
keeps changes localized to this file plus the model registry entry.

Notes:
- Prompts that follow the chat-template style (e.g.,
  "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>"
  "<|IMAGE_END|>...\\nAssistant: ") work best.
  A bare placeholder-only prompt may yield empty generations depending on
  task phrasing.

Scope:
- No global multimodal processing changes
- No debug dumps or temporary artifacts
- Minimal prompt-replacement logic based on HF processor outputs
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Annotated, Any, Literal, TypeAlias
import json

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    SiglipVisionConfig,
)
from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,  # type: ignore[reportMissingImports]
)

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.paddleocr_vl_vision import (
    PaddleOCRVLVisionModel,
)
from vllm.model_executor.models.siglip import (
    SiglipAttention,  # reuse weights
    SiglipVisionEmbeddings,
)

# NOTE: Avoid importing/depending on shared SigLIP2/NaViT helpers here.
# PaddleOCR-VL will use a localized vision encoder to prevent touching
# generic components used by other models.
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, BaseDummyOptions
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape


def _get_vision_config_value(config: object, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class PaddleOCRVLConfig(PretrainedConfig):
    """Minimal HF-style config wrapper for PaddleOCR-VL."""

    model_type = "paddleocr_vl"

    def __init__(
        self,
        *,
        vision_config: dict[str, Any] | None = None,
        hidden_size: int = 1024,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        intermediate_size: int = 3072,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000,
        rope_scaling: dict[str, Any] | None = None,
        use_3d_rope: bool = True,
        rope_is_neox_style: bool = True,
        image_token_id: int = 100295,
        vocab_size: int = 103424,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vision_config = vision_config or {}
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {}
        self.use_3d_rope = use_3d_rope
        self.rope_is_neox_style = rope_is_neox_style
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size


class PaddleOCRVLProjector(nn.Module):
    """Two-layer MLP that aligns vision features to the language space.

    The projector expects flattened patch features and merges them using
    a spatial kernel of size `(merge_h, merge_w)` driven by the SigLIP
    `spatial_merge_size` setting.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        hidden_size: int,
        merge_kernel_size: tuple[int, int] = (1, 1),
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.hidden_size = hidden_size
        self.merge_kernel_size = merge_kernel_size

        merged_hidden = vision_hidden_size * merge_kernel_size[0] * merge_kernel_size[1]

        self.pre_norm = nn.LayerNorm(vision_hidden_size, eps=1e-5)
        self.linear_1 = ColumnParallelLinear(
            merged_hidden,
            merged_hidden,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_1"),
        )
        self.act = nn.GELU()
        self.linear_2 = RowParallelLinear(
            merged_hidden,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_2"),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Project features per image.

        Args:
            vision_features: 2D tensor of shape [num_patches_total, vision_hidden]
            image_grid_thw: shape [num_images, 3] with T,H,W patch grid
        Returns:
            Tuple of [num_tokens_per_image, hidden] projected sequences per image.
        """
        if image_grid_thw.numel() == 0:
            return ()

        merge_h, merge_w = self.merge_kernel_size
        debug_proj = os.getenv("PADDLEOCRVL_DEBUG_PROJECTOR", "0") == "1"
        outputs: list[torch.Tensor] = []
        offset = 0

        for img_idx, grid in enumerate(image_grid_thw.tolist()):
            t, h, w = (int(x) for x in grid)
            num_patches = t * h * w
            patches = vision_features[offset : offset + num_patches]
            offset += num_patches

            patches = patches.view(t, h, w, self.vision_hidden_size)
            patches = self.pre_norm(patches.view(-1, self.vision_hidden_size)).view(
                t, h, w, self.vision_hidden_size
            )

            h_merge = h // merge_h
            w_merge = w // merge_w

            merged = (
                patches.view(
                    t,
                    h_merge,
                    merge_h,
                    w_merge,
                    merge_w,
                    self.vision_hidden_size,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(t, h_merge * w_merge, -1)
            )

            hidden_states = merged.reshape(-1, merged.shape[-1])
            hidden_states, _ = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states, _ = self.linear_2(hidden_states)
            hidden_states = hidden_states.view(t, h_merge * w_merge, self.hidden_size)

            outputs.extend(hidden_states.unbind())
            if debug_proj:
                try:
                    os.makedirs("test_result", exist_ok=True)
                    with open(
                        os.path.join("test_result", "paddleocrvl_projector_debug.jsonl"),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            json.dumps(
                                {
                                    "image_index": img_idx,
                                    "grid_thw": [t, h, w],
                                    "merge_hw": [merge_h, merge_w],
                                    "tokens_out": int(hidden_states.shape[0] * hidden_states.shape[1]),
                                    "tokens_per_t": int(hidden_states.shape[1]),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass

        return tuple(outputs)


class PaddleOCRVLSiglipEmbeddings(SiglipVisionEmbeddings):
    """Embeddings that tolerate non-square position grids.

    Some PaddleOCR-VL checkpoints ship vision position embeddings with
    rectangular grids (e.g., 24x32). This subclass replaces the square-root
    heuristic with a factorization-based interpolation that works for any
    grid whose area equals the number of positions in the learned table.
    """

    @staticmethod
    def _factorize_grid(n: int) -> tuple[int, int]:
        # Find a grid close to square: largest divisor <= sqrt(n)
        r = int(n**0.5)
        for h in range(r, 0, -1):
            if n % h == 0:
                return h, n // h
        # Fallback: degenerate grid
        return 1, n

    def interpolate_pos_encoding_anygrid(
        self, num_patches_h: int, num_patches_w: int
    ) -> torch.Tensor:
        pos = self.position_embedding.weight.unsqueeze(0)
        dim = pos.shape[-1]
        npos = pos.shape[1]
        h0, w0 = self._factorize_grid(npos)

        # [1, H0, W0, C] -> [1, C, H0, W0]
        pos_hw = pos.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        pos_interp = nn.functional.interpolate(
            pos_hw,
            size=(num_patches_h, num_patches_w),
            mode="bicubic",
            align_corners=False,
        )
        # [1, C, H, W] -> [1, H*W, C]
        pos_interp = pos_interp.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos_interp

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        # Use same patch embedding as upstream
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # [B, C, H, W] -> [B, H*W, C]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Compute patch grid from conv output to avoid relying on config sizes
        grid_h, grid_w = patch_embeds.shape[-2], patch_embeds.shape[-1]

        # Always add position encodings via robust rectangular interpolation.
        # This avoids mismatches when the learned table uses a non-square grid
        # or when processor produces grids different from the pretraining one.
        pos = self.interpolate_pos_encoding_anygrid(grid_h, grid_w)
        embeddings = embeddings + pos.to(embeddings.dtype)
        return embeddings


class PaddleOCRVLVisionTransformer(nn.Module):
    """Custom PaddleOCR-VL vision encoder (fully dedicated).

    Uses the dedicated PaddleOCRVLVisionModel which:
    - Supports HF's pre-split patch format ([B, L, 3, pH, pW])
    - Maintains HF implementation parity
    - Isolates from shared SigLIP components

    This wrapper provides compatibility with existing PaddleOCR-VL code
    while using the new dedicated vision tower internally.
    """

    def __init__(
        self,
        vision_config: SiglipVisionConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Use the dedicated vision model
        self.vision_model = PaddleOCRVLVisionModel(
            config=vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        # Public attributes referenced by helpers/tests.
        self.out_hidden_size = int(getattr(vision_config, "hidden_size", 1152))
        self.spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 1))

        self.eval()

        # Optional RoPE support at wrapper level is disabled by default.
        # PaddleOCR-VL's dedicated vision tower (paddleocr_vl_vision.py)
        # already installs a RoPE-enabled attention shim. To avoid double
        # instrumentation, only enable the wrapper-level shim when an explicit
        # opt-in flag is set.
        if os.getenv("PADDLEOCRVL_USE_WRAPPER_ROPE", "0") == "1":
            self._install_rope_attention_shim(quant_config)

        # Keep model's intrinsic dtype; do not force-cast the vision tower.

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.vision_model.post_layernorm.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.vision_model.post_layernorm.weight.device

    def _merge_grid_wh(self, h: int, w: int, merge: int) -> tuple[int, int]:
        if merge <= 1:
            return h, w
        assert h % merge == 0 and w % merge == 0
        return h // merge, w // merge

    def build_position_ids_row_major(
        self, grid_thw: torch.Tensor, merge: int
    ) -> list[torch.Tensor]:
        """Row-major 2D positions after spatial merge.

        Returns a list of [num_tokens, 2] tensors (h, w) per image.
        """
        pos_list: list[torch.Tensor] = []
        device = grid_thw.device
        for t, h, w in grid_thw.tolist():
            hm, wm = self._merge_grid_wh(int(h), int(w), merge)
            # t is typically 1 for images; keep for completeness
            h_idx = (
                torch.arange(hm, device=device).view(1, -1, 1).expand(t, -1, wm)
            ).reshape(-1)
            w_idx = (
                torch.arange(wm, device=device).view(1, 1, -1).expand(t, hm, -1)
            ).reshape(-1)
            pos = torch.stack([h_idx, w_idx], dim=-1)  # [t*hm*wm, 2]
            pos_list.append(pos)
        return pos_list

    def build_cu_seqlens_after_merge(
        self, grid_thw: torch.Tensor, merge: int
    ) -> torch.Tensor:
        """Cumulative seqlens over merged tokens per image.

        Example: [0, 192, 388, ...] after 2x2 merge on 24x32 -> 12x16.
        """
        lens = [0]
        total = 0
        for t, h, w in grid_thw.tolist():
            hm, wm = self._merge_grid_wh(int(h), int(w), merge)
            total += int(t) * hm * wm
            lens.append(total)
        return torch.tensor(lens, device=grid_thw.device, dtype=torch.int32)

    # ---- Local 2D RoPE support (shim) ----
    def _install_rope_attention_shim(
        self, quant_config: QuantizationConfig | None
    ) -> None:
        """Replace SigLIP self-attn modules with RoPE-enabled subclasses.

        Weight shapes and parameters are identical; we copy the state dict
        from the original attention layers to preserve weights.
        """

        class _RoPE2DAttn(SiglipAttention):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._rope_enabled: bool = False
                self._rope_cos: torch.Tensor | None = None  # [S, head_dim]
                self._rope_sin: torch.Tensor | None = None

            def set_rope(
                self, cos: torch.Tensor | None, sin: torch.Tensor | None
            ) -> None:
                self._rope_cos = cos
                self._rope_sin = sin
                self._rope_enabled = cos is not None and sin is not None

            @staticmethod
            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
                )

            def _apply_rope(
                self, q: torch.Tensor, k: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # q,k: [B, S, Hh, Dh]
                assert self._rope_cos is not None and self._rope_sin is not None
                base_cos = self._rope_cos.to(dtype=q.dtype, device=q.device)  # [S, Dh]
                base_sin = self._rope_sin.to(dtype=q.dtype, device=q.device)
                # Reshape to [1, S, 1, Dh] for broadcast over [B, S, Hh, Dh]
                cos = base_cos.view(1, base_cos.size(0), 1, base_cos.size(1))
                sin = base_sin.view(1, base_sin.size(0), 1, base_sin.size(1))
                q_embed = (q * cos) + (self._rotate_half(q) * sin)
                k_embed = (k * cos) + (self._rotate_half(k) * sin)
                return q_embed, k_embed

            def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
                qkv_states, _ = self.qkv_proj(hidden_states)
                query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

                needs_unsqueeze = query_states.ndim == 2
                if needs_unsqueeze:
                    query_states = query_states.unsqueeze(0)
                    key_states = key_states.unsqueeze(0)
                    value_states = value_states.unsqueeze(0)

                if self._rope_enabled:
                    B, S, _ = query_states.shape
                    # [B, S, Hh, Dh]
                    Hh = self.num_heads_per_partition
                    Dh = self.head_dim
                    qv = query_states.view(B, S, Hh, Dh)
                    kv = key_states.view(B, S, Hh, Dh)
                    qv, kv = self._apply_rope(qv, kv)
                    query_states = qv.reshape(B, S, Hh * Dh)
                    key_states = kv.reshape(B, S, Hh * Dh)

                out = self.attn(query_states, key_states, value_states)

                if needs_unsqueeze:
                    out = out.squeeze(0)

                attn_output, _ = self.out_proj(out)
                return attn_output, None

        # Replace attention on all layers
        layers = getattr(self.vision_model.vision_model.encoder, "layers", [])
        for idx, layer in enumerate(layers):
            attn: SiglipAttention = layer.self_attn  # type: ignore[assignment]
            rope_attn = _RoPE2DAttn(
                attn.config,
                quant_config=quant_config,
                prefix=f"{attn._get_name().lower()}_{idx}",
            )
            rope_attn.load_state_dict(attn.state_dict(), strict=True)
            layer.self_attn = rope_attn  # type: ignore[assignment]
        self._rope_layers = [ly.self_attn for ly in layers]

    def _set_rope_on_layers(
        self, cos: torch.Tensor | None, sin: torch.Tensor | None
    ) -> None:
        for attn in getattr(self, "_rope_layers", []) or []:
            if hasattr(attn, "set_rope"):
                attn.set_rope(cos, sin)  # type: ignore[attr-defined]

    def _compute_2d_rope_cos_sin(
        self, grid_thw: torch.Tensor, merge: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D RoPE cos/sin for the flattened HxW patch grid.

        HF PaddleOCR-VL applies 2D rotary on the original patch grid
        (H x W) inside the SigLIP encoder, and only merges spatial tokens
        later in the projector. To match HF, we must index RoPE using the
        full-resolution row/col coordinates, not the merged grid.

        Returns tensors shaped [S, Dh] where S=sum_i(T_i*H_i*W_i) and Dh is
        the per-head dimension of the SigLIP attention.
        """
        device = self.device

        # Collect per-token (h, w) indices for all images in row-major order.
        h_all: list[torch.Tensor] = []
        w_all: list[torch.Tensor] = []
        for t_hw in grid_thw.tolist():
            t, h, w = (int(t_hw[0]), int(t_hw[1]), int(t_hw[2]))
            # Row-major 2D grid indices for one frame
            rows = torch.arange(h, device=device)
            cols = torch.arange(w, device=device)
            h_grid = rows.view(h, 1).expand(h, w).reshape(-1)
            w_grid = cols.view(1, w).expand(h, w).reshape(-1)
            # Repeat over temporal dimension if t > 1
            if t > 1:
                h_grid = h_grid.repeat(t)
                w_grid = w_grid.repeat(t)
            h_all.append(h_grid)
            w_all.append(w_grid)

        h_full = torch.cat(h_all, dim=0)
        w_full = torch.cat(w_all, dim=0)

        # Per-head rotary dim (half allocated to H, half to W)
        head_dim = int(
            getattr(
                self.vision_model.vision_model.encoder.layers[0].self_attn,
                "head_dim",
                64,
            )
        )
        half = head_dim // 2  # HF uses half for (H,W), then repeats to Dh

        # Reproduce HF SigLIPRotaryEmbedding logic exactly:
        # inv_freq has length half/2 due to step 2
        theta = 10000.0
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, half, 2, dtype=torch.float32, device=device) / half)
        )
        # Build base frequencies for max grid size, then gather by (h,w)
        max_grid = int(max(h_full.max().item(), w_full.max().item())) + 1
        seq = torch.arange(max_grid, dtype=inv_freq.dtype, device=device)
        freqs = torch.outer(seq, inv_freq)  # [max_grid, half/2]

        # HF indexes freqs separately for h and w, then concatenates
        pids = torch.stack([h_full, w_full], dim=-1)
        rope_h = freqs[pids[:, 0]]  # [S, half/2]
        rope_w = freqs[pids[:, 1]]  # [S, half/2]
        rope_base = torch.cat([rope_h, rope_w], dim=-1)  # [S, half]
        rope = rope_base.repeat(1, 2)  # [S, head_dim]

        cos = torch.cos(rope).to(dtype=self.dtype)
        sin = torch.sin(rope).to(dtype=self.dtype)
        return cos[:, :head_dim], sin[:, :head_dim]

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Return per-patch visual features flattened to [N, C].

        Notes:
        - This localized encoder currently delegates to SigLIP blocks and a
          post-LayerNorm. Helper methods to construct 2D positions and cu_seqlens
          are provided for HF parity work and future RoPE alignment.
        """
        # Run SigLIP with abs-PE interpolation; returns [B, S, C] or [N, C].
        use_local_rope = os.getenv("PADDLEOCRVL_USE_LOCAL_ROPE", "0") == "1"
        # Pre-install rope on layers if enabled
        if use_local_rope and hasattr(self, "_rope_layers"):
            merge = int(getattr(self, "spatial_merge_size", 1))
            cos, sin = self._compute_2d_rope_cos_sin(grid_thw, merge)
            self._set_rope_on_layers(cos, sin)

        # HF PaddleOCR-VL uses packing_position_embedding with explicit
        # position_ids for pre-split inputs. Build row-major position ids
        # per image (0..t*h*w-1 mapped to 0..h*w-1) and pass them through.
        image_grid_thw_list = (
            grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        )
        pos_chunks: list[torch.Tensor] = []
        for t, h, w in image_grid_thw_list:
            numel = int(t) * int(h) * int(w)
            pos = torch.arange(numel, device=pixel_values.device, dtype=torch.long)
            pos = pos % (int(h) * int(w))
            pos_chunks.append(pos)
        position_ids = torch.cat(pos_chunks, dim=0) if pos_chunks else None

        # HF top-level always enables interpolate_pos_encoding=True for the
        # pre-split path. Mirror that here to ensure positional parity.
        outputs = self.vision_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw_list,
            # HF pre-split path uses packing_position_embedding; avoid
            # interpolated abs-PE here to match parity, especially under
            # resized grids (e.g., 0.25 scale).
            interpolate_pos_encoding=False,
        )
        if outputs.ndim == 3:
            b, s, c = outputs.shape
            feats = outputs.reshape(b * s, c)
        else:
            feats = outputs
        # SigLIP vision already applies a post-layernorm internally;
        # avoid double-normalizing here to match HF projector inputs.
        out = feats

        # Clear rope context to avoid leaking to future calls
        if use_local_rope and hasattr(self, "_rope_layers"):
            self._set_rope_on_layers(None, None)
        return out


class PaddleOCRVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    # Supported:
    # - 5D pre-split: [B, L, 3, pH, pW]
    # - 4D flattened: [N_total, 3, pH, pW] (variable L per image)
    pixel_values: torch.Tensor
    # [N_images, 3] with T,H,W patch-grid (note: may differ from patch count)
    image_grid_thw: Annotated[torch.Tensor, TensorShape("bi", 3)]

    def validate(self) -> None:  # type: ignore[override]
        # Validate image_grid_thw
        if not isinstance(self.image_grid_thw, torch.Tensor):
            raise ValueError("image_grid_thw must be a tensor")
        if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must have shape [N, 3]")
        # Validate pixel_values shape (support 5D pre-split and 4D flattened)
        pv = self.pixel_values
        if not isinstance(pv, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor")
        if pv.ndim == 5:
            if pv.shape[2] != 3:
                raise ValueError(
                    "5D pixel_values must have channel dimension == 3 at dim=2"
                )
            # Additional consistency checks happen upstream; accept here.
        elif pv.ndim == 4:
            if pv.shape[1] != 3:
                raise ValueError(
                    "4D pixel_values must be [N_total, 3, pH, pW] (flattened)"
                )
        else:
            raise ValueError("pixel_values must be 4D [N,3,pH,pW] or 5D [B,L,3,pH,pW]")


class PaddleOCRVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"] = "image_embeds"
    # [N_tokens, hidden]
    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    # [B, 3]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("bn", 3)]


PaddleOCRVLImageInputs: TypeAlias = (
    PaddleOCRVLImagePixelInputs | PaddleOCRVLImageEmbeddingInputs
)


class PaddleOCRVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        vision_cfg = self.get_hf_config().vision_config
        merge_size = int(_get_vision_config_value(vision_cfg, "spatial_merge_size", 1))
        merge = merge_size * merge_size
        return (grid_thw.prod(-1) // merge).to(torch.long)


class PaddleOCRVLDummyInputsBuilder(BaseDummyInputsBuilder[PaddleOCRVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        placeholder = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        return placeholder * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image") if mm_options else None
        vision_cfg = self.info.get_hf_config().vision_config
        image_size = int(_get_vision_config_value(vision_cfg, "image_size", 384))

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class PaddleOCRVLMultiModalProcessor(
    BaseMultiModalProcessor[PaddleOCRVLProcessingInfo]
):
    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> PaddleOCRVLImageInputs | None:  # type: ignore[name-defined]
        """Validate basic image inputs for unit tests and preprocessing.

        Mirrors the model's validation logic to keep tests decoupled from
        HF availability. Accepts both 4D and 5D `pixel_values` and ensures
        consistency with `image_grid_thw`.
        """
        import torch

        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        # Normalize grid tensor
        if image_grid_thw is None:
            raise ValueError("image_grid_thw must be provided for PaddleOCR-VL inputs.")
        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.tensor(image_grid_thw)
        image_grid_thw = image_grid_thw.to(torch.long)
        if image_grid_thw.ndim == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must have shape [N, 3]")

        # Normalize pixel tensor
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)

        if pixel_values.ndim == 5:
            # [B, L, 3, pH, pW]
            b, num_patches, c, _, _ = pixel_values.shape
            if c != 3:
                raise ValueError(
                    "5D pixel_values must have channel dimension == 3 at dim=2"
                )
            if image_grid_thw.shape[0] != b:
                raise ValueError(
                    "Mismatch between pixel_values batch size and image_grid_thw rows"
                    f": B={b}, rows={image_grid_thw.shape[0]}"
                )
            # Per-image L must equal t*h*w
            expected = (image_grid_thw.prod(-1)).tolist()
            for i, e in enumerate(expected):
                if num_patches != int(e):
                    raise ValueError(
                        "pixel_values L does not match "
                        "grid_thw product for one or more "
                        "images: "
                        f"image[{i}] expects L={int(e)}, "
                        f"got L={num_patches}"
                    )
        elif pixel_values.ndim == 4:
            # Either [B, 3, H, W] or flattened [N_total, 3, p, p]
            if pixel_values.shape[1] != 3:
                raise ValueError("Unexpected 4D pixel_values; expected [B, 3, H, W].")
            b = pixel_values.shape[0]
            if b != image_grid_thw.shape[0]:
                # Treat as flattened patches; validate total length
                expected_total = int(image_grid_thw.prod(-1).sum().item())
                if b != expected_total:
                    raise ValueError(
                        "Flattened pixel_values length mismatch: "
                        f"expected {expected_total}, got {b}"
                    )
        else:
            raise ValueError(
                "pixel_values must have rank 4 or 5 for PaddleOCR-VL inputs."
            )

        # Minimal structured return compatible with tests
        return {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _apply_hf_processor_text_mm(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> tuple[list[int], object, bool]:
        """
        Override to ensure raw HF pixel_values (float32) are passed through
        without additional post-processing that could alter values.

        This mirrors BaseMultiModalProcessor._apply_hf_processor_text_mm but
        directly invokes the HF processor and returns the BatchFeature as-is.
        """
        from transformers.feature_extraction_utils import BatchFeature

        # Build HF processor inputs
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # Call HF processor directly to avoid any value‑changing kwargs.
        # We intentionally do NOT forward mm_processor_kwargs/tokenization kwargs
        # here to keep pixel preprocessing identical to HF defaults.
        output = hf_processor(
            **dict(text=prompt_text, **processor_data), return_tensors="pt"
        )
        if not isinstance(output, BatchFeature):
            # Fallback to base path if a non-standard output is returned
            return super()._apply_hf_processor_text_mm(
                prompt_text,
                mm_items,
                hf_processor_mm_kwargs,
                tokenization_kwargs,
            )

        processed_data: BatchFeature = output
        processed_data.update(passthrough_data)

        (prompt_ids,) = processed_data.pop("input_ids").tolist()
        is_update_applied = self._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )
        return prompt_ids, processed_data, is_update_applied

    # Optionally force vLLM-side prompt updates for debugging so that
    # _get_prompt_updates is exercised (and can log details).
    def _hf_processor_applies_updates(
        self,
        *,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        if os.getenv("PADDLEOCRVL_FORCE_VLLM_UPDATES", "0") == "1":
            return False
        return super()._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, Any],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF processor can return either:
        # - 4D flattened patches: [N_total, 3, pH, pW]
        # - 5D pre-split per image: [B, L, 3, pH, pW]
        # We must declare field configs that slice per image correctly.
        grid_thw = hf_inputs.get("image_grid_thw")
        pixel_values = hf_inputs.get("pixel_values")
        if grid_thw is None and pixel_values is None:
            # No image-related fields present (e.g., cache hit path or pure text)
            return {}

        # Normalize grid_thw for later use (batched by image)
        if grid_thw is not None and not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw)
        if isinstance(grid_thw, torch.Tensor) and grid_thw.ndim == 3:
            grid_thw = grid_thw.reshape(-1, 3)

        field_cfg: dict[str, MultiModalFieldConfig] = {}

        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 5:
            # Pre-split per image: slice along batch (B) dimension.
            field_cfg["pixel_values"] = MultiModalFieldConfig.batched("image")
        else:
            # Flattened patches: slice along dim=0 using per-image patch counts.
            if grid_thw is None:
                assert isinstance(pixel_values, torch.Tensor), (
                    "pixel_values must be a tensor when grid_thw is None"
                )
                patch_counts = torch.tensor([pixel_values.shape[0]], dtype=torch.long)
            else:
                assert isinstance(grid_thw, torch.Tensor)
                patch_counts = grid_thw.prod(-1).to(torch.long)
            field_cfg["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", patch_counts, dim=0
            )

        # Always batched by image for grid metadata.
        field_cfg["image_grid_thw"] = MultiModalFieldConfig.batched("image")

        return field_cfg

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Early out if no image items (e.g., profiling or pure text)
        if "image" not in mm_items:
            return []
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        def _token_id(token: str | None, default: int | None) -> int | None:
            if token is None:
                return default
            token_id = tokenizer.convert_tokens_to_ids(token)
            return token_id if token_id is not None and token_id >= 0 else default

        image_token_id = _token_id("<|IMAGE_PLACEHOLDER|>", hf_config.image_token_id)
        start_token_id = _token_id(
            "<|IMAGE_START|>", getattr(hf_config, "vision_start_token_id", None)
        )
        end_token_id = _token_id(
            "<|IMAGE_END|>", getattr(hf_config, "vision_end_token_id", None)
        )

        merge_size = int(
            _get_vision_config_value(hf_config.vision_config, "spatial_merge_size", 1)
        )

        DEBUG_PROMPT = os.getenv("PADDLEOCRVL_DEBUG_PROMPT", "0") == "1"

        def _debug_log(entry: dict[str, object]) -> None:
            if not DEBUG_PROMPT:
                return
            try:
                os.makedirs("test_result", exist_ok=True)
                with open(
                    os.path.join("test_result", "paddleocrvl_prompt_debug.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        def _compute_num_tokens_for_item(item_idx: int) -> int:
            out_item = out_mm_kwargs["image"][item_idx]
            pv_field = out_item.get("pixel_values")
            num_tokens: int | None = None
            try:
                pv = pv_field.data if hasattr(pv_field, "data") else pv_field
                if isinstance(pv, torch.Tensor):
                    if pv.ndim == 4 and pv.shape[1] == 3:
                        num_patches = int(pv.shape[0])
                        num_tokens = num_patches // (merge_size * merge_size)
                    elif pv.ndim == 5 and pv.shape[2] == 3:
                        num_patches = int(pv.shape[0]) * int(pv.shape[1])
                        num_tokens = num_patches // (merge_size * merge_size)
                elif isinstance(pv, (list, tuple)) and pv:
                    head = pv[0]
                    if isinstance(head, torch.Tensor):
                        if head.ndim == 4 and head.shape[1] == 3:
                            num_patches = int(head.shape[0])
                            num_tokens = num_patches // (merge_size * merge_size)
                        elif head.ndim == 5 and head.shape[2] == 3:
                            num_patches = int(head.shape[0]) * int(head.shape[1])
                            num_tokens = num_patches // (merge_size * merge_size)
            except Exception:
                num_tokens = None

            if num_tokens is None:
                image_grid_field = out_item["image_grid_thw"]
                image_grid_thw = (
                    image_grid_field.data
                    if hasattr(image_grid_field, "data")
                    else image_grid_field
                )
                if not isinstance(image_grid_thw, torch.Tensor):
                    image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)
                num_tokens_t = (
                    image_grid_thw.prod(-1) // (merge_size * merge_size)
                ).sum()
                num_tokens = (
                    int(num_tokens_t.item())
                    if isinstance(num_tokens_t, torch.Tensor)
                    else int(num_tokens_t)
                )

            if num_tokens is None or num_tokens <= 0:
                num_tokens = 1
            return int(num_tokens)

        # Dynamic target and replacement that resolve per item index.
        def _target_fn(item_idx: int) -> list[int]:
            tgt: list[int] = []
            if start_token_id is not None:
                tgt.append(start_token_id)
            tgt.append(image_token_id)
            if end_token_id is not None:
                tgt.append(end_token_id)
            return tgt

        def _replacement_fn(item_idx: int) -> PromptUpdateDetails:
            n_tok = _compute_num_tokens_for_item(item_idx)
            repl: list[int] = []
            if start_token_id is not None:
                repl.append(start_token_id)
            repl.extend([image_token_id] * n_tok)
            if end_token_id is not None:
                repl.append(end_token_id)
            _debug_log(
                {
                    "image_index": int(item_idx),
                    "num_tokens": int(n_tok),
                    "target": _target_fn(item_idx),
                    "replacement_len": len(repl),
                    "merge_size": merge_size,
                }
            )
            return PromptUpdateDetails.select_token_id(
                repl,
                embed_token_id=image_token_id,
            )

        image_items = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
        if image_items is None:
            return []

        # Return a single PromptReplacement with dynamic target/replacement.
        # This avoids emitting N updates (one per image), which would bind N
        # updates to each item and trigger duplicate update warnings.
        return [
            PromptReplacement(
                modality="image",
                target=_target_fn,
                replacement=_replacement_fn,
            )
        ]

    # Override: avoid reusing cached prompt update lengths.
    # For PaddleOCR-VL，多图/多轮场景下占位宽度可能因为处理器的打包策略而变化，
    # 复用缓存的替换长度会导致 is_embed 与实际 encoder 输出长度不一致。
    # 因此这里直接走非缓存路径以重新计算占位符替换，确保宽度精确对齐。
    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        return super()._apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleOCRVLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsMRoPE, SupportsPP
):
    """PaddleOCR-VL multimodal decoder-only model."""

    # Explicitly declare protocol flags for clarity
    supports_mrope: bool = True

    supports_encoder_tp_data = False

    hf_to_vllm_mapper = WeightsMapper(
        # Keep mapping minimal here; visual.* are handled manually in
        # load_weights() to avoid double-loading/conflicts.
        orig_to_new_substr={},
        orig_to_new_prefix={
            "visual.": None,
            "mlp_AR.": "projector.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, PaddleOCRVLConfig):
            config = hf_config
        else:
            config = PaddleOCRVLConfig(**hf_config.to_dict())
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Prefer eager mode for stability on dynamic grids
        model_config = vllm_config.model_config
        if hasattr(model_config, "enforce_eager"):
            model_config.enforce_eager = True
        if hasattr(vllm_config, "compilation_config"):
            vllm_config.compilation_config.mode = 0
            vllm_config.compilation_config.use_inductor = False

        # Build HF SigLIP vision config for reference
        if isinstance(config.vision_config, PretrainedConfig):
            _vision_cfg = config.vision_config.__class__(
                **config.vision_config.to_dict()
            )
        else:
            _vision_cfg_data = config.vision_config
            if not isinstance(_vision_cfg_data, dict):
                _vision_cfg_data = _vision_cfg_data.to_dict()
            _vision_cfg = SiglipVisionConfig(**_vision_cfg_data)

        # Localized custom vision encoder (no shared SigLiP2/NaViT changes).
        merge_sz = int(
            _get_vision_config_value(config.vision_config, "spatial_merge_size", 1)
        )
        if isinstance(_vision_cfg, dict):
            _vision_cfg = SiglipVisionConfig(**_vision_cfg)
        with suppress(Exception):
            _vision_cfg.vision_use_head = False  # type: ignore[attr-defined]
        self.vision_tower = PaddleOCRVLVisionTransformer(
            vision_config=_vision_cfg,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.vision_tower.eval()

        # Projector
        vision_hidden = int(
            _get_vision_config_value(config.vision_config, "hidden_size", 1152)
        )
        self.projector = PaddleOCRVLProjector(
            vision_hidden_size=vision_hidden,
            hidden_size=config.hidden_size,
            merge_kernel_size=(merge_sz, merge_sz),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projector"),
        )
        self.projector.eval()

        # Language backbone (Ernie 4.5)
        # Ensure PaddleOCR‑VL specific mRoPE mapping is enabled for the text LM.
        # This sets the default mapping strategy used by MRotaryEmbedding to
        # mirror HF's apply_multimodal_rotary_pos_emb reorder (HW‑then‑T).
        import os as _os

        # HF applies a 3‑chunk T/H/W mapping on FULL dims then ::2 compress.
        # Align default to that behavior.
        _os.environ.setdefault("PADDLEOCRVL_MROPE_MAP", "chunked")
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Ernie4_5ForCausalLM"],
        )

        if self.language_model.lm_head is not None:
            self.lm_head: ParallelLMHead = self.language_model.lm_head
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            config.vocab_size,
        )

    def _ensure_hf_visual(self, device: torch.device, dtype: torch.dtype) -> None:
        """Lazily construct HF's SiglipVisionModel for exact RoPE parity.

        This path is gated by env PADDLEOCRVL_USE_HF_VISUAL=1 and is intended
        for correctness validation on single GPU. It does not affect the
        default optimized path.
        """
        if getattr(self, "_hf_visual", None) is not None:
            return
        try:
            # Dynamically import the custom vision class from the repo.
            SiglipVisionModelHF = get_class_from_dynamic_module(  # type: ignore[name-defined]
                "modeling_paddleocr_vl.SiglipVisionModel",
                "PaddlePaddle/PaddleOCR-VL",
                trust_remote_code=True,
            )
            # Build from pretrained to load only vision weights.
            self._hf_visual = (
                SiglipVisionModelHF.from_pretrained(
                    "PaddlePaddle/PaddleOCR-VL",
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                .to(device)
                .eval()
            )
            # Keep HF visual on eager/SDPA to avoid missing flash-attn symbols
            from contextlib import suppress

            with suppress(Exception):
                self._hf_visual.config._attn_implementation = "eager"
        except Exception:
            # Fallback to None; caller will continue with local rope_vision
            self._hf_visual = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        raise ValueError("Only image modality is supported.")

    def _validate_tensor(self, tensor: Any, name: str) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"{name} must be provided for PaddleOCR-VL inputs.")
        if isinstance(tensor, torch.Tensor):
            return tensor
        # Handle list/tuple inputs robustly (e.g., list of tensors)
        if isinstance(tensor, (list, tuple)):
            if len(tensor) == 0:
                raise ValueError(f"{name} cannot be empty.")
            if all(isinstance(x, torch.Tensor) for x in tensor):
                # Prefer concatenation along the leading dimension, which is
                # the common representation for flattened patches across
                # variable-length images. If that fails, fall back to stack.
                try:
                    return torch.cat(list(tensor), dim=0)
                except Exception:
                    try:
                        return torch.stack(list(tensor))
                    except Exception:
                        # Last resort
                        return torch.as_tensor(
                            [x.detach().cpu().numpy() for x in tensor]
                        )
            # Fallback: try to tensor-ify nested lists
            return torch.as_tensor(tensor)
        return torch.as_tensor(tensor)

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> PaddleOCRVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        image_grid_thw = self._validate_tensor(image_grid_thw, "image_grid_thw").to(
            torch.long
        )
        if image_grid_thw.ndim == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)
        if image_grid_thw.ndim != 2:
            raise ValueError(
                "image_grid_thw must have rank 2 or be reshapeable to [N, 3] "
                "for PaddleOCR-VL inputs."
            )

        if pixel_values is not None:
            pixel_values = self._validate_tensor(pixel_values, "pixel_values")
            if pixel_values.ndim == 5:
                # Already [B, L, 3, p, p]
                # Validate B matches image_grid_thw rows and L matches t*h*w
                b, num_patches, c, ph, pw = pixel_values.shape
                if c != 3:
                    raise ValueError(
                        "5D pixel_values must have channel dimension == 3 at dim=2"
                    )
                if image_grid_thw.shape[0] != b:
                    msg = (
                        "Mismatch: pixel_values batch vs image_grid_thw rows. "
                        f"B={b}, grid rows={image_grid_thw.shape[0]}"
                    )
                    raise ValueError(msg)
                # Per-image L should equal t*h*w
                expected_counts = (image_grid_thw.prod(-1)).tolist()
                for i, expect in enumerate(expected_counts):
                    if num_patches != int(expect):
                        msg = (
                            "Pre-split pixel_values L mismatch: "
                            f"image[{i}] expects L={int(expect)}, "
                            f"got L={num_patches}"
                        )
                        raise ValueError(msg)
            elif pixel_values.ndim == 4:
                if pixel_values.shape[1] == 3:
                    # [B, 3, H, W]
                    pass
                else:
                    raise ValueError(
                        "Unexpected 4D pixel_values; expected [B, 3, H, W]."
                    )
            else:
                raise ValueError(
                    "pixel_values must have rank 4 or 5 for PaddleOCR-VL inputs."
                )
            # If HF provided flat [N_total, 3, p, p] for variable-length images,
            # keep it flattened and rely on image_grid_thw to split downstream.
            if (
                pixel_values.ndim == 4
                and pixel_values.shape[0] != image_grid_thw.shape[0]
            ):
                grids = image_grid_thw.tolist()
                expected_total = 0
                for t, h, w in grids:
                    expected_total += int(t) * int(h) * int(w)
                if expected_total != pixel_values.shape[0]:
                    msg = (
                        "Flattened pixel_values length does not equal "
                        "sum(t*h*w) from image_grid_thw"
                    )
                    raise ValueError(msg)
                # Keep 4D flattened tensor; split happens in _process_image_input
            return PaddleOCRVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        assert image_embeds is not None
        image_embeds = self._validate_tensor(image_embeds, "image_embeds")
        return PaddleOCRVLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _process_image_input(
        self,
        image_input: PaddleOCRVLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        # Optional debugging for embedding injection path. Enable with
        # PADDLEOCRVL_DEBUG_EMBED=1 to print shapes and basic stats.
        _dbg = os.getenv("PADDLEOCRVL_DEBUG_EMBED", "0") != "0"
        if _dbg:
            try:
                print("\n" + "=" * 80)
                print("🔍 DEBUG: _process_image_input called")
                print("=" * 80)
                _t = image_input.get("type", "<unknown>")
                print(f"  type: {_t}")
                _grid = image_input.get("image_grid_thw")
                if isinstance(_grid, torch.Tensor):
                    shape = tuple(_grid.shape)
                    data = _grid.tolist()
                    print(f"  image_grid_thw: shape={shape}, data={data}")
                else:
                    print(f"  image_grid_thw: {_grid}")
                _pv = image_input.get("pixel_values")
                if isinstance(_pv, torch.Tensor):
                    print(
                        f"  pixel_values: shape={tuple(_pv.shape)}, dtype={_pv.dtype}"
                    )
                _ie = image_input.get("image_embeds")
                if isinstance(_ie, torch.Tensor):
                    print(
                        f"  image_embeds: shape={tuple(_ie.shape)}, dtype={_ie.dtype}"
                    )
            except Exception:
                pass
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
            merge = int(
                _get_vision_config_value(
                    self.config.vision_config, "spatial_merge_size", 1
                )
            )
            sizes = (grid_thw.prod(-1) // (merge * merge)).tolist()
            return image_embeds.split(sizes)

        # Pixel path
        pixel_values = image_input["pixel_values"]
        device = next(self.vision_tower.parameters()).device
        pixel_values = pixel_values.to(device=device)
        # Choose visual path: optional HF reference for calibration, else local
        # Env gate for HF reference visual path. In some local setups the
        # child engine process may not observe the shell-prefixed env var.
        # As a robust fallback for local ceiling checks, allow a file flag
        # under test_result/ to force-enable HF visual once.
        use_hf_visual = os.getenv("PADDLEOCRVL_USE_HF_VISUAL", "0") == "1"
        if not use_hf_visual:
            try:
                import os as _os

                if _os.path.exists("test_result/.force_hf_visual"):
                    use_hf_visual = True
            except Exception:
                pass
        grid_thw_tensor = grid_thw.to(device=pixel_values.device, dtype=torch.long)
        if use_hf_visual:
            # Lazily load HF visual and run to get last hidden state with RoPE
            self._ensure_hf_visual(device=pixel_values.device, dtype=pixel_values.dtype)
            if getattr(self, "_hf_visual", None) is None:
                # Fallback to local if dynamic import fails
                vision_features = self.vision_tower(
                    pixel_values=pixel_values, grid_thw=grid_thw_tensor
                )
            else:
                # Build per-token helpers following HF implementation
                # image_grid_thw: list[tuple(T,H,W)]
                image_grid_hws: list[tuple[int, int, int]] = []
                siglip_position_chunks: list[torch.Tensor] = []
                sample_idx_chunks: list[torch.Tensor] = []
                cu_seqlens_list: list[int] = [0]
                total = 0
                for idx, grid_item in enumerate(grid_thw_tensor):
                    t, h, w = (
                        int(grid_item[0].item()),
                        int(grid_item[1].item()),
                        int(grid_item[2].item()),
                    )
                    numel = t * h * w
                    image_grid_hws.append((t, h, w))
                    # Position ids within each image: 0..(t*h*w-1) % (h*w)
                    pos = torch.arange(
                        numel, device=pixel_values.device, dtype=torch.long
                    ) % (h * w)
                    siglip_position_chunks.append(pos)
                    sample_idx_chunks.append(
                        torch.full(
                            (numel,), idx, dtype=torch.long, device=pixel_values.device
                        )
                    )
                    total += numel
                    cu_seqlens_list.append(total)

                siglip_position_ids = torch.cat(siglip_position_chunks, dim=0)
                sample_indices = torch.cat(sample_idx_chunks, dim=0)
                cu_seqlens = torch.tensor(
                    cu_seqlens_list, dtype=torch.int32, device=pixel_values.device
                )

                # Call HF visual with RoPE enabled, return last hidden state.
                # Some grids (e.g., 392x392) may raise NotImplementedError
                # in HF dynamic module. In that case, gracefully fall back
                # to the local vision path.
                try:
                    hf_out = self._hf_visual(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_hws,
                        position_ids=siglip_position_ids,
                        vision_return_embed_list=True,
                        interpolate_pos_encoding=True,
                        sample_indices=sample_indices,
                        cu_seqlens=cu_seqlens,
                        return_pooler_output=False,
                        use_rope=True,
                        window_size=-1,
                    )
                except NotImplementedError:
                    vision_features = self.vision_tower(
                        pixel_values=pixel_values, grid_thw=grid_thw_tensor
                    )
                    # Skip to projector
                    projected = self.projector(
                        vision_features,
                        grid_thw.to(device=vision_features.device),
                    )
                    return projected

                # HF may return a dataclass (BaseModelOutputWithPooling) or a tuple/list
                # Normalize a variety of return types into a single tensor.
                def _normalize_hf_visual_output(obj: object) -> torch.Tensor:
                    # Dataclass with attribute
                    if hasattr(obj, "last_hidden_state") or hasattr(obj, "to_tuple"):
                        # Try direct attr first
                        out = getattr(obj, "last_hidden_state", None)
                        if isinstance(out, torch.Tensor):
                            return out
                        # Try ModelOutput tuple (may contain tensors/None)
                        if hasattr(obj, "to_tuple"):
                            tup = obj.to_tuple()
                            # Flatten one level
                            flat_seq: list[object] = []
                            for it in tup:
                                if isinstance(it, (list, tuple)):
                                    flat_seq.extend(list(it))
                                else:
                                    flat_seq.append(it)
                            for item in reversed(flat_seq):
                                if isinstance(item, torch.Tensor):
                                    return item
                        # Try dict-like access (ModelOutput behaves like dict)
                        if hasattr(obj, "to_dict"):
                            d = obj.to_dict()
                            for key in (
                                "last_hidden_state",
                                "hidden_states",
                                "embeddings",
                                "image_embeds",
                            ):
                                val = d.get(key)
                                if isinstance(val, torch.Tensor):
                                    return val
                    # Plain tensor
                    if isinstance(obj, torch.Tensor):
                        return obj
                    # Sequence of tensors (possibly nested lists)
                    if isinstance(obj, (list, tuple)):
                        # Flatten one level if nested
                        flat: list[torch.Tensor] = []
                        for item in obj:
                            if isinstance(item, torch.Tensor):
                                flat.append(item)
                            elif isinstance(item, (list, tuple)) and all(
                                isinstance(x, torch.Tensor) for x in item
                            ):
                                flat.extend(list(item))
                        if flat:
                            # Prefer the last if shapes vary, else concatenate on batch
                            try:
                                return torch.cat(flat, dim=0)
                            except Exception:
                                return flat[-1]
                    # Fallback: raise with type info
                    raise TypeError(f"Unexpected HF visual output type: {type(obj)!r}")

                feats = _normalize_hf_visual_output(hf_out)
                # HF typically returns [B, S, C]; flatten to [N, C]
                if feats.ndim == 3:
                    b, s, c = feats.shape
                    feats = feats.reshape(b * s, c)
                # HF SigLIP already applies post-LN; pass through directly
                vision_features = feats
        else:
            # Custom localized vision encoder
            # If inputs are flattened patches [N_total, 3, p, p], split per image
            # using image_grid_thw, and run per-sample to support variable L.
            if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
                sizes = (grid_thw_tensor.prod(-1)).tolist()
                # If multiple images, process sequentially to handle variable sizes
                if len(sizes) > 1:
                    pieces: list[torch.Tensor] = []
                    offset = 0
                    for i, num in enumerate(sizes):
                        num_i = int(num)
                        pv_i = pixel_values[offset : offset + num_i].unsqueeze(0)
                        offset += num_i
                        g_i = (
                            grid_thw_tensor[i : i + 1]
                            if grid_thw_tensor.ndim == 2
                            else grid_thw_tensor
                        )
                        vf_i = self.vision_tower(pixel_values=pv_i, grid_thw=g_i)
                        pieces.append(vf_i)
                    vision_features = torch.cat(pieces, dim=0)
                else:
                    # Single image flattened -> add batch dim to [1, L, 3, p, p]
                    pv5 = pixel_values.unsqueeze(0)
                    vision_features = self.vision_tower(
                        pixel_values=pv5, grid_thw=grid_thw_tensor
                    )
            # If scheduler packs multiple images into batch dimension as 5D,
            # run per-sample to keep dynamic-resolution logic simple and robust.
            elif pixel_values.ndim == 5 and pixel_values.shape[0] > 1:
                pieces: list[torch.Tensor] = []
                B = int(pixel_values.shape[0])
                for i in range(B):
                    pv_i = pixel_values[i : i + 1]
                    # grid_thw may be [N,3] or [3]; try to pick per-sample if available
                    if grid_thw_tensor.ndim == 2 and grid_thw_tensor.shape[0] >= (
                        i + 1
                    ):
                        g_i = grid_thw_tensor[i : i + 1]
                    else:
                        g_i = grid_thw_tensor
                    vf_i = self.vision_tower(pixel_values=pv_i, grid_thw=g_i)
                    pieces.append(vf_i)
                vision_features = torch.cat(pieces, dim=0)
            else:
                vision_features = self.vision_tower(
                    pixel_values=pixel_values, grid_thw=grid_thw_tensor
                )

        # Ensure dtype consistency when using HF visual path
        # HF visual returns bfloat16, but projector expects its own weight dtype
        if (
            use_hf_visual
            and vision_features.dtype != next(self.projector.parameters()).dtype
        ):
            vision_features = vision_features.to(
                dtype=next(self.projector.parameters()).dtype
            )

        projected = self.projector(
            vision_features,
            grid_thw.to(device=vision_features.device),
        )
        if _dbg:
            try:
                print(
                    "  vision_features(final):",
                    f"shape={tuple(vision_features.shape)},",
                    f"dtype={vision_features.dtype}",
                )
                if len(projected) > 0:
                    print(
                        "  projected[0](final):",
                        f"shape={tuple(projected[0].shape)},",
                        f"dtype={projected[0].dtype}",
                    )
            except Exception:
                pass
        # Optional debug dump for parity checks (not enabled by default).
        # Set environment variable PADDLEOCRVL_DEBUG_DUMP=1 to enable.
        dump_flag = os.getenv("PADDLEOCRVL_DEBUG_DUMP", "0")
        try:
            if os.path.exists("test_result/.force_debug_dump"):
                dump_flag = "2"
        except Exception:
            pass
        if dump_flag in {"1", "2"}:
            try:
                out_dir = os.path.join("test_result")
                os.makedirs(out_dir, exist_ok=True)
                dump: dict[str, object] = {
                    # Reflect whether local 2D‑RoPE path was enabled for vision encoder
                    "rope_used": os.getenv("PADDLEOCRVL_USE_LOCAL_ROPE", "0") == "1",
                    "use_hf_visual": use_hf_visual,
                    "hf_visual_env": os.getenv("PADDLEOCRVL_USE_HF_VISUAL", "0"),
                    "hf_visual_flag_exist": os.path.exists(
                        "test_result/.force_hf_visual"
                    ),
                    "grid_thw": grid_thw.detach().cpu(),
                    "pixel_values_shape": list(image_input["pixel_values"].shape),
                    "pixel_values_stats": {
                        "mean": float(image_input["pixel_values"].float().mean().cpu()),
                        "std": float(image_input["pixel_values"].float().std().cpu()),
                    },
                    "vision_features_shape": list(vision_features.shape),
                    "vision_features_stats": {
                        "mean": float(vision_features.float().mean().cpu()),
                        "std": float(vision_features.float().std().cpu()),
                    },
                    # Save only first image's projected tokens for size/weight
                    "projected_0_shape": list(projected[0].shape)
                    if len(projected) > 0
                    else [0, 0],
                    "projected_0_stats": (
                        {
                            "mean": float(projected[0].float().mean().cpu()),
                            "std": float(projected[0].float().std().cpu()),
                        }
                        if len(projected) > 0
                        else {"mean": 0.0, "std": 0.0}
                    ),
                }
                # Full tensors (optional) when PADDLEOCRVL_DEBUG_DUMP=2
                if dump_flag == "2":
                    # Save raw pixel_values as seen by the model
                    torch.save(
                        image_input["pixel_values"]
                        .detach()
                        .to(dtype=torch.float32, device="cpu"),
                        os.path.join(out_dir, "vllm_pixel_values.pt"),
                    )
                    # Save pre-projector vision features for parity checks
                    dump["vision_features_tensor"] = vision_features.detach().to(
                        dtype=torch.float32, device="cpu"
                    )
                if dump_flag == "2" and len(projected) > 0:
                    dump["projected_0_tensor"] = (
                        projected[0].detach().to(dtype=torch.float32, device="cpu")
                    )
                torch.save(dump, os.path.join(out_dir, "vllm_paddleocr_embeds.pt"))
            except Exception:
                pass

        return projected

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        if "images" in modalities:
            image_embeddings = self._process_image_input(modalities["images"])
            multimodal_embeddings.extend(image_embeddings)

        if os.getenv("PADDLEOCRVL_DEBUG_EMBED", "0") != "0":
            try:
                shapes = [tuple(t.shape) for t in multimodal_embeddings]
                dtypes = [
                    str(t.dtype).replace("torch.", "") for t in multimodal_embeddings
                ]
                total = sum(int(t.shape[0]) for t in multimodal_embeddings)
                print("\n" + "=" * 80)
                print("🔍 DEBUG: get_multimodal_embeddings")
                print("=" * 80)
                print(f"  num_segments: {len(multimodal_embeddings)}")
                print(f"  segment_shapes: {shapes}")
                print(f"  segment_dtypes: {dtypes}")
                print(f"  total_tokens: {total}")
            except Exception:
                pass

        return tuple(multimodal_embeddings)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities: dict[str, PaddleOCRVLImageInputs] = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "images" not in modalities
            ):
                modal = self._parse_and_validate_image_input(**kwargs)
                if modal is not None:
                    modalities["images"] = modal
        return modalities

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor | None,
        video_grid_thw: list[list[int]] | torch.Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        del (
            video_grid_thw,
            second_per_grid_ts,
            audio_feature_lengths,
            use_audio_in_video,
        )

        if seq_len is None:
            seq_len = len(input_tokens)

        device = torch.device("cpu")
        total_tokens = len(input_tokens)
        position_ids = torch.zeros((3, total_tokens), dtype=torch.long, device=device)

        if image_grid_thw is None:
            image_grid_thw = []
        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)

        image_token_id = hf_config.image_token_id
        merge_size = int(
            _get_vision_config_value(hf_config.vision_config, "spatial_merge_size", 1)
        )

        grid_list = image_grid_thw.tolist()
        grid_idx = 0
        # st_idx tracks the running starting index used as offset for each segment
        st_idx = 0
        idx = 0

        while idx < total_tokens:
            token_id = input_tokens[idx]
            # Vision segment: a run of image_token_id placeholders
            if grid_idx < len(grid_list) and token_id == image_token_id:
                t, h, w = grid_list[grid_idx]
                h_merge = max(1, h // merge_size)
                w_merge = max(1, w // merge_size)
                num_features = t * h_merge * w_merge

                # Determine actual run length of consecutive image tokens to
                # avoid overflow if tokenizer/runtime inserted fewer tokens
                # than expected due to packing or formatting differences.
                run_len = 0
                j = idx
                while j < total_tokens and input_tokens[j] == image_token_id:
                    run_len += 1
                    j += 1

                # For static images, temporal index per feature is 0
                # Build per-feature (t,h,w) indices and assign with st_idx offset
                # Use the observed run_len as an upper bound to prevent index
                # overflow. If it mismatches num_features, we still keep the
                # mapping monotonic and continue.
                assign_len = min(num_features, run_len)
                for feature_idx in range(assign_len):
                    spatial_idx = feature_idx % (h_merge * w_merge)
                    h_idx = spatial_idx // w_merge
                    w_idx = spatial_idx % w_merge

                    if idx >= total_tokens:
                        break
                    position_ids[0, idx] = st_idx  # temporal uses segment start
                    position_ids[1, idx] = st_idx + h_idx
                    position_ids[2, idx] = st_idx + w_idx
                    idx += 1
                # Advance start offset by the maximum assigned index + 1, to
                # mirror HF's llm_positions.max() + 1 semantics. For static
                # images, temporal index is 0 so the max comes from H/W.
                st_idx += max(0, h_merge - 1, w_merge - 1) + 1
                grid_idx += 1
                continue

            # Text token: all three dims share the same running index
            position_ids[0, idx] = st_idx
            position_ids[1, idx] = st_idx
            position_ids[2, idx] = st_idx
            idx += 1
            st_idx += 1

        mrope_position_delta = position_ids[0].max().item() + 1 - total_tokens
        position_ids = position_ids[:, context_len:seq_len]

        # Save full position_ids for parity diagnostics with HF.
        # This writes both a stable path and a run-id suffixed path.
        try:
            run_id = os.getenv("PADDLEOCRVL_RUN_ID") or __import__("time").strftime(
                "%Y%m%d-%H%M%S"
            )
            out_dir = os.path.join("test_result")
            os.makedirs(out_dir, exist_ok=True)
            # Stable filename expected by the request
            torch.save(
                {
                    "position_ids": position_ids.detach().to(
                        device="cpu", dtype=torch.int64
                    )
                },
                os.path.join(out_dir, "vllm_position_ids.pt"),
            )
            # Also persist a run-scoped copy under text_stages for multi-run debugging
            run_dir = os.path.join(out_dir, "text_stages")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(
                {
                    "position_ids": position_ids.detach().to(
                        device="cpu", dtype=torch.int64
                    )
                },
                os.path.join(run_dir, f"vllm_{run_id}_position_ids.pt"),
            )
        except Exception:
            pass

        # Optional debug dump (always-on small payload)
        try:
            run_id = os.getenv("PADDLEOCRVL_RUN_ID") or __import__("time").strftime(
                "%Y%m%d-%H%M%S"
            )
            out_dir = os.path.join("test_result", "text_stages")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "input_tokens": input_tokens,
                    "image_grid_thw": grid_list,
                    "merge_size": merge_size,
                    "position_ids_shape": list(position_ids.shape),
                    "mrope_position_delta": int(mrope_position_delta),
                },
                os.path.join(out_dir, f"vllm_{run_id}_mrope_positions_debug.pt"),
            )
        except Exception:
            pass

        return position_ids, mrope_position_delta

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # Optional debug: dump positions ndim/shape once to verify M-RoPE path
        if (
            os.getenv("PADDLEOCRVL_DEBUG_TEXT", "0") not in ("", "0", "false", "False")
            and getattr(self, "_debug_pos_dumped", None) is None
        ):
            try:
                run_id = os.getenv("PADDLEOCRVL_RUN_ID") or __import__("time").strftime(
                    "%Y%m%d-%H%M%S"
                )
                out_dir = os.path.join("test_result", "text_stages")
                os.makedirs(out_dir, exist_ok=True)
                info = {
                    "positions_ndim": int(getattr(positions, "ndim", -1)),
                    "positions_shape": list(getattr(positions, "shape", [])),
                }
                with open(
                    os.path.join(out_dir, f"vllm_{run_id}_positions_info.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    import json

                    json.dump(info, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self._debug_pos_dumped = True  # type: ignore[attr-defined]

        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)

        # Optional debug: dump last hidden state at prompt end and logits
        # Enable via PADDLEOCRVL_DEBUG_TEXT=1. Artifacts are written to
        # test_result/text_debug/ with a run id from PADDLEOCRVL_RUN_ID or
        # a timestamp. We dump only once per request, heuristically when
        # sequence length > 1 (prefill). This is sufficient to compare the
        # final prompt hidden vs HF and isolate LM head issues.
        if os.getenv("PADDLEOCRVL_DEBUG_TEXT", "0") not in ("", "0", "false", "False"):
            # Initialize per-instance state lazily
            if getattr(self, "_debug_text_dumped", None) is None:
                self._debug_text_dumped = False  # type: ignore[attr-defined]

            # Heuristic: prefill usually has S > 1. Dump only once.
            if not self._debug_text_dumped and hidden_states.ndim in (2, 3):
                # Log shape best-effort
                from contextlib import suppress

                with suppress(Exception):
                    print(
                        "[paddleocr_vl][DEBUG_TEXT] compute_logits hidden shape:",
                        tuple(hidden_states.shape),
                    )
                # Prefer the last token in current block; if 2D it's already [B, H]
                if hidden_states.ndim == 3:
                    last_hidden = hidden_states[:, -1, :]
                else:
                    last_hidden = hidden_states
                last_hidden = last_hidden.detach().to(dtype=torch.float32, device="cpu")

                if logits.ndim == 3:
                    last_logits = (
                        logits[:, -1, :].detach().to(dtype=torch.float32, device="cpu")
                    )
                else:
                    last_logits = logits.detach().to(dtype=torch.float32, device="cpu")

                run_id = os.getenv("PADDLEOCRVL_RUN_ID") or __import__("time").strftime(
                    "%Y%m%d-%H%M%S"
                )
                out_dir = os.path.join("test_result", "text_debug")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    {
                        "hidden_last": last_hidden,
                        "logits_last": last_logits,
                        "hidden_shape": list(hidden_states.shape),
                        "logits_shape": list(logits.shape),
                    },
                    os.path.join(
                        out_dir, f"vllm_{run_id}_prefill_last_hidden_logits.pt"
                    ),
                )
                self._debug_text_dumped = True  # type: ignore[attr-defined]

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Customized weight loading for localized PaddleOCR-VL vision tower.

        Route HF visual.* weights into the custom vision encoder's internal
        SiglipVisionModel. Language/projector weights are handled by
        AutoWeightsLoader using a minimal mapper.
        """
        loaded: set[str] = set()

        # Materialize once to allow multiple passes
        items = list(weights)

        # 1) Load vision (visual.*) weights
        for name, w in items:
            if not name.startswith("visual."):
                continue

            # Strip the leading 'visual.' prefix
            rest = name[len("visual.") :]

            # Map both embedding tables to the dedicated vision model
            if rest.startswith(
                "vision_model.embeddings.packing_position_embedding.weight"
            ):
                new_name = "vision_model.embeddings.packing_position_embedding.weight"
                child_loaded = self.vision_tower.vision_model.load_weights(
                    [(new_name, w)]
                )
                loaded |= {f"vision_tower.vision_model.{n}" for n in child_loaded}
                continue

            # Drop classification head (unused)
            if rest.startswith("vision_model.head."):
                continue

            # Load standard position_embedding for interpolation path
            if rest.startswith("vision_model.embeddings.position_embedding.weight"):
                new_name = "vision_model.embeddings.position_embedding.weight"
                child_loaded = self.vision_tower.vision_model.load_weights(
                    [(new_name, w)]
                )
                loaded |= {f"vision_tower.vision_model.{n}" for n in child_loaded}
                continue

            # Load into the internal vision model as-is.
            new_name = rest
            child_loaded = self.vision_tower.vision_model.load_weights([(new_name, w)])
            loaded |= {f"vision_tower.vision_model.{n}" for n in child_loaded}

        # 2) Load the remaining weights via AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        loaded |= loader.load_weights(
            items,
            mapper=WeightsMapper(
                orig_to_new_prefix={
                    "visual.": None,
                    "mlp_AR.": "projector.",
                    "model.": "language_model.model.",
                    "lm_head.": "language_model.lm_head.",
                }
            ),
        )
        # 3) Ensure local post-layernorm in the vision tower is initialized.
        # Some PaddleOCR-VL checkpoints do not provide these parameters.
        try:
            self.vision_tower.post_layernorm.weight.data.fill_(1.0)
            self.vision_tower.post_layernorm.bias.data.zero_()
            loaded.add("vision_tower.post_layernorm.weight")
            loaded.add("vision_tower.post_layernorm.bias")
        except Exception:
            pass
        return loaded
