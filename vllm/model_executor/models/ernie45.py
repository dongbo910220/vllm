# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Ernie 4.5 text model compatible with HF weights.

This class piggybacks on the Llama implementation but adjusts RoPE and
projection details to match Ernie 4.5:
 - Use non-Neox (interleaved) rotary style.
 - Use partial rotary dimension (Dh/2) for q/k instead of full Dh.
 - Remove bias in o_proj and enable skip_bias_add for parity.
"""

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.llama import LlamaForCausalLM

from .utils import PPMissingLayer


@support_torch_compile(
    # set dynamic_arg_dims to support mrope
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Ernie4_5ForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Hack Llama model to fit HF format Ernie4.5 dense implementation
        # Attention difference between Ernie and Llama:
        # 1. rotary_dim (Dh/2) and non-Neox (interleaved) style.
        # 2. There is no bias for o_proj in attention
        cfg = self.config
        rope_theta = getattr(cfg, "rope_theta", 10000)
        rope_scaling = getattr(cfg, "rope_scaling", None)
        max_position = getattr(cfg, "max_position_embeddings", 8192)

        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                # Recreate rotary embedding with Dh/2 and interleaved style
                head_dim = layer.self_attn.head_dim
                # Respect mRoPE section if present; its sum defines the HALF
                # rotary dimension. The rotary_dim passed to the embedding is
                # the FULL size, i.e., 2 * sum(mrope_section).
                if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
                    try:
                        rotary_dim = 2 * int(sum(rope_scaling["mrope_section"]))
                    except Exception:
                        rotary_dim = head_dim // 2
                else:
                    rotary_dim = head_dim // 2
                # HF PaddleOCR-VL 的 mRoPE 在 apply_multimodal_rotary_pos_emb 中
                # 采用“按 section 大小的块状重排（T块 | H块 | W块）”。
                # 这对应于本实现的非 interleaved 分支（mrope_interleaved=False）。
                # 因此不要强制开启 mrope_interleaved，除非权重明确指定。
                rs = rope_scaling
                if isinstance(rs, dict) and "mrope_interleaved" not in rs:
                    rs = {**rs, "mrope_interleaved": False}

                # 旋转风格按配置/环境选择：默认遵循模型配置中的
                # rope_is_neox_style；若设置了环境变量 PADDLEOCRVL_ROTATE_STYLE，
                # 则以环境变量为准（neox 或 gptj）。
                import os as _os

                env_rotate = _os.getenv("PADDLEOCRVL_ROTATE_STYLE", "").lower()
                if env_rotate in ("neox", "gptj"):
                    use_neox = env_rotate == "neox"
                else:
                    use_neox = bool(getattr(cfg, "rope_is_neox_style", True))
                layer.self_attn.rotary_emb = get_rope(
                    head_size=head_dim,
                    rotary_dim=rotary_dim,
                    max_position=max_position,
                    base=rope_theta,
                    is_neox_style=use_neox,
                    rope_scaling=rs,
                    dtype=None,
                )
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = True
