# Qwen2.5-VL mRoPE 实现对比分析 - 对 PaddleOCR-VL 的启发

## 概述

Qwen2.5-VL 和 PaddleOCR-VL 都实现了 3D mRoPE (Multimodal Rotary Position Embedding),用于处理图像/视频的 (temporal, height, width) 三维位置编码。通过对比两者的实现,我们发现了 **PaddleOCR-VL 当前实现的关键问题**。

---

## 核心差异

### Qwen2.5-VL 的 apply_multimodal_rotary_pos_emb (HF)

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # 关键步骤 1: 将 mrope_section 乘以 2 (转为 FULL 维度)
    mrope_section = mrope_section * 2  # [16, 24, 24] → [32, 48, 48]

    # 关键步骤 2: Split cos/sin 表
    cos_splits = cos.split(mrope_section, dim=-1)  # Split into [32, 48, 48]
    # cos_splits[0]: [batch, seq, 32]  - 第 0 段
    # cos_splits[1]: [batch, seq, 48]  - 第 1 段
    # cos_splits[2]: [batch, seq, 48]  - 第 2 段

    # 关键步骤 3: 循环选择 - 这是精髓!
    cos = torch.cat([
        m[i % 3] for i, m in enumerate(cos_splits)
    ], dim=-1)
    # i=0: cos_splits[0][0 % 3] = cos_splits[0][0] - 第 0 段选择索引 0 (T 轴)
    # i=1: cos_splits[1][1 % 3] = cos_splits[1][1] - 第 1 段选择索引 1 (H 轴)
    # i=2: cos_splits[2][2 % 3] = cos_splits[2][2] - 第 2 段选择索引 2 (W 轴)

    # 结果: cos = [T 段的 T 轴, H 段的 H 轴, W 段的 W 轴]

    # 应用 RoPE
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**关键理解**:
- `cos` 输入 shape: `[batch, seq, rotary_dim]` 其中 `rotary_dim` 是 FULL 维度
- `cos` 已经是**拼接后的完整表**,包含所有三个轴的值
- **Split 操作是在 FULL 维度上进行的**
- `m[i % 3]` 中的 `m` 是什么? 这需要理解输入 `cos` 的结构

---

### PaddleOCR-VL 的当前实现 (vLLM)

**Native 分支** (mrope.py line 613-624):

```python
cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)  # [3, tokens, 128]
# cos_full3[0]: T 轴的 FULL 表 [tokens, 128] = [freq0..63, freq0..63]
# cos_full3[1]: H 轴的 FULL 表 [tokens, 128]
# cos_full3[2]: W 轴的 FULL 表 [tokens, 128]

sections_full = [32, 48, 48]  # [2*t, 2*h, 2*w]
cos_parts = cos_full3.split(sections_full, dim=-1)
# cos_parts[0]: [3, tokens, 32] - 所有三轴的前 32 维
# cos_parts[1]: [3, tokens, 48] - 所有三轴的中 48 维
# cos_parts[2]: [3, tokens, 48] - 所有三轴的后 48 维

cos_full_mapped = torch.cat(
    [cos_parts[0][0], cos_parts[1][1], cos_parts[2][2]], dim=-1
)
# cos_parts[0][0]: T 轴的前 32 维
# cos_parts[1][1]: H 轴的中 48 维  ← 问题在这里!
# cos_parts[2][2]: W 轴的后 48 维  ← 问题在这里!
```

**问题**:
- 在 FULL 维度 [128] 上 split [32, 48, 48] 会**跨越 HALF 边界**
- `cos_full3[axis, token, :]` = `[freq0, freq1, ..., freq63, freq0, freq1, ..., freq63]`
- Split 后:
  - `cos_parts[0][axis]` = 前 32 维 = `freq[0:16]` (HALF 的前 16 个频率)
  - `cos_parts[1][axis]` = 中 48 维 = `freq[16:40]` (HALF 的 16-39 号频率)
  - `cos_parts[2][axis]` = 后 48 维 = `freq[40:63] + freq[0:23]` ← **跨边界了!**

这就是为什么会有 offset 错误!

---

## 关键启发

### 启发 1: Qwen2-VL 的 `m[i % 3]` 语义

让我们理解 `m[i % 3]` 的含义:

```python
cos_splits = cos.split(mrope_section, dim=-1)
# 假设 cos shape: [batch, seq, 128]
# mrope_section = [32, 48, 48]
# 那么 cos_splits 是一个 list: [cos1, cos2, cos3]
# cos1: [batch, seq, 32]
# cos2: [batch, seq, 48]
# cos3: [batch, seq, 48]

# 但是 cos 的输入是什么结构?
# 根据 Qwen2-VL 的代码,cos 是从三个轴的 freqs 拼接而来的:
# cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
# 其中 cos_t/cos_h/cos_w 是每个轴单独计算的

# 所以 m[i % 3] 的 m 必须是一个支持多维索引的张量!
```

**推测**: `m` 应该是 shape `[3, batch, seq, segment_size]`,其中第 0 维表示轴 (T/H/W)。

但这和标准的 RoPE 输入格式不同! 让我检查 Qwen2-VL 的 position 处理。

### 启发 2: Qwen2-VL 如何生成 cos/sin

关键在于理解 `cos` 的输入格式。让我查看 Qwen2-VL 如何构建 position_ids:

根据 vLLM 代码 (qwen2_5_vl.py line 806-819):
```python
def get_rope_by_thw(self, t, h, w):
    rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)  # 返回 3D RoPE
    # ...
    return rotary_pos_emb_thw, ...
```

这说明 Qwen2-VL 使用**专门的 3D RoPE 生成器**,直接生成拼接好的表,而不是像 PaddleOCR-VL 那样从通用 cache 索引。

### 启发 3: 正确的映射方式

**Qwen2-VL 的方法**:
1. 预先计算好完整的 3D RoPE 表 (已经按 T/H/W 拼接好)
2. 在应用时,通过 split + 循环选择来重新组织

**PaddleOCR-VL 应该采用的方法**:
1. 在 HALF 维度操作,避免跨边界
2. 从每个轴取对应长度,然后拼接

---

## 对 PaddleOCR-VL 的修复建议

### 方案 A: 模仿 Qwen2-VL 的 Cycle 逻辑 (不推荐)

```python
# 在 FULL 维度 split 和 cycle
mrope_section_full = [2*t, 2*h, 2*w]  # [32, 48, 48]
cos_splits = cos_full3.split(mrope_section_full, dim=-1)

# Cycle through axes
cos_mapped = torch.cat([
    cos_splits[i][i % 3] for i in range(3)
], dim=-1)
```

**问题**: 这假设 `cos_full3` 的结构是 `[3, tokens, 128]`,但实际上 `cos_splits[i]` 是 `[3, tokens, segment_size]`,所以 `cos_splits[i][i % 3]` 只会取到单个轴。

这个逻辑**只在 cos 已经是正确拼接格式时才有效**。

### 方案 B: 在 HALF 维度操作 (推荐 ✓)

```python
t, h, w = map(int, self.mrope_section)  # [16, 24, 24] - HALF
sections_half = [t, h, w]

# 在 HALF 维度分段
cos_half_chunks = cos_base.split(sections_half, dim=-1)  # [3, tokens, 16/24/24]
sin_half_chunks = sin_base.split(sections_half, dim=-1)

# 取对角线段 - 每个轴取自己对应的段
cos_t_half = cos_half_chunks[0][0]  # T 轴的前 16 个 HALF 频率
cos_h_half = cos_half_chunks[1][1]  # H 轴的前 24 个 HALF 频率
cos_w_half = cos_half_chunks[2][2]  # W 轴的前 24 个 HALF 频率

# 拼接 HALF 维度
cos_half_mapped = torch.cat([cos_t_half, cos_h_half, cos_w_half], dim=-1)  # [tokens, 64]

# 复制到 FULL 维度 (Neox-style)
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)  # [tokens, 128]
```

**优势**:
1. 避免跨越 HALF/FULL 边界
2. 逻辑清晰,符合 Neox-style 的 pairing 语义
3. 和 HF PaddleOCR-VL 的逻辑一致

### 方案 C: 直接从轴取 (最直接)

如果 `cos_base[axis]` 确实是该轴的独立频率表:

```python
# 直接从 HALF 维度取前 N 个
cos_t_half = cos_base[0, :, :t]   # T 轴前 16 个
cos_h_half = cos_base[1, :, :h]   # H 轴前 24 个
cos_w_half = cos_base[2, :, :w]   # W 轴前 24 个

cos_half_mapped = torch.cat([cos_t_half, cos_h_half, cos_w_half], dim=-1)
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)
```

---

## Qwen2-VL 和 PaddleOCR-VL 的架构差异

### Qwen2-VL

1. **Vision Encoder 自带 RoPE**: `Qwen2VisionRotaryEmbedding` 生成视觉部分的 3D RoPE
2. **LLM 部分使用标准 RoPE**: 继承 Llama 的 1D RoPE
3. **在 Attention 层应用 mRoPE**: 使用 `apply_multimodal_rotary_pos_emb` 统一处理

### PaddleOCR-VL

1. **统一的 mRoPE 层**: `MRotaryEmbedding` 同时处理视觉和文本
2. **在 Rotary Embedding 层内部完成映射**: forward 方法中处理 T/H/W 分段
3. **直接输出映射后的 cos/sin**: Attention 层直接使用

**关键差异**:
- Qwen2-VL: 映射逻辑在 Attention 层 (apply 时)
- PaddleOCR-VL: 映射逻辑在 Embedding 层 (生成时)

---

## 总结

### 为什么 Qwen2-VL 没有 offset 问题?

1. **预先生成正确的 3D RoPE 表**: 使用专门的 `rotary_pos_emb_thw(t, h, w)` 生成
2. **在应用层面做映射**: `apply_multimodal_rotary_pos_emb` 使用 split + cycle
3. **输入格式明确**: `cos` 已经是完整拼接好的表,不需要复杂的轴向操作

### PaddleOCR-VL 应该如何修复?

**推荐**: **方案 B** - 在 HALF 维度操作

修改位置:
- `vllm/model_executor/layers/rotary_embedding/mrope.py`
- Native 分支: line 613-624
- CUDA 分支: line 1040-1059

关键修改:
```python
# 改为在 HALF 维度 split
sections_half = [t, h, w]  # 不是 [2*t, 2*h, 2*w]
cos_half_chunks = cos_base.split(sections_half, dim=-1)

# 取对角线
cos_half_mapped = torch.cat([
    cos_half_chunks[0][0],
    cos_half_chunks[1][1],
    cos_half_chunks[2][2]
], dim=-1)

# 复制到 FULL
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)
```

### 验证方法

修复后,重新运行对比:
```bash
PADDLEOCRVL_DEBUG_MROPE=1 python <vllm_script>
python temp_examples/compare_mapped_full_tables.py
```

**预期结果**:
- T 段: cos_sim ≥ 0.99
- H 段: cos_sim ≥ 0.99
- W 段: cos_sim ≥ 0.99 (已经接近,应该进一步提升到 0.9999)

---

## 参考资料

- **Qwen2-VL HF**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
- **Qwen2.5-VL HF**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
- **vLLM Qwen2-VL**: `vllm/model_executor/models/qwen2_vl.py`
- **vLLM Qwen2.5-VL**: `vllm/model_executor/models/qwen2_5_vl.py`
- **PaddleOCR-VL mRoPE**: `vllm/model_executor/layers/rotary_embedding/mrope.py`
