# 给高手的消息：PaddleOCR-VL mRoPE T/H/W 段 Offset 错误

## 问题概述

在 PaddleOCR-VL 的 mRoPE 实现中,映射后的 FULL 表与 HuggingFace 参考实现存在**系统性的 offset 错误**。具体表现为:

- **T 段** (32维): cos_sim = 0.485,需要 +10 offset 才能达到 0.917
- **H 段** (48维): cos_sim = 0.845,需要 +1 offset 才能达到 0.993
- **W 段** (48维): cos_sim = 0.989,需要 +2 offset 才能达到 0.9999

W 段几乎完美对齐,说明基本逻辑正确,但存在索引偏移问题。

---

## 诊断数据

### 对比文件

- **HF 参考**: `test_result/hf_layer0_cos_full_last.pt` [128]
- **vLLM 当前**: `test_result/text_stages/vllm_textext_funsd_3_cos_full_chunked_last.pt` [128]
- **配置**: sections_full = [32, 48, 48] (T, H, W)

### 样本值对比 (T 段前 8 个)

```
HF:   [ 0.301, -0.602,  0.453,  0.214, -0.914,  0.295, -0.181,  0.621]
vLLM: [ 0.283, -0.598, -0.984, -0.906, -0.590, -0.221,  0.109,  0.371]
Diff: [ 0.018, -0.004,  1.438,  1.120, -0.324,  0.516, -0.290,  0.250]
```

**关键观察**: 前 2 个值非常接近 (diff < 0.02),但从第 3 个开始完全错位。

### 详细诊断报告

完整分析见: `test_result/segment_diagnosis/final_diagnosis_report.md`

---

## 问题根因分析

### 涉及代码位置

**文件**: `vllm/model_executor/layers/rotary_embedding/mrope.py`

#### CUDA 分支 (line 1040-1059):

```python
cos_full3 = torch.cat(
    [cos_base.to(torch.float32), cos_base.to(torch.float32)],
    dim=-1,
)  # [3, tokens, 128]

t, h, w = map(int, self.mrope_section)  # [16, 24, 24]
sections_full = [2 * t, 2 * h, 2 * w]   # [32, 48, 48]

cos_chunks = cos_full3.split(sections_full, dim=-1)
# cos_chunks[0]: [3, tokens, 32] - 所有三轴的前 32 维
# cos_chunks[1]: [3, tokens, 48] - 所有三轴的中 48 维
# cos_chunks[2]: [3, tokens, 48] - 所有三轴的后 48 维

cos_full_mapped = torch.cat(
    [cos_chunks[0][0], cos_chunks[1][1], cos_chunks[2][2]],
    dim=-1,
)
# cos_chunks[0][0]: T 轴的前 32 维
# cos_chunks[1][1]: H 轴的中 48 维
# cos_chunks[2][2]: W 轴的后 48 维
```

#### Native 分支 (line 613-624): 逻辑相同

### 问题根源推测

有两种可能:

#### 可能性 1: `cos_base` 的内容不正确

`cos_base` 来自:
```python
# Line 977 (CUDA) / 537 (Native):
cos_sin = self.cos_sin_cache[positions]  # positions: [3, tokens]
cos_base, sin_base = cos_sin.chunk(2, dim=-1)  # [3, tokens, 64]
```

**问题**: `self.cos_sin_cache` 是从父类 `RotaryEmbedding._compute_cos_sin_cache()` 继承的**1D 标准 RoPE cache**,不是专门为 mRoPE 设计的 3D cache。

当 `positions` 是 `[3, tokens]` 时:
- `positions[0]` 是 T 轴位置
- `positions[1]` 是 H 轴位置
- `positions[2]` 是 W 轴位置

`cos_sin_cache[positions]` 会对每个轴独立索引,得到 `[3, tokens, 128]`。但这个 cache 本身是否正确构建,需要验证。

**关键疑问**:
1. `cos_sin_cache` 是否为每个轴分别存储了不同的频率表?
2. 还是三个轴共享同一个频率表,只是 position 值不同?

根据标准 RoPE 实现,应该是**共享频率表,position 不同**。这理论上是正确的。

#### 可能性 2: 映射逻辑的索引错误

当前逻辑:
```python
cos_chunks = cos_full3.split([32, 48, 48], dim=-1)
cos_full_mapped = torch.cat(
    [cos_chunks[0][0], cos_chunks[1][1], cos_chunks[2][2]], dim=-1
)
```

**HF 的正确逻辑** (参考 `modeling_paddleocr_vl.py`):
```python
# HF 从每个轴取对应长度的段
cos_t = cos_axes[0, :t_full]  # T 轴的前 t_full=32 个
cos_h = cos_axes[1, :h_full]  # H 轴的前 h_full=48 个
cos_w = cos_axes[2, :w_full]  # W 轴的前 w_full=48 个
cos_full = torch.cat([cos_t, cos_h, cos_w], dim=-1)
```

对比 vLLM:
```python
# vLLM 先 split 全部三轴,再取对角线
cos_chunks = cos_full3.split([32, 48, 48], dim=-1)
# cos_chunks[i] 是所有三轴在第 i 个段的值
cos_full_mapped = torch.cat(
    [cos_chunks[0][0], cos_chunks[1][1], cos_chunks[2][2]], dim=-1
)
```

**关键差异**:
- HF: `cos_axes[axis, :length]` - 从**轴向表**切片
- vLLM: `cos_chunks[chunk_idx][axis]` - 从**拼接后的表**切片

**这可能是问题所在!** vLLM 的逻辑假设 `cos_full3[axis]` 已经是该轴的完整表,然后按段切分。但实际上 `cos_base` 可能不是纯轴向表。

---

## 验证方向

### 1. 检查 `cos_base` 的语义

在 `mrope.py` 的 forward 中添加调试代码:

```python
# 在 line 977 之后 (CUDA) 或 line 537 之后 (Native)
if os.getenv("PADDLEOCRVL_DEBUG_COSSBASE", "0") != "0":
    print(f"positions shape: {positions.shape}")
    print(f"positions values:\n{positions[:, :5]}")  # 前 5 个 token
    print(f"cos_base shape: {cos_base.shape}")
    print(f"cos_base[0, 0, :8]: {cos_base[0, 0, :8]}")  # T 轴第 0 个 token
    print(f"cos_base[1, 0, :8]: {cos_base[1, 0, :8]}")  # H 轴第 0 个 token
    print(f"cos_base[2, 0, :8]: {cos_base[2, 0, :8]}")  # W 轴第 0 个 token
```

**预期**: 如果 T/H/W 三个轴的 position 值不同,那么 `cos_base[0/1/2, 0, :]` 应该有显著差异。

### 2. 对比 HF 的轴向表构建

检查 HF 是如何构建 `cos_axes` 的:

```python
# HF modeling_paddleocr_vl.py (推测逻辑)
def get_mrope_cos_sin(positions, inv_freq):
    # positions: [3, tokens] - (T, H, W)
    # inv_freq: [dim//2]

    cos_axes = []
    sin_axes = []
    for axis in range(3):
        pos_axis = positions[axis]  # [tokens]
        freqs = pos_axis.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [tokens, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [tokens, dim] - FULL
        cos_axes.append(emb.cos())
        sin_axes.append(emb.sin())

    cos_axes = torch.stack(cos_axes, dim=0)  # [3, tokens, dim]
    return cos_axes, sin_axes
```

这和 vLLM 的 `cos_base` 应该是一致的语义。

### 3. 检查 HF 的分段拼接逻辑

关键代码在 HF 的 `apply_multimodal_rotary_pos_emb`:

```python
# 伪代码
cos_parts = []
sin_parts = []
for axis, length in enumerate([t_full, h_full, w_full]):
    cos_parts.append(cos_axes[axis, :, :length])
    sin_parts.append(sin_axes[axis, :, :length])

cos_full = torch.cat(cos_parts, dim=-1)  # [tokens, 128]
```

**注意**: HF 是从每个轴取**前 N 个维度**,不是取某个 offset 后的维度!

### 4. 验证 vLLM 的 split 逻辑

当前代码:
```python
cos_full3 = torch.cat([cos_base, cos_base], dim=-1)  # [3, tokens, 128]
cos_chunks = cos_full3.split([32, 48, 48], dim=-1)
```

假设 `cos_base[axis, token, :]` 是该轴该 token 的 64 个频率的 cos 值,那么:
- `cos_full3[axis, token, :]` = `[freq0, freq1, ..., freq63, freq0, freq1, ..., freq63]`
- `cos_chunks[0][axis, token, :]` = 前 32 维 = `[freq0, ..., freq15]` (HALF 维度的前 16 个频率)
- `cos_chunks[1][axis, token, :]` = 中 48 维 = `[freq16, ..., freq39]` (HALF 维度的 16-39 号频率)
- `cos_chunks[2][axis, token, :]` = 后 48 维 = `[freq40, ..., freq63, freq0, ..., freq23]` (跨越边界!)

**这可能就是问题!** vLLM 的 split 逻辑在 FULL 维度上 split,会跨越 HALF 的边界。

---

## 修复建议

### 方案 1: 修正 split 逻辑 (推荐)

**问题**: 当前是在 FULL 维度 [128] 上 split `[32, 48, 48]`,导致跨越 HALF 边界。

**修复**: 应该在 HALF 维度 [64] 上 split `[16, 24, 24]`,然后再复制到 FULL。

```python
# 修改 line 1040-1059 (CUDA) 和 613-624 (Native)

t, h, w = map(int, self.mrope_section)  # [16, 24, 24] - HALF
sections_half = [t, h, w]

# 在 HALF 维度分段
cos_half_chunks = cos_base.split(sections_half, dim=-1)  # [3, tokens, 16/24/24]
sin_half_chunks = sin_base.split(sections_half, dim=-1)

# 取对角线段
cos_t_half = cos_half_chunks[0][0]  # [tokens, 16]
cos_h_half = cos_half_chunks[1][1]  # [tokens, 24]
cos_w_half = cos_half_chunks[2][2]  # [tokens, 24]

# 拼接 HALF 维度
cos_half_mapped = torch.cat([cos_t_half, cos_h_half, cos_w_half], dim=-1)  # [tokens, 64]

# 复制到 FULL 维度
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)  # [tokens, 128]

# sin 同理
```

**优势**:
1. 逻辑清晰,先在 HALF 维度操作,再扩展到 FULL
2. 避免跨边界问题
3. 符合 HF 的语义

### 方案 2: 使用轴向 split (备选)

如果 `cos_base` 的语义确实是每个轴独立的频率表,可以:

```python
# 直接从 HALF 维度取前 N 个
cos_t_half = cos_base[0, :, :t]   # T 轴前 16 个
cos_h_half = cos_base[1, :, :h]   # H 轴前 24 个
cos_w_half = cos_base[2, :, :w]   # W 轴前 24 个

cos_half_mapped = torch.cat([cos_t_half, cos_h_half, cos_w_half], dim=-1)
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)
```

**优势**: 更直接,完全对齐 HF 逻辑

---

## 需要高手做的事

1. **确认 `cos_base` 的语义**:
   - 添加调试代码,dump `cos_base` 的值
   - 确认三个轴是否有正确的差异
   - 确认是否和 HF 的 `cos_axes` 一致

2. **根据上述分析修正映射逻辑**:
   - 优先尝试**方案 1**: 在 HALF 维度 split
   - 同时修改 Native 分支 (line 613-624) 和 CUDA 分支 (line 1040-1059)
   - 确保 sin 表也同步修改

3. **验证修复效果**:
   ```bash
   # 重新运行 vLLM 生成 dump
   PADDLEOCRVL_DEBUG_MROPE=1 python <your_vllm_script>

   # 对比修复后的结果
   python temp_examples/compare_mapped_full_tables.py
   ```

   **目标**: T/H/W 三个段的 cos_sim 均 ≥ 0.99

4. **如果方案 1 不work**:
   - 深入检查 `self.cos_sin_cache` 的构建逻辑
   - 可能需要重写 `_compute_cos_sin_cache()` 为 mRoPE 定制版本

---

## 参考资料

- **诊断报告**: `test_result/segment_diagnosis/final_diagnosis_report.md`
- **诊断脚本**: `temp_examples/detailed_segment_analysis.py`
- **HF 参考实现**: `test_result/hf_reference/modeling_paddleocr_vl.py`
- **Offset 检测结果**: 见 `detailed_segment_analysis.py` 的输出

---

## 当前环境

- 分支: `fix/paddleocrvl-mrope-full-dim`
- Python: 3.12
- vLLM 已安装: `VLLM_USE_PRECOMPILED=1 uv pip install --editable .`
- 虚拟环境: `.venv` (已激活: `source .venv/bin/activate`)

---

请高手按照上述分析进行修复。如有疑问,所有诊断数据和脚本都在 `test_result/segment_diagnosis/` 目录中。
