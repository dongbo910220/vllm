# PaddleOCR-VL mRoPE 修复已就绪 - 待验证

## 状态：✅ 代码已修复，等待测试验证

---

## 修复概述

已根据 Qwen2.5-VL 的启发和详细的 offset 诊断，修正了 PaddleOCR-VL mRoPE 的 T/H/W 段映射逻辑。

**核心问题**：旧代码在 FULL 维度 [128] 上按 [32, 48, 48] 分段，导致跨越 HALF 边界（位置 64），产生系统性 offset 错误。

**修复方案**：改为在 HALF 维度 [16, 24, 24] 分段，然后复制到 FULL，避免边界问题。

---

## 修复详情

### 修改文件

**文件**：`vllm/model_executor/layers/rotary_embedding/mrope.py`

### 修改位置

#### 1. Native 分支（line 612-638）

**旧代码逻辑**（已删除）：
```python
# 在 FULL 维度 [128] 上 split [32, 48, 48] - ✗ 跨边界！
cos_full3 = torch.cat([cos_half3_fp32, cos_half3_fp32], dim=-1)
sections_full = [2*t, 2*h, 2*w]  # [32, 48, 48]
cos_parts = cos_full3.split(sections_full, dim=-1)
```

**新代码逻辑**（已实现）：
```python
# 在 HALF 维度 [64] 上 split [16, 24, 24] - ✓ 不跨边界
t, h, w = map(int, self.mrope_section)  # [16, 24, 24]
sections_half = [t, h, w]

# 在 HALF 维度分段
cos_half_chunks = cos_half3_fp32.split(sections_half, dim=-1)
sin_half_chunks = sin_half3_fp32.split(sections_half, dim=-1)

# 取对角线段
cos_half_mapped = torch.cat(
    [cos_half_chunks[0][0], cos_half_chunks[1][1], cos_half_chunks[2][2]],
    dim=-1
)  # [tokens, 64] = [T前16, H前24, W前24]

# 复制到 FULL 维度 (Neox-style)
cos_full_mapped = torch.cat([cos_half_mapped, cos_half_mapped], dim=-1)
sin_full_mapped = torch.cat([sin_half_mapped, sin_half_mapped], dim=-1)
```

#### 2. CUDA 分支（line 1054-1080）

相同的修复逻辑，使用 `cos_base` 和 `sin_base` 代替 `cos_half3_fp32`。

---

## 验证步骤

### 1. 重新安装 vLLM

```bash
cd /home/bdong/projects/vllm-26900
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

**预计时间**：2-3 分钟

### 2. 重新运行 vLLM 生成 dump

```bash
# 使用你之前的 vLLM 测试脚本，确保开启调试
PADDLEOCRVL_DEBUG_MROPE=1 python <your_vllm_test_script>
```

**关键**：这会重新生成 `test_result/text_stages/vllm_*_cos_full_chunked_last.pt` 等文件。

### 3. 运行对比验证脚本

```bash
python temp_examples/compare_mapped_full_tables.py
```

**期望结果**：
```
=== Segment-wise Comparison ===
  Segment T (0:32):   cos_sim = 0.9900+  ✓ PASS
  Segment H (32:80):  cos_sim = 0.9900+  ✓ PASS
  Segment W (80:128): cos_sim = 0.9900+  ✓ PASS

Overall: ✓ PASS
```

### 4. （可选）详细段分析

```bash
python temp_examples/detailed_segment_analysis.py
```

**期望**：不再需要任何 offset，cos_sim 直接 ≥ 0.99。

---

## 验证标准

### ✅ 通过标准

- **T 段**: cos_sim(cos) ≥ 0.99，cos_sim(sin) ≥ 0.99
- **H 段**: cos_sim(cos) ≥ 0.99，cos_sim(sin) ≥ 0.99
- **W 段**: cos_sim(cos) ≥ 0.99，cos_sim(sin) ≥ 0.99
- **Overall**: cos_sim(cos) ≥ 0.99，cos_sim(sin) ≥ 0.99

### ❌ 如果仍然失败

1. 检查环境变量：
   ```bash
   # 确保使用 FULL 维度映射路径（不使用 HALF dup）
   unset PADDLEOCRVL_MROPE_USE_HALF_DUP
   ```

2. 检查 dump 文件是否更新：
   ```bash
   ls -lht test_result/text_stages/vllm_*_cos_full_chunked_last.pt
   ```
   确保时间戳是最新的。

3. 联系我，提供：
   - `compare_mapped_full_tables.py` 的完整输出
   - `detailed_segment_analysis.py` 的完整输出

---

## 修复原理说明

### 为什么旧代码会产生 offset 错误？

假设 `cos_base[axis, token, :]` 的 HALF 维度 [64] 内容为：
```
[freq0, freq1, ..., freq63]
```

复制到 FULL 维度 [128] 后：
```
[freq0, freq1, ..., freq63, freq0, freq1, ..., freq63]
```

**旧代码在 FULL 维度分段 [32, 48, 48]**：
- `cos_chunks[0]`: 维度 0:32 = `[freq0..15, freq0..15]` (对应 HALF 的 freq0..15)
- `cos_chunks[1]`: 维度 32:80 = `[freq16..39, freq16..39]` (对应 HALF 的 freq16..39)
- `cos_chunks[2]`: 维度 80:128 = `[freq40..63, freq0..23]` **← 跨边界了！**

第三段从 HALF 的 freq40 开始，但只有 24 个频率（到 freq63），剩余的 24 个是从 freq0 重新开始，导致错位！

### 为什么新代码正确？

**新代码在 HALF 维度分段 [16, 24, 24]**：
- `cos_half_chunks[0]`: HALF 的 0:16 = `[freq0..15]`
- `cos_half_chunks[1]`: HALF 的 16:40 = `[freq16..39]`
- `cos_half_chunks[2]`: HALF 的 40:64 = `[freq40..63]`

每个段都在 HALF 维度内，没有跨边界。然后：

```python
cos_half_mapped = [chunk0[0], chunk1[1], chunk2[2]]  # [tokens, 64]
                 = [T轴freq0..15, H轴freq16..39, W轴freq40..63]

cos_full_mapped = [cos_half_mapped, cos_half_mapped]  # [tokens, 128]
```

这样每个段的 FULL 维度都是正确的 Neox-style pairing：前后半部分完全相同。

---

## 参考文档

### 问题分析
- **根因报告**: `test_result/segment_diagnosis/final_diagnosis_report.md`
- **给专家的消息**: `MESSAGE_TO_EXPERT.md`

### 架构启发
- **Qwen2.5-VL 对比分析**: `QWEN2VL_MROPE_ANALYSIS.md`

### 诊断脚本
- `temp_examples/compare_mapped_full_tables.py` - 映射后 FULL 表对比
- `temp_examples/detailed_segment_analysis.py` - 详细段分析（含 offset 检测）

---

## 下一步（验证通过后）

一旦段对齐验证通过（cos_sim ≥ 0.99），需要继续：

1. **验证 post-RoPE Q/K 对齐**：
   - 对比 HF 和 vLLM 在 apply RoPE 后的 Q/K 值
   - 目标：cos_sim ≥ 0.99

2. **端到端生成验证**：
   - 使用 `temperature=0` 确保确定性
   - 对比 HF 和 vLLM 生成的 token 序列
   - 目标：完全一致

3. **代码清理**：
   - 移除调试代码（dump 逻辑）
   - 补充单元测试
   - Pre-commit 检查

---

## 当前环境

- **分支**: `fix/paddleocrvl-mrope-full-dim`
- **Python**: 3.12
- **虚拟环境**: `.venv`（已激活：`source .venv/bin/activate`）
- **待安装**: vLLM（修复后的代码）

---

## 联系方式

如有任何问题或验证失败，请提供：
1. 完整的验证脚本输出
2. vLLM 测试脚本的内容（如果有修改）
3. 环境变量设置（`env | grep PADDLE`）

祝验证顺利！🚀
