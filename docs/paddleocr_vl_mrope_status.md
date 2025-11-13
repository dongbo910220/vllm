SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# PaddleOCR‑VL mRoPE Status (fix/paddleocrvl-mrope-full-dim)

- Revert: d13f21774 (mrope: align chunked mapping with HF FULL diagonal selection)
  was reverted in cc37219d9 to restore green tests.
- Rationale: the FULL‑dim diagonal selection patch introduced syntax/indent
  issues and broke strict tests under test_common.
- Current result (strict comparator, stable sizes (() & 0.25)):
  - tests/models/multimodal/generation/test_common.py
  - Filter: "paddleocr_vl and (test_single_image_models or test_multi_image_models)"
  - Outcome: 4/4 PASS on RTX 4060 (8GB) using gpu_memory_utilization=0.8
- Logs:
  - test_result/fix/pytest_fix_branch_paddleocr_vl_strict.log

Notes

- Keep (0.25, 0.25) multi‑image disabled by default; investigate separately.
- If re‑attempting HF FULL‑dim alignment, do it on a new branch and add a
  minimal mrope unit test before switching default paths.

