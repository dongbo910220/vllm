PaddleOCR‑VL Cleanup Plan — Diff Classification and Steps

Scope
- Branch: fix/paddleocrvl-mrope-full-dim (rebased on origin/main)
- Goal: Minimize change footprint to align with upstream style (PR #27758), keep strict parity green for test_common, and remove debug-only paths/env gates.

Diff Categories (vs origin/main)
- Core model integration
  - vllm/model_executor/models/paddleocr_vl.py (core model wrapper, projector, processor hooks)
  - vllm/model_executor/models/registry.py (registry entry)

- Optional/extra vision code (candidate for removal or folding)
  - vllm/model_executor/models/paddleocr_vl_vision.py (dedicated vision wrapper)
  - vllm/model_executor/models/siglip2navit.py (unrelated to minimal integration)

- Shared core (should remain unchanged unless strictly necessary)
  - vllm/model_executor/layers/rotary_embedding/mrope.py (currently aligned with main)
  - vllm/model_executor/models/llama.py, ernie45.py (should match main)
  - vllm/multimodal/processing.py, vllm/v1/worker/* (keep changes minimal or none)

- Tests
  - tests/models/multimodal/generation/test_common.py (register PaddleOCR‑VL; strict comparator for local diagnostics)
  - tests/models/registry.py (registry validation)
  - Added focused tests (may trim if not required by upstream):
    - tests/models/test_paddleocr_vl_strict.py
    - tests/models/test_paddleocr_vl_multi_image_strict.py
    - tests/multimodal/processing/test_paddleocr_pixel_postprocess.py
    - tests/multimodal/processing/test_paddleocr_vl.py

- Local artifacts / debug-only (should not be tracked)
  - run_state/** (pre-commit state, caches)
  - vllm/_C.abi3.so.bak (binary backup)
  - IMPLEMENTATION_READY.md, MESSAGE_TO_EXPERT.md, QWEN2VL_MROPE_ANALYSIS.md (process notes; move or drop)

Cleanup Steps (incremental, test after each)
1) Artifacts hygiene (this change)
   - Ignore run_state/ and vllm/_C.abi3.so.bak, remove from index.

2) Tests alignment
   - Keep only minimal edits required to register PaddleOCR‑VL in test_common.
   - Keep strict comparator locally; do not upstream strict mode by default.

3) Model surface minimization
   - Prefer a single model file (paddleocr_vl.py) plus registry entry.
   - Remove/inline paddleocr_vl_vision.py if not strictly needed.
   - Avoid changes to shared components (mrope.py, llama.py, processing.py).

4) Remove env gates and debug hooks
   - Eliminate PADDLEOCRVL_* environment flags and temporary dumps.
   - Keep functionality intact for strict test_common.

5) Focused 0.25-scale diagnostics (if still failing)
   - Add temporary, env-gated prints/dumps for position_ids and placeholder widths; remove before finalizing.

Validation
- Run strict test_common after each step:
  python -m pytest tests/models/multimodal/generation/test_common.py -k "paddleocr_vl" -v
- Save logs under test_result/fix/ with descriptive filenames.

Notes
- Do not push; local commits must be DCO-signed.
- Keep origin/main pristine; avoid broad changes outside model + tests.

