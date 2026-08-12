# PORT-NOTES — overlay0261 (v0.26.0 overlay → main 48bada6ea4)

Forward-port of the production 2xSPARK-CLUSTER vLLM source overlay.

- OLD base: `v0.26.0` (`f2654939e6`)
- OLD overlay: `gx10-overlay-026-c7` (`ed042f4e70`) — authoritative source;
  the shipped `vllm-026-rebase/overlay026/vllm/` tree was diffed against it
  for all 15 files: **no disagreement**.
- NEW base: `48bada6ea4` (origin/main, v0.26.1rc0-590, "Fix chat completion
  500 on non-object JSON bodies (#51654)")
- Method per file: 3-way `git merge-file -p <main> <v0.26.0> <c7>`, conflicts
  resolved per policy; every output passes `python3 -m py_compile`.
- Result of the port: **13 whole-file overlays** (15 c7 files − 2 dropped-native);
  §14 later adds a 14th (`utils/deep_gemm.py`, post-build hotfix), and the #49731
  revert adds 3 more (documented in docs/14, not here) → **17 shipped** in
  `overlay0261/vllm/`.
  Production value carried: `nvfp4_ds_mla` dtype plumbing, Anemll SM120/121
  (B12X) MoE kernel adaptations, local guards.

## Per-file decisions

### 1. `vllm/config/cache.py` — CARRIED (clean merge)
- `"nvfp4_ds_mla"` added to `CacheDType` Literal at line 28, immediately
  after `"fp8_ds_mla"`. Diff vs main is exactly 1 line.
- Main drift note: main added `"nvfp4_4over6"`, `"turboquant_k8v4"`,
  `"turboquant_4bit_nc"` entries — left intact.

### 2. `vllm/config/speculative.py` — CARRIED (clean merge)
- c7's warn-only block replaced main's hard `raise ValueError` (from #49969)
  in the `method == "dspark"` validator, landing at lines 1091–1108.
  c7 text carried verbatim, including the
  `LOCAL OVERLAY (2xSPARK-CLUSTER, experiment E17; tracks upstream issue
  vllm-project/vllm#50012)` comment. Variable names unchanged on main
  (`dspark_block_size`, `self.num_speculative_tokens`) — no adaptation
  needed. Merge was conflict-free because main's raise block is identical to
  0.26.0's; the merged region was verified byte-identical to c7's.

### 3. `vllm/config/vllm.py` — ADAPTED (1 conflict, resolved)
- `validate_nvfp4_kv_cache_with_mla` (lines 2455–2465): main widened the gate
  from `cache_dtype == "nvfp4"` to `cache_dtype.startswith("nvfp4")`.
  Resolution: **main's widened condition kept**, body replaced with the
  overlay rewrite `self.cache_config.cache_dtype = "nvfp4_ds_mla"` instead of
  the raise. Added a 2-line `LOCAL OVERLAY (2xSPARK-CLUSTER)` comment
  (c7 had none here; added for greppability). Behavior: explicit
  `nvfp4_ds_mla` passes through as a harmless no-op self-assign; `nvfp4` (and
  any future `nvfp4_*` spelling) rewrites to `nvfp4_ds_mla`.

### 4. `vllm/envs.py` — CARRIED (clean merge)
- 4 env vars at TYPE_CHECKING lines 50–53 and environment_variables lambdas
  lines 1070–1079: `VLLM_USE_B12X_MOE` (bool, 0),
  `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` (int, 0),
  `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M` (int, 16),
  `VLLM_B12X_W4A16_FORCE_TILE_CONFIG` (str, ""). Placed adjacent to the same
  neighbors as in c7. Diff vs main is exactly these 4+4 entries.

### 5. `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` — CARRIED (clean merge)
All 9 `B12X` references landed; every anchor function keeps its 0.26.0 name
on main (no refactoring):
- (a) `B12X_MXFP4 = "B12X_MXFP4"` enum member — line 112.
- (b) `backend_to_kernel_cls` branch importing/returning `B12xExperts` — 187.
- (c) `map_mxfp4_backend`: `"flashinfer_b12x": [Mxfp4MoeBackend.B12X_MXFP4]` — 300.
- (d) `mxfp4_round_up_hidden_size_and_intermediate_size` round-up-128 branch — 686.
- (e) `convert_weight_to_mxfp4_moe_kernel_format` passthrough (returns native
  MXFP4 tensors; W4A16 packing happens in expert post-load) — 1323.
- (f) `make_mxfp4_moe_quant_config` backend list membership — 1786.
- (g) `make_mxfp4_moe_kernel`: `experts.process_weights_after_loading(layer)`
  for B12X — 1862.

### 6. `vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py` — CARRIED WHOLESALE
- 877-line new file, absent on main, copied byte-identical from c7.
- Import audit vs main (all resolve, **zero renames**):
  `vllm.envs`, `vllm.model_executor.layers.fused_moe.modular_kernel`,
  `.activation.MoEActivation` (activation.py:19),
  `.config.FusedMoEParallelConfig/FusedMoEQuantConfig/RoutingMethodType`
  (config.py:1036/214/102),
  `.topk_weight_and_reduce.TopKWeightAndReduceNoOP` (:44),
  `.quantization.utils.quant_utils.QuantKey/kMxfp4Static`
  (quant_utils.py:161/243), `vllm.model_executor.utils.replace_parameter`
  (:47), `vllm.platforms.current_platform`, `vllm.logger.init_logger`.

### 7. `vllm/models/deepseek_v4/attention.py` — CARRIED (nvfp4 hunks) + DROPPED-NATIVE (short-context path)
CARRIED (both landed clean; diff vs main is exactly 30 lines):
- (a) `_resolve_dsv4_kv_cache_dtype`: `kv_cache_dtype in ("nvfp4",
  "nvfp4_ds_mla")` branch returning `("nvfp4_ds_mla", torch.uint8)` with the
  `cache_config.cache_dtype` rewrite — lines 103–110. Main's function body is
  otherwise identical to 0.26.0's here.
- (b) `DeepseekV4Attention.get_kv_cache_spec` (line 653 def): the
  `uses_ds_mla_layout` / 3-way `alignment` (584 nvfp4_ds_mla / 576 fp8_ds_mla
  / 512) hunk — lines 658–678. The second `get_kv_cache_spec` (line 705,
  `DeepseekV4IndexerCache`) untouched.
DROPPED-NATIVE:
- `_fill_short_context_topk_indices` triton kernel + `from vllm.triton_utils
  import tl, triton` + the `DeepseekV4Indexer` short-context fast path.
  Evidence: main has the kernel at line 72 **byte-identical** and the fast
  path at line 834 with the **identical condition** `indexer_metadata.
  max_seq_len // self.compress_ratio <= self.topk_tokens`. Attribution
  correction: git evidence shows the deepseek_v4 kernel+fast path came from
  upstream **#49486** (`b0cb1da1bd`, "Skip topk and router when not needed"),
  not #48407 — #48407 (`ba702e978e`) touched
  `vllm/v1/layers/attention/{mla,sparse_mla}_attention.py` instead. Both are
  native on main.
- Semantic delta resolved: overlay returned `self.topk_indices_buffer`; main
  returns `None, None, None` under the #51430 forward() restructure (kernel
  still writes the buffer; the tuple return is the new contract). **Main's
  form kept** — this was the file's only merge conflict.
- #51430 forward() restructure and #49236 eager-scratch: not disturbed
  (nvfp4 hunks touch disjoint regions).

### 8. `vllm/models/deepseek_v4/compressor.py` — DROPPED-NATIVE (#48957), omitted
- Native equivalence verified element-by-element against `37e370fe93`:
  `_get_c128_boundary` def (main:48, body identical), metadata field
  `c128_boundary: bool | None = None` (:109), builder population gated on
  `self.block_size == 8` (:147–151), early-return skip conditioned on
  `cudagraph_runtime_mode != CUDAGraphMode.FULL and state_metadata.
  c128_boundary is False` (:388–395). Every old→c7 hunk exists verbatim in
  main; main's only extras are #49236 eager-scratch lines
  (`eager_scratch_pool` param, `compress_scratch` kwargs) which must not be
  reverted. File omitted from overlay0261.

### 9. `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` — CARRIED (3 features) + DROPPED-NATIVE (q-head ladder)
CARRIED:
- (a) `_as_sparse_cache` 256→64-token zero-copy page split with the
  `page_size % 64` and `expected_strides = (page_size * 584, 584, 584, 1)`
  ValueError guards and explanatory comment — lines 554–582, replacing main's
  simple `unsqueeze(-2)` form (main:550–552).
- (b) `_pad_decode_sparse_indices` static method + module-level
  `_FLASHINFER_DSV4_DECODE_TOPKS = (128, 512, 1024)` (:36) + `from vllm.logger
  import init_logger` + `logger =` + decode-path call site
  `swa_indices = self._pad_decode_sparse_indices(swa_indices)` (:811),
  immediately before `_prepare_query` — main's decode path is structurally
  identical to 0.26.0's here.
- (c) zero-token prefill chunk skip guard (`if q_chunk.shape[0] == 0: …
  logger.warning_once(…) continue`, "Local overlay, goal P2 fix-iter-2"
  comment) — lines 933–945, same anchor (`q_chunk = q[query_start:query_end]`
  in the prefill chunk loop); `chunk_idx/chunk_start/chunk_end` all in scope
  on main. `logger.warning_once` exists on main (logger.py:136).
DROPPED-NATIVE (#48047, `b49eaf205a`): `_SPARSE_MLA_SUPPORTED_Q_HEADS` /
`_pad_to_supported_q_heads` — main has them at lines 64–68 with both
`get_padded_num_q_heads` call sites (:181, :556). **Main's form kept**; the
one merge conflict (main's bare `get_padded_num_q_heads` vs c7's
`_pad_decode_sparse_indices` insertion point) was resolved by keeping the new
static method and main's bare ladder call — c7's stale "pre-#48047" TP2
comment was dropped with it. No duplicate definitions.
- #49236's 4 added lines: intact (not in merged regions).

### 10. `vllm/models/deepseek_v4/sparse_mla.py` — CARRIED (nvfp4) + DROPPED-NATIVE (#50004)
CARRIED (clean merge; diff vs main is exactly these 2 hunks):
- `"nvfp4_ds_mla"` in `supported_kv_cache_dtypes` — line 49.
- `get_kv_cache_shape`: `cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla")`
  584-byte shape condition — line 104.
DROPPED-NATIVE (#50004, `b2f9e4caa4`): `active_topk_width` adaptive width
(main:264–270, identical min/max/`next_power_of_2`/`_C128A_TOPK_ALIGNMENT`/
`c128a_max_compressed` formula; passed as `max_compressed_tokens=active_topk_width`
at :283) and the packed buffer-view hunks in `build_c128a_topk_metadata`
(`.view(-1)[: n*max_compressed_tokens].view(n, max_compressed_tokens)` for
both `global_decode` and `prefill_local`, stride args →
`max_compressed_tokens`) — main's implementation matches the overlay
semantics exactly.

### 11. `vllm/utils/torch_utils.py` — CARRIED (1 line) + SUBSUMED (1 hunk)
- CARRIED: `"nvfp4_ds_mla": torch.uint8` in `STR_DTYPE_TO_TORCH_DTYPE` —
  line 54, alongside main's new `"nvfp4_4over6": torch.uint8` (both kept).
- SUBSUMED (no delta needed): `is_quantized_kv_cache` — main widened the
  check to `kv_cache_dtype.startswith("nvfp4")`, which already accepts
  `"nvfp4_ds_mla"` (and `nvfp4_4over6`). c7's `in ("nvfp4", "nvfp4_ds_mla")`
  form is a strict subset of main's; main's form kept. This was a merge
  conflict resolved for main's side.

### 12. `vllm/v1/attention/backends/mla/flashmla_sparse.py` — CARRIED (clean merge; main == 0.26.0 for this file)
- `"nvfp4_ds_mla"` in `FlashMLASparseBackend.supported_kv_cache_dtypes` — :94.
- `FlashMLASparseImpl.__init__` assert relaxed to
  `kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")` with the updated message — :565.
- Prefill-workspace condition widened identically — :570.

### 13. `vllm/v1/attention/backends/mla/sparse_swa.py` — CARRIED (clean merge)
- `DeepseekV4SWACache.get_kv_cache_spec` (:87): 584-alignment hunk —
  `uses_nvfp4_ds_mla_layout` flag (:92–94) + 3-way `alignment=(584 if
  nvfp4_ds_mla else 576 if fp8_ds_mla else 512)` (:103–109), with c7's
  comment. Conditions extended, not restructured.
- `DeepseekSparseSWABackend.get_kv_cache_shape` (:148): condition widened to
  `cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla")` — :156.

### 14. `vllm/v1/kv_cache_interface.py` — CARRIED (2 widenings) + SUBSUMED (1 hunk)
- CARRIED: `MLAAttentionSpec.real_page_size_bytes` condition widened to
  `in ("fp8_ds_mla", "nvfp4_ds_mla")` (:409, c7 comment retained) and
  `SlidingWindowMLASpec.real_page_size_bytes` widened identically (:649–653).
- SUBSUMED (no delta): `get_kv_quant_mode` — main widened to
  `kv_cache_dtype.startswith("nvfp4")` → `KVQuantMode.NVFP4`, which covers
  `"nvfp4_ds_mla"`. Merge conflict resolved for main's side.

### 15. `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` — DROPPED-NATIVE (#50330), omitted
- c7's file is **byte-identical to main's**. #50330 (`fcdc7c2e9c`) natively
  merged: `get_draft_quant_config` import (main:22) and
  `draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)` (:42)
  both sit in `load_dspark_model` (def at :15) with the same comment.

## Backport disposition summary (all 5 native on main)

| PR | Native commit | Overlay file(s) | Disposition |
|---|---|---|---|
| #48957 | `37e370fe93` | compressor.py | dropped, semantics verified identical |
| #49486 | `b0cb1da1bd` | attention.py (kernel + fast path) | dropped, condition + kernel byte-identical |
| #50004 | `b2f9e4caa4` | sparse_mla.py | dropped, packed-view semantics verified identical |
| #48047 | `b49eaf205a` | flashinfer_sparse.py (q-head ladder) | dropped, main's form kept |
| #50330 | `fcdc7c2e9c` | dspark/utils.py | dropped, file byte-identical |

## Forward-drift audit

- `fp8_ds_mla` line multiset across `vllm/models/deepseek_v4`,
  `vllm/v1/attention/backends/mla`, `vllm/config`: **identical between
  v0.26.0 and main** (62 lines, same 12 files, same per-file counts, same
  content). No main-new gate/condition needs `nvfp4_ds_mla` acceptance beyond
  the carried hunks.
- `nvfp4` mentions in the same paths: main-new items are `"nvfp4_4over6"`
  (cache.py Literal + doc) and the `startswith("nvfp4")` gates in
  `vllm.py`/`kv_cache_interface.py`/`torch_utils.py` — all either handled
  (vllm.py rewrite carried) or beneficial (startswith gates subsume the
  overlay's enumerations). `vllm/config/attention.py` `IndexerKVDType` is
  indexer-KV, not cache_dtype — unchanged old→main, out of scope.
- Import spot-checks on main: `vllm.models.deepseek_v4.common.ops.
  build_flashinfer_mixed_sparse_indices` ✓ (common/ops/cache_utils.py:820);
  `vllm.v1.attention.backend.MultipleOf` ✓ (:49);
  `vllm.utils.flashinfer.flashinfer_trtllm_batch_decode_sparse_mla_dsv4` ✓
  (lazy-import wrapper, flashinfer.py:156, exported in `__all__`);
  `has_flashinfer_sparse_mla_sm120` ✓ (:216); `from vllm.triton_utils import
  tl, triton` ✓ — module became a **package** on main
  (`vllm/triton_utils/__init__.py`) but exports both names, import contract
  preserved.
- `DeepseekV4FlashInferSM120Attention` still lives in
  `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` with the same name
  (main:536) — whole-file replacement targets the right class.
- Out-of-scope note: the c7 branch also diffs two **test** files
  (`tests/kernels/attention/test_flashmla_sparse.py`,
  `tests/kernels/test_compressor_kv_cache.py`). The 0261 port covers `vllm/`
  only, matching the overlay026 deliverable shape.

## Merge conflicts encountered (all resolved, no markers in deliverables)

1. `config/vllm.py` — main widened gate condition; kept main's condition +
   overlay rewrite body.
2. `models/deepseek_v4/attention.py` — fast-path return shape; kept main's
   #51430 form.
3. `models/deepseek_v4/nvidia/flashinfer_sparse.py` — insertion-point
   adjacency; kept new static method + main's native ladder call.
4. `utils/torch_utils.py` ×2 — dict neighbors (both kept); is_quantized_kv_cache
   (main's `startswith` kept, subsumes overlay).
5. `v1/kv_cache_interface.py` — get_kv_quant_mode (main's `startswith` kept,
   subsumes overlay).

No conflict could not be resolved cleanly.

## Verification

- `python3 -m py_compile`: **13/13 port outputs pass** (the §14 hotfix and the 3
  revert files are additionally compiled by the image-build smoke).
- Every output diffed against `git show 48bada6ea4:<path>` to confirm the
  delta is exactly the documented hunks (nothing else carried, nothing lost).
- Shipped `overlay026/` tree vs `gx10-overlay-026-c7`: identical for all 15
  files (no disagreement to resolve).

## 14. `vllm/utils/deep_gemm.py` — CARRIED NEW (2026-08-11, post-build hotfix; 14th overlay file)

Not part of the c7→main port (this file is unoverlaid in overlay026). Added after the
first cand8 boot died at weight-load: main's vendored DeepGEMM pin moved from
deepseek-ai `a6b593d` (prod 0.26.0) to the vllm-project fork `e21c821` (#51003), whose
`csrc/apis/layout.hpp` **dropped every `arch_major == 12` branch** in
`transform_sf_into_required_layout` (SM100-only now, still true on DG fork main today)
→ `RuntimeError: Assertion error (layout.hpp:60): Unknown SF transformation` on GB10.
Verified: the C++ transform is only reached at weight-load; vLLM main's only live
caller on our path is `fp8_utils.py:1077` via the wrapper (the k-grouped shim has no
callers; `nvidia/model.py`'s MegaMoE raw-module call site is not our selected backend,
DEEPGEMM_MXFP4 is). DG ABI check: all 10 binding symbols main's python uses resolve
identically on both pins, so a pin revert was viable, but the fix here is cheaper:
a 1:1 torch port of the old pin's own reference impl
(`get_mn_major_tma_aligned_packed_ue8m0_tensor_torch`, smxx_layout.hpp:149-172)
intercepting the FP32 SF case on `capability.major == 12` inside the wrapper. Zero
runtime-perf effect (weight-load only). Bit-exactness vs the cand7 image's a6b593d
C++ binding verified on the head node before lane boot (shape sweep incl. odd mn/k).

### Verification result (2026-08-11, head node, verify-sf-layout.py)

Port vs the cand7 image's a6b593d C++ binding: **BIT-EXACT on all 12 production-class
cases** (2-D/3-D, aligned/odd mn and sf_k, UE8M0-class inputs = pure powers of two).
Caveat discovered during verification: the DG JIT kernel's overlapping-OR packing
(`v1>>15` etc.) differs from the clean packing for NON-power-of-2 fp32 inputs (mantissa
junk ORs into the previous byte) — that input class never occurs in this flow (the
checkpoint is scale_fmt=ue8m0, so `_upcast_e8m0_to_fp32` yields mantissa-zero values);
the torch port is the clean packing and matches the kernel exactly on the real class.
Image after hotfix: sha256:e4f297e4dfda… (parity both nodes).

## 15. DeepGEMM pin revert — runtime-layer swap to deepseek-ai@a6b593d2 (2026-08-11, hotfix #2)

Second e21c821 sm120 casualty, found on the first clean rendezvous boot: the head's
worker proc died in `profile_run` → first forward pass with
`RuntimeError: Assertion error (.../csrc/apis/hyperconnection.hpp:56): Unsupported
architecture` — `tf32_hc_prenorm_gemm` (DeepSeek V4 hyperconnection residual, hot path,
every layer). Full csrc diff e21c821 vs a6b593d2: the fork pin dropped the ENTIRE sm120
backend (0 sm120 refs in csrc vs 13 files: sm120_bf16_gemm / sm120_bmk_bnk_mn /
sm120_fp8_fp4_gemm_1d1d / sm120_tf32_hc_prenorm_gemm / heuristics/sm120.hpp). Per-branch
restoration was the wrong shape — this is a wholesale backend removal.

Fix: `Dockerfile.runtime-0261` swaps the vendored `vllm/third_party/deep_gemm/` package
to the 0.26.0 prod pin `deepseek-ai/DeepGEMM@a6b593d2` (nv-dev) at image-build time —
rebuilds the pybind11 `_C.so` (single g++ TU via `build_deepgemm_C.py`, copied verbatim
from `tools/`; the final image carries g++/nvcc/Python.h for runtime JIT) and replaces
the python package + JIT `include/` tree. Only cp312 is rebuilt (sole interpreter).
Safety of the revert: vllm's `utils/deep_gemm.py` wrapper is BYTE-IDENTICAL between
0.26.0 and main @48bada6ea4, and all 17 getattr-bound symbols exist in a6b593d2
(verified against its python package). No raw `deep_gemm` imports anywhere in main's
tree outside the wrapper (only the xpu model path, not ours). This also makes the DG
layer IDENTICAL to prod — removes a whole variable class from the A/B gate.

The section-14 torch SF-layout port stays in the overlay: bit-exact vs this same pin's
C++ binding, weight-load only, and it shadows the C++ path on cap-12 either way.

Smoke coverage added to build-0261-image.sh: cp312 `_C.so` presence, `sm120` marker
grep over the vendored `include/` (a6b593d2 = 11 files, e21c821 = 0), and a --gpus=all
import + hasattr probe (`tf32_hc_prenorm_gemm`, `transform_sf_into_required_layout`).

DG JIT cache note: `DG_JIT_CACHE_DIR` defaults to `$VLLM_CACHE_ROOT/deep_gemm` — under this
kit's lane root that is `/cache/huggingface/vllm-cache-c8r/deep_gemm` — kernels JIT'd from
e21c821 sources during the failed boots linger
there; wipe `$VLLM_CACHE_ROOT/deep_gemm` on both nodes before the first post-swap boot
(one-time; the swap changes kernel sources, and stale same-name entries are not worth
reasoning about).
