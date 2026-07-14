# ROCm GLM Implementation Plan — Updated

## Current State (all GLM dispatch functions implemented)

| Function | File | Status |
|---|---|---|
| GLM Router (select_tensor, select_batch) | `ds4_rocm_glm_router.cuh` | Real, included in `ds4_rocm.cu:131` |
| GLM MOE (one, batch, direct_scalar_q4) | `ds4_rocm_glm_moe.cuh` | Real, included in `ds4_rocm.cu:132` |
| GLM KV (7 functions) | `ds4_rocm_glm_kv.cuh` | Real, included in `ds4_rocm.cu:133` |
| GLM Indexer (7 functions) | `ds4_rocm_glm_indexer.cuh` | Real, included in `ds4_rocm.cu:134` |
| GLM Attention (11 functions) | `ds4_rocm_glm_attention.cuh` | Real, included in `ds4_rocm.cu:135` |
| Streaming expert cache | `ds4_rocm_current_api_compat.cuh` | Mostly real |
| Selected readback events | `ds4_rocm_current_api_compat.cuh` | Real |
| Model FD/map, SSD streaming | `ds4_rocm_current_api_compat.cuh` | Real |

## Completed Stubs

All stubs from the original plan are now resolved (commits `2ba3d99`, `8b75c08`, `a1fe3f1`):

| # | Function | Fix | Commit |
|---|---|---|---|
| 4 | `ds4_gpu_preload_q4_expert_tables` | Already returned 1 (no change needed) | `a1fe3f1` |
| 6 | `ds4_gpu_stream_expert_cache_seed_experts` | Full ROCm resident cache implementation (priority sort, alloc, async upload) | `a1fe3f1` |
| 7 | `ds4_gpu_stream_expert_cache_budget_for_expert_size` | Validate expert byte sizes, return `g_stream_expert_cache_budget` | `2ba3d99` |

## Bugs Fixed in Current Session

| # | Bug | Fix | Status |
|---|---|---|---|
| A | `seed_experts` return convention inverted: returns 1 on failure (caller checks `==0`), silently swallowing failures | Change failure paths to return 0 | Applied (not yet committed) |
| B | `cuda_model_range_ptr_from_fd` doesn't register mmap pointer as cached when arena alloc fails → `cuda_model_range_is_cached` returns false → `ds4_gpu_set_model_map_spans` fails | Register mmap pointer in `g_model_ranges` on arena OOM | Applied (not yet committed) |

## Known / Speculated Issues

### 1. Prefill batch cache seeding uses file FD reads per expert
`glm_batch_selected_prepare` now reads each unique expert's gate/up/down via `cuda_pread_full` into a pinned staging buffer, then async copies to GPU. For layers with 64 unique experts, this issues 192 `pread` syscalls. On fast NVMe this is fine; on slower storage it may bottleneck. Potential improvement: batch-read all expert weights for a layer into a single large staging buffer, then scatter to slot-ordered GPU buffers in one trip.

### 2. No sync between upload stream and default stream for batch cache
The batch cache seeding copies onto `g_model_upload_stream` (non-blocking), then waits with `cudaStreamSynchronize`. The MoE dispatch that follows uses the default stream. On ROCm, cross-stream visibility requires an implicit sync at the next default-stream operation. If the upload stream copy hasn't completed before the MoE kernel launch, the kernel may read stale data. Currently the `cudaStreamSynchronize` at line 1043 drains the upload stream before returning, so this should be safe — but if ROCm's `cudaStreamSynchronize` is broken for non-blocking streams, the MoE kernel could read garbage.

### 3. `ds4_gpu_flush_commands()` (cudaDeviceSynchronize) serializes pipeline
The flush added before batch cache seeding calls `cudaDeviceSynchronize`, which waits for ALL prior GPU work (attention, router, etc.) to complete. This serializes the prefill pipeline layer-by-layer. For large prompts, this may hide GPU parallelism. Alternative: use a lighter-weight sync (e.g., `cudaStreamSynchronize(0)` or an event) if only the default stream needs draining.

### 4. `stage_sync` disabled for large prefill
`glm_graph_small_prefill_stage_sync` returns false when `n_tokens > DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS` (likely 0), meaning no stage sync boundaries are inserted during long prefill. GPU commands queue up without intermediate flushes. If a kernel hangs (e.g., router select for 4096 tokens), the error is only detected at the end. Investigate whether enabling periodic syncs helps with stability.

### 5. Layer cache vs batch cache interaction
`ds4_gpu_glm_stream_expert_cache_seed_batch_selected` checks the layer cache first (line 1176). If the layer cache has data from a prior decode step, batch seeding is skipped and the MoE dispatch uses the layer cache directly. But for prefill, the layer cache is cold (no prior decode), so seeding always runs. After prefill, the layer cache may hold the last layer's experts — subsequent decode steps can reuse them. Verify the layer cache is not holding stale experts from a different layer.

### 6. IQ2_XXS kernel correctness
The IQ2_XXS MoE pair/down kernels were ported in `8b75c08` but have not been verified with real IQ2_XXS weights. The lookup table layout and dequant math should be validated with a test prompt.

### 7. IQ2_XXS OOM: model too large for GTT budget
On a 128 GiB Strix Halo APU with 110 GiB GTT, a 250 GiB IQ2_XXS model exceeds the expert cache budget (68.69 GiB for 7563 experts). The crash chain:
  1. `seed_experts` fails silently (return convention bug) → caller thinks seeding succeeded
  2. MoE dispatch finds no resident expert → falls back to `cuda_model_range_ptr`
  3. `cuda_model_range_ptr_from_fd` tries `cuda_model_arena_alloc` → `cudaMalloc` fails with OOM
  4. Falls back to direct mmap pointer (valid on APU) but `cuda_model_range_is_cached` returns false
  5. `ds4_gpu_set_model_map_spans` returns 0 → "SSD streaming failed to map layer model spans"

Fix A (return convention) prevents the silent swallow — caller sees failure and can use batch cache fallback. Fix B (register mmap as cached) prevents the "NOT CACHED" failure on APU.

Still open: the 16 GiB free reserve (`cuda_stream_resident_free_reserve_bytes`) leaves only ~94 GiB for model+cache on a 110 GiB GTT. With 250 GiB IQ2_XXS, the model map + expert cache can't coexist within this budget. The batch cache fallback (`glm_batch_selected_prepare`) may itself OOM when trying to `cudaMalloc` slot-ordered buffers.
