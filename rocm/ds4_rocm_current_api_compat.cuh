#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

extern "C" int ds4_gpu_tensor_read_after_selected_event(
        const ds4_gpu_tensor *tensor,
        uint64_t offset,
        void *data,
        uint64_t bytes,
        uint64_t event_value,
        const char *label) {
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset ||
        event_value == 0 ||
        !g_selected_readback_event) {
        return 0;
    }
    if (!g_selected_readback_stream) {
        cudaError_t err =
            cudaStreamCreateWithFlags(&g_selected_readback_stream,
                                      cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
#ifdef __HIP_PLATFORM_AMD__
    cudaError_t err = hipStreamWaitEvent(g_selected_readback_stream,
                                         g_selected_readback_event,
                                         0);
#else
    cudaError_t err = cudaStreamWaitEvent(g_selected_readback_stream,
                                          g_selected_readback_event,
                                          0);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback stream wait failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemcpyAsync(data,
                          (const char *)tensor->ptr + offset,
                          (size_t)bytes,
                          cudaMemcpyDeviceToHost,
                          g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback copy failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaStreamSynchronize(g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback sync failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    int ok = ds4_gpu_set_model_fd(fd);
    g_model_fd_host_base = model_map ? model_map : g_model_host_base;
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(
        ds4_gpu_tensor *dst,
        uint64_t dst_offset,
        const ds4_gpu_tensor *src,
        uint64_t src_offset,
        uint64_t count) {
    if (!dst || !src || !dst->ptr || !src->ptr) return 0;
    if ((dst_offset % sizeof(__half)) != 0 || (src_offset % sizeof(float)) != 0) return 0;
    if (dst_offset > dst->bytes || src_offset > src->bytes) return 0;
    if (count > (UINT64_MAX / sizeof(__half)) || count > (UINT64_MAX / sizeof(float))) return 0;
    uint64_t dst_bytes = count * sizeof(__half);
    uint64_t src_bytes = count * sizeof(float);
    if (dst_bytes > dst->bytes - dst_offset || src_bytes > src->bytes - src_offset) return 0;
    if (count == 0) return 1;
    f32_to_f16_kernel<<<(count + 255u) / 256u, 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32 to f16 launch");
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    /* Q4 expert tables are accessed via model_map at inference time;
     * no separate preload needed on ROCm (matches CUDA behavior). */
    (void)model_map;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_routed_moe_selected_override_n = 0;
    g_stream_selected_cache.loaded = 0;
    g_stream_batch_selected_cache.loaded = 0;
    static bool stats_atexit_registered = false;
    if (enabled && !stats_atexit_registered) {
        atexit(ds4_gpu_glm_moe_stats);
        stats_atexit_registered = true;
    }
}

extern "C" void ds4_gpu_set_glm_model(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_cache_budget = experts;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    /* ROCm on UMA (Strix Halo): cudaMemGetInfo returns only dedicated VRAM
     * (~1 GiB), not the GTT (~110 GiB shared system RAM).  Query host memory
     * instead so the streaming auto-cache can use the full GTT budget. */
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess &&
        total_b > 4ull * 1024ull * 1024ull * 1024ull) {
        /* total_b looks like real VRAM (not the 1 GiB Strix Halo stub) */
        return (uint64_t)total_b;
    }
    /* Fallback: use host memory (sysinfo on Linux, sysctl on macOS). */
#if defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        uint64_t total = (uint64_t)si.totalram * (uint64_t)si.mem_unit;
        if (total > 0) return total;
    }
#elif defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0 && mem > 0) return mem;
#endif
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return g_ssd_streaming_mode ? g_stream_expert_cache_budget : 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return (uint32_t)g_stream_resident_experts.size();
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_resident_cache_release();
}

extern "C" void ds4_gpu_stream_expert_cache_reset_selected(void) {
    fprintf(stderr, "ds4: diag reset selected active=%d loaded=%d override_n=%u\n",
            g_stream_selected_pending.active, g_stream_selected_cache.loaded,
            g_routed_moe_selected_override_n);
    g_stream_selected_cache.loaded = 0;
    g_routed_moe_selected_override_n = 0;
    memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_expert_bytes > UINT64_MAX / 2u ||
        down_expert_bytes > UINT64_MAX / 2u) {
        return 0;
    }
    return g_stream_expert_cache_budget;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    if (!cuda_stream_selected_load(table->model_map,
                                   table->model_size,
                                   table->layer,
                                   selected_ids,
                                   table->n_total_expert,
                                   n_selected,
                                   table->gate_offset,
                                   table->up_offset,
                                   table->down_offset,
                                   table->gate_expert_bytes,
                                   table->down_expert_bytes)) {
        return 0;
    }
    return cuda_stream_selected_finish_pending_missing(0);
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    return cuda_stream_selected_load(table->model_map,
                                     table->model_size,
                                     table->layer,
                                     selected_ids,
                                     table->n_total_expert,
                                     n_selected,
                                     table->gate_offset,
                                     table->up_offset,
                                     table->down_offset,
                                     table->gate_expert_bytes,
                                     table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!table) return 0;
    const ds4_gpu_tensor *selected_exec = NULL;
    const char **gate_ptrs = NULL;
    const char **up_ptrs = NULL;
    const char **down_ptrs = NULL;
    uint32_t unique = 0;
    return cuda_stream_batch_selected_prepare_from_host(table->model_map,
                                                        table->model_size,
                                                        table->layer,
                                                        selected_ids,
                                                        n_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes,
                                                        &selected_exec,
                                                        &gate_ptrs,
                                                        &up_ptrs,
                                                        &down_ptrs,
                                                        &unique,
                                                        1);
}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_load(table->model_map,
                                               table->model_size,
                                               table->layer,
                                               table->n_total_expert,
                                               table->gate_offset,
                                               table->up_offset,
                                               table->down_offset,
                                               table->gate_expert_bytes,
                                               table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_seed_selected(table->model_map,
                                                        table->layer,
                                                        selected,
                                                        n_tokens,
                                                        n_seed_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    cuda_stream_layer_expert_cache_release();
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !expert_ids || n_experts == 0) return 0;

    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    /* Priority sort: highest priority first.  If no priorities given,
     * use reverse index (last expert highest). */
    std::vector<uint32_t> order;
    try {
        order.reserve(n_experts);
    } catch (...) {
        return 1;
    }
    for (uint32_t i = 0; i < n_experts; i++) {
        const uint32_t prio = expert_priorities ? expert_priorities[i]
                            : (n_experts - i);
        uint32_t pos = 0;
        while (pos < order.size()) {
            const uint32_t other_prio = expert_priorities
                ? expert_priorities[order[pos]]
                : (n_experts - order[pos]);
            if (prio > other_prio) break;
            pos++;
        }
        order.insert(order.begin() + pos, i);
    }

    for (uint32_t ri = 0; ri < (uint32_t)order.size(); ri++) {
        const uint32_t i = order[ri];
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) continue;

        /* Already resident? */
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) continue;

        /* Allocate and load.  We pass no selected_ids context so
         * eviction doesn't protect any in-flight selected experts. */
        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         expert,
                                         NULL,
                                         0,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) return 1;

        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        const uint64_t expert_u64 = (uint64_t)(uint32_t)expert;
        const uint64_t gate_src =
            gate_offset + expert_u64 * gate_expert_bytes;
        const uint64_t down_src =
            down_offset + expert_u64 * down_expert_bytes;

        /* Load gate, up, down via upload stream. */
        cudaError_t err = cudaMemcpyAsync(entry.gate,
            (const char *)model_map + gate_src,
            (size_t)gate_expert_bytes,
            cudaMemcpyHostToDevice,
            g_model_upload_stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.up,
                (const char *)model_map + up_offset + expert_u64 * gate_expert_bytes,
                (size_t)gate_expert_bytes,
                cudaMemcpyHostToDevice,
                g_model_upload_stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.down,
                (const char *)model_map + down_src,
                (size_t)down_expert_bytes,
                cudaMemcpyHostToDevice,
                g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                "streaming hotlist seed copy failed at layer=%u expert=%d: %s\n",
                layer, expert, cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 1;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(
        const int32_t *selected,
        uint32_t n_selected) {
    if (n_selected > DS4_ROCM_N_EXPERT_USED || (!selected && n_selected != 0)) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_routed_moe_selected_override[i] = selected[i];
    }
    g_routed_moe_selected_override_n = n_selected;
    return 1;
}

// --- GLM 5.2 stub implementations (remaining: non-GLM stubs) ---

extern "C" int ds4_gpu_flush_encoder(void) { return 1; }

extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!out || !model_map || n_embd == 0 || (n_embd & 31u) != 0) return 0;
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    const uint64_t table_bytes = (uint64_t)n_vocab * row_bytes;
    if (weight_offset > model_size || table_bytes > model_size - weight_offset ||
        out->bytes < (uint64_t)n_embd * sizeof(float)) {
        return 0;
    }
    if (token >= n_vocab) token = 0;
    const char *wptr = cuda_model_range_ptr(model_map,
                                            weight_offset + (uint64_t)token * row_bytes,
                                            row_bytes,
                                            "glm token_embd row");
    if (!wptr) return 0;
    embed_token_q8_0_kernel<<<(n_embd + 255u) / 256u, 256>>>(
        (float *)out->ptr, (const unsigned char *)wptr, n_embd);
    return cuda_ok(cudaGetLastError(), "glm embed token q8_0 launch");
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!out || !tokens || !model_map || n_embd == 0 || (n_embd & 31u) != 0 ||
        n_tokens == 0) {
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    const uint64_t table_bytes = (uint64_t)n_vocab * row_bytes;
    if (weight_offset > model_size || table_bytes > model_size - weight_offset ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            table_bytes, "glm token_embd");
    if (!wptr) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    embed_tokens_q8_0_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>(
        (float *)out->ptr,
        (const int32_t *)tokens->ptr,
        (const unsigned char *)wptr,
        n_vocab, n_tokens, n_embd, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm embed tokens q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        float clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim, x, clamp);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        float clamp) {
    static ds4_gpu_tensor scratch_gate = {NULL, 0, 0};
    static ds4_gpu_tensor scratch_up = {NULL, 0, 0};
    if (!mid || out_dim == 0 || out_dim > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t bytes = out_dim * sizeof(float);
    if (scratch_gate.bytes < bytes) {
        if (scratch_gate.ptr) (void)cudaFree(scratch_gate.ptr);
        if (scratch_up.ptr) (void)cudaFree(scratch_up.ptr);
        scratch_gate.ptr = NULL;
        scratch_up.ptr = NULL;
        scratch_gate.bytes = 0;
        scratch_up.bytes = 0;
        void *g = NULL;
        void *u = NULL;
        if (!cuda_ok(cudaMalloc(&g, (size_t)bytes), "shared mid gate scratch") ||
            !cuda_ok(cudaMalloc(&u, (size_t)bytes), "shared mid up scratch")) {
            if (g) (void)cudaFree(g);
            return 0;
        }
        scratch_gate.ptr = g;
        scratch_gate.bytes = bytes;
        scratch_up.ptr = u;
        scratch_up.bytes = bytes;
    }
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&scratch_gate, &scratch_up,
                                                     mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim, x, clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        float clamp) {
    if (!gate || !up || !mid || !x || !model_map || n_tok == 0 ||
        out_dim == 0 || n_tok > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok * out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                           gate_offset, up_offset,
                                           in_dim, out_dim, out_dim,
                                           x, n_tok) &&
           ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(n_tok * out_dim), clamp, 1.0f);
}
