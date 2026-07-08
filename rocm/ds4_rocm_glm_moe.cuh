// GLM 5.2 routed MoE kernels
// Gate+up pair (Q2_K/Q4_K) + down projection (Q2_K/Q4_K/Q5_K/Q6_K)
// 8 selected experts, model_map + offset weight addressing

// Quant type constants (matches gguf-tools/quants.h values)
enum {
    DS4Q_TYPE_Q2_K = 10,
    DS4Q_TYPE_Q4_K = 12,
};

// Helper: compute SwiGLU activation
__device__ static float glm_silu(float g, float u, float w) {
    float gs = 1.0f / (1.0f + expf(-g));
    return gs * g * u * w;
}

// Helper: expert weight pointer from model_map
__device__ static const char *glm_expert_ptr(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        int32_t     expert_id,
        uint64_t    expert_bytes,
        uint32_t    row,
        uint64_t    row_bytes) {
    uint64_t addr = offset + (uint64_t)expert_id * expert_bytes + (uint64_t)row * row_bytes;
    if (addr > model_size) return NULL;
    return (const char *)model_map + addr;
}

// Q4_K gate+up pair kernel: 256 threads, each thread processes one row of mid_dim
// Grid: (1, n_expert, n_tokens) — single block per expert per token

// Q4_K gate+up pair: 256 threads, each thread processes one block of mid_dim
__global__ static void glm_q4_K_pair_swiglu_kernel(
        float       *mid,
        const char  *gate_base,
        const char  *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint32_t      n_expert,
        uint32_t      expert_in_dim,
        uint32_t      expert_mid_dim,
        uint64_t      gate_row_bytes,
        uint64_t      up_row_bytes,
        uint32_t      n_tokens) {
    uint32_t token = blockIdx.x;
    uint32_t expert = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (token >= n_tokens || expert >= n_expert) return;
    if (tid >= expert_mid_dim) return;

    int32_t e = selected[expert];
    float w = weights[expert];

    // Gate and up row pointers for this expert
    const char *gate_row = gate_base +
        (uint64_t)e * (uint64_t)expert_mid_dim * gate_row_bytes +
        (uint64_t)tid * gate_row_bytes;
    const char *up_row = up_base +
        (uint64_t)e * (uint64_t)expert_mid_dim * up_row_bytes +
        (uint64_t)tid * up_row_bytes;

    // Input x quantized to Q8_K blocks
    const cuda_block_q8_K *xq = (const cuda_block_q8_K *)(x + (uint64_t)token * expert_in_dim);
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;

    // Accumulate gate and up dot products across all blocks of x
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const cuda_block_q4_K *gblk = (const cuda_block_q4_K *)(gate_row + (uint64_t)b * sizeof(cuda_block_q4_K));
        const cuda_block_q4_K *ublk = (const cuda_block_q4_K *)(up_row + (uint64_t)b * sizeof(cuda_block_q4_K));
        gate_acc += dev_dot_q4_K_q8_K_block(gblk, xq + b);
        up_acc += dev_dot_q4_K_q8_K_block(ublk, xq + b);
    }

    // SwiGLU: mid = silu(gate) * gate * up * weight
    uint32_t mid_offset = (uint64_t)token * (uint64_t)n_expert * expert_mid_dim +
                          (uint64_t)expert * expert_mid_dim + tid;
    mid[mid_offset] = glm_silu(gate_acc, up_acc, w);
}

// Q2_K gate+up pair kernel
__global__ static void glm_q2_K_pair_swiglu_kernel(
        float       *mid,
        const char  *gate_base,
        const char  *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint32_t      n_expert,
        uint32_t      expert_in_dim,
        uint32_t      expert_mid_dim,
        uint64_t      gate_row_bytes,
        uint64_t      up_row_bytes,
        uint32_t      n_tokens) {
    uint32_t token = blockIdx.x;
    uint32_t expert = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (token >= n_tokens || expert >= n_expert) return;
    if (tid >= expert_mid_dim) return;

    int32_t e = selected[expert];
    float w = weights[expert];

    const char *gate_row = gate_base +
        (uint64_t)e * (uint64_t)expert_mid_dim * gate_row_bytes +
        (uint64_t)tid * gate_row_bytes;
    const char *up_row = up_base +
        (uint64_t)e * (uint64_t)expert_mid_dim * up_row_bytes +
        (uint64_t)tid * up_row_bytes;

    const cuda_block_q8_K *xq = (const cuda_block_q8_K *)(x + (uint64_t)token * expert_in_dim);
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const cuda_block_q2_K *gblk = (const cuda_block_q2_K *)(gate_row + (uint64_t)b * sizeof(cuda_block_q2_K));
        const cuda_block_q2_K *ublk = (const cuda_block_q2_K *)(up_row + (uint64_t)b * sizeof(cuda_block_q2_K));
        gate_acc += dev_dot_q2_K_q8_K_block(gblk, xq + b);
        up_acc += dev_dot_q2_K_q8_K_block(ublk, xq + b);
    }

    uint32_t mid_offset = (uint64_t)token * (uint64_t)n_expert * expert_mid_dim +
                          (uint64_t)expert * expert_mid_dim + tid;
    mid[mid_offset] = glm_silu(gate_acc, up_acc, w);
}

// Down projection kernel using Q8_K mid and Q4_K/Q2_K weights
__global__ static void glm_q4_K_down_block_kernel(
        float       *out,
        const cuda_block_q8_K *midq,
        const char  *down_base,
        const int32_t *selected,
        const float *weights,
        uint32_t      n_expert,
        uint32_t      expert_mid_dim,
        uint32_t      out_dim,
        uint64_t      down_row_bytes,
        uint32_t      n_tokens) {
    uint32_t token = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (token >= n_tokens) return;
    if (tid >= out_dim) return;

    float sum = 0.0f;
    uint32_t mid_blocks = expert_mid_dim / CUDA_QK_K;
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        int32_t e = selected[expert];
        float w = weights[expert];

        const char *down_row = down_base +
            (uint64_t)e * (uint64_t)out_dim * down_row_bytes +
            (uint64_t)tid * down_row_bytes;

        const cuda_block_q8_K *mid_src = midq +
            (uint64_t)token * (uint64_t)n_expert * mid_blocks +
            (uint64_t)expert * mid_blocks;

        float expert_out = 0.0f;
        for (uint32_t b = 0; b < mid_blocks; b++) {
            const cuda_block_q4_K *dblk = (const cuda_block_q4_K *)(down_row + (uint64_t)b * sizeof(cuda_block_q4_K));
            expert_out += dev_dot_q4_K_q8_K_block(dblk, mid_src + b);
        }
        sum += expert_out * w;
    }
    out[(uint64_t)token * out_dim + tid] = sum;
}

__global__ static void glm_q2_K_down_block_kernel(
        float       *out,
        const cuda_block_q8_K *midq,
        const char  *down_base,
        const int32_t *selected,
        const float *weights,
        uint32_t      n_expert,
        uint32_t      expert_mid_dim,
        uint32_t      out_dim,
        uint64_t      down_row_bytes,
        uint32_t      n_tokens) {
    uint32_t token = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (token >= n_tokens) return;
    if (tid >= out_dim) return;

    float sum = 0.0f;
    uint32_t mid_blocks = expert_mid_dim / CUDA_QK_K;
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        int32_t e = selected[expert];
        float w = weights[expert];

        const char *down_row = down_base +
            (uint64_t)e * (uint64_t)out_dim * down_row_bytes +
            (uint64_t)tid * down_row_bytes;

        const cuda_block_q8_K *mid_src = midq +
            (uint64_t)token * (uint64_t)n_expert * mid_blocks +
            (uint64_t)expert * mid_blocks;

        float expert_out = 0.0f;
        for (uint32_t b = 0; b < mid_blocks; b++) {
            const cuda_block_q2_K *dblk = (const cuda_block_q2_K *)(down_row + (uint64_t)b * sizeof(cuda_block_q2_K));
            expert_out += dev_dot_q2_K_q8_K_block(dblk, mid_src + b);
        }
        sum += expert_out * w;
    }
    out[(uint64_t)token * out_dim + tid] = sum;
}

// Dispatch functions

static int glm_moe_validate_tensors(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x,
        uint32_t              n_expert,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        uint32_t              n_tokens) {
    if (!out || !mid || !selected || !weights || !x) return 0;
    if (!cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(mid, n_tokens, (uint64_t)n_expert * expert_mid_dim, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(selected, 1, n_expert, sizeof(int32_t))) return 0;
    if (!cuda_tensor_has_elems2(weights, 1, n_expert, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float))) return 0;
    return 1;
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident) {
    (void)layer_index;
    (void)force_resident;
    if (!glm_moe_validate_tensors(out, mid, selected, weights, x,
                                  n_expert, expert_in_dim, expert_mid_dim, out_dim, 1)) return 0;
    if (!model_map || n_total_expert == 0 || n_expert == 0 ||
        gate_type != up_type ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        expert_in_dim % CUDA_QK_K != 0 ||
        expert_mid_dim % CUDA_QK_K != 0 ||
        out_dim % CUDA_QK_K != 0) return 0;

    // Validate model range
    uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    uint64_t up_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (!cuda_model_range_fits(model_size, gate_offset, gate_bytes)) return 0;
    if (!cuda_model_range_fits(model_size, up_offset, up_bytes)) return 0;
    if (!cuda_model_range_fits(model_size, down_offset, down_bytes)) return 0;

    const char *gate_base = (const char *)cuda_model_range_ptr(
            model_map, gate_offset, gate_bytes, "glm_moe_gate");
    const char *up_base = (const char *)cuda_model_range_ptr(
            model_map, up_offset, up_bytes, "glm_moe_up");
    const char *down_base = (const char *)cuda_model_range_ptr(
            model_map, down_offset, down_bytes, "glm_moe_down");
    if (!gate_base || !up_base || !down_base) return 0;

    // Quantize input x to Q8_K
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    // Use mid buffer as temp for quantized x (size: 1 * xq_blocks * sizeof(cuda_block_q8_K))
    cuda_block_q8_K *xq = (cuda_block_q8_K *)mid->ptr;
    {
        dim3 grid(xq_blocks, 1, 1);
        q8_K_quantize_kernel<<<grid, 256>>>(
                xq, (const float *)x->ptr, expert_in_dim, 1);
        if (!cuda_ok(cudaGetLastError(), "glm_moe xq quantize")) return 0;
    }

    int ok = 1;

    // Gate+up pair kernel
    uint32_t pair_threads = 256;
    uint32_t pair_grid_x = (expert_mid_dim + pair_threads - 1) / pair_threads;
    if (gate_type == DS4Q_TYPE_Q4_K) {
        glm_q4_K_pair_swiglu_kernel<<<dim3(1, n_expert, 1), pair_threads>>>(
                (float *)mid->ptr,
                gate_base, up_base,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                n_expert, expert_in_dim, expert_mid_dim,
                gate_row_bytes, up_row_bytes, 1);
        ok = cuda_ok(cudaGetLastError(), "glm_moe q4 pair");
    } else if (gate_type == DS4Q_TYPE_Q2_K) {
        glm_q2_K_pair_swiglu_kernel<<<dim3(1, n_expert, 1), pair_threads>>>(
                (float *)mid->ptr,
                gate_base, up_base,
                (const float *)x->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                n_expert, expert_in_dim, expert_mid_dim,
                gate_row_bytes, up_row_bytes, 1);
        ok = cuda_ok(cudaGetLastError(), "glm_moe q2 pair");
    } else {
        return 0; // unsupported gate type
    }
    if (!ok) return 0;

    // Quantize mid to Q8_K for down projection
    uint32_t mid_blocks = expert_mid_dim / CUDA_QK_K;
    uint32_t midq_count = (uint32_t)n_expert * mid_blocks;
    // Use out buffer as temp for quantized mid (size: midq_count * sizeof(cuda_block_q8_K))
    cuda_block_q8_K *midq = (cuda_block_q8_K *)out->ptr;
    {
        dim3 grid(mid_blocks, n_expert, 1);
        q8_K_quantize_kernel<<<grid, 256>>>(
                midq, (const float *)mid->ptr, expert_mid_dim, n_expert);
        if (!cuda_ok(cudaGetLastError(), "glm_moe midq quantize")) return 0;
    }

    // Down projection kernel
    uint32_t down_threads = 256;
    uint32_t down_grid_x = (out_dim + down_threads - 1) / down_threads;
    if (down_type == DS4Q_TYPE_Q4_K) {
        glm_q4_K_down_block_kernel<<<dim3(1, 1, 1), down_threads>>>(
                (float *)out->ptr,
                midq, down_base,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                n_expert, expert_mid_dim, out_dim,
                down_row_bytes, 1);
        ok = cuda_ok(cudaGetLastError(), "glm_moe q4 down");
    } else if (down_type == DS4Q_TYPE_Q2_K) {
        glm_q2_K_down_block_kernel<<<dim3(1, 1, 1), down_threads>>>(
                (float *)out->ptr,
                midq, down_base,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                n_expert, expert_mid_dim, out_dim,
                down_row_bytes, 1);
        ok = cuda_ok(cudaGetLastError(), "glm_moe q2 down");
    } else {
        return 0; // unsupported down type
    }

    return ok;
}
