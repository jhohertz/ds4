// GLM 5.2 router kernels
// Matches Metal kernel_glm_router_select_one: sigmoid + bitonic sort top-k + normalize

__device__ static float ds4_glm_router_sigmoid(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

__device__ static bool ds4_glm_router_better(
        const float *scores,
        int32_t a,
        int32_t b) {
    float sa = scores[(uint32_t)a];
    float sb = scores[(uint32_t)b];
    return sa > sb || (sa == sb && a < b);
}

__global__ static void glm_router_select_kernel(
        int32_t *selected,
        float   *weights,
        float   *probs,
        const float *bias,
        const float *logits,
        uint32_t     n_expert,
        uint32_t     n_expert_used,
        float        expert_weight_scale,
        uint32_t     n_tokens) {
    uint32_t token = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (token >= n_tokens) return;

    // Shared memory: scores[256] + idx[256]
    __shared__ float shared_scores[256];
    __shared__ int32_t shared_idx[256];

    const float *token_logits = logits + (uint64_t)token * n_expert;
    int32_t *token_selected = selected + (uint64_t)token * n_expert_used;
    float *token_weights = weights + (uint64_t)token * n_expert_used;
    float *token_probs = probs + (uint64_t)token * n_expert;

    uint32_t n_exp = min(n_expert, 256u);
    bool active = tid < n_exp;
    float p = active ? ds4_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    if (active) token_probs[tid] = p;
    shared_scores[tid] = active ? p + bias[tid] : -INFINITY;
    shared_idx[tid] = (int32_t)tid;
    __syncthreads();

    // Bitonic sort
    for (uint32_t k = 2; k <= 256u; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            uint32_t other = tid ^ j;
            if (other > tid) {
                int32_t a = shared_idx[tid];
                int32_t b = shared_idx[other];
                bool descending = (tid & k) == 0u;
                bool swap = descending
                    ? ds4_glm_router_better(shared_scores, b, a)
                    : ds4_glm_router_better(shared_scores, a, b);
                if (swap) {
                    shared_idx[tid] = b;
                    shared_idx[other] = a;
                }
            }
            __syncthreads();
        }
    }

    uint32_t k_used = min(n_expert_used, n_exp);
    if (tid < k_used) {
        token_selected[tid] = shared_idx[tid];
    }
    __syncthreads();

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < k_used; i++) {
            sum += token_probs[(uint32_t)token_selected[i]];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        token_weights[tid] = token_probs[(uint32_t)token_selected[tid]] / sum * expert_weight_scale;
    }
}

static int glm_router_validate(
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *probs,
        uint32_t              n_expert,
        uint32_t              n_expert_used) {
    if (!selected || !weights || !probs) return 0;
    if (!cuda_tensor_has_elems2(selected, 1, n_expert_used, sizeof(int32_t))) return 0;
    if (!cuda_tensor_has_elems2(weights, 1, n_expert_used, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(probs, 1, n_expert, sizeof(float))) return 0;
    return 1;
}

extern "C" int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    if (!glm_router_validate(selected, weights, probs, n_expert, n_expert_used)) return 0;
    if (!logits || !model_map || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_expert || !(expert_weight_scale > 0.0f)) return 0;
    if (!cuda_tensor_has_f32(logits, n_expert)) return 0;

    const float *bias = NULL;
    if (bias_offset != 0) {
        uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (!cuda_model_range_fits(model_size, bias_offset, bias_bytes)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, bias_bytes, "glm_router_bias");
        if (!bias) return 0;
    }

    glm_router_select_kernel<<<1, 256>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert, n_expert_used, expert_weight_scale, 1);
    return cuda_ok(cudaGetLastError(), "glm_router_select launch");
}

extern "C" int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_expert || !(expert_weight_scale > 0.0f)) return 0;
    if (!cuda_tensor_has_elems2(logits, n_tokens, n_expert, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(selected, n_tokens, n_expert_used, sizeof(int32_t))) return 0;
    if (!cuda_tensor_has_elems2(weights, n_tokens, n_expert_used, sizeof(float))) return 0;
    if (!cuda_tensor_has_elems2(probs, n_tokens, n_expert, sizeof(float))) return 0;

    const float *bias = NULL;
    if (bias_offset != 0) {
        uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (!cuda_model_range_fits(model_size, bias_offset, bias_bytes)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, bias_bytes, "glm_router_bias");
        if (!bias) return 0;
    }

    glm_router_select_kernel<<<n_tokens, 256>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert, n_expert_used, expert_weight_scale, n_tokens);
    return cuda_ok(cudaGetLastError(), "glm_router_select_batch launch");
}
