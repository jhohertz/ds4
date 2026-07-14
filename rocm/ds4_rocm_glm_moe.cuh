// GLM 5.2 routed MoE kernels — full port of ds4_cuda_glm_moe.inc for ROCm
// Includes: per-element dequant, router, utility kernels, template MoE pair/down,
// streaming expert cache, batch launcher.

/* =========================================================================
 * K-quant block layouts (q5_K, q6_K — q2_K/q4_K already in ds4_rocm.h)
 * ========================================================================= */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[CUDA_QK_K / 8];
    uint8_t qs[CUDA_QK_K / 2];
} glm_block_q5_K;

typedef struct {
    uint8_t ql[CUDA_QK_K / 2];
    uint8_t qh[CUDA_QK_K / 4];
    int8_t scales[CUDA_QK_K / 16];
    uint16_t d;
} glm_block_q6_K;

/* GLM MoE cache hit/miss counters for end-of-inference stats */
static int g_glm_moe_selected_cache_hits;
static int g_glm_moe_batch_cache_hits;
static int g_glm_moe_layer_cache_hits;
static int g_glm_moe_fallback_range_ptr;
static int g_glm_moe_fallback_mmap;

extern "C" void ds4_gpu_glm_moe_stats(void) {
    fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM MoE stats: selected_cache=%d batch_cache=%d layer_cache=%d range_ptr=%d mmap=%d\n",
            g_glm_moe_selected_cache_hits,
            g_glm_moe_batch_cache_hits,
            g_glm_moe_layer_cache_hits,
            g_glm_moe_fallback_range_ptr,
            g_glm_moe_fallback_mmap);
    g_glm_moe_selected_cache_hits = 0;
    g_glm_moe_batch_cache_hits = 0;
    g_glm_moe_layer_cache_hits = 0;
    g_glm_moe_fallback_range_ptr = 0;
    g_glm_moe_fallback_mmap = 0;
}

/* IQ2_XXS tables — ported from ds4_iq2_tables_cuda.inc */
#define DS4_ROCKM_IQ2_KSIGN_SIZE 128u
#define DS4_ROCKM_IQ2_GRID_SIZE 256u

__device__ __constant__ uint8_t glm_iq2_ksigns[DS4_ROCKM_IQ2_KSIGN_SIZE] = {
       0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
     144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
     160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
      48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
     192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
      80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
      96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
     240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ __constant__ uint64_t glm_iq2xxs_grid[DS4_ROCKM_IQ2_GRID_SIZE] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* Per-element dequantization structs — port of ds4_cuda_glm_moe.inc */
/* ========================================================================= */

__device__ __forceinline__ static float glm_half_bits_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ static void glm_get_scale_min_k4_just2(
        uint32_t j, const uint8_t *q, uint32_t *sc, uint32_t *mn) {
    if (j < 4u) {
        *sc = q[j] & 63u;
        *mn = q[j + 4u] & 63u;
    } else {
        *sc = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] & 0xC0u) >> 2u);
        *mn = (q[j + 4u] >> 4u) | ((q[j] & 0xC0u) >> 2u);
    }
}

struct glm_deq_q2_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q2_K *xb =
            (const cuda_block_q2_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 16u;
        const uint32_t l = idx & 15u;
        const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        const uint32_t shift = ((group / 2u) & 3u) * 2u;
        const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
        const uint32_t sc = (uint32_t)xb->scales[group];
        return glm_half_bits_to_f32(xb->d) * (float)(sc & 0x0Fu) * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)(sc >> 4u);
    }
};

struct glm_deq_iq2_xxs {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_iq2_xxs *xb =
            (const cuda_block_iq2_xxs *)blocks + (k >> 8);
        const uint32_t within = k & 255u;
        const uint32_t ib32 = within >> 5;
        const uint32_t kk = within & 31u;
        const uint32_t l = kk >> 3;
        const uint32_t j = kk & 7u;
        const uint16_t *q2 = xb->qs + 4u * ib32;
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint32_t grid_idx = (aux0 >> (8u * l)) & 0xffu;
        const uint32_t sign_idx = (aux1 >> (7u * l)) & 127u;
        const uint64_t g = glm_iq2xxs_grid[grid_idx];
        const int32_t gj = (int32_t)(uint32_t)(uint8_t)(g >> (8u * j));
        const uint8_t signs = glm_iq2_ksigns[sign_idx];
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        return 0.125f * dev_f16_to_f32(xb->d) * (float)ls * (float)gj * sign;
    }
};

struct glm_deq_q4_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q4_K *xb =
            (const cuda_block_q4_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 32u;
        const uint32_t l = idx & 31u;
        uint32_t sc, mn;
        glm_get_scale_min_k4_just2(group, xb->scales, &sc, &mn);
        const uint32_t byte_off = (group >> 1u) * 32u + l;
        const uint32_t shift = (group & 1u) * 4u;
        const uint32_t q = ((uint32_t)xb->qs[byte_off] >> shift) & 0x0Fu;
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)mn;
    }
};

struct glm_deq_q5_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const glm_block_q5_K *xb =
            (const glm_block_q5_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 32u;
        const uint32_t l = idx & 31u;
        uint32_t sc, mn;
        glm_get_scale_min_k4_just2(group, xb->scales, &sc, &mn);
        const uint32_t ql_base = (group >> 1u) * 32u + l;
        const uint32_t shift = (group & 1u) * 4u;
        uint32_t q = ((uint32_t)xb->qs[ql_base] >> shift) & 0x0Fu;
        q += (xb->qh[l] & (uint8_t)(1u << group)) ? 16u : 0u;
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)mn;
    }
};

struct glm_deq_q6_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const glm_block_q6_K *xb =
            (const glm_block_q6_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t n128 = idx >> 7u;
        const uint32_t r = idx & 127u;
        const uint32_t l = r & 31u;
        const uint32_t quarter = r >> 5u;
        const uint32_t ql_base = n128 * 64u;
        const uint32_t qh_base = n128 * 32u;
        const uint32_t sc_base = n128 * 8u;
        uint32_t q;
        int sc;
        if (quarter == 0u) {
            q = (xb->ql[ql_base + l] & 0x0Fu) |
                (((xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 0u];
        } else if (quarter == 1u) {
            q = (xb->ql[ql_base + 32u + l] & 0x0Fu) |
                (((xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 2u];
        } else if (quarter == 2u) {
            q = (xb->ql[ql_base + l] >> 4u) |
                (((xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 4u];
        } else {
            q = (xb->ql[ql_base + 32u + l] >> 4u) |
                (((xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
            sc = (int)(xb->scales[sc_base + (l >> 4u) + 6u]);
        }
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)((int)q - 32);
    }
};

/* Tensor type ids (GGUF) */
enum {
    GLM_CUDA_TENSOR_Q2_K = 10,
    GLM_CUDA_TENSOR_Q4_K = 12,
    GLM_CUDA_TENSOR_Q5_K = 13,
    GLM_CUDA_TENSOR_Q6_K = 14,
    GLM_CUDA_TENSOR_IQ2_XXS = 16
};

/* =========================================================================
 * Small utility kernels.
 * ========================================================================= */

__global__ static void glm_add3_kernel(
        float *out, const float *a, const float *b, const float *c, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i] + c[i];
}

__global__ static void glm_add_rms_norm_weight_kernel(
        float *norm_out,
        float *sum_out,
        const float *a,
        const float *b,
        const float *w,
        uint32_t n,
        float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a[i] + b[i];
        sum_out[i] = v;
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        norm_out[i] = sum_out[i] * scale * w[i];
    }
}

__global__ static void glm_sort_i32_rows_asc_kernel(
        int32_t *dst, const int32_t *src, uint32_t row_width, uint32_t n_rows) {
    extern __shared__ int32_t glm_sort_row_tmp[];
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t n_threads = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)row * row_width;

    for (uint32_t i = tid; i < row_width; i += n_threads) {
        glm_sort_row_tmp[i] = src[base + i];
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= row_width; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < row_width; i += n_threads) {
                const uint32_t other = i ^ j;
                if (other > i && other < row_width) {
                    const int32_t va = glm_sort_row_tmp[i];
                    const int32_t vb = glm_sort_row_tmp[other];
                    const bool up = (i & k) == 0u;
                    if ((up && va > vb) || (!up && va < vb)) {
                        glm_sort_row_tmp[i] = vb;
                        glm_sort_row_tmp[other] = va;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < row_width; i += n_threads) {
        dst[base + i] = glm_sort_row_tmp[i];
    }
}

/* =========================================================================
 * GLM router: sigmoid probabilities, selection score = prob + bias, top-k
 * ========================================================================= */

__device__ __forceinline__ static float glm_router_sigmoid_dev(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ static bool glm_router_better_dev(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void glm_router_select_warp_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t k_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)token * n_expert;
    float *prob = probs + (uint64_t)token * n_expert;
    int32_t *sel = selected + (uint64_t)token * k_used;
    float *w = weights + (uint64_t)token * k_used;

    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float p = glm_router_sigmoid_dev(log[e]);
            local_prob[j] = p;
            local_score[j] = p + bias[e];
            prob[e] = p;
        } else {
            local_prob[j] = 0.0f;
            local_score[j] = -INFINITY;
        }
    }
    __syncwarp();

    float sum = 0.0f;
    for (uint32_t k = 0; k < k_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (glm_router_better_dev(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(FULL_WARP_MASK, best_score, mask);
            const float other_prob = __shfl_xor_sync(FULL_WARP_MASK, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(FULL_WARP_MASK, best_idx, mask);
            if (glm_router_better_dev(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            sel[k] = (int32_t)best_idx;
            w[k] = best_prob;
        }
        sum += best_prob;
    }

    if (lane == 0) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++) {
            w[k] = w[k] / sum * expert_weight_scale;
        }
    }
}

static int glm_router_select_launch(
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs,
        const void *model_map,
        uint64_t model_size,
        uint64_t bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }
    if (logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM router received undersized buffers\n");
        return 0;
    }
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM router bias range is outside the mapped model\n");
        return 0;
    }
    const float *bias = (const float *)cuda_model_range_ptr(model_map,
            bias_offset, bias_bytes, "glm_router_bias");
    if (!bias) return 0;

    dim3 block(32, 4, 1);
    glm_router_select_warp_kernel<<<(n_tokens + 3u) / 4u, block>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert,
            n_expert_used,
            expert_weight_scale,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "glm router select launch");
}

/* =========================================================================
 * GLM routed MoE kernels (template, per-element dequant).
 * ========================================================================= */

template <typename DEQ>
__global__ static void glm_moe_pair_swiglu_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint32_t in_dim,
        uint32_t mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t slot = blockIdx.y;
    const uint32_t token = blockIdx.z;
    if (row >= mid_dim || slot >= n_expert_used || token >= n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * n_expert_used + slot;
    const uint64_t mid_off = (uint64_t)token * mid_token_stride +
                             (uint64_t)slot * mid_dim + row;
    const int expert = selected[selected_off];
    if (expert < 0 || (uint32_t)expert >= n_total_expert) {
        if (lane == 0) mid[mid_off] = 0.0f;
        return;
    }

    const char *gate_row = gate +
        (uint64_t)(uint32_t)expert * gate_expert_bytes +
        (uint64_t)row * gate_row_bytes;
    const char *up_row = up +
        (uint64_t)(uint32_t)expert * up_expert_bytes +
        (uint64_t)row * up_row_bytes;
    const float *token_x = x + (uint64_t)token * in_dim;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint32_t k = lane; k < in_dim; k += 32u) {
        const float xv = token_x[k];
        acc_gate += DEQ::value(gate_row, k) * xv;
        acc_up += DEQ::value(up_row, k) * xv;
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc_gate += __shfl_xor_sync(FULL_WARP_MASK, acc_gate, mask);
        acc_up += __shfl_xor_sync(FULL_WARP_MASK, acc_up, mask);
    }
    if (lane == 0) {
        const float g = acc_gate;
        const float sw = g / (1.0f + expf(-g));
        mid[mid_off] = sw * acc_up * weights[selected_off];
    }
}

template <typename DEQ>
__global__ static void glm_moe_down_kernel(
        float *out,
        const char *down,
        const int32_t *selected,
        const float *mid,
        uint32_t mid_dim,
        uint32_t out_dim,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= n_tokens) return;

    const uint64_t selected_base = (uint64_t)token * n_expert_used;
    const uint64_t mid_base = (uint64_t)token * mid_token_stride;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) continue;
        const char *down_row = down +
            (uint64_t)(uint32_t)expert * down_expert_bytes +
            (uint64_t)row * down_row_bytes;
        const float *slot_mid = mid + mid_base + (uint64_t)slot * mid_dim;
        for (uint32_t k = lane; k < mid_dim; k += 32u) {
            acc += DEQ::value(down_row, k) * slot_mid[k];
        }
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc += __shfl_xor_sync(FULL_WARP_MASK, acc, mask);
    }
    if (lane == 0) {
        out[(uint64_t)token * out_dim + row] = acc;
    }
}

static bool glm_gate_pair_type_supported(uint32_t gate_type, uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == GLM_CUDA_TENSOR_Q2_K ||
            gate_type == GLM_CUDA_TENSOR_Q4_K ||
            gate_type == GLM_CUDA_TENSOR_Q5_K ||
            gate_type == GLM_CUDA_TENSOR_IQ2_XXS);
}

static bool glm_down_type_supported(uint32_t down_type) {
    return down_type == GLM_CUDA_TENSOR_IQ2_XXS ||
           down_type == GLM_CUDA_TENSOR_Q2_K ||
           down_type == GLM_CUDA_TENSOR_Q4_K ||
           down_type == GLM_CUDA_TENSOR_Q5_K ||
           down_type == GLM_CUDA_TENSOR_Q6_K;
}

/* =========================================================================
 * GLM routed MoE launcher (decode + batch) — port of glm_routed_moe_launch
 * Uses ROCm streaming cache infrastructure via compat.cuh globals.
 * ========================================================================= */

static int glm_routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        bool force_resident) {
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_tokens > 65535u ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % CUDA_QK_K) != 0 ||
        (expert_mid_dim % CUDA_QK_K) != 0 ||
        !glm_gate_pair_type_supported(gate_type, up_type) ||
        !glm_down_type_supported(down_type)) {
        return 0;
    }

    const uint64_t per_token_mid = (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM routed MoE received inconsistent expert strides\n");
        return 0;
    }
    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM routed MoE tensor range is outside the mapped model\n");
        return 0;
    }
    const uint64_t mid_values =
        (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
    const uint64_t slot_values = (uint64_t)n_tokens * n_expert;
    if (x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        mid->bytes < mid_values * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        selected->bytes < slot_values * sizeof(int32_t) ||
        weights->bytes < slot_values * sizeof(float)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM routed MoE received undersized activation buffers\n");
        return 0;
    }

    /* Weight source resolution (ROCm streaming cache via compat.cuh globals) */
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const int streaming = g_ssd_streaming_mode && !force_resident;

    /* The shared streaming caches store one gate/up stride; GLM gate and up
     * always share type and dims so the strides match by construction. */
    const int uniform_gate_up = gate_expert_bytes == up_expert_bytes;

    int use_selected_cache = 0;
    for (int attempt = 0; attempt < 2 && !use_selected_cache; attempt++) {
        use_selected_cache =
            streaming && uniform_gate_up &&
            g_stream_selected_cache.loaded &&
            g_stream_selected_cache.model_map == model_map &&
            g_stream_selected_cache.layer == layer_index &&
            g_stream_selected_cache.n_total_expert == n_total_expert &&
            g_stream_selected_cache.n_selected == n_expert &&
            g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
            g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
            g_stream_selected_cache.gate &&
            g_stream_selected_cache.up &&
            g_stream_selected_cache.down &&
            g_stream_selected_cache.slot_tensor.ptr &&
            g_stream_selected_cache.slot_tensor.bytes >=
                slot_values * sizeof(int32_t);
        if (use_selected_cache || attempt == 1 || !streaming || !uniform_gate_up) break;
        /* Safety net: decode reached the MoE without an early selected load.
         * Load the selected experts now. */
        const char *lg = NULL, *lu = NULL, *ld = NULL;
        if (n_tokens != 1u || n_expert > 8u ||
            cuda_stream_layer_expert_cache_apply(model_map, layer_index,
                                                 n_total_expert,
                                                 gate_offset, up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes,
                                                 &lg, &lu, &ld)) {
            break;
        }
        int32_t ids[8];
        if (!ds4_gpu_tensor_read(selected, 0, ids,
                                 (uint64_t)n_expert * sizeof(ids[0]))) {
            return 0;
        }
        const ds4_gpu_stream_expert_table table = {
            model_map, model_size, layer_index, n_total_expert,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
        };
        if (!ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, n_expert)) {
            return 0;
        }
    }

    /* GLM batch selected cache: used during prefill (n_tokens > 1) when the
     * single-token selected cache doesn't apply. Populated by
     * glm_batch_selected_prepare before the MoE dispatch. */
    int use_batch_cache = 0;
    if (!use_selected_cache && streaming && uniform_gate_up &&
        n_tokens > 1u &&
        g_glm_batch_selected_cache.loaded &&
        g_glm_batch_selected_cache.model_map == model_map &&
        g_glm_batch_selected_cache.layer == layer_index &&
        g_glm_batch_selected_cache.n_total_expert == n_total_expert &&
        g_glm_batch_selected_cache.n_selected == n_expert &&
        g_glm_batch_selected_cache.n_tokens == n_tokens &&
        g_glm_batch_selected_cache.gate_expert_bytes == gate_expert_bytes &&
        g_glm_batch_selected_cache.down_expert_bytes == down_expert_bytes &&
        g_glm_batch_selected_cache.gate &&
        g_glm_batch_selected_cache.up &&
        g_glm_batch_selected_cache.down &&
        g_glm_batch_selected_cache.slot_tensor.ptr &&
        g_glm_batch_selected_cache.slot_tensor.bytes >=
            slot_values * sizeof(int32_t)) {
        use_batch_cache = 1;
    }

    if (use_selected_cache) {
        g_glm_moe_selected_cache_hits++;
        selected_exec = &g_stream_selected_cache.slot_tensor;
        gate_w = g_stream_selected_cache.gate;
        up_w = g_stream_selected_cache.up;
        down_w = g_stream_selected_cache.down;
    } else if (use_batch_cache) {
        g_glm_moe_batch_cache_hits++;
        selected_exec = &g_glm_batch_selected_cache.slot_tensor;
        gate_w = g_glm_batch_selected_cache.gate;
        up_w = g_glm_batch_selected_cache.up;
        down_w = g_glm_batch_selected_cache.down;
    } else if (streaming && uniform_gate_up &&
               cuda_stream_layer_expert_cache_apply(model_map, layer_index,
                                                       n_total_expert,
                                                       gate_offset, up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       &gate_w, &up_w, &down_w)) {
        g_glm_moe_layer_cache_hits++;
    } else if (streaming) {
        /* Layer cache MISS — load synchronously on-demand */
        /* cache MISS — load on-demand */
        if (cuda_stream_layer_expert_cache_load(model_map, model_size,
                                                layer_index, n_total_expert,
                                                gate_offset, up_offset,
                                                down_offset,
                                                gate_expert_bytes,
                                                down_expert_bytes)) {
            /* Retry apply after load */
            cuda_stream_layer_expert_cache_apply(model_map, layer_index,
                                                 n_total_expert,
                                                 gate_offset, up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes,
                                                 &gate_w, &up_w, &down_w);
        }
        if (!gate_w || !up_w || !down_w) {
            g_glm_moe_fallback_range_ptr++;
            gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_tensor_bytes,
                                           "glm_moe_gate");
            up_w = cuda_model_range_ptr(model_map, up_offset, up_tensor_bytes,
                                         "glm_moe_up");
            down_w = cuda_model_range_ptr(model_map, down_offset, down_tensor_bytes,
                                           "glm_moe_down");
        }
    } else {
        g_glm_moe_fallback_mmap++;
        gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_tensor_bytes,
                                       "glm_moe_gate");
        up_w = cuda_model_range_ptr(model_map, up_offset, up_tensor_bytes,
                                     "glm_moe_up");
        down_w = cuda_model_range_ptr(model_map, down_offset, down_tensor_bytes,
                                       "glm_moe_down");
    }
    if (!gate_w || !up_w || !down_w) {
        /* ptrs FAILED */
        return 0;
    }

    const uint32_t rows_per_block = 4u;
    dim3 block(32, rows_per_block, 1);
    dim3 pair_grid((expert_mid_dim + rows_per_block - 1u) / rows_per_block,
                   n_expert,
                   n_tokens);
    dim3 down_grid((out_dim + rows_per_block - 1u) / rows_per_block,
                   n_tokens,
                   1);
    const int32_t *sel_ptr = (const int32_t *)selected_exec->ptr;
    const float *w_ptr = (const float *)weights->ptr;
    const float *x_ptr = (const float *)x->ptr;
    float *mid_ptr = (float *)mid->ptr;
    float *out_ptr = (float *)out->ptr;

    /* debug: x and gate/up values omitted */

    switch (gate_type) {
    case GLM_CUDA_TENSOR_IQ2_XXS:
        glm_moe_pair_swiglu_kernel<glm_deq_iq2_xxs><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q2_K:
        glm_moe_pair_swiglu_kernel<glm_deq_q2_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q4_K:
        glm_moe_pair_swiglu_kernel<glm_deq_q4_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    default: /* Q5_K, validated above */
        glm_moe_pair_swiglu_kernel<glm_deq_q5_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    }
    if (!cuda_ok(cudaGetLastError(), "glm routed moe pair launch")) return 0;

    switch (down_type) {
    case GLM_CUDA_TENSOR_IQ2_XXS:
        glm_moe_down_kernel<glm_deq_iq2_xxs><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q2_K:
        glm_moe_down_kernel<glm_deq_q2_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q4_K:
        glm_moe_down_kernel<glm_deq_q4_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q5_K:
        glm_moe_down_kernel<glm_deq_q5_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    default: /* Q6_K, validated above */
        glm_moe_down_kernel<glm_deq_q6_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    }
    return cuda_ok(cudaGetLastError(), "glm routed moe down launch");
}

/* Seed the GLM batch selected cache for prefill (n_tokens > 1).
 * Reads all selected IDs from the GPU tensor, deduplicates, copies unique
 * expert weights to slot-ordered GPU buffers, and creates a re-indexed slot
 * tensor so the GLM MoE kernel uses slot-based indexing.
 */
static int glm_batch_selected_prepare(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode || !model_map || !selected ||
        n_tokens <= 1 || n_tokens > 65535u ||
        n_total_expert == 0 || n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 || n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }

    /* Read all selected IDs from GPU to host */
    const uint64_t n_ids = (uint64_t)n_tokens * n_selected;
    int32_t *ids = (int32_t *)malloc(n_ids * sizeof(int32_t));
    if (!ids) return 0;
    if (!ds4_gpu_tensor_read(selected, 0, ids, n_ids * sizeof(int32_t))) {
        free(ids);
        return 0;
    }

    /* Deduplicate: collect unique expert IDs */
    int32_t unique[DS4_ROCM_N_EXPERT_USED * 64]; /* max 512 unique experts */
    uint32_t n_unique = 0;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e < 0 || (uint32_t)e >= n_total_expert) continue;
        uint32_t j;
        for (j = 0; j < n_unique; j++) {
            if (unique[j] == e) break;
        }
        if (j == n_unique && n_unique < DS4_ROCM_N_EXPERT_USED * 64u) {
            unique[n_unique++] = e;
        }
    }
    if (n_unique == 0) {
        free(ids);
        return 0;
    }

    /* Allocate or reuse slot-ordered GPU buffers for gate, up, down */
    const uint64_t gate_bytes = (uint64_t)n_unique * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_unique * down_expert_bytes;
    if (g_glm_batch_selected_cache.gate_capacity < gate_bytes) {
        if (g_glm_batch_selected_cache.gate) (void)cudaFree(g_glm_batch_selected_cache.gate);
        if (g_glm_batch_selected_cache.up) (void)cudaFree(g_glm_batch_selected_cache.up);
        g_glm_batch_selected_cache.gate = NULL;
        g_glm_batch_selected_cache.up = NULL;
        g_glm_batch_selected_cache.gate_capacity = 0;
        cudaError_t err = cudaMalloc((void **)&g_glm_batch_selected_cache.gate, (size_t)gate_bytes);
        if (err != cudaSuccess) {
            free(ids);
            return 0;
        }
        err = cudaMalloc((void **)&g_glm_batch_selected_cache.up, (size_t)gate_bytes);
        if (err != cudaSuccess) {
            (void)cudaFree(g_glm_batch_selected_cache.gate);
            g_glm_batch_selected_cache.gate = NULL;
            free(ids);
            return 0;
        }
        g_glm_batch_selected_cache.gate_capacity = gate_bytes;
    }
    if (g_glm_batch_selected_cache.down_capacity < down_bytes) {
        if (g_glm_batch_selected_cache.down) (void)cudaFree(g_glm_batch_selected_cache.down);
        g_glm_batch_selected_cache.down = NULL;
        g_glm_batch_selected_cache.down_capacity = 0;
        cudaError_t err = cudaMalloc((void **)&g_glm_batch_selected_cache.down, (size_t)down_bytes);
        if (err != cudaSuccess) {
            free(ids);
            return 0;
        }
        g_glm_batch_selected_cache.down_capacity = down_bytes;
    }

    /* Ensure the upload stream is initialized */
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream,
                                                    cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            free(ids);
            return 0;
        }
    }

    /* Copy unique expert weights from host to slot-ordered GPU buffers.
     * Gate and up share the same expert_bytes stride (verified by caller).
     * Use the model file descriptor when available to avoid slow mmap page
     * faults: read into a pinned staging buffer, then async copy to GPU. */
    int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);
    void *stage_buf = NULL;
    size_t stage_size = 0;
    if (use_fd) {
        stage_size = gate_expert_bytes > down_expert_bytes
            ? (size_t)gate_expert_bytes : (size_t)down_expert_bytes;
        cudaError_t err = cudaMallocHost(&stage_buf, stage_size);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            /* fall through to mmap path */
            use_fd = 0;
        }
    }
    for (uint32_t i = 0; i < n_unique; i++) {
        const uint64_t expert = (uint64_t)(uint32_t)unique[i];
        const uint64_t src_gate_off = gate_offset + expert * gate_expert_bytes;
        const uint64_t src_up_off   = up_offset   + expert * gate_expert_bytes;
        const uint64_t src_down_off = down_offset + expert * down_expert_bytes;
        const uint64_t dst_off = (uint64_t)i * gate_expert_bytes;
        cudaError_t err;
        if (use_fd) {
            /* Read gate weights from file into staging buffer */
            if (!cuda_pread_full(g_model_fd, stage_buf, gate_expert_bytes,
                                 src_gate_off)) {
                free(ids);
                if (stage_buf) (void)cudaFreeHost(stage_buf);
                return 0;
            }
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.gate + dst_off,
                                  stage_buf,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); if (stage_buf) (void)cudaFreeHost(stage_buf); return 0; }
            /* Read up weights from file into staging buffer */
            if (!cuda_pread_full(g_model_fd, stage_buf, gate_expert_bytes,
                                 src_up_off)) {
                free(ids);
                if (stage_buf) (void)cudaFreeHost(stage_buf);
                return 0;
            }
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.up + dst_off,
                                  stage_buf,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); if (stage_buf) (void)cudaFreeHost(stage_buf); return 0; }
            /* Read down weights from file into staging buffer */
            const uint64_t dst_down_off = (uint64_t)i * down_expert_bytes;
            if (!cuda_pread_full(g_model_fd, stage_buf, down_expert_bytes,
                                 src_down_off)) {
                free(ids);
                if (stage_buf) (void)cudaFreeHost(stage_buf);
                return 0;
            }
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.down + dst_down_off,
                                  stage_buf,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); if (stage_buf) (void)cudaFreeHost(stage_buf); return 0; }
        } else {
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.gate + dst_off,
                                  (const char *)model_map + src_gate_off,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); return 0; }
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.up + dst_off,
                                  (const char *)model_map + src_up_off,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); return 0; }
            const uint64_t dst_down_off = (uint64_t)i * down_expert_bytes;
            err = cudaMemcpyAsync(g_glm_batch_selected_cache.down + dst_down_off,
                                  (const char *)model_map + src_down_off,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess) { free(ids); return 0; }
        }
    }
    if (stage_buf) (void)cudaFreeHost(stage_buf);

    /* Wait for copies to complete */
    cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) { free(ids); return 0; }

    /* Create re-indexed slot tensor on GPU:
     * for each token+slot, store the slot index of the corresponding expert. */
    const uint64_t slot_entries = n_ids;
    const uint64_t slot_bytes = slot_entries * sizeof(int32_t);
    if (g_glm_batch_selected_cache.slot_capacity < slot_bytes) {
        if (g_glm_batch_selected_cache.slot_ids)
            (void)cudaFree(g_glm_batch_selected_cache.slot_ids);
        g_glm_batch_selected_cache.slot_ids = NULL;
        g_glm_batch_selected_cache.slot_capacity = 0;
        err = cudaMalloc((void **)&g_glm_batch_selected_cache.slot_ids,
                         (size_t)slot_bytes);
        if (err != cudaSuccess) { free(ids); return 0; }
        g_glm_batch_selected_cache.slot_capacity = slot_bytes;
    }

    int32_t *slot_host = (int32_t *)malloc(slot_bytes);
    if (!slot_host) { free(ids); return 0; }
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        uint32_t s;
        for (s = 0; s < n_unique; s++) {
            if (unique[s] == e) break;
        }
        slot_host[i] = s < n_unique ? (int32_t)s : -1;
    }
    err = cudaMemcpy(g_glm_batch_selected_cache.slot_ids,
                     slot_host,
                     (size_t)slot_bytes,
                     cudaMemcpyHostToDevice);
    free(slot_host);
    free(ids);
    if (err != cudaSuccess) return 0;

    g_glm_batch_selected_cache.slot_tensor.ptr = g_glm_batch_selected_cache.slot_ids;
    g_glm_batch_selected_cache.slot_tensor.bytes = slot_bytes;
    g_glm_batch_selected_cache.slot_tensor.owner = 0;

    /* Populate cache header */
    g_glm_batch_selected_cache.loaded = 1;
    g_glm_batch_selected_cache.model_map = model_map;
    g_glm_batch_selected_cache.layer = layer;
    g_glm_batch_selected_cache.n_total_expert = n_total_expert;
    g_glm_batch_selected_cache.n_selected = n_selected;
    g_glm_batch_selected_cache.n_tokens = n_tokens;
    g_glm_batch_selected_cache.n_unique = n_unique;
    g_glm_batch_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_glm_batch_selected_cache.down_expert_bytes = down_expert_bytes;
    return 1;
}

/* Streaming layer cache lookup — delegates to runtime's existing cache. */

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected) {
    if (!g_ssd_streaming_mode ||
        getenv("DS4_CUDA_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL ||
        getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        /* early_load SKIP */
        return 1;
    }
    if (!table || !selected || n_selected == 0 || n_selected > 8u) {
        /* early_load INVALID_PARAMS */
        return 1;
    }
    const char *g = NULL, *u = NULL, *d = NULL;
    if (cuda_stream_layer_expert_cache_apply(table->model_map,
                                             table->layer,
                                             table->n_total_expert,
                                             table->gate_offset,
                                             table->up_offset,
                                             table->down_offset,
                                             table->gate_expert_bytes,
                                             table->down_expert_bytes,
                                             &g, &u, &d)) {
        /* early_load cache HIT */
        return 1;
    }
    /* early_load cache MISS, reading selected_ids */

    int32_t selected_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (!ds4_gpu_tensor_read(selected, 0, selected_ids,
                             (uint64_t)n_selected * sizeof(selected_ids[0]))) {
        /* early_load tensor_read FAILED */
        return 0;
    }
    /* selected_ids debug omitted */
    /* Use seed_selected (load + finish_pending) instead of begin_selected_load
     * so the read pool is idle before the next layer starts. */
    int ret = ds4_gpu_stream_expert_cache_seed_selected(table,
                                                        selected_ids,
                                                        n_selected);
    /* early_load seed_selected result logged */
    return ret;
}

extern "C" int ds4_gpu_glm_stream_expert_cache_seed_batch_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_selected) {
    if (!g_ssd_streaming_mode ||
        getenv("DS4_CUDA_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL ||
        getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        /* batch_seed SKIP */
        return 1;
    }
    if (!table || !selected || n_tokens <= 1 || n_selected == 0 || n_selected > 8u) {
        /* batch_seed INVALID_PARAMS */
        return 1;
    }
    /* Check if the batch cache already matches (reuse from previous layer).
     * This is the fast path: if the same layer+tokens were already seeded,
     * skip the expensive prepare step. */
    if (g_glm_batch_selected_cache.loaded &&
        g_glm_batch_selected_cache.model_map == table->model_map &&
        g_glm_batch_selected_cache.layer == table->layer &&
        g_glm_batch_selected_cache.n_tokens == n_tokens &&
        g_glm_batch_selected_cache.n_selected == n_selected &&
        g_glm_batch_selected_cache.gate_expert_bytes == table->gate_expert_bytes &&
        g_glm_batch_selected_cache.down_expert_bytes == table->down_expert_bytes &&
        g_glm_batch_selected_cache.gate &&
        g_glm_batch_selected_cache.slot_tensor.ptr) {
        /* batch_seed CACHE_HIT */
        return 1;
    }
    /* Check if the layer cache is already loaded — if so, we can skip the
     * batch cache entirely because the MoE dispatch will use the layer cache. */
    const char *lg = NULL, *lu = NULL, *ld = NULL;
    if (cuda_stream_layer_expert_cache_apply(table->model_map,
                                             table->layer,
                                             table->n_total_expert,
                                             table->gate_offset,
                                             table->up_offset,
                                             table->down_offset,
                                             table->gate_expert_bytes,
                                             table->down_expert_bytes,
                                             &lg, &lu, &ld)) {
        /* batch_seed LAYER_CACHE_HIT */
        return 1;
    }
    /* Seed the batch cache with the selected experts for all tokens.
     * This copies only the needed experts to GPU-local memory. */
    if (!glm_batch_selected_prepare(table->model_map,
                                    table->model_size,
                                    table->layer,
                                    selected,
                                    n_tokens,
                                    table->n_total_expert,
                                    n_selected,
                                    table->gate_offset,
                                    table->up_offset,
                                    table->down_offset,
                                    table->gate_expert_bytes,
                                    table->down_expert_bytes)) {
        /* batch_seed PREPARE_FAILED */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "glm batch seed FAILED il=%u\n", table->layer);
        return 0;
    }
    /* batch_seed OK */
    return 1;
}

/* =========================================================================
 * extern "C" backend entry points
 * ========================================================================= */

extern "C" int ds4_gpu_add3_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t n) {
    if (!out || !a || !b || !c || n == 0 ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        c->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    glm_add3_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out->ptr, (const float *)a->ptr,
            (const float *)b->ptr, (const float *)c->ptr, n);
    return cuda_ok(cudaGetLastError(), "add3 launch");
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor *norm_out,
        ds4_gpu_tensor *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n,
        float eps) {
    if (!norm_out || !sum_out || !a || !b || !model_map || n == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    if (norm_out->bytes < row_bytes || sum_out->bytes < row_bytes ||
        a->bytes < row_bytes || b->bytes < row_bytes ||
        weight_offset > model_size || row_bytes > model_size - weight_offset) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map,
            weight_offset, row_bytes, "add_rms_weight");
    if (!w) return 0;
    glm_add_rms_norm_weight_kernel<<<1, 256>>>(
            (float *)norm_out->ptr,
            (float *)sum_out->ptr,
            (const float *)a->ptr,
            (const float *)b->ptr,
            w, n, eps);
    return cuda_ok(cudaGetLastError(), "add rms_norm_weight launch");
}

extern "C" int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t row_width,
        uint32_t n_rows) {
    if (!dst || !src || row_width == 0 || n_rows == 0 ||
        (row_width & (row_width - 1u)) != 0) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)row_width * n_rows * sizeof(int32_t);
    if (src->bytes < bytes || dst->bytes < bytes) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "row sort received undersized buffers\n");
        return 0;
    }
    const uint32_t shared_bytes = row_width * (uint32_t)sizeof(int32_t);
    if (shared_bytes > 48u * 1024u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "row sort scratch exceeds shared memory limit\n");
        return 0;
    }
    uint32_t threads = row_width < 256u ? row_width : 256u;
    glm_sort_i32_rows_asc_kernel<<<n_rows, threads, shared_bytes>>>(
            (int32_t *)dst->ptr, (const int32_t *)src->ptr, row_width, n_rows);
    return cuda_ok(cudaGetLastError(), "sort i32 rows asc launch");
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        bool force_resident) {
    if (n_expert == 0 || (uint64_t)n_expert * expert_mid_dim > UINT32_MAX) return 0;
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 1u,
                                 n_expert * expert_mid_dim,
                                 force_resident);
}

extern "C" int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 n_tokens, mid_token_stride, false);
}

extern "C" int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 n_tokens, mid_token_stride, false);
}
