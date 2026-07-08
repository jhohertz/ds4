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

/* IQ2_XXS tables — ported from ds4_iq2_tables_cuda.inc */
#define DS4_ROCKM_IQ2_KSIGN_SIZE 128u
#define DS4_ROCKM_IQ2_GRID_SIZE 256u

__device__ __constant__ uint8_t glm_iq2_ksigns[DS4_ROCKM_IQ2_KSIGN_SIZE] = {
    0xFF, 0xD7, 0xE3, 0xA2, 0xED, 0x95, 0xB5, 0x44, 0x7E, 0x57, 0x67, 0x22, 0x76, 0x34, 0x54, 0x10,
    0xFB, 0xD2, 0xE2, 0xA1, 0xEA, 0x93, 0xB3, 0x42, 0x7D, 0x55, 0x65, 0x21, 0x74, 0x32, 0x52, 0x11,
    0xFA, 0xD1, 0xE1, 0xA0, 0xE9, 0x92, 0xB2, 0x41, 0x7C, 0x54, 0x64, 0x20, 0x73, 0x31, 0x51, 0x10,
    0xF9, 0xD0, 0xE0, 0xA0, 0xE8, 0x91, 0xB1, 0x40, 0x7B, 0x53, 0x63, 0x20, 0x72, 0x30, 0x50, 0x10,
    0xEF, 0xC7, 0xD3, 0x92, 0xDD, 0x85, 0xA5, 0x34, 0x6E, 0x47, 0x57, 0x12, 0x66, 0x24, 0x44, 0x00,
    0xEE, 0xC6, 0xD2, 0x91, 0xDC, 0x84, 0xA4, 0x33, 0x6D, 0x46, 0x56, 0x11, 0x65, 0x23, 0x43, 0x00,
    0xED, 0xC5, 0xD1, 0x90, 0xDB, 0x83, 0xA3, 0x32, 0x6C, 0x45, 0x55, 0x10, 0x64, 0x22, 0x42, 0x00,
    0xEC, 0xC4, 0xD0, 0x90, 0xDA, 0x82, 0xA2, 0x31, 0x6B, 0x44, 0x54, 0x10, 0x63, 0x21, 0x41, 0x00,
};

__device__ __constant__ uint64_t glm_iq2xxs_grid[DS4_ROCKM_IQ2_GRID_SIZE] = {
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
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

    if (use_selected_cache) {
        selected_exec = &g_stream_selected_cache.slot_tensor;
        gate_w = g_stream_selected_cache.gate;
        up_w = g_stream_selected_cache.up;
        down_w = g_stream_selected_cache.down;
    } else if (streaming && uniform_gate_up &&
               cuda_stream_layer_expert_cache_apply(model_map, layer_index,
                                                    n_total_expert,
                                                    gate_offset, up_offset,
                                                    down_offset,
                                                    gate_expert_bytes,
                                                    down_expert_bytes,
                                                    &gate_w, &up_w, &down_w)) {
        /* full-layer cache hit */
    } else {
        gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_tensor_bytes,
                                      "glm_moe_gate");
        up_w = cuda_model_range_ptr(model_map, up_offset, up_tensor_bytes,
                                    "glm_moe_up");
        down_w = cuda_model_range_ptr(model_map, down_offset, down_tensor_bytes,
                                      "glm_moe_down");
    }
    if (!gate_w || !up_w || !down_w) return 0;

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

/* Streaming layer cache lookup — delegates to runtime's existing cache. */

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected) {
    if (!g_ssd_streaming_mode ||
        getenv("DS4_CUDA_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL ||
        getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        return 1;
    }
    if (!table || !selected || n_selected == 0 || n_selected > 8u) {
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
        return 1;
    }

    int32_t selected_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (!ds4_gpu_tensor_read(selected, 0, selected_ids,
                             (uint64_t)n_selected * sizeof(selected_ids[0]))) {
        return 0;
    }
    return ds4_gpu_stream_expert_cache_begin_selected_load(table,
                                                           selected_ids,
                                                           n_selected);
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
