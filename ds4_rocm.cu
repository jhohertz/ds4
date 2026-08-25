#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} cuda_block_mxfp4;

static_assert(sizeof(cuda_block_mxfp4) == 17, "cuda_block_mxfp4 must match the GGUF MXFP4 block layout");

/* Twice the MXFP4 values so each 32-value sub-block can use signed-int8
 * dp4a; the factor of 1/2 is folded into the sub-block scale. */
__device__ __constant__ static const int8_t cuda_mxfp4_values_x2[16] = {
     0,  1,  2,  3,  4,  6,  8,  12,
     0, -1, -2, -3, -4, -6, -8, -12,
};

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

#include "rocm/ds4_rocm_tp.cuh"

/* Stage-1 test entries for the TP combine kernels (tests/test_tp_combine_rocm).
 * Loopback only: a writer kernel or the host plays the peer; the NHI pool
 * import path arrives with the transport backend stage. */
extern "C" int ds4_gpu_tp_test_combine(uint32_t n, uint32_t iterations) {
    if (n == 0 || n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX) return 0;
    const size_t bytes = (size_t)n * sizeof(float);
    float *h_acc = (float *)malloc(bytes);
    float *h_remote = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *d_acc = NULL, *d_remote = NULL;
    int pass = 1;
    if (!h_acc || !h_remote || !h_ref || !h_out ||
        hipMalloc(&d_acc, bytes) != hipSuccess ||
        hipMalloc(&d_remote, bytes) != hipSuccess) {
        pass = 0;
        goto done;
    }
    for (uint32_t it = 0; pass && it < iterations; it++) {
        for (uint32_t i = 0; i < n; i++) {
            h_acc[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
            h_remote[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
            h_ref[i] = h_acc[i] + h_remote[i];
        }
        if (hipMemcpy(d_acc, h_acc, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(d_remote, h_remote, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            pass = 0;
            break;
        }
        const uint32_t block = 256;
        const uint32_t grid = (n + block - 1) / block;
        hipLaunchKernelGGL(dsv4_tp_combine_f32_kernel, dim3(grid), dim3(block),
                           0, 0, d_acc, d_remote, n);
        if (hipDeviceSynchronize() != hipSuccess ||
            hipMemcpy(h_out, d_acc, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
            memcmp(h_out, h_ref, bytes) != 0) {
            pass = 0;
            break;
        }
    }
done:
    free(h_acc); free(h_remote); free(h_ref); free(h_out);
    if (d_acc) (void)hipFree(d_acc);
    if (d_remote) (void)hipFree(d_remote);
    return pass;
}

extern "C" int ds4_gpu_tp_test_spin_exchange(int uncached_pool,
                                             int host_stamp,
                                             uint32_t n,
                                             uint32_t seqs) {
    if (n == 0 || n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX || seqs == 0) return 0;
    const size_t payload_bytes = (size_t)n * sizeof(float);
    const size_t pool_bytes = (size_t)seqs * DS4_ROCM_TP_SLOT_BYTES;
    unsigned char *d_pool = NULL;
    float *d_acc = NULL, *d_src = NULL;
    uint32_t *h_stamp = NULL, *d_stamp_view = NULL;
    int *d_timeout = NULL;
    float *h_acc = (float *)malloc(payload_bytes);
    float *h_src = (float *)malloc(payload_bytes);
    float *h_ref = (float *)malloc(payload_bytes);
    float *h_out = (float *)malloc(payload_bytes);
    hipStream_t spin_stream = NULL, fill_stream = NULL;
    int pass = 1;
    int h_timeout = 0;

    if (!h_acc || !h_src || !h_ref || !h_out) { pass = 0; goto done; }
    if (uncached_pool) {
        if (hipExtMallocWithFlags((void **)&d_pool, pool_bytes,
                                  hipDeviceMallocUncached) != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP test: uncached pool allocation unsupported\n");
            pass = 0;
            goto done;
        }
    } else if (hipMalloc(&d_pool, pool_bytes) != hipSuccess) {
        pass = 0;
        goto done;
    }
    if (hipMalloc(&d_acc, payload_bytes) != hipSuccess ||
        hipMalloc(&d_src, payload_bytes) != hipSuccess ||
        hipMalloc(&d_timeout, sizeof(int)) != hipSuccess ||
        hipMemset(d_pool, 0, pool_bytes) != hipSuccess ||
        hipStreamCreate(&spin_stream) != hipSuccess ||
        hipStreamCreate(&fill_stream) != hipSuccess) {
        pass = 0;
        goto done;
    }
    if (host_stamp) {
        if (hipHostMalloc((void **)&h_stamp, sizeof(uint32_t), 0) != hipSuccess ||
            hipHostGetDevicePointer((void **)&d_stamp_view, h_stamp, 0) != hipSuccess) {
            pass = 0;
            goto done;
        }
        *h_stamp = 0;
    }

    for (uint32_t i = 0; i < n; i++) {
        h_acc[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        h_ref[i] = h_acc[i];
    }
    if (hipMemcpy(d_acc, h_acc, payload_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        pass = 0;
        goto done;
    }

    for (uint32_t seq = 1; pass && seq <= seqs; seq++) {
        unsigned char *slot = d_pool + (size_t)(seq - 1) * DS4_ROCM_TP_SLOT_BYTES;
        for (uint32_t i = 0; i < n; i++) {
            h_src[i] = ((float)(rand() % 1000) - 500.0f) * 0.03125f + (float)seq;
            h_ref[i] += h_src[i];
        }
        if (hipMemcpy(d_src, h_src, payload_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemset(d_timeout, 0, sizeof(int)) != hipSuccess) {
            pass = 0;
            break;
        }
        /* The spin kernel launches FIRST and must wait; the payload arrives
         * behind it, proving the stamp actually gates the combine. */
        hipLaunchKernelGGL(dsv4_tp_spin_combine_f32_kernel, dim3(1), dim3(256),
                           0, spin_stream, d_acc, slot, n, seq,
                           (unsigned long long)400000000ull,
                           host_stamp ? d_stamp_view : NULL, d_timeout);
        if (host_stamp) {
            if (hipMemcpyAsync(slot, h_src, payload_bytes, hipMemcpyHostToDevice,
                               fill_stream) != hipSuccess ||
                hipStreamSynchronize(fill_stream) != hipSuccess) {
                pass = 0;
            } else {
                __atomic_store_n(h_stamp, seq, __ATOMIC_RELEASE);
            }
        } else {
            hipLaunchKernelGGL(dsv4_tp_slot_fill_kernel, dim3(1), dim3(256),
                               0, fill_stream, slot, d_src, n, seq);
        }
        if (!pass) break;
        if (hipStreamSynchronize(spin_stream) != hipSuccess ||
            hipStreamSynchronize(fill_stream) != hipSuccess) {
            pass = 0;
            break;
        }
        if (hipMemcpy(&h_timeout, d_timeout, sizeof(int),
                      hipMemcpyDeviceToHost) != hipSuccess || h_timeout != 0) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP test: spin timeout at seq %u\n", seq);
            pass = 0;
            break;
        }
    }
    if (pass &&
        (hipMemcpy(h_out, d_acc, payload_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
         memcmp(h_out, h_ref, payload_bytes) != 0)) {
        pass = 0;
    }

done:
    free(h_acc); free(h_src); free(h_ref); free(h_out);
    if (h_stamp) (void)hipHostFree(h_stamp);
    if (d_pool) (void)hipFree(d_pool);
    if (d_acc) (void)hipFree(d_acc);
    if (d_src) (void)hipFree(d_src);
    if (d_timeout) (void)hipFree(d_timeout);
    if (spin_stream) (void)hipStreamDestroy(spin_stream);
    if (fill_stream) (void)hipStreamDestroy(fill_stream);
    return pass;
}

/* Stage-2 GPU helpers for the NHI TP transport (ds4_tp_nhi.c and the
 * two-node live test): uncached pool allocation for wave-resident polling
 * (contract rules 1 and 6), and stream-based fill/spin-combine wrappers
 * whose fill-side stream synchronize is the dispatch-per-send NHI release
 * (rule 8). */
static hipStream_t g_tp_fill_stream;
static hipStream_t g_tp_spin_stream;
static float *g_tp_fill_scratch;
static size_t g_tp_fill_scratch_bytes;
static int *g_tp_spin_timeout_flag;

static int cuda_tp_streams_ready(void) {
    if (!g_tp_fill_stream &&
        hipStreamCreate(&g_tp_fill_stream) != hipSuccess) return 0;
    if (!g_tp_spin_stream &&
        hipStreamCreate(&g_tp_spin_stream) != hipSuccess) return 0;
    if (!g_tp_spin_timeout_flag &&
        hipMalloc(&g_tp_spin_timeout_flag, sizeof(int)) != hipSuccess) return 0;
    return 1;
}

extern "C" int ds4_gpu_tp_pool_alloc_export_uc(uint64_t bytes,
                                               void **dev_ptr,
                                               int *dmabuf_fd) {
    if (dev_ptr) *dev_ptr = NULL;
    if (dmabuf_fd) *dmabuf_fd = -1;
    if (!dev_ptr || !dmabuf_fd || bytes == 0 ||
        (uint64_t)(size_t)bytes != bytes)
        return 0;
    void *ptr = NULL;
    if (!cuda_ok(hipExtMallocWithFlags(&ptr, (size_t)bytes,
                                       hipDeviceMallocUncached),
                 "uncached TP pool allocation"))
        return 0;
    hipDeviceptr_t base = NULL;
    size_t range = 0;
    hipError_t err = hipMemGetAddressRange(&base, &range, (hipDeviceptr_t)ptr);
    if (err != hipSuccess || base != (hipDeviceptr_t)ptr ||
        range != (size_t)bytes) {
        if (err != hipSuccess) (void)cuda_ok(err, "TP pool address range");
        else
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "TP pool is not a dedicated allocation "
                    "(base=%p self=%p range=%zu bytes=%llu)\n",
                    (void *)base, ptr, range, (unsigned long long)bytes);
        (void)hipFree(ptr);
        return 0;
    }
    int fd = -1;
    err = hipMemGetHandleForAddressRange(&fd, (hipDeviceptr_t)ptr,
                                         (size_t)bytes,
                                         hipMemRangeHandleTypeDmaBufFd, 0);
    if (err != hipSuccess || fd < 0) {
        if (err != hipSuccess) (void)cuda_ok(err, "TP pool DMA-BUF export");
        (void)hipFree(ptr);
        return 0;
    }
    *dev_ptr = ptr;
    *dmabuf_fd = fd;
    return 1;
}

extern "C" int ds4_gpu_tp_dev_buf_create(const float *init_host, uint32_t n,
                                         void **dev_ptr) {
    if (dev_ptr) *dev_ptr = NULL;
    if (!dev_ptr || n == 0) return 0;
    float *d = NULL;
    const size_t bytes = (size_t)n * sizeof(float);
    if (hipMalloc(&d, bytes) != hipSuccess) return 0;
    if (init_host && hipMemcpy(d, init_host, bytes,
                               hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(d);
        return 0;
    }
    *dev_ptr = d;
    return 1;
}

extern "C" int ds4_gpu_tp_dev_buf_read(const void *dev_ptr, float *out_host,
                                       uint32_t n) {
    if (!dev_ptr || !out_host || n == 0) return 0;
    return hipMemcpy(out_host, dev_ptr, (size_t)n * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess;
}

extern "C" void ds4_gpu_tp_dev_buf_free(void *dev_ptr) {
    if (dev_ptr) (void)hipFree(dev_ptr);
}

extern "C" int ds4_gpu_tp_fill_release(void *slot_dev, const float *src_host,
                                       uint32_t n, uint32_t stamp) {
    if (!slot_dev || !src_host || n == 0 ||
        n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return 0;
    if (!cuda_tp_streams_ready()) return 0;
    const size_t bytes = (size_t)n * sizeof(float);
    if (g_tp_fill_scratch_bytes < bytes) {
        if (g_tp_fill_scratch) (void)hipFree(g_tp_fill_scratch);
        g_tp_fill_scratch = NULL;
        g_tp_fill_scratch_bytes = 0;
        if (hipMalloc(&g_tp_fill_scratch, bytes) != hipSuccess) return 0;
        g_tp_fill_scratch_bytes = bytes;
    }
    if (hipMemcpyAsync(g_tp_fill_scratch, src_host, bytes,
                       hipMemcpyHostToDevice, g_tp_fill_stream) != hipSuccess)
        return 0;
    hipLaunchKernelGGL(dsv4_tp_slot_fill_kernel, dim3(1), dim3(256), 0,
                       g_tp_fill_stream, (unsigned char *)slot_dev,
                       g_tp_fill_scratch, n, stamp);
    /* Dispatch-per-send design: the stream synchronize is the NHI
     * visibility release (contract rule 8). */
    return cuda_ok(hipStreamSynchronize(g_tp_fill_stream),
                   "TP fill release synchronize");
}

extern "C" int ds4_gpu_tp_spin_combine_start(void *acc_dev,
                                             const void *slot_dev,
                                             uint32_t n,
                                             uint32_t expect_stamp,
                                             unsigned long long max_spins) {
    if (!acc_dev || !slot_dev || n == 0 ||
        n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return 0;
    if (!cuda_tp_streams_ready()) return 0;
    if (hipMemsetAsync(g_tp_spin_timeout_flag, 0, sizeof(int),
                       g_tp_spin_stream) != hipSuccess)
        return 0;
    hipLaunchKernelGGL(dsv4_tp_spin_combine_f32_kernel, dim3(1), dim3(256), 0,
                       g_tp_spin_stream, (float *)acc_dev,
                       (const unsigned char *)slot_dev, n, expect_stamp,
                       max_spins, (const uint32_t *)NULL,
                       g_tp_spin_timeout_flag);
    return 1;
}

extern "C" int ds4_gpu_tp_spin_combine_wait(int *timed_out) {
    if (timed_out) *timed_out = 1;
    if (!g_tp_spin_stream || !g_tp_spin_timeout_flag) return 0;
    if (hipStreamSynchronize(g_tp_spin_stream) != hipSuccess) return 0;
    int flag = 0;
    if (hipMemcpy(&flag, g_tp_spin_timeout_flag, sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return 0;
    if (timed_out) *timed_out = flag;
    return 1;
}

/* Tensor-parallel gates are Metal-only; stubs keep shared graph code
 * linkable (TP option validation rejects non-Metal backends). */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)full_in_dim; (void)k_off; (void)k_cnt; (void)out_dim; (void)x;
    (void)x_elem_off;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    (void)out; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups_total; (void)group0; (void)group_cnt; (void)out_dim;
    (void)heads;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add; (void)residual_hc;
    (void)post; (void)comb; (void)n_embd; (void)n_hc;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}
