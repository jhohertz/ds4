/* Stage-1 tests for the ROCm tensor-parallel combine kernels.
 *
 * The combine (world-of-two partial add) must be bit-exact against a CPU
 * float reference: TP's acceptable divergence lives in the reduction order
 * of the split matvecs, never in the combine.  The spin-combine variants
 * additionally prove the in-band stamp actually gates the combine: the
 * spinning kernel is launched before the payload exists and must wait for
 * the writer (a second-stream kernel, or the host via a pinned stamp).
 *
 * Loopback only: the NHI imported-pool path is exercised by the transport
 * stage on the two-node rig; this suite runs on any single ROCm device.
 */

#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failures;

static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        fprintf(stderr, "FAIL %s\n", what);
    } else {
        printf("ROCm TP %-44s PASS\n", what);
    }
}

static void fill_q8_block(unsigned char *block, uint32_t seed) {
    block[0] = 0x00u;
    block[1] = 0x3cu; /* IEEE fp16 1.0 */
    int8_t *q = (int8_t *)(block + 2u);
    for (uint32_t i = 0; i < 32u; i++)
        q[i] = (int8_t)((int)((seed + i * 3u) % 7u) - 3);
}

/* Rank 1's output groups are compact at heads[0], even though their A/B
 * weight window starts at group 1.  Poison the unused upper half so this
 * catches an accidental group0 shift while also exercising Q8 K-slice. */
static int test_attention_slice(void) {
    enum { GD = 32, RANK = 32, GROUPS = 2, OUT = 16 };
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * 34u;
    const uint64_t b_off = (a_bytes + 255u) & ~255ull;
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * 34u;
    const uint64_t model_bytes = (b_off + b_bytes + 4095u) & ~4095ull;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    FILE *model_file = NULL;
    float heads[GROUPS * GD], expected[OUT], actual[OUT];
    ds4_gpu_tensor *heads_t = NULL, *low_t = NULL, *out_t = NULL;
    int ok = model != NULL;
    if (!ok) return 0;

    for (uint32_t g = 0; g < GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++)
            fill_q8_block(model + ((uint64_t)g * RANK + r) * 34u,
                          11u + g * 97u + r * 5u);
    }
    for (uint32_t r = 0; r < OUT; r++) {
        for (uint32_t b = 0; b < GROUPS; b++)
            fill_q8_block(model + b_off + ((uint64_t)r * GROUPS + b) * 34u,
                          23u + r * 13u + b * 71u);
    }
    for (uint32_t i = 0; i < GD; i++) {
        heads[i] = (float)((int)(i % 7u) - 3);
        heads[GD + i] = 1000.0f + (float)i; /* must never be read */
    }
    float low_ref[RANK];
    for (uint32_t r = 0; r < RANK; r++) {
        const int8_t *q = (const int8_t *)(model +
            ((uint64_t)RANK + r) * 34u + 2u);
        float sum = 0.0f;
        for (uint32_t i = 0; i < GD; i++) sum += (float)q[i] * heads[i];
        low_ref[r] = sum;
    }
    for (uint32_t r = 0; r < OUT; r++) {
        const int8_t *q = (const int8_t *)(model + b_off +
            ((uint64_t)r * GROUPS + 1u) * 34u + 2u);
        float sum = 0.0f;
        for (uint32_t i = 0; i < RANK; i++) sum += (float)q[i] * low_ref[i];
        expected[r] = sum;
    }

    setenv("DS4_ROCM_DSV4_PREQUANT_DECODE", "0", 1);
    const uint64_t offsets[2] = {0, b_off};
    const uint64_t sizes[2] = {a_bytes, b_bytes};
    model_file = tmpfile();
    ok = model_file != NULL &&
         fwrite(model, 1, (size_t)model_bytes, model_file) == model_bytes &&
         fflush(model_file) == 0 &&
         ds4_gpu_set_model_map(model, model_bytes) &&
         ds4_gpu_set_model_fd_for_map(fileno(model_file), model) &&
         ds4_gpu_set_model_map_spans(model, model_bytes, offsets, sizes, 2,
                                      a_bytes > b_bytes ? a_bytes : b_bytes);
    if (ok) heads_t = ds4_gpu_tensor_alloc(sizeof(heads));
    if (ok) low_t = ds4_gpu_tensor_alloc((uint64_t)GROUPS * RANK * sizeof(float));
    if (ok) out_t = ds4_gpu_tensor_alloc(sizeof(actual));
    ok = ok && heads_t && low_t && out_t &&
         ds4_gpu_tensor_write(heads_t, 0, heads, sizeof(heads)) &&
         ds4_gpu_attention_output_q8_tp_tensor(
             out_t, low_t, model, model_bytes, 0, b_off,
             GD, RANK, GROUPS, 1, 1, OUT, heads_t) &&
         ds4_gpu_tensor_read(out_t, 0, actual, sizeof(actual));
    if (ok) ok = memcmp(actual, expected, sizeof(actual)) == 0;

    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(low_t);
    ds4_gpu_tensor_free(heads_t);
    if (model_file) fclose(model_file);
    free(model);
    return ok;
}

int main(void) {
    srand(20260825);
    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_tp_combine_rocm: no ROCm device available\n");
        return 1;
    }

    /* Bit-exact combine at DS4's hidden size, an odd size, and a small one. */
    check(ds4_gpu_tp_test_combine(7168, 8), "combine 7168 (hidden) bitexact");
    check(ds4_gpu_tp_test_combine(8191, 4), "combine 8191 (odd) bitexact");
    check(ds4_gpu_tp_test_combine(256, 4), "combine 256 bitexact");

    /* Spin-combine, writer-kernel arm: in-band stamp in a plain pool
     * (kernel-to-kernel visibility goes through L2 and needs no UC). */
    check(ds4_gpu_tp_test_spin_exchange(0, 0, 7168, 16),
          "spin-combine kernel-writer plain pool");

    /* Spin-combine on an UNCACHED pool: the production MTYPE for
     * wave-resident polling of NHI-written slots (contract rule 6). */
    check(ds4_gpu_tp_test_spin_exchange(1, 0, 7168, 16),
          "spin-combine kernel-writer uncached pool");

    /* Host-mediated stamp (pinned control word, contract rule 8), payload
     * arriving by copy engine while the wave spins. */
    check(ds4_gpu_tp_test_spin_exchange(1, 1, 7168, 8),
          "spin-combine host-stamp uncached pool");

    check(test_attention_slice(),
          "rank1 compact-head Q8 attention/K-slice");

    printf("test_tp_combine_rocm: %d/%d checks passed (%d failed)\n",
           g_checks - g_failures, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
