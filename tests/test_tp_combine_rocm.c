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
#include <stdio.h>
#include <stdlib.h>

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

    printf("test_tp_combine_rocm: %d/%d checks passed (%d failed)\n",
           g_checks - g_failures, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
