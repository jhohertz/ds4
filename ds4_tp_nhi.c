/* NHI transport core for ROCm tensor parallelism (Window C, stage 2).
 *
 * Owns the thunderbolt-stream device and the two uncached GPU frame pools
 * (one per direction) that the per-layer TP exchange rides on.  The GPU
 * writes partial block outputs into transmit slots and spin-polls in-band
 * stamps in receive slots (rocm/ds4_rocm_tp.cuh); this module owns
 * everything host-side: pool import, slot arithmetic, submission, event
 * reaping and receive-frame reposting, and ordered teardown.
 *
 * Every rule here comes from the validated transport-integration contract
 * (strix-rdma docs/TP_TRANSPORT_CONTRACT.md):
 *   - pools are dedicated uncached allocations imported tx+rx in one call
 *     after open() and strictly before ZC_ENABLE (rules 1-3);
 *   - slots advance strictly in order, message = 8 frames, and both peers
 *     submit in the same order because the receive side only has arrival
 *     order (rules 9-10);
 *   - events are reaped nonblocking off the critical path and receive
 *     frames repost in consumption order (rule 11);
 *   - the caller must run an out-of-band barrier between both sides'
 *     enable and the first submit -- transmitting toward an inactive peer
 *     can wedge the whole Thunderbolt connection (rule 13);
 *   - teardown logs kernel stats first (with a flush so unit stops cannot
 *     eat the line), then closes the device, then frees the pools --
 *     the kernel's DMA-BUF attachment drops at close (rules 4, 16).
 */

#ifdef __linux__

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "ds4_gpu.h"
#include "ds4_tbstream_uapi.h"
#include "ds4_tp_nhi.h"

#define DS4_TP_NHI_MSG_FRAMES 8u
#define DS4_TP_NHI_REAP_BATCH 64u

struct ds4_tp_nhi {
    int device_fd;
    char *device_path;
    uint32_t ring_size;
    uint32_t frame_size;
    uint32_t msg_frames;
    uint32_t msgs;            /* ring_size / msg_frames */
    uint64_t pool_bytes;      /* per direction */
    void *tx_pool;            /* uncached GPU allocation, imported as TX */
    void *rx_pool;            /* uncached GPU allocation, imported as RX */
    uint64_t tx_submitted;     /* next global TX sequence */
    uint64_t tx_done_seen;     /* validated TX_DONE events, strict FIFO */
    uint64_t rx_events_seen;   /* validated complete RX messages */
    uint64_t rx_consumed_seen; /* GPU final-reader completions, strict FIFO */
    uint64_t rx_reposted;      /* messages returned to the RX FIFO */
    int peer_closed;
};

static void tp_nhi_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

int ds4_tp_nhi_open(ds4_tp_nhi **out,
                    const char *device_path,
                    uint32_t ring_frames,
                    char *err,
                    size_t errlen) {
    if (out) *out = NULL;
    if (!out || !device_path || ring_frames < DS4_TP_NHI_MSG_FRAMES ||
        (ring_frames % DS4_TP_NHI_MSG_FRAMES) != 0) {
        tp_nhi_set_err(err, errlen, "invalid TP NHI open arguments");
        return 0;
    }

    ds4_tp_nhi *t = calloc(1, sizeof(*t));
    if (!t) {
        tp_nhi_set_err(err, errlen, "out of memory");
        return 0;
    }
    t->device_fd = -1;
    t->msg_frames = DS4_TP_NHI_MSG_FRAMES;
    t->pool_bytes = (uint64_t)ring_frames * TBSTREAM_ZC_FRAME_SIZE;
    t->device_path = strdup(device_path);
    if (!t->device_path) goto fail_oom;

    int tx_fd = -1, rx_fd = -1;
    if (!ds4_gpu_tp_pool_alloc_export_uc(t->pool_bytes, &t->tx_pool, &tx_fd) ||
        !ds4_gpu_tp_pool_alloc_export_uc(t->pool_bytes, &t->rx_pool, &rx_fd)) {
        if (tx_fd >= 0) close(tx_fd);
        tp_nhi_set_err(err, errlen,
                       "uncached TP pool allocation/export failed "
                       "(dedicated BO required)");
        goto fail;
    }

    t->device_fd = open(device_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (t->device_fd < 0) {
        tp_nhi_set_err(err, errlen, "open %s: %s", device_path,
                       strerror(errno));
        close(tx_fd);
        close(rx_fd);
        goto fail;
    }

    /* Both directions imported in one call, before ENABLE (rule 2). */
    struct tbstream_zc_import imp;
    memset(&imp, 0, sizeof(imp));
    imp.version = TBSTREAM_ZC_IMPORT_VERSION;
    imp.tx.fd = tx_fd;
    imp.tx.offset = 0;
    imp.tx.length = t->pool_bytes;
    imp.rx.fd = rx_fd;
    imp.rx.offset = 0;
    imp.rx.length = t->pool_bytes;
    if (ioctl(t->device_fd, TBSTREAM_ZC_IMPORT, &imp) != 0) {
        tp_nhi_set_err(err, errlen,
                       "%s: TBSTREAM_ZC_IMPORT (tx+rx): %s (needs patch-14+ "
                       "module, zc_diagnostic_dmabuf=1, CAP_SYS_RAWIO, "
                       "matching ring size)",
                       device_path, strerror(errno));
        close(tx_fd);
        close(rx_fd);
        goto fail;
    }
    close(tx_fd);
    close(rx_fd);

    if (ioctl(t->device_fd, TBSTREAM_ZC_ENABLE) != 0) {
        tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_ENABLE: %s",
                       device_path, strerror(errno));
        goto fail;
    }
    struct tbstream_zc_info info;
    memset(&info, 0, sizeof(info));
    if (ioctl(t->device_fd, TBSTREAM_ZC_GET_INFO, &info) != 0) {
        tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_GET_INFO: %s",
                       device_path, strerror(errno));
        goto fail;
    }
    if (info.frame_size != TBSTREAM_ZC_FRAME_SIZE ||
        info.ring_size != ring_frames) {
        char detail[96];
        snprintf(detail, sizeof(detail), "ring %u frame %u (wanted %u/%u)",
                 info.ring_size, info.frame_size, ring_frames,
                 TBSTREAM_ZC_FRAME_SIZE);
        tp_nhi_set_err(err, errlen, "%s: geometry mismatch: %s",
                       device_path, detail);
        goto fail;
    }
    t->ring_size = info.ring_size;
    t->frame_size = info.frame_size;
    t->msgs = t->ring_size / t->msg_frames;
    *out = t;
    return 1;

fail_oom:
    tp_nhi_set_err(err, errlen, "out of memory");
fail:
    ds4_tp_nhi_close(t);
    return 0;
}

uint32_t ds4_tp_nhi_msg_frames(const ds4_tp_nhi *t) {
    return t ? t->msg_frames : 0;
}

uint32_t ds4_tp_nhi_msgs(const ds4_tp_nhi *t) {
    return t ? t->msgs : 0;
}

/* Device VAs for the slot a given 0-based sequence occupies.  Slots advance
 * strictly in order on both sides (rule 9). */
void *ds4_tp_nhi_tx_slot(ds4_tp_nhi *t, uint64_t seq) {
    if (!t) return NULL;
    const uint64_t slot = (seq % t->msgs) * (uint64_t)t->msg_frames *
                          t->frame_size;
    return (unsigned char *)t->tx_pool + slot;
}

const void *ds4_tp_nhi_rx_slot(ds4_tp_nhi *t, uint64_t seq) {
    if (!t) return NULL;
    const uint64_t slot = (seq % t->msgs) * (uint64_t)t->msg_frames *
                          t->frame_size;
    return (const unsigned char *)t->rx_pool + slot;
}

static uint32_t tp_nhi_frame_first(const ds4_tp_nhi *t, uint64_t seq) {
    return (uint32_t)((seq % t->msgs) * t->msg_frames);
}

static uint64_t tp_nhi_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* POST_RX is legal only after the complete RX event and the GPU's final
 * slot reader have both advanced past a message.  Batch every contiguous
 * ready prefix; repost order is the receive-ring address mapping (rule 11). */
static int tp_nhi_repost_ready(ds4_tp_nhi *t, char *err, size_t errlen) {
    const uint64_t ready_through = t->rx_events_seen < t->rx_consumed_seen
        ? t->rx_events_seen : t->rx_consumed_seen;
    if (ready_through < t->rx_reposted) {
        tp_nhi_set_err(err, errlen,
                       "%s: TP NHI RX repost accounting regressed",
                       t->device_path);
        return 0;
    }
    uint64_t ready = ready_through - t->rx_reposted;
    if (ready > t->msgs) {
        tp_nhi_set_err(err, errlen,
                       "%s: TP NHI RX repost lag %llu exceeds %u slots",
                       t->device_path, (unsigned long long)ready, t->msgs);
        return 0;
    }
    while (ready != 0) {
        const uint64_t batch = ready > t->msgs ? t->msgs : ready;
        uint32_t frames = (uint32_t)batch * t->msg_frames;
        if (ioctl(t->device_fd, TBSTREAM_ZC_POST_RX, &frames) != 0) {
            tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_POST_RX: %s",
                           t->device_path, strerror(errno));
            return 0;
        }
        t->rx_reposted += batch;
        ready -= batch;
    }
    return 1;
}

/* Drain driver events without blocking (rule 11).  Every RX and TX_DONE
 * record is checked against strict global FIFO sequence; a geometry drift
 * would otherwise make a later stamp appear in the wrong GPU slot. */
int ds4_tp_nhi_reap(ds4_tp_nhi *t, char *err, size_t errlen) {
    if (!t) return 0;
    struct tbstream_zc_event events[DS4_TP_NHI_REAP_BATCH];
    for (;;) {
        struct tbstream_zc_reap reap;
        memset(&reap, 0, sizeof(reap));
        reap.max = DS4_TP_NHI_REAP_BATCH;
        reap.flags = TBSTREAM_ZC_REAP_NONBLOCK;
        reap.events = (uint64_t)(uintptr_t)events;
        const int count = ioctl(t->device_fd, TBSTREAM_ZC_REAP, &reap);
        if (count < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN)
                return tp_nhi_repost_ready(t, err, errlen);
            tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_REAP: %s",
                           t->device_path, strerror(errno));
            return 0;
        }
        if (count == 0) return tp_nhi_repost_ready(t, err, errlen);
        if (count > (int)DS4_TP_NHI_REAP_BATCH) {
            tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_REAP: %s",
                           t->device_path, "event count out of range");
            return 0;
        }
        for (int i = 0; i < count; i++) {
            const struct tbstream_zc_event *ev = &events[i];
            uint64_t seq;
            switch (ev->type) {
            case TBSTREAM_ZC_EV_RX:
                seq = t->rx_events_seen;
                if (ev->first != tp_nhi_frame_first(t, seq) ||
                    ev->nframes != t->msg_frames ||
                    ev->bytes != t->msg_frames * t->frame_size) {
                    tp_nhi_set_err(err, errlen,
                                   "%s: RX event %llu geometry "
                                   "first=%u frames=%u bytes=%u",
                                   t->device_path, (unsigned long long)seq,
                                   ev->first, ev->nframes, ev->bytes);
                    return 0;
                }
                t->rx_events_seen++;
                break;
            case TBSTREAM_ZC_EV_TX_DONE:
                seq = t->tx_done_seen;
                if (seq >= t->tx_submitted ||
                    ev->first != tp_nhi_frame_first(t, seq) ||
                    ev->nframes != t->msg_frames ||
                    ev->bytes != t->msg_frames * t->frame_size) {
                    tp_nhi_set_err(err, errlen,
                                   "%s: TX_DONE event %llu geometry "
                                   "first=%u frames=%u bytes=%u",
                                   t->device_path, (unsigned long long)seq,
                                   ev->first, ev->nframes, ev->bytes);
                    return 0;
                }
                t->tx_done_seen++;
                break;
            case TBSTREAM_ZC_EV_CLOSE:
                t->peer_closed = 1;
                break;
            default:
                tp_nhi_set_err(err, errlen,
                               "%s: unknown TP NHI event type %u",
                               t->device_path, ev->type);
                return 0;
            }
        }
        if (count < (int)DS4_TP_NHI_REAP_BATCH)
            return tp_nhi_repost_ready(t, err, errlen);
    }
}

/* Submit one GPU-released TX message.  `first` is kernel output, not an
 * address request: validating it proves that the driver's ring head still
 * matches the global gate sequence (rules 9 and 12). */
int ds4_tp_nhi_submit(ds4_tp_nhi *t, uint64_t seq, char *err, size_t errlen) {
    if (!t || t->peer_closed) return 0;
    if (seq != t->tx_submitted) {
        tp_nhi_set_err(err, errlen,
                       "%s: out-of-order TP NHI submit %llu (want %llu)",
                       t->device_path, (unsigned long long)seq,
                       (unsigned long long)t->tx_submitted);
        return 0;
    }

    const uint64_t deadline = tp_nhi_now_ns() + 30000000000ull;
    while (seq >= t->msgs && t->tx_done_seen <= seq - t->msgs) {
        if (!ds4_tp_nhi_reap(t, err, errlen) || t->peer_closed) return 0;
        if (tp_nhi_now_ns() >= deadline) {
            tp_nhi_set_err(err, errlen,
                           "%s: timed out waiting for TX slot %u credit",
                           t->device_path, tp_nhi_frame_first(t, seq));
            return 0;
        }
        const struct timespec pause = {0, 50000};
        nanosleep(&pause, NULL);
    }

    for (;;) {
        struct tbstream_zc_tx tx;
        memset(&tx, 0, sizeof(tx));
        tx.nframes = t->msg_frames;
        tx.last_len = t->frame_size;
        if (ioctl(t->device_fd, TBSTREAM_ZC_SUBMIT_TX, &tx) == 0) {
            const uint32_t expected = tp_nhi_frame_first(t, seq);
            if (tx.first != expected) {
                tp_nhi_set_err(err, errlen,
                               "%s: TX submit %llu returned first=%u (want %u)",
                               t->device_path, (unsigned long long)seq,
                               tx.first, expected);
                return 0;
            }
            t->tx_submitted++;
            return 1;
        }
        if (errno != ENOBUFS && errno != EAGAIN && errno != EINTR) {
            tp_nhi_set_err(err, errlen, "%s: TBSTREAM_ZC_SUBMIT_TX: %s",
                           t->device_path, strerror(errno));
            return 0;
        }
        if (!ds4_tp_nhi_reap(t, err, errlen) || t->peer_closed) return 0;
        if (tp_nhi_now_ns() >= deadline) {
            tp_nhi_set_err(err, errlen,
                           "%s: TBSTREAM_ZC_SUBMIT_TX credit timeout",
                           t->device_path);
            return 0;
        }
        const struct timespec pause = {0, 50000};
        nanosleep(&pause, NULL);
    }
}

/* Record completion of the GPU's final read from one RX slot.  Reap joins
 * this monotonic sequence with validated RX events before POST_RX. */
int ds4_tp_nhi_consumed(ds4_tp_nhi *t, uint64_t seq,
                        char *err, size_t errlen) {
    if (!t) return 0;
    if (seq != t->rx_consumed_seen) {
        tp_nhi_set_err(err, errlen,
                       "%s: out-of-order TP NHI consume %llu (want %llu)",
                       t->device_path, (unsigned long long)seq,
                       (unsigned long long)t->rx_consumed_seen);
        return 0;
    }
    t->rx_consumed_seen++;
    const uint64_t deadline = tp_nhi_now_ns() + 30000000000ull;
    while (t->rx_reposted <= seq) {
        if (!ds4_tp_nhi_reap(t, err, errlen) || t->peer_closed) return 0;
        if (tp_nhi_now_ns() >= deadline) {
            tp_nhi_set_err(err, errlen,
                           "%s: timed out waiting for RX event %llu",
                           t->device_path, (unsigned long long)seq);
            return 0;
        }
        const struct timespec pause = {0, 50000};
        nanosleep(&pause, NULL);
    }
    return 1;
}

int ds4_tp_nhi_peer_closed(const ds4_tp_nhi *t) {
    return t ? t->peer_closed : 1;
}

int ds4_tp_nhi_quiesce(ds4_tp_nhi *t, char *err, size_t errlen) {
    if (!t) return 0;
    const uint64_t deadline = tp_nhi_now_ns() + 30000000000ull;
    while (t->tx_done_seen < t->tx_submitted ||
           t->rx_reposted < t->rx_consumed_seen) {
        if (!ds4_tp_nhi_reap(t, err, errlen)) return 0;
        if (t->peer_closed &&
            (t->tx_done_seen < t->tx_submitted ||
             t->rx_reposted < t->rx_consumed_seen)) {
            tp_nhi_set_err(err, errlen,
                           "%s: peer closed with TP NHI ownership outstanding",
                           t->device_path);
            return 0;
        }
        if (tp_nhi_now_ns() >= deadline) {
            tp_nhi_set_err(err, errlen,
                           "%s: TP NHI quiesce timeout "
                           "tx=%llu/%llu rx=%llu/%llu",
                           t->device_path,
                           (unsigned long long)t->tx_done_seen,
                           (unsigned long long)t->tx_submitted,
                           (unsigned long long)t->rx_reposted,
                           (unsigned long long)t->rx_consumed_seen);
            return 0;
        }
        const struct timespec pause = {0, 50000};
        nanosleep(&pause, NULL);
    }
    return 1;
}

/* Stats first (with a flush so a unit stop cannot eat the line), then the
 * device (drops the kernel DMA-BUF attachment), then the pools (rules 4
 * and 16). */
void ds4_tp_nhi_close(ds4_tp_nhi *t) {
    if (!t) return;
    if (t->device_fd >= 0) {
        struct tbstream_zc_stats stats;
        memset(&stats, 0, sizeof(stats));
        stats.version = TBSTREAM_ZC_STATS_VERSION;
        stats.struct_size = sizeof(stats);
        if (ioctl(t->device_fd, TBSTREAM_ZC_GET_STATS, &stats) == 0) {
            fprintf(stderr,
                    "ds4: TP NHI stats %s: flags=0x%x%s%s failures=%llu "
                    "event_drops=%llu crc=%llu overrun=%llu "
                    "tx=%llu/%llu rx=%llu/%llu\n",
                    t->device_path, stats.flags,
                    (stats.flags & TBSTREAM_ZC_STATS_F_TX_IMPORTED)
                        ? " TX_IMPORTED" : "",
                    (stats.flags & TBSTREAM_ZC_STATS_F_RX_IMPORTED)
                        ? " RX_IMPORTED" : "",
                    (unsigned long long)stats.failures,
                    (unsigned long long)stats.event_drops,
                    (unsigned long long)stats.crc_errors,
                    (unsigned long long)stats.overrun_errors,
                    (unsigned long long)stats.tx.descriptors_posted,
                    (unsigned long long)stats.tx.descriptors_completed,
                    (unsigned long long)stats.rx.descriptors_posted,
                    (unsigned long long)stats.rx.descriptors_completed);
        }
        fflush(stderr);
        close(t->device_fd);
        t->device_fd = -1;
    }
    if (t->tx_pool) {
        (void)ds4_gpu_pool_free_exported(t->tx_pool);
        t->tx_pool = NULL;
    }
    if (t->rx_pool) {
        (void)ds4_gpu_pool_free_exported(t->rx_pool);
        t->rx_pool = NULL;
    }
    free(t->device_path);
    free(t);
}

#endif /* __linux__ */
