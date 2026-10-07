// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_RING_H
#define DSLCAP_RING_H
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
struct dsl_ring {
    uint8_t *data;
    size_t capacity, read_pos, write_pos, used, high_water;
    uint64_t written, drain_deadline, stall_ns;
    int buffered, writer_done;
    atomic_int cancel;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_t writer;
    int fd, saved_flags, closed, result, drain_timed_out, drain_cancelled;
};
uint64_t dsl_now_ns(void);
int dsl_ring_start(struct dsl_ring *r, size_t capacity, int fd);
int dsl_ring_start_buffer(struct dsl_ring *r, size_t capacity, int fd, double stall_seconds);
int dsl_ring_push(void *context, const uint8_t *data, size_t length);
int dsl_ring_status(struct dsl_ring *r);
// Close producer, drain with stream fixed/buffer progress policy, join, restore fd flags.
int dsl_ring_finish(struct dsl_ring *r);
#endif
