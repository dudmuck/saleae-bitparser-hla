// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_RING_H
#define DSLCAP_RING_H
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
struct dsl_ring {
    uint8_t *data;
    size_t capacity, read_pos, write_pos, used, high_water;
    uint64_t written, drain_deadline;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_t writer;
    int fd, saved_flags, closed, result, drain_timed_out;
};
uint64_t dsl_now_ns(void);
int dsl_ring_start(struct dsl_ring *r, size_t capacity, int fd);
int dsl_ring_push(void *context, const uint8_t *data, size_t length);
int dsl_ring_status(struct dsl_ring *r);
// Close producer, drain for at most two seconds, join writer, restore fd flags.
int dsl_ring_finish(struct dsl_ring *r);
#endif
