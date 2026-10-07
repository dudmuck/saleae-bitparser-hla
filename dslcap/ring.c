// SPDX-License-Identifier: GPL-3.0-or-later
#include "ring.h"
#include "status.h"
#include "control.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef DSLCAP_TEST_TIME_SCALE
#define DSLCAP_TEST_TIME_SCALE 1.0
#endif
#define DRAIN_HARD_GRACE ((uint64_t)(1e10 * DSLCAP_TEST_TIME_SCALE))

uint64_t dsl_now_ns(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (uint64_t)time.tv_sec * 1000000000 + (uint64_t)time.tv_nsec;
}

static void close_locked(struct dsl_ring *r)
{
    if (!r->closed) {
        r->closed = 1;
        r->drain_deadline = dsl_now_ns() + (r->buffered ? r->stall_ns : UINT64_C(2000000000));
    }
    pthread_cond_broadcast(&r->ready);
}

static void fail(struct dsl_ring *r, int result)
{
    pthread_mutex_lock(&r->mutex);
    if (!r->result) r->result = result;
    close_locked(r);
    pthread_mutex_unlock(&r->mutex);
}

static void writer_done(struct dsl_ring *r)
{
    pthread_mutex_lock(&r->mutex);
    r->writer_done = 1;
    pthread_cond_broadcast(&r->ready);
    pthread_mutex_unlock(&r->mutex);
}

static void *writer(void *context)
{
    struct dsl_ring *r = context;
    uint8_t chunk[65536];
    for (;;) {
        pthread_mutex_lock(&r->mutex);
        while (!r->used && !r->closed) pthread_cond_wait(&r->ready, &r->mutex);
        if (!r->used && r->closed) { pthread_mutex_unlock(&r->mutex); break; }
        size_t n = r->used < sizeof(chunk) ? r->used : sizeof(chunk);
        size_t first = r->capacity - r->read_pos;
        if (first > n) first = n;
        memcpy(chunk, r->data + r->read_pos, first);
        memcpy(chunk + first, r->data, n - first);
        r->read_pos = (r->read_pos + n) % r->capacity;
        r->used -= n;
        pthread_mutex_unlock(&r->mutex);
        size_t offset = 0;
        while (offset < n) {
            pthread_mutex_lock(&r->mutex);
            uint64_t deadline = r->drain_deadline;
            int cancel = r->cancel;
            pthread_mutex_unlock(&r->mutex);
            if (cancel) { r->drain_cancelled = 1; writer_done(r); return NULL; }
            if (deadline && dsl_now_ns() >= deadline) {
                r->drain_timed_out = 1;
                fail(r, DSL_DRAIN_TIMEOUT); writer_done(r); return NULL;
            }
            ssize_t result = write(r->fd, chunk + offset, n - offset);
            if (result > 0) {
                offset += (size_t)result;
                pthread_mutex_lock(&r->mutex);
                r->written += (uint64_t)result;
                if (r->buffered && r->closed) {
                    r->drain_deadline = dsl_now_ns() + r->stall_ns;
                    dsl_control_deadline(r->drain_deadline + DRAIN_HARD_GRACE);
                }
                pthread_cond_broadcast(&r->ready);
                pthread_mutex_unlock(&r->mutex);
                continue;
            }
            if (result < 0 && errno == EINTR) continue;
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                struct pollfd pollfd = {.fd = r->fd, .events = POLLOUT};
                if (poll(&pollfd, 1, 50) < 0 && errno != EINTR) {
                    fail(r, DSL_OUTPUT_ERROR); writer_done(r); return NULL;
                }
                continue;
            }
            fail(r, DSL_OUTPUT_ERROR); writer_done(r); return NULL;
        }
    }
    writer_done(r);
    return NULL;
}

static int start(struct dsl_ring *r, size_t capacity, int fd, double stall_seconds)
{
    memset(r, 0, sizeof(*r));
    if (!capacity) return -1;
    r->capacity = capacity; r->fd = fd;
    r->buffered = stall_seconds > 0;
    r->stall_ns = (uint64_t)(stall_seconds * 1e9);
    r->data = malloc(capacity);
    if (!r->data) return -1;
    r->saved_flags = fcntl(fd, F_GETFL);
    if (r->saved_flags < 0 || fcntl(fd, F_SETFL, r->saved_flags | O_NONBLOCK) < 0) goto free_data;
    if (pthread_mutex_init(&r->mutex, NULL)) goto restore_flags;
    pthread_condattr_t attr;
    if (pthread_condattr_init(&attr)) goto free_mutex;
    if (pthread_condattr_setclock(&attr, CLOCK_MONOTONIC)) { pthread_condattr_destroy(&attr); goto free_mutex; }
    int cond_result = pthread_cond_init(&r->ready, &attr);
    pthread_condattr_destroy(&attr);
    if (cond_result) goto free_mutex;
    if (pthread_create(&r->writer, NULL, writer, r)) goto free_cond;
    return 0;
free_cond: pthread_cond_destroy(&r->ready);
free_mutex: pthread_mutex_destroy(&r->mutex);
restore_flags: fcntl(fd, F_SETFL, r->saved_flags);
free_data: free(r->data); r->data = NULL; return -1;
}

int dsl_ring_start(struct dsl_ring *r, size_t capacity, int fd)
{ return start(r, capacity, fd, 0); }
int dsl_ring_start_buffer(struct dsl_ring *r, size_t capacity, int fd, double stall_seconds)
{ return start(r, capacity, fd, stall_seconds); }

int dsl_ring_push(void *context, const uint8_t *data, size_t length)
{
    struct dsl_ring *r = context;
    pthread_mutex_lock(&r->mutex);
    if (r->closed || length > r->capacity - r->used) {
        if (!r->closed && !r->result) r->result = DSL_RING_FULL;
        close_locked(r);
        pthread_mutex_unlock(&r->mutex);
        return -1;
    }
    size_t first = r->capacity - r->write_pos;
    if (first > length) first = length;
    memcpy(r->data + r->write_pos, data, first);
    memcpy(r->data, data + first, length - first);
    r->write_pos = (r->write_pos + length) % r->capacity;
    r->used += length;
    if (r->used > r->high_water) r->high_water = r->used;
    pthread_cond_signal(&r->ready);
    pthread_mutex_unlock(&r->mutex);
    return 0;
}

int dsl_ring_status(struct dsl_ring *r)
{
    pthread_mutex_lock(&r->mutex);
    int result = r->result;
    pthread_mutex_unlock(&r->mutex);
    return result;
}

int dsl_ring_finish(struct dsl_ring *r)
{
    pthread_mutex_lock(&r->mutex);
    close_locked(r);
    if (r->buffered) {
        dsl_control_deadline(r->drain_deadline + DRAIN_HARD_GRACE);
        while (!r->writer_done) {
            if (dsl_signal || r->cancel) r->cancel = 1;
            uint64_t wake = dsl_now_ns() + UINT64_C(50000000);
            struct timespec timeout = {.tv_sec = (time_t)(wake / 1000000000), .tv_nsec = (long)(wake % 1000000000)};
            pthread_cond_timedwait(&r->ready, &r->mutex, &timeout);
        }
    }
    pthread_mutex_unlock(&r->mutex);
    pthread_join(r->writer, NULL);
    if (fcntl(r->fd, F_SETFL, r->saved_flags) < 0 && !r->result) r->result = DSL_OUTPUT_ERROR;
    pthread_cond_destroy(&r->ready);
    pthread_mutex_destroy(&r->mutex);
    free(r->data); r->data = NULL;
    return r->result;
}
