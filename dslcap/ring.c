// SPDX-License-Identifier: GPL-3.0-or-later
#include "ring.h"
#include "status.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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
        r->drain_deadline = dsl_now_ns() + 2000000000;
    }
    pthread_cond_signal(&r->ready);
}

static void fail(struct dsl_ring *r, int result)
{
    pthread_mutex_lock(&r->mutex);
    if (!r->result) r->result = result;
    close_locked(r);
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
            pthread_mutex_unlock(&r->mutex);
            if (deadline && dsl_now_ns() >= deadline) {
                r->drain_timed_out = 1;
                fail(r, DSL_DRAIN_TIMEOUT); return NULL;
            }
            ssize_t result = write(r->fd, chunk + offset, n - offset);
            if (result > 0) { offset += (size_t)result; r->written += (uint64_t)result; continue; }
            if (result < 0 && errno == EINTR) continue;
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                struct pollfd pollfd = {.fd = r->fd, .events = POLLOUT};
                if (poll(&pollfd, 1, 50) < 0 && errno != EINTR) {
                    fail(r, DSL_OUTPUT_ERROR); return NULL;
                }
                continue;
            }
            fail(r, DSL_OUTPUT_ERROR); return NULL;
        }
    }
    return NULL;
}

int dsl_ring_start(struct dsl_ring *r, size_t capacity, int fd)
{
    memset(r, 0, sizeof(*r));
    if (!capacity) return -1;
    r->capacity = capacity; r->fd = fd;
    r->data = malloc(capacity);
    if (!r->data) return -1;
    r->saved_flags = fcntl(fd, F_GETFL);
    if (r->saved_flags < 0 || fcntl(fd, F_SETFL, r->saved_flags | O_NONBLOCK) < 0) goto free_data;
    if (pthread_mutex_init(&r->mutex, NULL)) goto restore_flags;
    if (pthread_cond_init(&r->ready, NULL)) goto free_mutex;
    if (pthread_create(&r->writer, NULL, writer, r)) goto free_cond;
    return 0;
free_cond: pthread_cond_destroy(&r->ready);
free_mutex: pthread_mutex_destroy(&r->mutex);
restore_flags: fcntl(fd, F_SETFL, r->saved_flags);
free_data: free(r->data); r->data = NULL; return -1;
}

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
    pthread_mutex_unlock(&r->mutex);
    pthread_join(r->writer, NULL);
    if (fcntl(r->fd, F_SETFL, r->saved_flags) < 0 && !r->result) r->result = DSL_OUTPUT_ERROR;
    pthread_cond_destroy(&r->ready);
    pthread_mutex_destroy(&r->mutex);
    free(r->data); r->data = NULL;
    return r->result;
}
