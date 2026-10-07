// SPDX-License-Identifier: GPL-3.0-or-later
// Independent review verifier; reproducible command is in VALIDATION.md.
#include "ring.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const size_t total = 16 * 1024 * 1024 + 137;
static int input_fd;
static void *consume(void *unused) {
    (void)unused;
    uint8_t data[7777];
    size_t count = 0;
    ssize_t n;
    while ((n = read(input_fd, data, sizeof data)) > 0) {
        for (ssize_t i = 0; i < n; i++, count++)
            assert(data[i] == (uint8_t)(count % 251));
    }
    assert(n == 0 && count == total);
    return NULL;
}

int main(void) {
    int fds[2];
    assert(pipe(fds) == 0);
    input_fd = fds[0];
    pthread_t reader;
    assert(pthread_create(&reader, NULL, consume, NULL) == 0);
    struct dsl_ring ring;
    const size_t capacity = 98317;
    assert(dsl_ring_start(&ring, capacity, fds[1]) == 0);
    uint8_t data[8192];
    size_t count = 0, pushes = 0;
    while (count < total) {
        size_t n = 1 + (pushes * 7919) % sizeof data;
        if (n > total - count) n = total - count;
        for (size_t i = 0; i < n; i++) data[i] = (uint8_t)((count+i) % 251);
        for (;;) {
            pthread_mutex_lock(&ring.mutex);
            size_t free_space = ring.capacity - ring.used;
            pthread_mutex_unlock(&ring.mutex);
            if (free_space >= n) break;
            usleep(100);
        }
        assert(dsl_ring_push(&ring, data, n) == 0);
        count += n;
        pushes++;
    }
    assert(dsl_ring_finish(&ring) == 0);
    assert(ring.written == total);
    assert(ring.read_pos == total % capacity);
    assert(ring.write_pos == total % capacity);
    close(fds[1]);
    assert(pthread_join(reader, NULL) == 0);
    close(fds[0]);
    printf("ring wrap PASS: %zu bytes, %zu irregular pushes, %zu capacity crossings, byte oracle modulo 251\n",
           total, pushes, total/capacity);
    return 0;
}
