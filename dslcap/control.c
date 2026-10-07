// SPDX-License-Identifier: GPL-3.0-or-later
#include "control.h"
#include "ring.h"
#include "status.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
volatile sig_atomic_t dsl_signal;
static atomic_uint_fast64_t deadline;
static atomic_bool done;
static atomic_uint signal_grace = 5;
static pthread_t watchdog;

static void signal_handler(int signal_number) { dsl_signal = signal_number; }

static void *watch(void *unused)
{
    (void)unused;
    uint64_t signal_deadline = 0;
    while (!atomic_load(&done)) {
        uint64_t now = dsl_now_ns(), bound = atomic_load(&deadline);
        if (dsl_signal && !signal_deadline) signal_deadline = now + (uint64_t)atomic_load(&signal_grace) * 1000000000;
        if (signal_deadline && now >= signal_deadline) {
            int flags = fcntl(STDERR_FILENO, F_GETFL);
            if (flags >= 0) fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK);
            dprintf(STDERR_FILENO, "dslcap: interrupted driver did not stop within signal grace; process exit releases USB resources\n");
            _exit(128 + dsl_signal);
        }
        if (!dsl_signal && bound && now >= bound) {
            int flags = fcntl(STDERR_FILENO, F_GETFL);
            if (flags >= 0) fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK);
            dprintf(STDERR_FILENO, "dslcap: driver watchdog timed out; process exit releases USB resources; verify reopen\n");
            _exit(DSL_DRIVER_TIMEOUT);
        }
        struct timespec delay = {.tv_nsec = 100000000};
        nanosleep(&delay, NULL);
    }
    return NULL;
}

void dsl_control_deadline(uint64_t absolute_ns) { atomic_store(&deadline, absolute_ns); }
void dsl_control_signal_grace(unsigned seconds) { atomic_store(&signal_grace, seconds); }

void dsl_control_bound(uint64_t seconds)
{
    uint64_t now = dsl_now_ns();
    uint64_t value = seconds > (UINT64_MAX - now) / 1000000000 ? UINT64_MAX :
                     now + seconds * UINT64_C(1000000000);
    atomic_store(&deadline, seconds ? value : 0);
}

int dsl_control_start(void)
{
    struct sigaction action = {.sa_handler = signal_handler};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL)) return -1;
    action.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &action, NULL)) return -1;
    atomic_store(&done, 0);
    dsl_control_bound(30);
    return pthread_create(&watchdog, NULL, watch, NULL) ? -1 : 0;
}

void dsl_control_finish(void)
{
    atomic_store(&done, 1);
    pthread_join(watchdog, NULL);
}
