// SPDX-License-Identifier: GPL-3.0-or-later
// BCM2711 W1 generator. Offline use MUST specify --dry-run.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static const unsigned pins[] = {21, 19, 20, 16, 13, 6, 22, 25};
#define ALLOWED_MASK ((1u<<6)|(1u<<13)|(1u<<16)|(1u<<19)|(1u<<20)|(1u<<21)|(1u<<22)|(1u<<25))
static volatile uint32_t *gpio;
static volatile sig_atomic_t stopping;
static int dry_run, changed, failed;
static uint64_t index_state, max_states, simulated_ns, deadline;
static unsigned step_us = 10;
static FILE *manifest;
static unsigned previous_state;
static void restore(void);

static uint64_t now_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); restore(); exit(1); }
    return (uint64_t)t.tv_sec * 1000000000u + (unsigned)t.tv_nsec;
}

static void stop_signal(int sig) { stopping = sig; }

static unsigned function(unsigned pin)
{
    return (gpio[pin / 10] >> (3 * (pin % 10))) & 7;
}

static void set_function(unsigned pin, unsigned value)
{
    unsigned shift = 3 * (pin % 10);
    gpio[pin / 10] = (gpio[pin / 10] & ~(7u << shift)) | value << shift;
}

static void set_pull(unsigned pin, unsigned value)
{
    unsigned reg = 0xe4 / 4 + pin / 16, shift = 2 * (pin % 16);
    gpio[reg] = (gpio[reg] & ~(3u << shift)) | value << shift;
}

static void restore(void)
{
    if (!gpio) return;
    if (changed) {
        for (unsigned i = 0; i < 8; i++) {
            set_function(pins[i], 0);
            set_pull(pins[i], pins[i] == 6 ? 1 : 2); // BCM2711 up=1, down=2.
        }
        // Let the input levels settle under the restored pulls before reading
        // back configuration and GPLEV0, including on signal cleanup paths.
        struct timespec settle = {.tv_sec = 0, .tv_nsec = 1000000};
        while (nanosleep(&settle, &settle) && errno == EINTR) {}
        for (unsigned i = 0; i < 8; i++) {
            unsigned pin = pins[i], pull = (gpio[0xe4 / 4 + pin / 16] >> (2 * (pin % 16))) & 3;
            if (function(pin) || pull != (pin == 6 ? 1u : 2u)) {
                fprintf(stderr, "restore verification failed GPIO%u\n", pin);
                failed = 1;
            }
        }
        fprintf(stderr, "restored W1 inputs/default pulls; levels=0x%08x\n", gpio[0x34 / 4] & ALLOWED_MASK);
    }
    munmap((void *)gpio, 4096);
    gpio = NULL;
}

static int running(void)
{
    return !stopping && !failed && (!max_states || index_state < max_states) &&
        (dry_run ? simulated_ns < deadline : now_ns() < deadline);
}

static int emit(unsigned state, int delay)
{
    if (!running()) return 0;
    if (dry_run) {
        if (printf("{\"index\":%llu,\"state\":%u}\n", (unsigned long long)index_state, state) < 0)
            failed = 1;
        simulated_ns += (uint64_t)step_us * 1000;
    } else {
        uint32_t bits = 0;
        for (unsigned i = 0; i < 8; i++) if (state & (1u << i)) bits |= 1u << pins[i];
        uint32_t previous_bits = 0;
        for (unsigned i = 0; i < 8; i++) if (previous_state & (1u << i)) previous_bits |= 1u << pins[i];
        // Always clear before set, skipping empty masks. An analyzer can
        // observe only previous_state & state between the two writes.
        // SPI emits data setup, SCLK rise, SCLK fall as separate states;
        // its SCLK writes never change data at the same time.
        uint32_t clear_mask = previous_bits & ~bits, set_mask = bits & ~previous_bits;
        if (clear_mask) gpio[0x28 / 4] = clear_mask;
        if (set_mask) gpio[0x1c / 4] = set_mask;
        if (delay) {
            struct timespec t = {.tv_sec = step_us / 1000000, .tv_nsec = (step_us % 1000000) * 1000};
            while (nanosleep(&t, &t) && errno == EINTR && !stopping) {}
        }
    }
    previous_state = state;
    index_state++;
    return !failed;
}

static unsigned prbs8(unsigned state)
{
    unsigned feedback = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1;
    return ((state << 1) | feedback) & 255;
}

static unsigned prbs9_byte(unsigned *state)
{
    unsigned byte = 0;
    for (unsigned bit = 0; bit < 8; bit++) {
        byte = byte << 1 | (*state & 1);
        unsigned feedback = ((*state >> 8) ^ (*state >> 4)) & 1;
        *state = ((*state << 1) | feedback) & 511;
    }
    return byte;
}

static int spi_frame(unsigned frame, unsigned *mosi_seed, unsigned *miso_seed)
{
    unsigned mosi[16], miso[16];
    for (unsigned i = 0; i < 16; i++) {
        mosi[i] = prbs9_byte(mosi_seed); miso[i] = prbs9_byte(miso_seed);
    }
    if (!emit(8, 1) || !emit(0, 1)) return 0;
    for (unsigned i = 0; i < 16; i++)
        for (int bit = 7; bit >= 0; bit--) {
            unsigned data = ((mosi[i] >> bit) & 1) << 2 | ((miso[i] >> bit) & 1) << 1;
            if (!emit(data, 1) || !emit(data | 1, 1) || !emit(data, 1)) return 0;
        }
    if (!emit(8, 1)) return 0;
    if (fprintf(manifest, "{\"frame\":%u,\"mosi\":\"", frame) < 0) failed = 1;
    for (unsigned i = 0; i < 16; i++) if (fprintf(manifest, "%02x", mosi[i]) < 0) failed = 1;
    if (fprintf(manifest, "\",\"miso\":\"") < 0) failed = 1;
    for (unsigned i = 0; i < 16; i++) if (fprintf(manifest, "%02x", miso[i]) < 0) failed = 1;
    if (fprintf(manifest, "\"}\n") < 0 || fflush(manifest)) failed = 1;
    // Fixed inter-frame gap, 16 state intervals including the NSS rise.
    for (unsigned i = 0; i < 15; i++) if (!emit(8, 1)) return 0;
    return !failed;
}

static unsigned long number(const char *text)
{
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 0);
    if (!*text || *text == '-' || *end || errno) { fprintf(stderr, "invalid integer: %s\n", text); exit(2); }
    return value;
}

int main(int argc, char **argv)
{
    const char *pattern = NULL, *manifest_path = NULL;
    double duration = 0;
    unsigned seed = 1;
    uint32_t mask = ALLOWED_MASK;
    enum { DURATION = 256, STEP, SEED, DRY, MANIFEST, STATES, MASK };
    const struct option options[] = {
        {"duration", required_argument, NULL, DURATION}, {"step-us", required_argument, NULL, STEP},
        {"seed", required_argument, NULL, SEED}, {"dry-run", no_argument, NULL, DRY},
        {"manifest", required_argument, NULL, MANIFEST}, {"states", required_argument, NULL, STATES},
        {"mask", required_argument, NULL, MASK}, {NULL, 0, NULL, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "", options, NULL)) != -1) {
        switch (c) {
        case DURATION: {
            char *end; errno = 0; duration = strtod(optarg, &end);
            if (errno || end == optarg || *end || !isfinite(duration) || duration <= 0 || duration > 3600) return 2;
            break;
        }
        case STEP: { unsigned long v = number(optarg); if (v < 10 || v > 1000000) return 2; step_us = (unsigned)v; break; }
        case SEED: { unsigned long v = number(optarg); if (!v || v > 255) return 2; seed = (unsigned)v; break; }
        case STATES: max_states = number(optarg); if (!max_states) return 2; break;
        case MASK: { unsigned long v = number(optarg); if (v > UINT32_MAX) return 2; mask = (uint32_t)v; break; }
        case DRY: dry_run = 1; break;
        case MANIFEST: manifest_path = optarg; break;
        default: return 2;
        }
    }
    if (optind + 1 == argc) pattern = argv[optind];
    if (!pattern || (strcmp(pattern, "walk") && strcmp(pattern, "prbs") && strcmp(pattern, "spi")) || duration <= 0) {
        fprintf(stderr, "Usage: pi_pattern walk|prbs|spi --duration SECONDS [--dry-run] [--step-us N>=10] [--states N] [--seed 1..255] [--manifest FILE]\n");
        return 2;
    }
    if (mask & ~ALLOWED_MASK) { fprintf(stderr, "refusing GPIO outside W1 allowed mask\n"); return 2; }
    if (mask != ALLOWED_MASK) { fprintf(stderr, "patterns require all eight W1 lines\n"); return 2; }
    if (!strcmp(pattern, "spi") && !dry_run && !manifest_path) { fprintf(stderr, "spi requires --manifest FILE\n"); return 2; }
    manifest = stdout;
    if (manifest_path && !(manifest = fopen(manifest_path, "wx"))) { perror("manifest"); return 1; }
    struct sigaction action = {.sa_handler = stop_signal};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL) || sigaction(SIGPIPE, &action, NULL)) { perror("sigaction"); return 1; }
    if (!dry_run) {
        int fd = open("/dev/gpiomem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (fd < 0) { perror("/dev/gpiomem"); return 1; }
        void *mapping = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (mapping == MAP_FAILED) { perror("mmap"); return 1; }
        gpio = mapping;
        for (unsigned i = 0; i < 8; i++)
            if (function(pins[i])) { fprintf(stderr, "refusing non-input GPIO%u\n", pins[i]); restore(); return 1; }
        if (stopping) { restore(); return 128 + stopping; }
        changed = 1;
        // Establish initial idle levels before enabling any output.
        gpio[0x28 / 4] = ALLOWED_MASK;
        if (!strcmp(pattern, "spi")) gpio[0x1c / 4] = 1u << pins[3];
        for (unsigned i = 0; i < 8 && !stopping; i++) { set_pull(pins[i], 0); set_function(pins[i], 1); }
    }
    previous_state = !strcmp(pattern, "spi") ? 8 : 0;
    deadline = (uint64_t)(duration * 1e9) + (dry_run ? 0 : now_ns());
    FILE *header = dry_run ? stdout : stderr;
    if (fprintf(header, "{\"pattern\":\"%s\",\"seed\":%u,\"polynomial\":\"x^8+x^6+x^5+x^4+1\",\"spi_seeds\":[1,341],\"spi_polynomial\":\"x^9+x^5+1\",\"frame_bytes\":16,\"step_us\":%u,\"write_order\":[\"GPCLR0\",\"GPSET0\"],\"intermediate\":\"prev & next\",\"pins\":[21,19,20,16,13,6,22,25]}\n", pattern, seed, step_us) < 0) failed = 1;
    if (!strcmp(pattern, "walk")) {
        while (running()) {
            for (unsigned i = 0; i < 8 && running(); i++) emit(1u << i, 1);
            for (unsigned i = 0; i < 256 && running(); i++) emit(i ^ (i >> 1), 1);
        }
    } else if (!strcmp(pattern, "prbs")) {
        unsigned state = seed;
        while (running()) {
            emit(state, 0);
            unsigned next = prbs8(state);
            while (next == state) next = prbs8(next);
            state = next;
        }
    } else {
        unsigned mosi = 1, miso = 341, frame = 0;
        while (running() && spi_frame(frame, &mosi, &miso)) frame++;
    }
    restore();
    if (manifest_path && fclose(manifest)) failed = 1;
    if (fflush(stdout) || fflush(stderr)) failed = 1;
    return stopping ? 128 + stopping : failed ? 1 : 0;
}
