// SPDX-License-Identifier: GPL-3.0-or-later
// GPCLK0 on GPIO20 only. Offline callers MUST use --dry-run.
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define PAGE_BYTES 4096
#define PERIPHERAL_BASE UINT64_C(0xfe000000)
#define CM_PAGE (PERIPHERAL_BASE + 0x101000)
#define GP0CTL (0x70 / 4)
#define GP0DIV (0x74 / 4)
#define FSEL20 (0x08 / 4)
#define PULL20 (0xe8 / 4)
#define LEVEL0 (0x34 / 4)
#define PASSWORD UINT32_C(0x5a000000)
#define LOW24 UINT32_C(0x00ffffff)
#define ENAB (1u << 4)
#define KILL (1u << 5)
#define BUSY (1u << 7)
#define BUSY_TIMEOUT UINT64_C(100000000)

static volatile uint32_t *cm, *gpio;
static uint32_t simulated_cm[PAGE_BYTES / 4], simulated_gpio[PAGE_BYTES / 4];
static volatile sig_atomic_t stopping;
static int dry_run, failed, owned;
static uint32_t saved_ctl, saved_div;

static void on_signal(int sig) { stopping = sig; }

static int now_ns(uint64_t *value)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); return -1; }
    *value = (uint64_t)t.tv_sec * 1000000000u + (unsigned)t.tv_nsec;
    return 0;
}

static int sleep_ns(uint64_t ns)
{
    struct timespec t = {.tv_sec = (time_t)(ns / 1000000000), .tv_nsec = (long)(ns % 1000000000)};
    while (nanosleep(&t, &t)) {
        if (errno != EINTR) { perror("nanosleep"); return -1; }
        // Cleanup waits must finish even if another termination signal arrives.
    }
    return 0;
}

static int write_clock(unsigned reg, uint32_t value)
{
    // These are the only permitted clock-manager writes.
    if (reg != GP0CTL && reg != GP0DIV) abort();
    value = PASSWORD | (value & LOW24);
    cm[reg] = dry_run ? value & LOW24 : value;
    if (printf("{\"event\":\"write\",\"register\":\"%s\",\"address\":\"0x%08" PRIx64 "\",\"value\":\"0x%08" PRIx32 "\"}\n",
               reg == GP0CTL ? "CM_GP0CTL" : "CM_GP0DIV", CM_PAGE + reg * 4, value) < 0) {
        failed = 1;
        return -1;
    }
    return 0;
}

static int write_gpio(unsigned reg, uint32_t value)
{
    if (reg != FSEL20 && reg != PULL20) abort();
    gpio[reg] = value;
    if (printf("{\"event\":\"write\",\"register\":\"%s\",\"offset\":\"0x%02x\",\"value\":\"0x%08" PRIx32 "\"}\n",
               reg == FSEL20 ? "GPFSEL2" : "GPIO_PUP_PDN_CNTRL_REG1", reg * 4, value) < 0) {
        failed = 1;
        return -1;
    }
    return 0;
}

static int gpio_function(unsigned function)
{
    // GPIO20 alone: GPFSEL2 bits 0..2. Every other field is preserved.
    return write_gpio(FSEL20, (gpio[FSEL20] & ~7u) | function);
}

static int gpio_pull(unsigned pull)
{
    // GPIO20 alone: pull register 1 bits 8..9. up=1, down=2.
    return write_gpio(PULL20, (gpio[PULL20] & ~(3u << 8)) | (pull << 8));
}

static int wait_idle(int interruptible)
{
    if (dry_run) {
        if (printf("{\"event\":\"wait_busy_clear\",\"timeout_ms\":100}\n") < 0) failed = 1;
        return failed ? -1 : 0;
    }
    uint64_t start, current;
    if (now_ns(&start)) return -1;
    while (cm[GP0CTL] & BUSY) {
        if (interruptible && stopping) return -1;
        if (now_ns(&current)) return -1;
        if (current - start >= BUSY_TIMEOUT) return 1;
        if (sleep_ns(100000)) return -1;
    }
    return 0;
}

static void stop_clock(void)
{
    if (!owned) return;
    // Isolate the pin before touching clock control, including error paths.
    if (gpio_function(0)) failed = 1;
    if (gpio_pull(2)) failed = 1;
    uint32_t ctl = cm[GP0CTL] & LOW24 & ~(ENAB | BUSY | KILL);
    if (write_clock(GP0CTL, ctl)) failed = 1;
    int idle = wait_idle(0);
    if (idle < 0) failed = 1;
    if (idle == 1) {
        fprintf(stderr, "GPCLK0 did not stop normally; bounded KILL recovery\n");
        if (write_clock(GP0CTL, ctl | KILL)) failed = 1;
        int killed = wait_idle(0);
        if (write_clock(GP0CTL, ctl)) failed = 1; // Always clear KILL.
        idle = wait_idle(0);
        if (killed != 0 || idle != 0) failed = 1;
    }
    if (idle == 0) {
        if (write_clock(GP0DIV, saved_div)) failed = 1;
        if (write_clock(GP0CTL, saved_ctl & ~ENAB)) failed = 1;
    } else {
        // Never change a divider/source if even KILL cannot stop the clock.
        fprintf(stderr, "GPCLK0 remains busy; GPIO20 is input, clock restoration incomplete\n");
        failed = 1;
    }
    if (!dry_run && sleep_ns(1000000)) failed = 1;
    if (dry_run && printf("{\"event\":\"settle\",\"milliseconds\":1}\n") < 0) failed = 1;
    unsigned function = gpio[FSEL20] & 7, pull = (gpio[PULL20] >> 8) & 3;
    unsigned level = (gpio[LEVEL0] >> 20) & 1;
    uint32_t actual_ctl = cm[GP0CTL] & LOW24, actual_div = cm[GP0DIV] & LOW24;
    int matches = function == 0 && pull == 2 && level == 0 &&
        actual_ctl == (saved_ctl & ~ENAB) && actual_div == saved_div;
    if (printf("{\"event\":\"readback\",\"gpio20_function\":%u,\"gpio20_pull\":%u,\"gpio20_level\":%u,"
               "\"ctl\":\"0x%08" PRIx32 "\",\"div\":\"0x%08" PRIx32 "\",\"matches\":%s}\n",
               function, pull, level, actual_ctl, actual_div, matches ? "true" : "false") < 0) failed = 1;
    if (!matches) { fprintf(stderr, "GPCLK0/GPIO20 restore readback mismatch\n"); failed = 1; }
    owned = 0;
}

static int read_file(const char *path, unsigned char *data, size_t capacity)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror(path); return -1; }
    size_t used = 0;
    while (used < capacity) {
        ssize_t n = read(fd, data + used, capacity - used);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { perror(path); close(fd); return -1; }
        if (!n) { close(fd); return (int)used; }
        used += (size_t)n;
    }
    close(fd);
    fprintf(stderr, "%s: property too long\n", path);
    return -1;
}

static uint32_t be32(const unsigned char *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

static unsigned cells(const char *path)
{
    unsigned char data[5];
    if (read_file(path, data, sizeof(data)) != 4) return 0;
    uint32_t n = be32(data);
    return n >= 1 && n <= 2 ? n : 0;
}

static uint64_t address(const unsigned char *p, unsigned n)
{
    return n == 2 ? (uint64_t)be32(p) << 32 | be32(p + 4) : be32(p);
}

static int platform_check(void)
{
    unsigned char data[4096];
    int length = read_file("/proc/device-tree/compatible", data, sizeof(data));
    int compatible = 0;
    if (length < 0) return -1;
    const char target[] = "brcm,bcm2711";
    for (int i = 0; i + (int)sizeof(target) <= length; i++)
        if ((!i || !data[i - 1]) && !memcmp(data + i, target, sizeof(target))) compatible = 1;
    if (!compatible) { fprintf(stderr, "refusing non-BCM2711 platform\n"); return -1; }
    unsigned child = cells("/proc/device-tree/soc/#address-cells");
    unsigned parent = cells("/proc/device-tree/#address-cells");
    unsigned size = cells("/proc/device-tree/soc/#size-cells");
    if (!child || !parent || !size) { fprintf(stderr, "unsupported device-tree cells\n"); return -1; }
    length = read_file("/proc/device-tree/soc/ranges", data, sizeof(data));
    unsigned stride = 4 * (child + parent + size);
    if (length <= 0 || (unsigned)length % stride) { fprintf(stderr, "malformed soc/ranges\n"); return -1; }
    int found = 0;
    for (unsigned i = 0; i < (unsigned)length; i += stride) {
        uint64_t bus = address(data + i, child);
        uint64_t arm = address(data + i + child * 4, parent);
        uint64_t span = address(data + i + (child + parent) * 4, size);
        if (bus <= UINT64_C(0x7e000000) && UINT64_C(0x7e000000) - bus < span) {
            uint64_t offset = UINT64_C(0x7e000000) - bus;
            if (arm > UINT64_MAX - offset || arm + offset != PERIPHERAL_BASE || found) {
                fprintf(stderr, "refusing unexpected/ambiguous peripheral base\n"); return -1;
            }
            found = 1;
        }
    }
    if (!found) fprintf(stderr, "no soc/ranges mapping for bus 0x7e000000\n");
    return found ? 0 : -1;
}

static volatile uint32_t *map_page(const char *device, uint64_t offset)
{
    int fd = open(device, O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) { perror(device); return NULL; }
    void *p = mmap(NULL, PAGE_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)offset);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return NULL; }
    return p;
}

int main(int argc, char **argv)
{
    unsigned source = 1, divi = 2;
    double duration = 0;
    enum { SOURCE = 256, DIVI, DURATION, DRY };
    const struct option options[] = {{"source", required_argument, NULL, SOURCE},
        {"divi", required_argument, NULL, DIVI}, {"duration", required_argument, NULL, DURATION},
        {"dry-run", no_argument, NULL, DRY}, {NULL, 0, NULL, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "", options, NULL)) != -1) {
        char *end;
        switch (c) {
        case SOURCE:
            if (!strcmp(optarg, "osc")) source = 1;
            else if (!strcmp(optarg, "plld")) source = 6;
            else goto usage;
            break;
        case DIVI: {
            errno = 0; unsigned long n = strtoul(optarg, &end, 10);
            if (errno || end == optarg || *end || *optarg == '-' || n < 2 || n > 4095) goto usage;
            divi = (unsigned)n;
            break;
        }
        case DURATION:
            errno = 0; duration = strtod(optarg, &end);
            if (errno || end == optarg || *end || !isfinite(duration) || duration <= 0 || duration > 60) goto usage;
            break;
        case DRY: dry_run = 1; break;
        default: goto usage;
        }
    }
    if (optind != argc || duration <= 0) goto usage;
    struct sigaction action = {.sa_handler = on_signal};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL) || sigaction(SIGPIPE, &action, NULL)) {
        perror("sigaction"); return 1;
    }
    if (dry_run) {
        // No platform files/devices are opened. These explicit simulated
        // snapshots make every restoration write numeric and testable.
        cm = simulated_cm; gpio = simulated_gpio;
        gpio[PULL20] = 2u << 8;
    } else {
        if (platform_check()) return 1;
        if (sysconf(_SC_PAGESIZE) != PAGE_BYTES) { fprintf(stderr, "requires 4096-byte pages\n"); return 1; }
        // The only /dev/mem mapping is this single clock-manager page.
        cm = map_page("/dev/mem", CM_PAGE);
        if (!cm) return 1;
        gpio = map_page("/dev/gpiomem", 0);
        if (!gpio) { munmap((void *)cm, PAGE_BYTES); return 1; }
    }
    saved_ctl = cm[GP0CTL] & LOW24; saved_div = cm[GP0DIV] & LOW24;
    if (saved_ctl & (ENAB | BUSY) || (gpio[FSEL20] & 7)) {
        fprintf(stderr, "refusing active GPCLK0 or non-input GPIO20\n"); failed = 1; goto done;
    }
    if (stopping) goto done;
    if (printf("{\"event\":\"header\",\"dry_run\":%s,\"source\":\"%s\",\"divi\":%u,\"nominal_hz\":%.9f,"
               "\"duration_s\":%.9f,\"cm_page\":\"0xfe101000\",\"saved_ctl\":\"0x%08" PRIx32 "\","
               "\"saved_div\":\"0x%08" PRIx32 "\",\"ctl_disabled\":\"0x%08x\",\"ctl_enabled\":\"0x%08x\","
               "\"divider\":\"0x%08x\",\"snapshot\":\"%s\"}\n",
               dry_run ? "true" : "false", source == 1 ? "osc" : "plld", divi,
               (source == 1 ? 54000000.0 : 750000000.0) / divi, duration,
               saved_ctl, saved_div, PASSWORD | source, PASSWORD | source | ENAB,
               PASSWORD | divi << 12, dry_run ? "simulated inactive clock/input pin/pull-down" : "hardware") < 0) {
        failed = 1; goto done;
    }
    owned = 1;
    if (write_clock(GP0CTL, source) || wait_idle(1) || stopping ||
        write_clock(GP0DIV, divi << 12) || write_clock(GP0CTL, source) ||
        write_clock(GP0CTL, source | ENAB) || stopping || gpio_pull(0) || gpio_function(2)) {
        failed = !stopping; goto done;
    }
    if (dry_run) {
        if (printf("{\"event\":\"run\",\"duration_s\":%.9f}\n", duration) < 0) failed = 1;
        if (printf("{\"event\":\"conditional_stop_fallback\",\"condition\":\"BUSY timeout\","
                   "\"kill_ctl\":\"0x%08x\",\"clear_kill_ctl\":\"0x%08x\","
                   "\"timeout_ms\":100,\"restore_only_when_idle\":true}\n",
                   PASSWORD | source | KILL, PASSWORD | source) < 0) failed = 1;
    } else {
        uint64_t start, current;
        if (now_ns(&start)) { failed = 1; goto done; }
        uint64_t end = start + (uint64_t)(duration * 1e9);
        while (!stopping) {
            if (now_ns(&current)) { failed = 1; break; }
            if (current >= end) break;
            uint64_t remaining = end - current;
            if (sleep_ns(remaining < 10000000 ? remaining : 10000000)) { failed = 1; break; }
        }
    }
done:
    stop_clock();
    if (!dry_run) {
        if (gpio) munmap((void *)gpio, PAGE_BYTES);
        if (cm) munmap((void *)cm, PAGE_BYTES);
    }
    if (fflush(stdout) || fflush(stderr)) failed = 1;
    return stopping ? 128 + stopping : failed ? 1 : 0;
usage:
    fprintf(stderr, "Usage: pi_gpclk --duration S (0 < S <= 60) [--source osc|plld] [--divi 2..4095] [--dry-run]\n");
    return 2;
}
