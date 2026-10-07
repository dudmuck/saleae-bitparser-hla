// SPDX-License-Identifier: GPL-3.0-or-later
#include "options.h"
#include "convert.h"
#include "ring.h"
#include "status.h"
#include "control.h"
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
struct output { uint8_t bytes[4096]; size_t size; int fail; };
static int emit(void *context, const uint8_t *data, size_t length)
{
    struct output *output = context;
    if (output->fail) return -1;
    CHECK(output->size + length <= sizeof(output->bytes));
    memcpy(output->bytes + output->size, data, length); output->size += length;
    return 0;
}

static void conversion(uint16_t mask, size_t split, uint64_t limit)
{
    uint16_t expected[256];
    uint8_t source[512] = {0}, packed[512];
    unsigned channels[16], count = 0, unitsize = mask & 0xff00 ? 2 : 1;
    for (unsigned i = 0; i < 16; i++) if (mask & (1u << i)) channels[count++] = i;
    // Independent sample-first oracle; combine walking bits, boundary changes
    // and a deterministic mixed-bit sequence. Encode raw bitplanes slowly.
    for (unsigned sample = 0; sample < 256; sample++) {
        expected[sample] = (uint16_t)(((sample * 19427u) ^ (sample << 7) ^ (1u << (sample % 16))) & mask);
        packed[sample * unitsize] = (uint8_t)expected[sample];
        if (unitsize == 2) packed[sample * unitsize + 1] = (uint8_t)(expected[sample] >> 8);
        for (unsigned ordinal = 0; ordinal < count; ordinal++)
            if (expected[sample] & (1u << channels[ordinal]))
                source[(sample / 64) * count * 8 + ordinal * 8 + (sample % 64) / 8] |= (uint8_t)(1u << (sample % 8));
    }
    struct output output = {0}; struct dsl_converter converter;
    CHECK(!dsl_converter_init(&converter, mask, limit, emit, &output));
    size_t total = count * 32, offset = 0;
    while (offset < total) {
        size_t n = split < total - offset ? split : total - offset;
        CHECK(!dsl_converter_feed(&converter, source + offset, n)); offset += n;
    }
    CHECK(!dsl_converter_finish(&converter));
    size_t samples = limit ? (size_t)limit : 256;
    CHECK(output.size == samples * unitsize);
    CHECK(!memcmp(output.bytes, packed, output.size));
}

static void conversion_tests(void)
{
    uint16_t masks[] = {0xff, 0xfff, 0xffff, 0x8209, 0x8000, 1};
    for (size_t m = 0; m < sizeof(masks)/sizeof(*masks); m++)
        for (size_t split = 1; split <= 133; split++) {
            conversion(masks[m], split, 0);
            conversion(masks[m], split, 1);
            conversion(masks[m], split, 65);
            conversion(masks[m], split, 191);
        }
    struct output output = {0}; struct dsl_converter c; uint8_t zeros[128] = {0};
    CHECK(!dsl_converter_init(&c, 0xfff, 64, emit, &output));
    CHECK(!dsl_converter_feed(&c, zeros, 95));
    CHECK(dsl_converter_finish(&c) == -1);
    CHECK(!dsl_converter_feed(&c, zeros, 1)); CHECK(!dsl_converter_finish(&c));
    CHECK(!dsl_converter_init(&c, 0xff, 100, emit, &output));
    CHECK(!dsl_converter_feed(&c, zeros, 64)); CHECK(dsl_converter_finish(&c) == -2);
    output.fail = 1;
    CHECK(!dsl_converter_init(&c, 0xff, 0, emit, &output));
    CHECK(dsl_converter_feed(&c, zeros, 64) == -1);
    CHECK(dsl_converter_feed(&c, NULL, 1) == -1);
}

static void parser_tests(void)
{
    uint16_t mask; uint64_t value;
    CHECK(!dsl_parse_channels("0,3,9,15", &mask) && mask == 0x8209);
    const char *bad_channels[] = {"", "0,", ",1", "-1", "0-16", "3-1", "1,1", "0-3,2", "0x1", "65536"};
    for (size_t i = 0; i < sizeof(bad_channels)/sizeof(*bad_channels); i++) CHECK(dsl_parse_channels(bad_channels[i], &mask));
    CHECK(!dsl_parse_quantity("25M", &value) && value == 25000000);
    CHECK(!dsl_parse_quantity("1.5K", &value) && value == 1500);
    const char *bad_quantities[] = {"0", "-1", "nan", "inf", "1.1", "1foo", " 1", "18446744073709551616", "1e100G"};
    for (size_t i = 0; i < sizeof(bad_quantities)/sizeof(*bad_quantities); i++) CHECK(dsl_parse_quantity(bad_quantities[i], &value));
}

static void fill_pipe(int fd)
{
    int flags = fcntl(fd, F_GETFL); CHECK(flags >= 0);
    CHECK(!fcntl(fd, F_SETFL, flags | O_NONBLOCK));
    uint8_t buffer[4096] = {0};
    while (write(fd, buffer, sizeof(buffer)) > 0) {}
    CHECK(!fcntl(fd, F_SETFL, flags));
}

static void ring_tests(void)
{
    signal(SIGPIPE, SIG_IGN);
    FILE *file = tmpfile(); CHECK(file);
    int fd = fileno(file), flags = fcntl(fd, F_GETFL);
    struct dsl_ring r; uint8_t bytes[200000];
    for (size_t i = 0; i < sizeof(bytes); i++) bytes[i] = (uint8_t)(i * 17 + (i >> 8));
    CHECK(!dsl_ring_start(&r, 262144, fd));
    CHECK(!dsl_ring_push(&r, bytes, 30001)); CHECK(!dsl_ring_push(&r, bytes + 30001, sizeof(bytes) - 30001));
    CHECK(!dsl_ring_finish(&r)); CHECK(fcntl(fd, F_GETFL) == flags);
    CHECK(!fseek(file, 0, SEEK_SET)); uint8_t output[sizeof(bytes)];
    CHECK(fread(output, 1, sizeof(output), file) == sizeof(output));
    CHECK(!memcmp(bytes, output, sizeof(bytes))); fclose(file);
    int pipefd[2]; CHECK(!pipe(pipefd)); close(pipefd[0]);
    CHECK(!dsl_ring_start(&r, 65536, pipefd[1])); CHECK(!dsl_ring_push(&r, bytes, 10));
    CHECK(dsl_ring_finish(&r) == DSL_OUTPUT_ERROR); close(pipefd[1]);
    CHECK(!pipe(pipefd)); fill_pipe(pipefd[1]); flags = fcntl(pipefd[1], F_GETFL);
    CHECK(!dsl_ring_start(&r, 65536, pipefd[1])); CHECK(!dsl_ring_push(&r, bytes, 32768));
    uint64_t begin = dsl_now_ns(); CHECK(dsl_ring_finish(&r) == DSL_DRAIN_TIMEOUT);
    CHECK(dsl_now_ns() - begin < UINT64_C(3000000000)); CHECK(fcntl(pipefd[1], F_GETFL) == flags);
    close(pipefd[0]); close(pipefd[1]);
    CHECK(!pipe(pipefd)); fill_pipe(pipefd[1]); CHECK(!dsl_ring_start(&r, 65536, pipefd[1]));
    int full = 0;
    for (int i = 0; i < 100; i++) if (dsl_ring_push(&r, bytes, 4096)) { full = 1; break; }
    CHECK(full); CHECK(dsl_ring_finish(&r) == DSL_RING_FULL); // Preserve first failure, not drain timeout.
    close(pipefd[0]); close(pipefd[1]);
}

static int discard(void *context, const uint8_t *data, size_t length)
{
    (void)data; *(uint64_t *)context += length; return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "--watchdog")) {
        CHECK(!dsl_control_start()); dsl_control_bound(1); sleep(4); return 1;
    }
    if (argc > 1 && !strcmp(argv[1], "--signal-watchdog")) {
        CHECK(!dsl_control_start()); raise(SIGTERM); sleep(8); return 1;
    }
    if (argc > 1 && !strcmp(argv[1], "--benchmark")) {
        uint8_t raw[65536] = {0}; uint64_t count = 0; struct dsl_converter c;
        CHECK(!dsl_converter_init(&c, 0xff, 0, discard, &count));
        uint64_t begin = dsl_now_ns();
        for (int i = 0; i < 4096; i++) CHECK(!dsl_converter_feed(&c, raw, sizeof(raw)));
        double seconds = (double)(dsl_now_ns() - begin) / 1e9;
        printf("8-channel transpose: %.1f MB/s (%.3fs, %llu bytes)\n", count / seconds / 1e6, seconds, (unsigned long long)count);
        return 0;
    }
    conversion_tests(); parser_tests(); ring_tests();
    puts("conversion (3192 split/trim cases), parser, writer/broken-pipe/backpressure tests passed");
    return 0;
}
