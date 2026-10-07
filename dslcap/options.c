// SPDX-License-Identifier: GPL-3.0-or-later
#include "options.h"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int dsl_parse_quantity(const char *text, uint64_t *value)
{
    if (!text || !isdigit((unsigned char)*text)) return -1;
    errno = 0;
    char *end;
    long double number = strtold(text, &end), multiplier = 1;
    if (*end) {
        if (end[1]) return -1;
        switch (*end) {
        case 'k': case 'K': multiplier = 1000; break;
        case 'M': multiplier = 1000000; break;
        case 'G': multiplier = 1000000000; break;
        default: return -1;
        }
    }
    number *= multiplier;
    if (errno || !isfinite(number) || number < 1 || number > UINT64_MAX ||
        floorl(number) != number) return -1;
    *value = (uint64_t)number;
    return 0;
}

int dsl_parse_channels(const char *text, uint16_t *mask)
{
    uint16_t result = 0;
    if (!text || !*text) return -1;
    while (*text) {
        if (!isdigit((unsigned char)*text)) return -1;
        char *end;
        errno = 0;
        unsigned long first = strtoul(text, &end, 10), last = first;
        if (errno || first > 15) return -1;
        if (*end == '-') {
            text = end + 1;
            if (!isdigit((unsigned char)*text)) return -1;
            last = strtoul(text, &end, 10);
            if (errno || last < first || last > 15) return -1;
        }
        for (unsigned long i = first; i <= last; i++) {
            uint16_t bit = (uint16_t)(1u << i);
            if (result & bit) return -1;
            result |= bit;
        }
        if (!*end) break;
        if (*end != ',' || !end[1]) return -1;
        text = end + 1;
    }
    *mask = result;
    return result ? 0 : -1;
}

void dsl_usage(void)
{
    puts("Usage: dslcap [--scan] [--fw-dir DIR] [-v|-vv]\n"
         "       dslcap --samplerate R --channels LIST (--samples N|--time T|--continuous)\n"
         "              [--mode stream|buffer] [--vth V]\n"
         "       dslcap --test-pattern\n"
         "Rates/sample counts accept K/M/G suffixes; time accepts s/ms/us.\n"
         "Defaults: 25M, channels 0-7, stream, VTH 1.6V. No options: bring-up only.\n"
         "Output: META samplerate: N, then physical channel bits in 1-byte samples\n"
         "or 2-byte little-endian samples when any selected channel is >=8.\n"
         "--test-pattern forces 16-channel buffer mode, 100M and driver depth.\n"
         "--scan verifies activation/security/HDL and prints the device.\n"
         "--fw-dir defaults to /usr/local/share/DSView/res. Logs use stderr.");
}

int dsl_parse_options(int argc, char **argv, struct dsl_options *o)
{
    *o = (struct dsl_options){.fw_dir = "/usr/local/share/DSView/res",
        .rate = 25000000, .channels = 0xff, .vth = 1.6, .stream = 1};
    enum { RATE = 256, CHANNELS, SAMPLES, TIME, CONTINUOUS, MODE, VTH, PATTERN };
    static const struct option options[] = {
        {"scan", no_argument, NULL, 's'}, {"fw-dir", required_argument, NULL, 'f'},
        {"help", no_argument, NULL, 'h'}, {"samplerate", required_argument, NULL, RATE},
        {"channels", required_argument, NULL, CHANNELS}, {"samples", required_argument, NULL, SAMPLES},
        {"time", required_argument, NULL, TIME}, {"continuous", no_argument, NULL, CONTINUOUS},
        {"mode", required_argument, NULL, MODE}, {"vth", required_argument, NULL, VTH},
        {"test-pattern", no_argument, NULL, PATTERN}, {NULL, 0, NULL, 0}};
    int option, duration_seen = 0;
    long double seconds = 0;
    optind = 1;
    while ((option = getopt_long(argc, argv, "vh", options, NULL)) != -1) {
        char *end;
        switch (option) {
        case 's': o->scan = 1; break;
        case 'f': o->fw_dir = optarg; break;
        case 'v': if (o->verbosity < 2) o->verbosity++; break;
        case 'h': dsl_usage(); return 1;
        case RATE: if (dsl_parse_quantity(optarg, &o->rate)) goto invalid; o->capture = 1; break;
        case CHANNELS: if (dsl_parse_channels(optarg, &o->channels)) goto invalid; o->capture = 1; break;
        case SAMPLES:
            if (duration_seen++ || dsl_parse_quantity(optarg, &o->samples)) goto invalid;
            o->capture = 1; break;
        case TIME:
            if (duration_seen++ || !isdigit((unsigned char)*optarg)) goto invalid;
            errno = 0; seconds = strtold(optarg, &end);
            if (!strcmp(end, "ms")) seconds /= 1000;
            else if (!strcmp(end, "us")) seconds /= 1000000;
            else if (*end && strcmp(end, "s")) goto invalid;
            if (errno || !isfinite(seconds) || seconds <= 0) goto invalid;
            o->capture = 1; break;
        case CONTINUOUS: if (duration_seen++) goto invalid; o->continuous = o->capture = 1; break;
        case MODE:
            if (!strcmp(optarg, "stream")) o->stream = 1;
            else if (!strcmp(optarg, "buffer")) o->stream = 0;
            else goto invalid;
            o->capture = 1; break;
        case VTH:
            errno = 0; o->vth = strtod(optarg, &end);
            if (errno || end == optarg || *end || !isfinite(o->vth) || o->vth < 0 || o->vth > 2.5) goto invalid;
            o->capture = 1; break;
        case PATTERN: o->pattern = o->capture = 1; break;
        default: goto invalid;
        }
    }
    if (optind != argc || (o->scan && o->capture)) goto invalid;
    if (o->pattern) {
        o->channels = 0xffff; o->stream = 0; o->continuous = 0;
    } else {
        if (o->capture && !duration_seen) goto invalid;
        if (o->continuous && !o->stream) goto invalid;
        if (seconds) {
            long double samples = ceill(seconds * o->rate);
            if (!isfinite(samples) || samples < 1 || samples > UINT64_MAX - 4096) goto invalid;
            o->samples = (uint64_t)samples;
        }
        if (o->samples > UINT64_MAX - 4096) goto invalid;
    }
    return 0;
invalid:
    fprintf(stderr, "dslcap: invalid capture arguments; use --help\n");
    return 2;
}
