// SPDX-License-Identifier: GPL-3.0-or-later
#include "options.h"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <inttypes.h>
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

int dsl_parse_duration(const char *text, double *seconds, int allow_zero)
{
    if (!text || !isdigit((unsigned char)*text)) return -1;
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    if (!strcmp(end, "ms")) value /= 1000;
    else if (!strcmp(end, "us")) value /= 1000000;
    else if (*end && strcmp(end, "s")) return -1;
    if (errno || !isfinite(value) || value < 0 || (!allow_zero && !value)) return -1;
    *seconds = value;
    return 0;
}

static int parse_trigger(const char *text, struct dsl_options *o)
{
    if (o->trigger || !text || !*text) return -1;
    while (*text) {
        char *end;
        if (!isdigit((unsigned char)*text)) return -1;
        unsigned long ch = strtoul(text, &end, 10);
        if (ch > 15 || *end++ != ':' || !*end || o->trigger_conditions[ch]) return -1;
        char condition = *end++;
        switch (condition) {
        case 'r': case 'R': condition = 'R'; break;
        case 'f': case 'F': condition = 'F'; break;
        case '1': case 'h': condition = '1'; break;
        case '0': case 'l': condition = '0'; break;
        case 'e': condition = 'C'; break;
        default: return -1;
        }
        o->trigger_conditions[ch] = condition;
        if (!*end) break;
        if (*end++ != ',' || !*end) return -1;
        text = end;
    }
    o->trigger = 1;
    return 0;
}

int dsl_buffer_capacity(uint64_t samples, unsigned width, uint64_t rate, int trigger, size_t *capacity)
{
    char line[80];
    int first = snprintf(line, sizeof(line), "META samplerate: %" PRIu64 "\n", rate);
    int second = trigger ? snprintf(line, sizeof(line), "META trigger: %" PRIu64 "\n",
            samples ? samples - 1 : 0) : 0;
    if (trigger && second < (int)strlen("META trigger: none\n")) second = (int)strlen("META trigger: none\n");
    size_t extra = (size_t)(first + second);
    if (!samples || (width != 1 && width != 2) || samples > (SIZE_MAX - extra) / width) return -1;
    *capacity = (size_t)samples * width + extra;
    return 0;
}

uint64_t dsl_trigger_position(unsigned percent, uint64_t arm_limit, uint64_t channel_depth)
{
    uint64_t position = (uint32_t)(percent / 100.0 * arm_limit);
    if (position < 64) position = 64;
    uint64_t cap = channel_depth * 90 / 100;
    if (position > cap) position = cap;
    return position & ~UINT64_C(63);
}

void dsl_usage(void)
{
    puts("Usage: dslcap [--scan] [--fw-dir DIR] [-v|-vv]\n"
         "       dslcap --samplerate R --channels LIST (--samples N|--time T|--continuous)\n"
         "              [--mode stream|buffer] [--vth V]\n"
         "              [--trigger CH:COND,...] [--trigger-pos 0..90]\n"
         "              [--trigger-timeout T] [--on-timeout fail|upload] [--drain-timeout S]\n"
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
        .rate = 25000000, .channels = 0xff, .vth = 1.6, .stream = 1, .trigger_pos = 10, .drain_timeout = 30};
    enum { RATE = 256, CHANNELS, SAMPLES, TIME, CONTINUOUS, MODE, VTH, PATTERN, TRIGGER, POS, TIMEOUT, ON_TIMEOUT, DRAIN };
    static const struct option options[] = {
        {"scan", no_argument, NULL, 's'}, {"fw-dir", required_argument, NULL, 'f'},
        {"help", no_argument, NULL, 'h'}, {"samplerate", required_argument, NULL, RATE},
        {"channels", required_argument, NULL, CHANNELS}, {"samples", required_argument, NULL, SAMPLES},
        {"time", required_argument, NULL, TIME}, {"continuous", no_argument, NULL, CONTINUOUS},
        {"mode", required_argument, NULL, MODE}, {"vth", required_argument, NULL, VTH},
        {"test-pattern", no_argument, NULL, PATTERN},
        {"trigger", required_argument, NULL, TRIGGER}, {"trigger-pos", required_argument, NULL, POS},
        {"trigger-timeout", required_argument, NULL, TIMEOUT}, {"on-timeout", required_argument, NULL, ON_TIMEOUT},
        {"drain-timeout", required_argument, NULL, DRAIN}, {NULL, 0, NULL, 0}};
    int option, duration_seen = 0, trigger_modifiers = 0, drain_seen = 0;
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
        case TRIGGER: if (parse_trigger(optarg, o)) goto invalid; o->capture = 1; break;
        case POS: {
            if (!isdigit((unsigned char)*optarg)) goto invalid;
            errno = 0;
            unsigned long pos = strtoul(optarg, &end, 10);
            if (errno || *end || pos > 90) goto invalid;
            o->trigger_pos = (unsigned)pos; trigger_modifiers = 1; break;
        }
        case TIMEOUT:
            if (dsl_parse_duration(optarg, &o->trigger_timeout, 1) || o->trigger_timeout > UINT64_MAX / 1e9 - 60) goto invalid;
            o->timeout_set = trigger_modifiers = 1; break;
        case ON_TIMEOUT:
            if (!strcmp(optarg, "upload")) o->timeout_upload = 1;
            else if (strcmp(optarg, "fail")) goto invalid;
            trigger_modifiers = 1; break;
        case DRAIN:
            if (dsl_parse_duration(optarg, &o->drain_timeout, 0) ||
                o->drain_timeout < 1 || o->drain_timeout > 3600) goto invalid;
            drain_seen = 1; break;
        case PATTERN: o->pattern = o->capture = 1; break;
        default: goto invalid;
        }
    }
    if (optind != argc || (o->scan && o->capture) || (trigger_modifiers && !o->trigger) ||
        (drain_seen && o->stream) || (o->trigger && (o->stream || o->continuous || o->pattern))) goto invalid;
    for (unsigned ch = 0; ch < 16; ch++)
        if (o->trigger_conditions[ch] && !(o->channels & (1u << ch))) goto invalid;
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
