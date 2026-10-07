// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_OPTIONS_H
#define DSLCAP_OPTIONS_H
#include <stdint.h>
#include <stddef.h>
struct dsl_serial_trigger {
    uint8_t channels[4]; // start, stop, clock, data (roles may share a channel).
    char conditions[3];
    uint16_t value;
    unsigned bits;
};
struct dsl_options {
    const char *fw_dir;
    uint64_t rate, samples;
    uint16_t channels;
    double vth, trigger_timeout, drain_timeout;
    uint64_t arm_limit, trigger_effective;
    char trigger_conditions[16];
    unsigned trigger_pos;
    int trigger, serial, timeout_set, timeout_upload;
    struct dsl_serial_trigger serial_trigger;
    int scan, verbosity, stream, continuous, pattern, capture;
};
// 0 success, 1 help printed, 2 invalid usage (diagnostic on stderr).
int dsl_parse_options(int argc, char **argv, struct dsl_options *options);
int dsl_parse_channels(const char *text, uint16_t *mask);
int dsl_parse_quantity(const char *text, uint64_t *value);
int dsl_parse_duration(const char *text, double *seconds, int allow_zero);
int dsl_buffer_capacity(uint64_t samples, unsigned width, uint64_t rate, int trigger, size_t *capacity);
uint64_t dsl_trigger_position(unsigned percent, uint64_t arm_limit, uint64_t channel_depth);
int dsl_parse_serial(const char *text, struct dsl_serial_trigger *serial);
void dsl_usage(void);
#endif
