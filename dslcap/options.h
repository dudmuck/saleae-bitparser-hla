// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_OPTIONS_H
#define DSLCAP_OPTIONS_H
#include <stdint.h>
struct dsl_options {
    const char *fw_dir;
    uint64_t rate, samples;
    uint16_t channels;
    double vth;
    int scan, verbosity, stream, continuous, pattern, capture;
};
// 0 success, 1 help printed, 2 invalid usage (diagnostic on stderr).
int dsl_parse_options(int argc, char **argv, struct dsl_options *options);
int dsl_parse_channels(const char *text, uint16_t *mask);
int dsl_parse_quantity(const char *text, uint64_t *value);
void dsl_usage(void);
#endif
