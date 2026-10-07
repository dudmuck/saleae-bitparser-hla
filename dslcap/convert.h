// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_CONVERT_H
#define DSLCAP_CONVERT_H
#include <stddef.h>
#include <stdint.h>
typedef int (*dsl_emit_fn)(void *context, const uint8_t *data, size_t length);
struct dsl_converter {
    unsigned int channels[16], count, unitsize;
    uint8_t carry[128], output[16384];
    size_t carry_size, output_size;
    uint64_t samples, limit;
    dsl_emit_fn emit;
    void *context;
};
int dsl_converter_init(struct dsl_converter *c, uint16_t mask, uint64_t limit,
                       dsl_emit_fn emit, void *context);
int dsl_converter_feed(struct dsl_converter *c, const void *data, size_t length);
// 0 complete, -1 incomplete source group, -2 finite sample count mismatch.
int dsl_converter_finish(const struct dsl_converter *c);
#endif
