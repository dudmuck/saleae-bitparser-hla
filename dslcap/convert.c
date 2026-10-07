// SPDX-License-Identifier: GPL-3.0-or-later
#include "convert.h"
#include <string.h>

static uint64_t transpose8(uint64_t x)
{
    uint64_t t = (x ^ (x >> 7)) & UINT64_C(0x00aa00aa00aa00aa);
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & UINT64_C(0x0000cccc0000cccc);
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & UINT64_C(0x00000000f0f0f0f0);
    return x ^ t ^ (t << 28);
}

int dsl_converter_init(struct dsl_converter *c, uint16_t mask, uint64_t limit,
                       dsl_emit_fn emit, void *context)
{
    if (!mask || !emit) return -1;
    memset(c, 0, sizeof(*c));
    for (unsigned i = 0; i < 16; i++)
        if (mask & (1u << i)) c->channels[c->count++] = i;
    c->unitsize = mask & 0xff00 ? 2 : 1;
    c->limit = limit; c->emit = emit; c->context = context;
    return 0;
}

static int group(struct dsl_converter *c, const uint8_t *input)
{
    size_t samples = 64;
    if (c->limit && c->limit - c->samples < samples)
        samples = (size_t)(c->limit - c->samples);
    for (unsigned byte = 0; byte < 8 && byte * 8 < samples; byte++) {
        uint64_t low = 0, high = 0;
        for (unsigned ch = 0; ch < c->count; ch++) {
            unsigned phys = c->channels[ch];
            uint64_t bits = (uint64_t)input[ch * 8 + byte] << ((phys & 7) * 8);
            if (phys < 8) low |= bits; else high |= bits;
        }
        low = transpose8(low); high = transpose8(high);
        for (unsigned bit = 0; bit < 8 && byte * 8 + bit < samples; bit++) {
            c->output[c->output_size++] = (uint8_t)(low >> (bit * 8));
            if (c->unitsize == 2) c->output[c->output_size++] = (uint8_t)(high >> (bit * 8));
        }
    }
    c->samples += samples;
    if (c->output_size + 128 > sizeof(c->output)) {
        if (c->emit(c->context, c->output, c->output_size)) return -1;
        c->output_size = 0;
    }
    return 0;
}

int dsl_converter_feed(struct dsl_converter *c, const void *data, size_t length)
{
    if (length && !data) return -1;
    if (!length) return 0;
    const uint8_t *input = data;
    size_t group_size = c->count * 8;
    if (c->carry_size) {
        size_t n = group_size - c->carry_size;
        if (n > length) n = length;
        memcpy(c->carry + c->carry_size, input, n);
        c->carry_size += n; input += n; length -= n;
        if (c->carry_size == group_size) {
            if (group(c, c->carry)) return -1;
            c->carry_size = 0;
        }
    }
    while (length >= group_size) {
        if (group(c, input)) return -1;
        input += group_size; length -= group_size;
    }
    if (length) {
        memcpy(c->carry, input, length);
        c->carry_size = length;
    }
    if (c->output_size) {
        if (c->emit(c->context, c->output, c->output_size)) return -1;
        c->output_size = 0;
    }
    return 0;
}

int dsl_converter_finish(const struct dsl_converter *c)
{
    if (c->carry_size) return -1;
    if (c->limit && c->samples != c->limit) return -2;
    return 0;
}
