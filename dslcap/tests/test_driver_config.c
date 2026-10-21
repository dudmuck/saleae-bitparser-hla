// SPDX-License-Identifier: GPL-3.0-or-later
#include "driver_config.h"
#include <hardware/DSL/dsl.h>
#include <assert.h>

static void exact_rates(const struct DSL_profile *p)
{
    // Port of DSView dsl.c:1078-1085 for LOGIC. Decode the actual divider
    // registers rather than repeating the chooser's divisibility predicate.
    for (const uint64_t *r = p->dev_caps.samplerates; *r; r++)
        for (int stream = 0; stream <= 1; stream++)
            for (unsigned mask = 1; mask <= UINT16_MAX; mask++) {
                int id = dsl_choose_mode(p, stream, *r, (uint16_t)mask);
                if (id < 0) continue;
                const struct DSL_channels *m = &channel_modes[id];
                uint32_t tmp_u32 = (uint32_t)ceil(m->hw_max_samplerate * 1.0 / *r);
                uint16_t div_h = (tmp_u32 >= m->pre_div ? m->pre_div - 1U : tmp_u32 - 1U) << 8;
                tmp_u32 = (uint32_t)ceil(tmp_u32 * 1.0 / m->pre_div);
                uint16_t div_l = tmp_u32 & 0xffff;
                div_h += tmp_u32 >> 16;
                uint64_t divisor = ((div_h >> 8) + 1) *
                    (((uint64_t)(div_h & 0xff) << 16) | div_l);
                unsigned factor = 1;
                if (*r == p->dev_caps.half_samplerate)
                    factor = p->dev_caps.feature_caps & CAPS_FEATURE_ADF4360 ? 1 : 2;
                if (*r == p->dev_caps.quarter_samplerate)
                    factor = p->dev_caps.feature_caps & CAPS_FEATURE_ADF4360 ? 2 : 4;
                assert(m->hw_max_samplerate * factor == *r * divisor);
            }
}

static void u3_buffer(const struct DSL_profile *p, const struct DSL_profile *plus)
{
    // Fixed register images derived from dsl.c:1078-1085 (LOGIC, pre_div=5).
    const struct { uint64_t rate; unsigned div_h, div_l; } rates[] = {
        {1000000, 0x400, 100}, {2000000, 0x400, 50}, {4000000, 0x400, 25},
        {5000000, 0x400, 20}, {10000000, 0x400, 10}, {20000000, 0x400, 5},
        {25000000, 0x400, 4}, {50000000, 0x400, 2}, {100000000, 0x400, 1},
        {125000000, 0x300, 1}, {250000000, 0x100, 1}, {500000000, 0, 1},
        {1000000000, 0, 1},
    };
    for (size_t i = 0; i < ARRAY_SIZE(rates); i++) {
        uint64_t rate = rates[i].rate;
        uint16_t mask = rate == 1000000000 ? 0xff : 0xffff;
        int id = dsl_choose_mode(p, 0, rate, mask);
        assert(id == (rate == 1000000000 ? DSL_BUFFER1000x8 : DSL_BUFFER500x16));
        const struct DSL_channels *m = &channel_modes[id];
        unsigned tmp = (unsigned)ceil(m->hw_max_samplerate * 1.0 / rate);
        unsigned div_h = (tmp >= m->pre_div ? m->pre_div - 1U : tmp - 1U) << 8;
        tmp = (unsigned)ceil(tmp * 1.0 / m->pre_div);
        div_h |= tmp >> 16;
        assert(div_h == rates[i].div_h && (tmp & 0xffff) == rates[i].div_l);
    }
    for (const uint64_t *r = p->dev_caps.samplerates; *r; r++) {
        int listed = 0;
        for (size_t i = 0; i < ARRAY_SIZE(rates); i++) if (*r == rates[i].rate) listed = 1;
        assert((dsl_choose_mode(p, 0, *r, 1) >= 0) == listed);
    }
    for (unsigned lane = 8; lane < 16; lane++)
        assert(dsl_choose_mode(p, 0, 1000000000, (uint16_t)(1u << lane)) == -1);
    assert(dsl_choose_mode(p, 0, 500000000, 0xffff) >= 0);
    assert(dsl_choose_mode(p, 0, 1000000000, 0x1ff) == -1);
    for (unsigned count = 1; count <= 16; count++) {
        uint16_t mask = (uint16_t)((1u << count) - 1);
        uint64_t depth = (UINT64_C(2147483648) / count) & ~UINT64_C(1023);
        assert(dsl_buffer_depth(p, mask) == depth);
        assert(dsl_buffer_fits(p, mask, depth));
        assert(!dsl_buffer_fits(p, mask, depth + 1));
        size_t capacity;
        assert(!dsl_buffer_capacity(depth, count > 8 ? 2 : 1, 500000000, 0, &capacity));
        assert(capacity > depth * (count > 8 ? 2u : 1u));
    }
    assert(dsl_buffer_depth(p, 0xffff) == 134217728);
    assert(dsl_buffer_depth(p, 0xff) == 268435456);
    assert(dsl_buffer_depth(plus, 0xffff) == 16777216);
    assert(!dsl_buffer_fits(p, 0xffff, UINT64_MAX));
    assert(!dsl_buffer_fits(p, 0xffff, 0));
    struct dsl_options o = {.profile = p, .rate = 500000000, .samples = 134217728, .channels = 0xffff};
    assert(dsl_capture_budget(&o) == 108); // 30 + ceil(256 MiB/4 MB/s) + 10.
    o.rate = 1000000000; o.samples = 268435456; o.channels = 0xff;
    assert(dsl_capture_budget(&o) == 108);
    o.profile = plus; o.rate = 100000000; o.samples = 16777216; o.channels = 0xffff;
    assert(dsl_capture_budget(&o) == 30);
    o.continuous = 1;
    assert(dsl_capture_budget(&o) == 0);
}

int main(void)
{
    const struct DSL_profile *plus = dsl_profile(dsl_model_at(0)->pid, LIBUSB_SPEED_HIGH);
    const struct DSL_profile *super = dsl_profile(dsl_model_at(1)->pid, LIBUSB_SPEED_SUPER);
    const struct DSL_profile *high = dsl_profile(dsl_model_at(1)->pid, LIBUSB_SPEED_HIGH);
    assert(plus && super && high && super != high);
    assert(dsl_profile_needs_security(plus));
    assert(!dsl_profile_needs_security(super) && !dsl_profile_needs_security(high));
    assert(!dsl_profile(dsl_model_at(0)->pid, LIBUSB_SPEED_SUPER));
    // Fixed Plus expectations from the pre-Phase-1 chooser and existing CLI
    // capture/trigger/serial cases: mode identity, not just acceptance.
    const struct { int stream; uint64_t rate; uint16_t mask; int mode; } cases[] = {
        {1, 25000000, 0xff, DSL_STREAM25x12},
        {1, 25000000, 0xfff, DSL_STREAM25x12},
        {1, 20000000, 0xffff, DSL_STREAM20x16},
        {1, 25000000, 0x8209, DSL_STREAM25x12},
        {1, 50000000, 0xff, -1},
        {1, 60000000, 3, -1},
        {1, 3000000, 0xff, -1},
        {0, 100000000, 0xff, DSL_BUFFER100x16},
        {0, 100000000, 0x800f, DSL_BUFFER100x16},
        {0, 200000000, 0x8000, -1},
        {0, 200000000, 0xffff, -1},
        {0, 200000000, 0xff, DSL_BUFFER200x8},
        {0, 400000000, 0xff, -1},
        {0, 400000000, 0xf, DSL_BUFFER400x4},
        {1, 50000, 0xffff, DSL_STREAM20x16},
    };
    for (size_t i = 0; i < ARRAY_SIZE(cases); i++)
        assert(dsl_choose_mode(plus, cases[i].stream, cases[i].rate, cases[i].mask) == cases[i].mode);
    assert(dsl_choose_mode(high, 1, 20000000, 0xffff) == DSL_STREAM20x16_3DN2);
    assert(dsl_choose_mode(super, 1, 125000000, 0xffff) == DSL_STREAM125x16_16);
    assert(dsl_choose_mode(NULL, 1, 25000000, 0xff) == -1);
    assert(dsl_choose_mode(super, 1, 250000000, 0x0fff) == DSL_STREAM250x12_16);
    assert(dsl_choose_mode(super, 1, 250000000, 0x3f3f) == DSL_STREAM250x12_16);
    assert(dsl_choose_mode(super, 1, 500000000, 0x003f) == DSL_STREAM500x6);
    assert(dsl_choose_mode(super, 1, 500000000, 0x0025) == DSL_STREAM500x6);
    assert(dsl_choose_mode(super, 1, 1000000000, 0x0025) == DSL_STREAM1000x3);
    for (unsigned lane = 8; lane < 16; lane++)
        assert(dsl_choose_mode(super, 1, 1000000000, (uint16_t)(1u << lane)) == -1);
    assert(dsl_choose_mode(super, 1, 500000, 1) == -1);
    assert(dsl_choose_mode(plus, 1, 500000, 1) >= 0);
    assert(dsl_choose_mode(high, 1, 100000, 0xffff) == DSL_STREAM20x16_3DN2);
    assert(dsl_choose_mode(high, 1, 50000, 1) == -1);
    assert(dsl_choose_mode(super, 1, 250000000, 0x1fff) == -1);
    assert(dsl_choose_mode(super, 1, 500000000, 0x007f) == -1);
    assert(dsl_choose_mode(super, 1, 1000000000, 0x000f) == -1);
    exact_rates(plus); exact_rates(high); exact_rates(super);
    u3_buffer(super, plus);
    puts("Plus mode choices preserved; PID/speed profiles distinct");
    return 0;
}
