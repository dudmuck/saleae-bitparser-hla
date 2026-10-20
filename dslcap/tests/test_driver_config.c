// SPDX-License-Identifier: GPL-3.0-or-later
#include "driver_config.h"
#include <hardware/DSL/dsl.h>
#include <assert.h>

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
    puts("Plus mode choices preserved; PID/speed profiles distinct");
    return 0;
}
