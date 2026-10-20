// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_DRIVER_CONFIG_H
#define DSLCAP_DRIVER_CONFIG_H
#include "options.h"
struct DSL_profile;
struct dsl_model {
    uint16_t pid;
    const char *name, *bitstream;
    int needs_security;
};
// Table order is the default selection priority; option aliases use its indices.
const struct dsl_model *dsl_model_at(unsigned index);
const struct DSL_profile *dsl_profile(uint16_t pid, int usb_speed);
int dsl_profile_needs_security(const struct DSL_profile *profile);
// Updates options to the driver's effective forced internal-pattern values.
int dsl_configure(struct dsl_options *options);
// Pure mode chooser against an explicit static profile; no USB access.
int dsl_choose_mode(const struct DSL_profile *profile, int stream, uint64_t rate, uint16_t mask);
#endif
