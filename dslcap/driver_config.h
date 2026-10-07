// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_DRIVER_CONFIG_H
#define DSLCAP_DRIVER_CONFIG_H
#include "options.h"
// Updates options to the driver's effective forced internal-pattern values.
int dsl_configure(struct dsl_options *options);
// Pure mode chooser against the pinned 0034 profile, usable before USB init.
int dsl_choose_mode(int stream, uint64_t rate, uint16_t mask);
#endif
