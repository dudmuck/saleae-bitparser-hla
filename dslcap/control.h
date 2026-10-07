// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef DSLCAP_CONTROL_H
#define DSLCAP_CONTROL_H
#include <signal.h>
#include <stdint.h>
extern volatile sig_atomic_t dsl_signal;
int dsl_control_start(void);
void dsl_control_bound(uint64_t seconds);
void dsl_control_deadline(uint64_t absolute_ns);
void dsl_control_signal_grace(unsigned seconds);
void dsl_control_finish(void);
#endif
