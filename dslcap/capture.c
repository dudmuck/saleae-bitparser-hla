// SPDX-License-Identifier: GPL-3.0-or-later
#include "capture.h"
#include "control.h"
#include "convert.h"
#include "driver_config.h"
#include "ring.h"
#include "status.h"
#include <libsigrok.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

struct capture_state {
    struct dsl_converter converter;
    struct dsl_ring ring;
    atomic_int error, ended, running, data_end, device_event;
};
static struct capture_state state;
static _Atomic(struct capture_state *) current;

static void error(struct capture_state *state, int code)
{
    int expected = 0;
    atomic_compare_exchange_strong(&state->error, &expected, code);
}

static void receive(const struct sr_dev_inst *device, const struct sr_datafeed_packet *packet)
{
    (void)device;
    struct capture_state *state = atomic_load(&current);
    if (!state || atomic_load(&state->error)) return;
    if (!packet || packet->status != SR_PKT_OK) { error(state, DSL_DATA_ERROR); return; }
    if (packet->type == SR_DF_OVERFLOW) { error(state, DSL_FPGA_OVERFLOW); return; }
    if (packet->type == SR_DF_END) { atomic_store(&state->data_end, 1); return; }
    if (packet->type != SR_DF_LOGIC) return;
    const struct sr_datafeed_logic *logic = packet->payload;
    if (atomic_load(&state->data_end) || !logic || logic->format != LA_CROSS_DATA ||
        logic->length > SIZE_MAX || (logic->length && !logic->data)) {
        error(state, DSL_DATA_ERROR); return;
    }
    if (dsl_converter_feed(&state->converter, logic->data, (size_t)logic->length)) {
        int ring_error = dsl_ring_status(&state->ring);
        error(state, ring_error ? ring_error : DSL_DATA_ERROR);
    }
}

static void event(int event_code)
{
    struct capture_state *state = atomic_load(&current);
    if (!state) return;
    switch (event_code) {
    case DS_EV_DEVICE_RUNNING: atomic_store(&state->running, 1); break;
    case DS_EV_COLLECT_TASK_END: atomic_store(&state->ended, 1); break;
    case DS_EV_COLLECT_TASK_END_BY_ERROR:
    case DS_EV_COLLECT_TASK_END_BY_DETACHED:
    case DS_EV_DEVICE_SPEED_NOT_MATCH:
    case DS_EV_CURRENT_DEVICE_DETACH:
        atomic_store(&state->device_event, event_code);
        error(state, DSL_DEVICE_ERROR); atomic_store(&state->ended, 1); break;
    default: break;
    }
}

static const char *cause(int code)
{
    switch (code) {
    case DSL_RING_FULL: return "output ring full; consumer could not keep up";
    case DSL_FPGA_OVERFLOW: return "FPGA overflow; detection may be delayed, data since overflow is suspect";
    case DSL_DEVICE_ERROR: return "device collection error, detach or incompatible USB speed";
    case DSL_OUTPUT_ERROR: return "output write failed or consumer closed pipe";
    case DSL_DATA_ERROR: return "malformed/truncated input or incomplete finite capture";
    case DSL_DRAIN_TIMEOUT: return "output did not drain within 2s";
    default: return "capture failure";
    }
}

int dsl_capture(struct dsl_options *options)
{
    int rc = dsl_configure(options);
    if (rc) return rc;
    // One capture per process. Storage outlives async detach event callbacks.
    if (dsl_ring_start(&state.ring, (size_t)256 * 1024 * 1024, STDOUT_FILENO)) {
        fprintf(stderr, "dslcap: cannot allocate/start output ring\n");
        return DSL_OUTPUT_ERROR;
    }
    dsl_converter_init(&state.converter, options->channels,
                       options->continuous ? 0 : options->samples, dsl_ring_push, &state.ring);
    char metadata[80];
    int n = snprintf(metadata, sizeof(metadata), "META samplerate: %" PRIu64 "\n", options->rate);
    if (dsl_ring_push(&state.ring, (const uint8_t *)metadata, (size_t)n)) rc = DSL_OUTPUT_ERROR;
    atomic_store(&current, &state);
    ds_set_datafeed_callback(receive);
    ds_set_event_callback(event);
    if (!rc && ds_start_collect() != SR_OK) rc = DSL_DEVICE_ERROR;
    dsl_control_bound(30); // Driver start/arm/transfer setup may contain polling.
    int running = 0;
    while (!rc && !atomic_load(&state.ended)) {
        if (dsl_signal) { rc = 128 + dsl_signal; break; }
        int ring_error = dsl_ring_status(&state.ring);
        if (ring_error) error(&state, ring_error);
        rc = atomic_load(&state.error);
        if (!running && atomic_load(&state.running)) {
            running = 1;
            dsl_control_bound(options->continuous ? 0 : options->samples / options->rate + 30);
        }
        g_usleep(10000);
    }
    if (!rc) rc = atomic_load(&state.error);
    // Only the main control thread stops; ds_stop_collect joins the worker.
    dsl_control_bound(10);
    if (ds_is_collecting() && ds_stop_collect() != SR_OK && !rc) rc = DSL_DEVICE_ERROR;
    if (!rc && (!atomic_load(&state.data_end) || dsl_converter_finish(&state.converter))) {
        fprintf(stderr, "dslcap: finite/data-END validation failed: expected=%" PRIu64
                " decoded=%" PRIu64 " carry=%zu END=%d\n", state.converter.limit,
                state.converter.samples, state.converter.carry_size, atomic_load(&state.data_end));
        rc = DSL_DATA_ERROR;
    }
    int writer_error = dsl_ring_finish(&state.ring);
    if (!rc) rc = writer_error;
    if (state.ring.drain_timed_out)
        fprintf(stderr, "dslcap: drain timed out after 2s; buffered output truncated; preserving exit %d\n", rc);
    ds_set_datafeed_callback(NULL);
    // Static state remains valid even for a late asynchronous detach event.
    ds_set_event_callback(NULL);
    fprintf(stderr, "dslcap: samples=%" PRIu64 ", output bytes=%" PRIu64 ", ring high-water=%zu\n",
            state.converter.samples, state.ring.written, state.ring.high_water);
    if (rc == DSL_DEVICE_ERROR) {
        int code = atomic_load(&state.device_event);
        fprintf(stderr, "dslcap: device event=%d: %s\n", code,
                code == DS_EV_DEVICE_SPEED_NOT_MATCH ? "USB speed mismatch" :
                code == DS_EV_COLLECT_TASK_END_BY_DETACHED || code == DS_EV_CURRENT_DEVICE_DETACH ?
                "analyzer detached" : "collection failed");
    }
    if (rc && rc < 128) fprintf(stderr, "dslcap: %s (exit %d)\n", cause(rc), rc);
    else if (rc) fprintf(stderr, "dslcap: interrupted by signal %d\n", rc - 128);
    return rc;
}
