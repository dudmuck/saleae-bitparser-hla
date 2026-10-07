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
#include <string.h>
#include <unistd.h>

#ifndef DSLCAP_TEST_TIME_SCALE
#define DSLCAP_TEST_TIME_SCALE 1.0
#endif
static uint64_t span(double seconds) { return (uint64_t)(seconds * 1e9 * DSLCAP_TEST_TIME_SCALE); }
static uint64_t add_ns(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
struct capture_state {
    struct dsl_converter converter;
    struct dsl_ring ring;
    const struct dsl_options *options;
    pthread_mutex_t packet_mutex;
    atomic_int error, ended, running, data_end, device_event, bad_data;
    int header, logic_seen, committed, force_intent, complete;
    uint32_t trigger_status, real_pos;
    uint64_t actual, emitted, decoded, header_time;
};
static struct capture_state state;
static _Atomic(struct capture_state *) current;

static void error(struct capture_state *s, int code)
{
    if (code == DSL_DATA_ERROR) atomic_store(&s->bad_data, 1);
    int expected = 0;
    atomic_compare_exchange_strong(&s->error, &expected, code);
}

static void receive_locked(struct capture_state *s, const struct sr_datafeed_packet *packet)
{
    if (s->committed) return; // Abort discards pending driver data irreversibly.
    if (atomic_load(&s->error)) return;
    if (!packet || packet->status != SR_PKT_OK) { error(s, DSL_DATA_ERROR); return; }
    if (packet->type == SR_DF_OVERFLOW) { error(s, DSL_FPGA_OVERFLOW); return; }
    if (packet->type == SR_DF_END) {
        if (s->options->trigger) {
            if (!s->header || dsl_converter_finish(&s->converter)) error(s, DSL_DATA_ERROR);
            else s->complete = 1;
        }
        atomic_store(&s->data_end, 1); return;
    }
    if (packet->type == SR_DF_TRIGGER && s->options->trigger) {
        struct ds_trigger_pos header;
        if (!packet->payload || s->header || s->logic_seen || atomic_load(&s->data_end)) {
            error(s, DSL_DATA_ERROR); return;
        }
        // DSView frees the backing transfer immediately after this callback.
        memcpy(&header, packet->payload, sizeof(header));
        uint64_t remain = ((uint64_t)header.remain_cnt_h << 32) | header.remain_cnt_l;
        if (header.check_id != UINT32_C(0x55555555) || remain > s->options->arm_limit) {
            error(s, DSL_DATA_ERROR); return;
        }
        uint64_t actual = (s->options->arm_limit - remain) & ~UINT64_C(1023);
        uint64_t emitted = actual < s->options->samples ? actual : s->options->samples;
        if (!emitted || (!s->force_intent && actual < s->options->samples) ||
            ((header.status & 1) && header.real_pos >= emitted)) {
            error(s, DSL_DATA_ERROR); return;
        }
        s->converter.limit = emitted;
        char metadata[80];
        int n = (header.status & 1) ? snprintf(metadata, sizeof(metadata), "META trigger: %u\n", header.real_pos) :
                                    snprintf(metadata, sizeof(metadata), "META trigger: none\n");
        if (dsl_ring_push(&s->ring, (const uint8_t *)metadata, (size_t)n)) {
            error(s, dsl_ring_status(&s->ring)); return;
        }
        fprintf(stderr, "dslcap: trigger header remain=%" PRIu64 " actual=%" PRIu64 " emitted=%" PRIu64 " status=0x%x real-pos=%u\n",
                remain, actual, emitted, header.status, header.real_pos);
        s->actual = actual; s->emitted = emitted;
        s->trigger_status = header.status; s->real_pos = header.real_pos;
        s->header_time = dsl_now_ns();
        s->header = 1; // Mutex publication follows converter and META initialization.
        return;
    }
    if (packet->type != SR_DF_LOGIC) return;
    if (s->options->trigger && !s->header) { error(s, DSL_DATA_ERROR); return; }
    s->logic_seen = 1;
    const struct sr_datafeed_logic *logic = packet->payload;
    if (atomic_load(&s->data_end) || !logic || logic->format != LA_CROSS_DATA ||
        logic->length > SIZE_MAX || (logic->length && !logic->data)) {
        error(s, DSL_DATA_ERROR); return;
    }
    if (dsl_converter_feed(&s->converter, logic->data, (size_t)logic->length)) {
        int ring_error = dsl_ring_status(&s->ring);
        error(s, ring_error ? ring_error : DSL_DATA_ERROR);
    }
    s->decoded = s->converter.samples;
}

static void receive(const struct sr_dev_inst *device, const struct sr_datafeed_packet *packet)
{
    (void)device;
    struct capture_state *s = atomic_load(&current);
    if (!s) return;
    pthread_mutex_lock(&s->packet_mutex);
    receive_locked(s, packet);
    pthread_mutex_unlock(&s->packet_mutex);
}

static void event(int event_code)
{
    struct capture_state *s = atomic_load(&current);
    if (!s) return;
    switch (event_code) {
    case DS_EV_DEVICE_RUNNING: atomic_store(&s->running, 1); break;
    case DS_EV_COLLECT_TASK_END: atomic_store(&s->ended, 1); break;
    case DS_EV_COLLECT_TASK_END_BY_ERROR:
    case DS_EV_COLLECT_TASK_END_BY_DETACHED:
    case DS_EV_DEVICE_SPEED_NOT_MATCH:
    case DS_EV_CURRENT_DEVICE_DETACH:
        atomic_store(&s->device_event, event_code);
        error(s, DSL_DEVICE_ERROR); atomic_store(&s->ended, 1);
        if (s->ring.buffered) atomic_store(&s->ring.cancel, 1);
        break;
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
    case DSL_DRAIN_TIMEOUT: return "output drain stalled";
    case DSL_TRIGGER_TIMEOUT: return "No trigger within --trigger-timeout";
    default: return "capture failure";
    }
}

static int read_hit(struct sr_status *status)
{
    memset(status, 0, sizeof(*status));
    return ds_get_actived_device_status(status, TRUE) == SR_OK;
}

// Main owns device lifecycle; packet_mutex is never held during driver calls.
static int wait_trigger(struct capture_state *s)
{
    const struct dsl_options *o = s->options;
    uint64_t start = 0, deadline = 0, phase_start = 0, last_poll = 0;
    uint64_t grace = span(.340), unchanged_since = 0, longest = 0;
    uint32_t previous_count = 0;
    int previous_hit = 0, have_status = 0, phase = 0, forced = 0, rc = 0;
    unsigned status_failures = 0;
    // phase: 0 arm, 1 wait, 2 post-trigger, 3 force, 4 upload.
    while (!atomic_load(&s->ended)) {
        uint64_t now = dsl_now_ns();
        if (dsl_signal) { rc = 128 + dsl_signal; break; }
        int ring_error = dsl_ring_status(&s->ring);
        if (ring_error) error(s, ring_error);
        if ((rc = atomic_load(&s->error))) break;
        if (!phase && atomic_load(&s->running)) {
            start = now; phase = 1;
            deadline = o->timeout_set ? add_ns(start, add_ns(span(o->trigger_timeout), grace)) : 0;
            dsl_control_deadline(deadline ? add_ns(deadline, span(30)) : 0);
            unchanged_since = now;
        }
        pthread_mutex_lock(&s->packet_mutex);
        int header = s->header;
        int complete = s->complete;
        uint64_t actual = s->emitted, header_time = s->header_time;
        pthread_mutex_unlock(&s->packet_mutex);
        if (complete) break;
        if (header && phase != 4) {
            phase = 4; phase_start = header_time;
            deadline = phase_start + span((double)actual * (o->channels & 0xff00 ? 2 : 1) / 4000000 + 10);
            dsl_control_deadline(add_ns(deadline, span(10)));
        }
        int hit = 0;
        if (phase && phase < 4 && (!last_poll || now - last_poll >= span(.050))) {
            struct sr_status status;
            int ok = read_hit(&status);
            last_poll = now;
            if (!ok) status_failures++;
            else {
                hit = status.trig_hit & 1;
                uint32_t count = ((uint32_t)status.captured_cnt3 << 24) |
                    ((uint32_t)status.captured_cnt2 << 16) | ((uint32_t)status.captured_cnt1 << 8) | status.captured_cnt0;
                if (!have_status || previous_count != count || previous_hit != hit) {
                    if (now - unchanged_since > longest) longest = now - unchanged_since;
                    unchanged_since = now;
                    previous_count = count; previous_hit = hit; have_status = 1;
                    if (o->verbosity) fprintf(stderr, "dslcap: host-observed cached status change captured=%u hit=%d (refresh age unknown)\n", count, hit);
                }
            }
        }
        if (phase == 1 && hit) {
            phase = 2; phase_start = now;
            deadline = now + span((1 - o->trigger_pos / 100.0) * o->samples / o->rate + .340 + 5);
            dsl_control_deadline(add_ns(deadline, span(10)));
        }
        if (phase == 1 && deadline && now >= deadline) {
            // Reconcile callback publication AND the cached hit immediately before commitment.
            struct sr_status status;
            int ok = read_hit(&status);
            if (!ok) status_failures++;
            pthread_mutex_lock(&s->packet_mutex);
            if (s->header) { pthread_mutex_unlock(&s->packet_mutex); continue; }
            if (ok && (status.trig_hit & 1)) {
                pthread_mutex_unlock(&s->packet_mutex);
                phase = 2; deadline = now + span((1 - o->trigger_pos / 100.0) * o->samples / o->rate + .340 + 5);
                dsl_control_deadline(add_ns(deadline, span(10))); continue;
            }
            fprintf(stderr, "dslcap: timeout decision after nominal grace; %s; status-read failures=%u\n",
                    status_failures || now - unchanged_since >= grace ? "freshness unknown" : "host-observed cache changes", status_failures);
            if (!o->timeout_upload) {
                s->committed = 1;
                pthread_mutex_unlock(&s->packet_mutex);
                rc = DSL_TRIGGER_TIMEOUT; break;
            }
            // The config read can itself start callbacks: publish intent first.
            s->force_intent = 1;
            pthread_mutex_unlock(&s->packet_mutex);
            GVariant *value = NULL;
            int result = ds_get_actived_device_config(NULL, NULL, SR_CONF_WAIT_UPLOAD, &value);
            if (result != SR_OK || !value || !g_variant_is_of_type(value, G_VARIANT_TYPE_BOOLEAN)) {
                if (value) g_variant_unref(value);
                rc = DSL_DEVICE_ERROR; break;
            }
            forced = g_variant_get_boolean(value); g_variant_unref(value);
            pthread_mutex_lock(&s->packet_mutex);
            s->force_intent = forced;
            if (!forced && s->header && s->actual < o->samples) error(s, DSL_DATA_ERROR);
            pthread_mutex_unlock(&s->packet_mutex);
            phase = 3; phase_start = now;
            deadline = now + (forced ? span(10) : grace);
            dsl_control_deadline(add_ns(deadline, span(10)));
        } else if (phase > 1 && deadline && now >= deadline) {
            rc = phase == 3 && !forced ? DSL_DEVICE_ERROR : DSL_DATA_ERROR;
            break;
        }
        g_usleep(1000);
    }
    if (start) {
        uint64_t tail = dsl_now_ns() - unchanged_since;
        if (tail > longest) longest = tail;
        fprintf(stderr, "dslcap: cached status failures=%u longest unchanged=%.3fs; %s (nominal grace 340ms, no age guarantee)\n",
                status_failures, longest / 1e9, status_failures || longest >= grace ? "freshness unknown" : "host-observed cache changes");
    }
    return rc ? rc : atomic_load(&s->error);
}

int dsl_capture(struct dsl_options *options)
{
    int rc = dsl_configure(options);
    if (rc) return rc;
    state.options = options;
    if (pthread_mutex_init(&state.packet_mutex, NULL)) return DSL_CONFIG_ERROR;
    size_t capacity = (size_t)256 * 1024 * 1024;
    if (!options->stream && dsl_buffer_capacity(options->samples, options->channels & 0xff00 ? 2 : 1,
            options->rate, options->trigger, &capacity)) return DSL_CONFIG_ERROR;
    int ring_result = options->stream ? dsl_ring_start(&state.ring, capacity, STDOUT_FILENO) :
        dsl_ring_start_buffer(&state.ring, capacity, STDOUT_FILENO, options->drain_timeout);
    if (ring_result) {
        fprintf(stderr, "dslcap: cannot allocate/start output ring\n");
        return options->stream ? DSL_OUTPUT_ERROR : DSL_CONFIG_ERROR;
    }
    if (!options->stream) dsl_control_signal_grace(15);
    dsl_converter_init(&state.converter, options->channels,
                       options->continuous ? 0 : options->samples, dsl_ring_push, &state.ring);
    char metadata[80];
    int n = snprintf(metadata, sizeof(metadata), "META samplerate: %" PRIu64 "\n", options->rate);
    if (dsl_ring_push(&state.ring, (const uint8_t *)metadata, (size_t)n)) rc = DSL_OUTPUT_ERROR;
    atomic_store(&current, &state);
    ds_set_datafeed_callback(receive);
    ds_set_event_callback(event);
    dsl_control_bound(30);
    if (!rc && ds_start_collect() != SR_OK) rc = DSL_DEVICE_ERROR;
    if (!rc && options->trigger) rc = wait_trigger(&state);
    else {
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
    }
    if (!rc) rc = atomic_load(&state.error);
    dsl_control_bound(10);
    if (ds_is_collecting() && ds_stop_collect() != SR_OK) rc = DSL_DEVICE_ERROR;
    // Worker is joined before accessing converter state. Signal/device/data facts win.
    if (atomic_load(&state.device_event) || rc == DSL_DEVICE_ERROR) rc = DSL_DEVICE_ERROR;
    else if (atomic_load(&state.bad_data)) rc = DSL_DATA_ERROR;
    if (!rc && ((options->trigger && !state.header) || !atomic_load(&state.data_end) ||
                (options->trigger ? !state.complete : dsl_converter_finish(&state.converter)))) {
        fprintf(stderr, "dslcap: finite/data-END validation failed: expected=%" PRIu64
                " decoded=%" PRIu64 " carry=%zu END=%d\n", state.converter.limit,
                state.converter.samples, state.converter.carry_size, atomic_load(&state.data_end));
        rc = DSL_DATA_ERROR;
    }
    int writer_error = dsl_ring_finish(&state.ring);
    if (!rc) rc = writer_error;
    if (atomic_load(&state.device_event) || rc == DSL_DEVICE_ERROR) rc = DSL_DEVICE_ERROR;
    if (state.ring.drain_timed_out || state.ring.drain_cancelled)
        fprintf(stderr, "dslcap: buffered output truncated during drain; preserving exit %d\n", dsl_signal ? 128 + dsl_signal : rc);
    if (dsl_signal) rc = 128 + dsl_signal;
    ds_set_datafeed_callback(NULL);
    ds_set_event_callback(NULL);
    fprintf(stderr, "dslcap: samples=%" PRIu64 ", output bytes=%" PRIu64 ", ring high-water=%zu\n",
            state.converter.samples, state.ring.written, state.ring.high_water);
    if (rc == DSL_DEVICE_ERROR) fprintf(stderr, "dslcap: device event=%d\n", atomic_load(&state.device_event));
    if (rc && rc < 128) fprintf(stderr, "dslcap: %s (exit %d)\n", cause(rc), rc);
    else if (rc) fprintf(stderr, "dslcap: interrupted by signal %d\n", rc - 128);
    return rc;
}
