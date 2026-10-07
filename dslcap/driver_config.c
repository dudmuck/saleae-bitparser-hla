// SPDX-License-Identifier: GPL-3.0-or-later
#include "driver_config.h"
#include "status.h"
#include <hardware/DSL/dsl.h>

static const struct DSL_profile *profile(void)
{
    for (size_t i = 0; supported_DSLogic[i].vid; i++)
        if (supported_DSLogic[i].vid == 0x2a0e && supported_DSLogic[i].pid == 0x0034)
            return &supported_DSLogic[i];
    return NULL;
}

int dsl_choose_mode(int stream, uint64_t rate, uint16_t mask)
{
    const struct DSL_profile *p = profile();
    unsigned enabled = 0;
    for (unsigned i = 0; i < 16; i++) if (mask & (1u << i)) enabled++;
    int chosen = -1;
    if (!p || !enabled) return -1;
    int listed = 0;
    for (const uint64_t *r = p->dev_caps.samplerates; *r; r++) if (*r == rate) listed = 1;
    if (!listed) return -1;
    for (unsigned i = 0; i < ARRAY_SIZE(channel_modes); i++) {
        const struct DSL_channels *mode = &channel_modes[i];
        if (!(p->dev_caps.channels & (UINT64_C(1) << i)) ||
            mode->mode != LOGIC || mode->stream != stream ||
            mode->vld_num < enabled || rate < mode->min_samplerate || rate > mode->max_samplerate)
            continue;
        if (mode->num < 16 && (mask >> mode->num)) continue;
        // FPGA arm rounds the hardware clock divider up; cached rate
        // readback alone does not prove the advertised timebase is real.
        uint64_t divider_rate = rate;
        // DSLogic's 200M/400M buffer packing uses the half/quarter flags
        // at a 100M hardware clock, as in dsl_fpga_arm's mode bits.
        if (!stream && rate == p->dev_caps.half_samplerate) divider_rate /= 2;
        if (!stream && rate == p->dev_caps.quarter_samplerate) divider_rate /= 4;
        if (mode->hw_max_samplerate % divider_rate) continue;
        uint64_t divisor = mode->hw_max_samplerate / divider_rate;
        if (divisor >= mode->pre_div && divisor % mode->pre_div) continue;
        if (chosen < 0 || mode->vld_num > channel_modes[chosen].vld_num) chosen = (int)i;
    }
    return chosen;
}

static int set(int key, GVariant *value)
{
    g_variant_ref_sink(value);
    int rc = ds_set_actived_device_config(NULL, NULL, key, value);
    g_variant_unref(value);
    if (rc != SR_OK) fprintf(stderr, "dslcap: configuration key %d failed (%d)\n", key, rc);
    return rc == SR_OK ? 0 : -1;
}

static int get(int key, const GVariantType *type, GVariant **value)
{
    *value = NULL;
    if (ds_get_actived_device_config(NULL, NULL, key, value) != SR_OK || !*value ||
        !g_variant_is_of_type(*value, type)) {
        if (*value) g_variant_unref(*value);
        *value = NULL;
        fprintf(stderr, "dslcap: cannot read configuration key %d\n", key);
        return -1;
    }
    return 0;
}

static int configure_serial(const struct dsl_serial_trigger *serial)
{
    char values[4][2][32]; // Exactly 16 space-separated probe characters + NUL.
    for (unsigned stage = 0; stage < 4; stage++)
        for (unsigned side = 0; side < 2; side++) {
            for (unsigned position = 0; position < 31; position++)
                values[stage][side][position] = position & 1 ? ' ' : 'X';
            values[stage][side][31] = '\0';
        }
    values[0][0][30 - 2 * serial->channels[0]] = serial->conditions[0];
    values[0][1][30 - 2 * serial->channels[1]] = serial->conditions[1];
    values[1][0][30 - 2 * serial->channels[2]] = serial->conditions[2];
    values[2][0][30 - 2 * serial->channels[3]] = '0';
    for (unsigned bit = 0; bit < serial->bits; bit++)
        values[3][0][30 - 2 * bit] = serial->value & (1u << bit) ? '1' : '0';
    // DSView's default stage selector is 1; commit_trigger stores selector-1.
    // Serial roles are fixed at 0..3 independently of that advanced selector.
    if (ds_trigger_set_stage(0) != SR_OK) return DSL_CONFIG_ERROR;
    for (unsigned stage = 0; stage < 4; stage++)
        if (ds_trigger_stage_set_value((uint16_t)stage, 16, values[stage][0], values[stage][1]) != SR_OK)
            return DSL_CONFIG_ERROR;
    for (unsigned stage = 0; stage < 4; stage++)
        if (ds_trigger_stage_set_logic((uint16_t)stage, 16, 1) != SR_OK) return DSL_CONFIG_ERROR;
    for (unsigned stage = 0; stage < 4; stage++)
        if (ds_trigger_stage_set_inv((uint16_t)stage, 16, 0, 0) != SR_OK) return DSL_CONFIG_ERROR;
    if (ds_trigger_stage_set_count(1, 16, 1, 0) != SR_OK ||
        ds_trigger_stage_set_count(3, 16, serial->bits - 1, 0) != SR_OK) return DSL_CONFIG_ERROR;
    return 0;
}

int dsl_configure(struct dsl_options *o)
{
    int mode = o->pattern ? profile()->dev_caps.intest_channel :
               dsl_choose_mode(o->stream, o->rate, o->channels);
    if (mode < 0) {
        fprintf(stderr, "dslcap: impossible samplerate/channel combination\n");
        return DSL_CONFIG_ERROR;
    }
    if (!o->stream && !o->pattern) {
        unsigned count = 0;
        for (unsigned i = 0; i < 16; i++) if (o->channels & (1u << i)) count++;
        uint64_t aligned = (o->samples + SAMPLES_ALIGN) & ~SAMPLES_ALIGN;
        if (aligned > profile()->dev_caps.hw_depth / count) {
            fprintf(stderr, "dslcap: finite buffer limit exceeds channel hardware depth\n");
            return DSL_CONFIG_ERROR;
        }
    }
    if (set(SR_CONF_OPERATION_MODE, g_variant_new_int16(o->pattern ? LO_OP_INTEST :
            o->stream ? LO_OP_STREAM : LO_OP_BUFFER))) return DSL_CONFIG_ERROR;
    if (!o->pattern) {
        uint64_t driver_limit = o->continuous ? o->rate : o->samples;
        // Buffer trigger-position parsing rounds captured counts down to
        // SAMPLES_ALIGN; arm an aligned count and trim to the requested N.
        if (!o->stream) driver_limit = (driver_limit + SAMPLES_ALIGN) & ~SAMPLES_ALIGN;
        o->arm_limit = driver_limit;
        if (set(SR_CONF_CHANNEL_MODE, g_variant_new_int16((int16_t)mode)) ||
            set(SR_CONF_SAMPLERATE, g_variant_new_uint64(o->rate)) ||
            set(SR_CONF_LIMIT_SAMPLES, g_variant_new_uint64(driver_limit)))
            return DSL_CONFIG_ERROR;
    } else {
        GVariant *value;
        if (get(SR_CONF_SAMPLERATE, G_VARIANT_TYPE_UINT64, &value)) return DSL_CONFIG_ERROR;
        o->rate = g_variant_get_uint64(value); g_variant_unref(value);
        if (get(SR_CONF_LIMIT_SAMPLES, G_VARIANT_TYPE_UINT64, &value)) return DSL_CONFIG_ERROR;
        o->samples = g_variant_get_uint64(value); g_variant_unref(value);
        if (o->rate != 100000000 || o->samples != profile()->dev_caps.hw_depth / 16) {
            fprintf(stderr, "dslcap: unexpected internal-pattern forced configuration\n");
            return DSL_CONFIG_ERROR;
        }
        fprintf(stderr, "dslcap: test-pattern overrides mode/rate/limit/channels: buffer 100M, %" PRIu64 " samples, 16 channels\n", o->samples);
    }
    if (o->stream && set(SR_CONF_LOOP_MODE, g_variant_new_boolean(o->continuous))) return DSL_CONFIG_ERROR;
    if (set(SR_CONF_VTH, g_variant_new_double(o->vth))) return DSL_CONFIG_ERROR;
    for (GSList *l = ds_get_actived_device_channels(); l; l = l->next) {
        const struct sr_channel *channel = l->data;
        if (channel->index > 15 || ds_enable_device_channel_index(channel->index,
                (o->channels & (1u << channel->index)) != 0) != SR_OK) return DSL_CONFIG_ERROR;
    }
    GVariant *value;
    if (get(SR_CONF_SAMPLERATE, G_VARIANT_TYPE_UINT64, &value)) return DSL_CONFIG_ERROR;
    uint64_t effective_rate = g_variant_get_uint64(value); g_variant_unref(value);
    if (effective_rate != o->rate) {
        fprintf(stderr, "dslcap: samplerate readback mismatch: requested=%" PRIu64 " actual=%" PRIu64 "\n", o->rate, effective_rate);
        return DSL_CONFIG_ERROR;
    }
    if (get(SR_CONF_VLD_CH_NUM, G_VARIANT_TYPE_INT16, &value)) return DSL_CONFIG_ERROR;
    int valid_channels = g_variant_get_int16(value); g_variant_unref(value);
    uint16_t enabled_mask = 0;
    int enabled = 0;
    for (GSList *l = ds_get_actived_device_channels(); l; l = l->next) {
        const struct sr_channel *channel = l->data;
        if (channel->enabled) {
            if (channel->index > 15) return DSL_CONFIG_ERROR;
            enabled_mask |= (uint16_t)(1u << channel->index); enabled++;
        }
    }
    if (enabled_mask != o->channels || enabled > valid_channels) {
        fprintf(stderr, "dslcap: channel readback mismatch or enabled count exceeds valid channel count\n");
        return DSL_CONFIG_ERROR;
    }
    o->arm_limit = o->stream ? o->samples : (o->samples + SAMPLES_ALIGN) & ~SAMPLES_ALIGN;
    // Ensure optional RLE and trigger state cannot alter the raw cross format.
    if (set(SR_CONF_RLE_SUPPORT, g_variant_new_boolean(FALSE)) || ds_trigger_reset() != SR_OK)
        return DSL_CONFIG_ERROR;
    if (o->trigger) {
        o->trigger_effective = dsl_trigger_position(o->trigger_pos, o->arm_limit,
                (profile()->dev_caps.hw_depth / (unsigned)enabled) & ~SAMPLES_ALIGN);
        if (ds_trigger_set_pos((uint16_t)o->trigger_pos) != SR_OK ||
            ds_trigger_set_mode(o->serial ? SERIAL_TRIGGER : SIMPLE_TRIGGER) != SR_OK) return DSL_CONFIG_ERROR;
        if (o->serial && configure_serial(&o->serial_trigger)) return DSL_CONFIG_ERROR;
        for (unsigned ch = 0; ch < 16; ch++)
            if (o->trigger_conditions[ch] && ds_trigger_probe_set((uint16_t)ch,
                    (unsigned char)o->trigger_conditions[ch], 'X') != SR_OK) return DSL_CONFIG_ERROR;
        if (o->timeout_upload && set(SR_CONF_BUFFER_OPTIONS, g_variant_new_int16(1) /* DSView dslogic.c private SR_BUF_UPLOAD */))
            return DSL_CONFIG_ERROR;
        if (ds_trigger_set_en(1) != SR_OK) return DSL_CONFIG_ERROR;
        fprintf(stderr, "dslcap: %s trigger position=%u%% effective=%" PRIu64 " arm-limit=%" PRIu64 "\n",
                o->serial ? "serial (MSB-first assumed)" : "simple AND", o->trigger_pos, o->trigger_effective, o->arm_limit);
    }
    fprintf(stderr, "dslcap: capture configured: %" PRIu64 " Hz, mask=0x%04x, unitsize=%d, %s, %s\n",
            o->rate, o->channels, o->channels & 0xff00 ? 2 : 1,
            o->stream ? "stream" : "buffer", o->continuous ? "continuous" : "finite");
    return 0;
}
