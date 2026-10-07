// SPDX-License-Identifier: GPL-3.0-or-later
// Fake only the hardware-facing calls; use DSView's actual xlog/log.c.
#include <libsigrok.h>
#include <libusb.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

extern xlog_writer *sr_log;
static ds_device_handle active;
static int activation_count;
static int last_error;
static int released;
static const char *scenario;
static struct sr_channel channels[16];
static GSList *channel_list;
static uint64_t rate = 25000000, sample_limit;
static int operation, setter_count;
static atomic_int collecting, stop_requested;
static pthread_t collect_thread;
static int thread_started;
static ds_datafeed_callback_t feed_callback;
static dslib_event_callback_t event_callback;

static int is(const char *value)
{
    return scenario && strcmp(scenario, value) == 0;
}

void ds_set_firmware_resource_dir(const char *directory)
{
    // Assert the frontend never reaches DSView's unsafe strcpy with a
    // directory that overflows DS_RES_PATH or is empty.
    if (!directory[0] || strlen(directory) >= 500)
        abort();
}

int ds_lib_init(void)
{
    scenario = getenv("DSLCAP_TEST_SCENARIO");
    fprintf(stderr, "FAKE init\n");
    for (unsigned i = 0; i < 16; i++) {
        channels[i].index = (uint16_t)i; channels[i].enabled = TRUE;
        channel_list = g_slist_append(channel_list, &channels[i]);
    }
    return is("init-fail") ? SR_ERR : SR_OK;
}

int ds_lib_exit(void)
{
    if (is("init-fail"))
        abort(); // DSView's partial-init exit may use an uninitialized mutex.
    fprintf(stderr, "FAKE cleanup active=%llu\n", active);
    if (thread_started) pthread_join(collect_thread, NULL);
    g_slist_free(channel_list); channel_list = NULL;
    active = 0;
    return is("cleanup-fail") ? SR_ERR : SR_OK;
}

int ds_get_device_list(struct ds_device_base_info **list, int *count)
{
    if (is("list-fail"))
        return SR_ERR;
    if (is("no-device")) {
        *list = NULL;
        *count = 0;
        return SR_OK;
    }
    *count = is("multiple") ? 3 : 2;
    *list = calloc((size_t)*count, sizeof(**list));
    if (!*list)
        return SR_ERR;
    (*list)[0].handle = 1;
    strcpy((*list)[0].name, "Demo Device");
    (*list)[1].handle = 2;
    strcpy((*list)[1].name, "DSLogic PLus");
    if (*count == 3) {
        (*list)[2].handle = 3;
        strcpy((*list)[2].name, "DSLogic PLus");
    }
    return SR_OK;
}

int LIBUSB_CALL libusb_get_device_descriptor(libusb_device *device,
                                            struct libusb_device_descriptor *descriptor)
{
    if ((uintptr_t)device == 1)
        abort(); // The demo handle must never be treated as a USB pointer.
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->idVendor = 0x2a0e;
    descriptor->idProduct = is("wrong-pid") ? 0x0020 : 0x0034;
    return 0;
}

uint8_t LIBUSB_CALL libusb_get_bus_number(libusb_device *device)
{
    (void)device;
    return 1;
}

uint8_t LIBUSB_CALL libusb_get_device_address(libusb_device *device)
{
    (void)device;
    return 6;
}

int ds_active_device(ds_device_handle handle)
{
    activation_count++;
    fprintf(stderr, "FAKE activation=%d\n", activation_count);
    if (activation_count == 2 && !released)
        abort(); // HDL reopen must follow a release.
    active = handle;
    last_error = SR_OK;
    if (is("activation-fail"))
        return SR_ERR;
    if (is("fallback"))
        active = 1; // SR_OK despite falling back to demo, as in DSView.
    if (is("last-error"))
        last_error = SR_ERR;
    if (is("hdl-fail") && activation_count == 2)
        return SR_ERR;
    if (is("security-fail"))
        xlog_err(sr_log, "Security check failed!");
    else if (is("both-security")) {
        xlog_err(sr_log, "Security check failed!");
        xlog_err(sr_log, "Security check pass!");
    } else if (!is("no-security") &&
               !(is("no-security-reopen") && activation_count == 2))
        xlog_err(sr_log, "Security check pass!");
    return SR_OK;
}

int ds_get_last_error(void)
{
    return last_error;
}

int ds_get_actived_device_info(struct ds_device_full_info *info)
{
    if (is("info-fail"))
        return SR_ERR;
    memset(info, 0, sizeof(*info));
    info->handle = active;
    info->dev_type = active == 1 ? DEV_TYPE_DEMO : DEV_TYPE_USB;
    info->di = (struct sr_dev_inst *)(uintptr_t)active;
    return SR_OK;
}

int dsl_hdl_version(const struct sr_dev_inst *device, uint8_t *value)
{
    if ((uintptr_t)device != 2 || activation_count != 2)
        abort();
    *value = is("hdl-mismatch") ? 0x0d : 0x0e;
    return is("hdl-read-fail") ? SR_ERR : SR_OK;
}

int ds_release_actived_device(void)
{
    fprintf(stderr, "FAKE release\n");
    released = 1;
    return is("release-fail") ? SR_ERR : SR_OK;
}

int ds_set_actived_device_config(const struct sr_channel *ch, const struct sr_channel_group *group,
                                 int key, GVariant *value)
{
    (void)ch; (void)group;
    fprintf(stderr, "FAKE config=%d step=%d\n", key, ++setter_count);
    if (is("config-set-fail")) return SR_ERR;
    switch (key) {
    case SR_CONF_OPERATION_MODE:
        if (setter_count != 1) abort();
        operation = g_variant_get_int16(value);
        for (unsigned i = 0; i < 16; i++) channels[i].enabled = TRUE;
        if (operation == LO_OP_INTEST) { rate = 100000000; sample_limit = 16777216; }
        break;
    case SR_CONF_CHANNEL_MODE:
        if (setter_count != 2) abort();
        for (unsigned i = 0; i < 16; i++) channels[i].enabled = TRUE;
        break;
    case SR_CONF_SAMPLERATE: if (setter_count != 3) abort(); rate = g_variant_get_uint64(value); break;
    case SR_CONF_LIMIT_SAMPLES: if (setter_count != 4) abort(); sample_limit = g_variant_get_uint64(value); break;
    default: break;
    }
    return SR_OK;
}

int ds_get_actived_device_config(const struct sr_channel *ch, const struct sr_channel_group *group,
                                 int key, GVariant **value)
{
    (void)ch; (void)group;
    if (is("config-read-fail")) return SR_ERR;
    if (key == SR_CONF_SAMPLERATE) *value = g_variant_new_uint64(is("config-clamp") ? rate / 2 : rate);
    else if (key == SR_CONF_LIMIT_SAMPLES) *value = g_variant_new_uint64(sample_limit);
    else if (key == SR_CONF_VLD_CH_NUM) *value = g_variant_new_int16(is("config-valid-count") ? 1 : 16);
    else return SR_ERR;
    g_variant_ref_sink(*value);
    return SR_OK;
}

int ds_enable_device_channel_index(int index, gboolean enabled)
{
    if (setter_count < (operation == LO_OP_INTEST ? 2 : operation == LO_OP_STREAM ? 6 : 5)) abort();
    if (is("config-enable-fail")) return SR_ERR;
    if (!is("config-enable-ignored")) channels[index].enabled = enabled;
    return SR_OK;
}

GSList *ds_get_actived_device_channels(void) { return channel_list; }
int ds_trigger_reset(void) { return SR_OK; }
void ds_set_datafeed_callback(ds_datafeed_callback_t callback) { feed_callback = callback; }
void ds_set_event_callback(dslib_event_callback_t callback) { event_callback = callback; }
int ds_is_collecting(void) { return atomic_load(&collecting); }

static void *collect(void *unused)
{
    (void)unused;
    if (event_callback) event_callback(DS_EV_DEVICE_RUNNING);
    unsigned count = 0;
    for (unsigned i = 0; i < 16; i++) if (channels[i].enabled) count++;
    uint8_t data[128] = {0};
    for (unsigned i = 0; i < count; i++) memset(data + i * 8, (i & 1) ? 0xaa : 0x55, 8);
    uint64_t groups = (sample_limit + 63) / 64;
    if (is("capture-short")) groups = 1;
    if (is("capture-hang")) {
        while (!atomic_load(&stop_requested)) usleep(1000);
    } else {
        for (uint64_t i = 0; i < groups && !atomic_load(&stop_requested); i++) {
            struct sr_datafeed_logic logic = {.length = count * 8,
                .format = is("capture-format") ? LA_SPLIT_DATA : LA_CROSS_DATA, .data = data};
            if (is("capture-partial")) logic.length--;
            struct sr_datafeed_packet packet = {.type = is("capture-overflow") ? SR_DF_OVERFLOW : SR_DF_LOGIC,
                .status = is("capture-status") ? SR_PKT_DATA_ERROR : SR_PKT_OK, .payload = &logic};
            feed_callback(NULL, &packet);
        }
    }
    if (!is("capture-no-end")) {
        struct sr_datafeed_packet packet = {.type = SR_DF_END, .status = SR_PKT_OK};
        feed_callback(NULL, &packet);
    }
    atomic_store(&collecting, 0);
    if (event_callback) event_callback(is("capture-device-error") ? DS_EV_COLLECT_TASK_END_BY_ERROR :
        is("capture-detach") ? DS_EV_COLLECT_TASK_END_BY_DETACHED :
        is("capture-speed") ? DS_EV_DEVICE_SPEED_NOT_MATCH : DS_EV_COLLECT_TASK_END);
    return NULL;
}

int ds_start_collect(void)
{
    if (is("capture-start-fail")) return SR_ERR;
    atomic_store(&collecting, 1);
    if (pthread_create(&collect_thread, NULL, collect, NULL)) abort();
    thread_started = 1;
    return SR_OK;
}

int ds_stop_collect(void)
{
    if (pthread_equal(pthread_self(), collect_thread)) abort();
    atomic_store(&stop_requested, 1);
    pthread_join(collect_thread, NULL); thread_started = 0;
    return SR_OK;
}
