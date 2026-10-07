// SPDX-License-Identifier: GPL-3.0-or-later
// Fake only the hardware-facing calls; use DSView's actual xlog/log.c.
#include <libsigrok.h>
#include <libusb.h>
#include <stdlib.h>
#include <string.h>

extern xlog_writer *sr_log;
static ds_device_handle active;
static int activation_count;
static int last_error;
static int released;
static const char *scenario;

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
    return is("init-fail") ? SR_ERR : SR_OK;
}

int ds_lib_exit(void)
{
    if (is("init-fail"))
        abort(); // DSView's partial-init exit may use an uninitialized mutex.
    fprintf(stderr, "FAKE cleanup active=%llu\n", active);
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
