// SPDX-License-Identifier: GPL-3.0-or-later
#include <errno.h>
#include <getopt.h>
#include <libsigrok.h>
#include <libusb.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "options.h"
#include "capture.h"
#include "driver_config.h"
#include "control.h"
#include "status.h"

#define DEFAULT_FW_DIR "/usr/local/share/DSView/res"
#define BITSTREAM "DSLogicPlus-pgl12-2.bin"
#define TARGET_VID 0x2a0e
#define TARGET_PID 0x0034
#define EXPECTED_HDL_VERSION 0x0e
// DSView lib_main.c copies this argument into DS_RES_PATH[500].
#define RESOURCE_PATH_CAPACITY 500

// DSView log.c leaves a shared-context writer owned by its caller.
extern xlog_writer *sr_log;
// Pinned DSView 1.3.2 internal read-only helper (dsl.c); not declared in
// the public API. Check explicitly because dev_open skips this on a load.
extern int dsl_hdl_version(const struct sr_dev_inst *sdi, uint8_t *value);
#ifdef DSLCAP_TEST_RELOAD_FPGA
int dslcap_test_reload_fpga(const struct sr_dev_inst *device, const char *fw_dir);
#endif
static atomic_bool security_pass;
static atomic_bool security_fail;

enum exit_code {
    EXIT_USAGE = 2,
    EXIT_FIRMWARE = 3,
    EXIT_DEVICE = 4,
    EXIT_ACTIVATION = 5,
    EXIT_SECURITY = 6,
    EXIT_CLEANUP = 7
};

static void receive_log(const char *data, int length)
{
    if (!data || length <= 0)
        return;
    // xlog's receiver supplies a length, not a terminating NUL.
    if (g_strstr_len(data, length, "Security check pass!"))
        atomic_store(&security_pass, 1);
    if (g_strstr_len(data, length, "Security check failed!"))
        atomic_store(&security_fail, 1);
    fwrite(data, 1, (size_t)length, stderr);
}

static int validate_firmware(const char *directory)
{
    struct stat st;
    if (strlen(directory) >= RESOURCE_PATH_CAPACITY || !directory[0]) {
        fprintf(stderr, "dslcap: firmware directory must contain 1..499 bytes\n");
        return EXIT_FIRMWARE;
    }
    char *path = g_build_filename(directory, BITSTREAM, NULL);
    int rc = 0;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "dslcap: cannot stat bitstream %s: %s\n", path, strerror(errno));
        rc = EXIT_FIRMWARE;
    } else if (!S_ISREG(st.st_mode) || st.st_size <= 0 || access(path, R_OK) != 0) {
        fprintf(stderr, "dslcap: bitstream must be a nonempty readable regular file: %s\n", path);
        rc = EXIT_FIRMWARE;
    }
    g_free(path);
    return rc;
}

static int activate(ds_device_handle handle, const char *stage)
{
    struct ds_device_full_info info = {0};
    atomic_store(&security_pass, 0);
    atomic_store(&security_fail, 0);
    int ret = ds_active_device(handle);
    int last_error = ds_get_last_error();
    if (ret != SR_OK || last_error != SR_OK ||
        ds_get_actived_device_info(&info) != SR_OK ||
        info.handle != handle || info.dev_type != DEV_TYPE_USB) {
        fprintf(stderr, "dslcap: %s failed (activation=%d, last_error=%d); "
                "requested USB device must remain active\n", stage, ret, last_error);
        return EXIT_ACTIVATION;
    }
    if (atomic_load(&security_fail) || !atomic_load(&security_pass)) {
        fprintf(stderr, "dslcap: %s rejected: %s\n", stage,
                atomic_load(&security_fail) ? "security check failed" :
                "no explicit security-pass evidence");
        return EXIT_SECURITY;
    }
    fprintf(stderr, "dslcap: %s: requested USB device active; security=pass\n", stage);
    return 0;
}

int main(int argc, char **argv)
{
    struct dsl_options options;
    int parse_result = dsl_parse_options(argc, argv, &options);
    if (parse_result) return parse_result == 1 ? 0 : parse_result;
    const char *fw_dir = options.fw_dir;
    int scan = options.scan, verbosity = options.verbosity;
    if (options.capture && !options.pattern &&
        dsl_choose_mode(options.stream, options.rate, options.channels) < 0) {
        fprintf(stderr, "dslcap: impossible samplerate/channel combination\n");
        return DSL_CONFIG_ERROR;
    }
    int rc = validate_firmware(fw_dir);
    if (rc)
        return rc;

    xlog_context *log = xlog_new2(0); // Never use xlog's stdout console.
    if (!log || xlog_add_receiver(log, receive_log, NULL) != 0) {
        fprintf(stderr, "dslcap: cannot create log receiver\n");
        if (log) xlog_free(log);
        return EXIT_ACTIVATION;
    }
    int level = verbosity == 0 ? XLOG_LEVEL_ERR :
                verbosity == 1 ? XLOG_LEVEL_INFO : XLOG_LEVEL_DBG;
    ds_log_level(level);
    ds_log_set_context(log);
    if (dsl_control_start()) {
        fprintf(stderr, "dslcap: cannot start driver watchdog\n");
        xlog_free_writer(sr_log); sr_log = NULL; xlog_free(log);
        return EXIT_ACTIVATION;
    }

    // Initialization scans immediately, so set the resource path first.
    ds_set_firmware_resource_dir(fw_dir);
    struct ds_device_base_info *devices = NULL;
    ds_device_handle target = NULL_HANDLE;
    unsigned int bus = 0, address = 0;
    int count = 0, matches = 0;
    int init_ret = ds_lib_init();
    int library_initialized = init_ret == SR_OK;
    if (init_ret != SR_OK) {
        fprintf(stderr, "dslcap: driver initialization failed (%d)\n", init_ret);
        rc = EXIT_ACTIVATION;
        goto cleanup;
    }
    if (ds_get_device_list(&devices, &count) != SR_OK) {
        fprintf(stderr, "dslcap: cannot obtain device list\n");
        rc = EXIT_DEVICE;
        goto cleanup;
    }
    for (int i = 0; i < count; i++) {
        // dsdevice.c uses the profile's model as name. The 0034 profile
        // spells it "DSLogic PLus". Demo handles are not libusb pointers;
        // filter the model before reading USB descriptors. dslogic.c stores
        // the libusb_device pointer as the USB device handle.
        if (strcmp(devices[i].name, "DSLogic PLus") != 0)
            continue;
        libusb_device *usb = (libusb_device *)(uintptr_t)devices[i].handle;
        struct libusb_device_descriptor descriptor;
        if (libusb_get_device_descriptor(usb, &descriptor) != 0 ||
            descriptor.idVendor != TARGET_VID || descriptor.idProduct != TARGET_PID)
            continue;
        target = devices[i].handle;
        bus = libusb_get_bus_number(usb);
        address = libusb_get_device_address(usb);
        matches++;
    }
    if (matches != 1) {
        fprintf(stderr, "dslcap: expected exactly one 2a0e:0034; found %d\n", matches);
        rc = EXIT_DEVICE;
        goto cleanup;
    }
    rc = activate(target, "initial activation");
    if (rc)
        goto cleanup;

#ifdef DSLCAP_TEST_RELOAD_FPGA
    struct ds_device_full_info reload_info = {0};
    if (ds_get_actived_device_info(&reload_info) != SR_OK ||
        reload_info.handle != target || reload_info.dev_type != DEV_TYPE_USB ||
        !reload_info.di || dslcap_test_reload_fpga(reload_info.di, fw_dir) != SR_OK) {
        fprintf(stderr, "dslcap: hardware-only volatile FPGA reload failed\n");
        rc = EXIT_ACTIVATION;
        goto cleanup;
    }
#endif

    // dsl_dev_open checks HDL only in its already-configured FPGA branch.
    // Reopen after initial activation to exercise that check after a load.
    if (ds_release_actived_device() != SR_OK) {
        fprintf(stderr, "dslcap: cannot release device for HDL verification\n");
        rc = EXIT_CLEANUP;
        goto cleanup;
    }
    rc = activate(target, "HDL verification reopen");
    if (rc)
        goto cleanup;
    struct ds_device_full_info info = {0};
    uint8_t hdl_version = 0;
    if (ds_get_actived_device_info(&info) != SR_OK ||
        info.handle != target || info.dev_type != DEV_TYPE_USB || !info.di ||
        dsl_hdl_version(info.di, &hdl_version) != SR_OK ||
        hdl_version != EXPECTED_HDL_VERSION) {
        fprintf(stderr, "dslcap: explicit HDL version check failed (device=0x%02x expected=0x%02x)\n",
                hdl_version, EXPECTED_HDL_VERSION);
        rc = EXIT_ACTIVATION;
        goto cleanup;
    }
    fprintf(stderr, "dslcap: explicit HDL version=0x%02x (expected 0x%02x)\n",
            hdl_version, EXPECTED_HDL_VERSION);
    if (dsl_signal) rc = 128 + dsl_signal;
    else if (options.capture) rc = dsl_capture(&options);
cleanup:
    free(devices);
    if (library_initialized) {
        dsl_control_bound(10);
        int exit_ret = ds_lib_exit();
        if (exit_ret != SR_OK) {
            fprintf(stderr, "dslcap: driver cleanup failed (%d)\n", exit_ret);
            if (!rc) rc = EXIT_CLEANUP;
        }
    } else {
        // lib_main.c can fail before initializing its mutex; ds_lib_exit
        // would then use/destroy that uninitialized mutex. No USB scan or
        // hotplug thread has started on these failure paths. This CLI exits
        // immediately and the process reclaims any partial library context.
        fprintf(stderr, "dslcap: driver cleanup skipped after partial initialization; "
                "process exit reclaims resources\n");
    }
    xlog_free_writer(sr_log);
    sr_log = NULL;
    xlog_free(log);
    dsl_control_finish();
    if (dsl_signal && !rc) rc = 128 + dsl_signal;
    if (!rc) {
        if (!options.capture) fprintf(stderr, "dslcap: 2a0e:0034 bring-up complete; driver HDL check passed on reopen\n");
        if (scan && printf("2a0e:0034 DSLogic PLus bus=%u address=%u activated security=pass hdl=checked\n",
                           bus, address) < 0)
            rc = EXIT_CLEANUP;
    }
    if (fflush(stdout) == EOF && !rc)
        rc = EXIT_CLEANUP;
    return rc;
}
