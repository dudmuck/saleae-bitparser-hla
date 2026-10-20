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
#define TARGET_VID 0x2a0e
#define EXPECTED_HDL_VERSION 0x0e
// DSView lib_main.c copies this argument into DS_RES_PATH[500].
#define RESOURCE_PATH_CAPACITY 500

// DSView log.c leaves a shared-context writer owned by its caller.
extern xlog_writer *sr_log;
// Pinned DSView 1.3.2 internal read-only helper (dsl.c); not declared in
// the public API. Check explicitly because dev_open skips this on a load.
extern int dsl_hdl_version(const struct sr_dev_inst *sdi, uint8_t *value);
#ifdef DSLCAP_TEST_RELOAD_FPGA
int dslcap_test_reload_fpga(const struct sr_dev_inst *device, const char *fw_dir, const char *bitstream);
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

static int validate_firmware(const char *directory, const char *bitstream)
{
    struct stat st;
    if (strlen(directory) >= RESOURCE_PATH_CAPACITY || !directory[0]) {
        fprintf(stderr, "dslcap: firmware directory must contain 1..499 bytes\n");
        return EXIT_FIRMWARE;
    }
    if (!bitstream) {
        if (stat(directory, &st) || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "dslcap: firmware directory is not accessible: %s\n", directory);
            return EXIT_FIRMWARE;
        }
        return 0;
    }
    char *path = g_build_filename(directory, bitstream, NULL);
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

static int activate(ds_device_handle handle, const char *stage, int needs_security)
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
    if (atomic_load(&security_fail) || (needs_security && !atomic_load(&security_pass))) {
        fprintf(stderr, "dslcap: %s rejected: %s\n", stage,
                atomic_load(&security_fail) ? "security check failed" :
                "no explicit security-pass evidence");
        return EXIT_SECURITY;
    }
    fprintf(stderr, "dslcap: %s: requested USB device active; security=%s\n", stage,
            needs_security ? "pass" : "n/a");
    return 0;
}

int main(int argc, char **argv)
{
    struct dsl_options options;
    int parse_result = dsl_parse_options(argc, argv, &options);
    if (parse_result) return parse_result == 1 ? 0 : parse_result;
    if (options.list_modes) return dsl_list_modes(options.model, options.speed);
    const char *fw_dir = options.fw_dir;
    int scan = options.scan, verbosity = options.verbosity;
    // Reject combinations impossible for every eligible static profile before
    // USB initialization. Validate again against the selected speed below.
    if (options.capture && !options.pattern) {
        int possible = 0;
        const int speeds[] = {LIBUSB_SPEED_HIGH, LIBUSB_SPEED_SUPER};
        for (unsigned m = 0; dsl_model_at(m); m++) {
            if (options.model >= 0 && options.model != (int)m) continue;
            for (unsigned i = 0; i < sizeof(speeds) / sizeof(speeds[0]); i++)
                if (dsl_choose_mode(dsl_profile(dsl_model_at(m)->pid, speeds[i]),
                        options.stream, options.rate, options.channels) >= 0)
                    possible = 1;
        }
        if (!possible) {
            fprintf(stderr, "dslcap: impossible samplerate/channel combination\n");
            return DSL_CONFIG_ERROR;
        }
    }
    int rc = validate_firmware(fw_dir, NULL);
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
    int count = 0, matches = 0, speed = 0, needs_security = 0;
    const struct dsl_model *selected = NULL;
    unsigned found[2] = {0};
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
    // Demo handles are not USB pointers. Only inspect descriptors after a
    // supported model-name match, as dsdevice.c names devices from profiles.
    for (unsigned m = 0; dsl_model_at(m); m++) {
        const struct dsl_model *model = dsl_model_at(m);
        for (int i = 0; i < count; i++) {
            if (strcmp(devices[i].name, model->name)) continue;
            libusb_device *usb = (libusb_device *)(uintptr_t)devices[i].handle;
            struct libusb_device_descriptor descriptor;
            if (libusb_get_device_descriptor(usb, &descriptor) ||
                descriptor.idVendor != TARGET_VID || descriptor.idProduct != model->pid)
                continue;
            found[m]++;
            if ((options.model >= 0 && options.model != (int)m) ||
                (selected && selected != model)) continue;
            selected = model;
            target = devices[i].handle;
            bus = libusb_get_bus_number(usb);
            address = libusb_get_device_address(usb);
            speed = libusb_get_device_speed(usb);
            matches++;
        }
    }
    if (matches != 1) {
        const struct dsl_model *wanted = options.model < 0 ? selected :
                                         dsl_model_at((unsigned)options.model);
        fprintf(stderr, "dslcap: expected exactly one %s; found %d\n",
                wanted ? wanted->name : "supported analyzer", matches);
        for (unsigned m = 0; dsl_model_at(m); m++)
            fprintf(stderr, "dslcap: found %u %s\n", found[m], dsl_model_at(m)->name);
        for (int i = 0; i < count; i++)
            fprintf(stderr, "dslcap: enumerated %s\n", devices[i].name);
        rc = EXIT_DEVICE;
        goto cleanup;
    }
    if (options.model < 0)
        for (unsigned m = 0; dsl_model_at(m); m++)
            if (found[m] && dsl_model_at(m) != selected)
                fprintf(stderr, "dslcap: selected %s; ignoring %u %s\n",
                        selected->name, found[m], dsl_model_at(m)->name);
    options.profile = dsl_profile(selected->pid, speed);
    if (!options.profile) {
        fprintf(stderr, "dslcap: no profile for %s at USB speed %d\n", selected->name, speed);
        rc = EXIT_DEVICE;
        goto cleanup;
    }
    needs_security = dsl_profile_needs_security(options.profile);
    if (needs_security != selected->needs_security) {
        fprintf(stderr, "dslcap: profile security mismatch for %s\n", selected->name);
        rc = EXIT_DEVICE;
        goto cleanup;
    }
    if (options.capture)
        fprintf(stderr, "dslcap: selected %04x:%04x %s speed=%s\n",
                TARGET_VID, selected->pid, selected->name,
                speed == LIBUSB_SPEED_SUPER ? "super" : "high");
#ifdef DSLCAP_TEST_RELOAD_FPGA
    if (selected != dsl_model_at(0)) {
        fprintf(stderr, "dslcap: forced FPGA reload is supported only on the Plus; refusing %s\n", selected->name);
        rc = DSL_CONFIG_ERROR;
        goto cleanup;
    }
#endif
    if (selected != dsl_model_at(0) && options.capture &&
        (!options.stream || options.trigger || options.serial || options.pattern)) {
        fprintf(stderr, "dslcap: buffer/trigger/test-pattern not validated on U3Pro16 (Phase 3)\n");
        rc = DSL_CONFIG_ERROR;
        goto cleanup;
    }
    rc = validate_firmware(fw_dir, selected->bitstream);
    if (rc) goto cleanup;
    if (options.capture && !options.pattern &&
        dsl_choose_mode(options.profile, options.stream, options.rate, options.channels) < 0) {
        fprintf(stderr, "dslcap: impossible samplerate/channel combination\n");
        rc = DSL_CONFIG_ERROR;
        goto cleanup;
    }
    rc = activate(target, "initial activation", needs_security);
    if (rc)
        goto cleanup;

#ifdef DSLCAP_TEST_RELOAD_FPGA
    struct ds_device_full_info reload_info = {0};
    if (ds_get_actived_device_info(&reload_info) != SR_OK ||
        reload_info.handle != target || reload_info.dev_type != DEV_TYPE_USB ||
        !reload_info.di || dslcap_test_reload_fpga(reload_info.di, fw_dir, selected->bitstream) != SR_OK) {
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
    rc = activate(target, "HDL verification reopen", needs_security);
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
    if (dsl_signal && (!rc || (options.capture && !options.stream))) rc = 128 + dsl_signal;
    if (!rc) {
        if (!options.capture) fprintf(stderr, "dslcap: %04x:%04x %s bring-up complete; driver HDL check passed on reopen\n",
                TARGET_VID, selected->pid, selected->name);
        if (scan && printf("%04x:%04x %s bus=%u address=%u speed=%s activated security=%s hdl=checked\n",
                           TARGET_VID, selected->pid, selected->name, bus, address,
                           speed == LIBUSB_SPEED_SUPER ? "super" : "high",
                           needs_security ? "pass" : "n/a") < 0)
            rc = EXIT_CLEANUP;
    }
    if (fflush(stdout) == EOF && !rc)
        rc = EXIT_CLEANUP;
    return rc;
}
