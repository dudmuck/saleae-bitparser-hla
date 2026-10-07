// SPDX-License-Identifier: GPL-3.0-or-later
// Explicit hardware-only test helper. Not part of the production binary,
// and never registered with CTest. Uses the pinned DSView 1.3.2 internals.
#include <hardware/DSL/dsl.h>

int dslcap_test_reload_fpga(const struct sr_dev_inst *device, const char *fw_dir)
{
    if (!device || device->dev_type != DEV_TYPE_USB || !device->conn)
        return SR_ERR;
    struct sr_usb_dev_inst *usb = device->conn;
    if (!usb->devhdl)
        return SR_ERR;

    char *path = g_build_filename(fw_dir, "DSLogicPlus-pgl12-2.bin", NULL);
    fprintf(stderr, "dslcap: HARDWARE TEST: reload volatile FPGA from %s\n", path);
    // dsl.c:1296-1486 uses PROG_B/LED/INTRDY/BULK_WR/WORDWIDE controls
    // plus endpoint 2 bulk OUT; no DSL_CTL_NVM or EEPROM-writing call.
    int ret = dsl_fpga_config(usb->devhdl, path);
    g_free(path);
    if (ret == SR_OK)
        fprintf(stderr, "dslcap: HARDWARE TEST: volatile FPGA upload completed\n");
    return ret;
}
