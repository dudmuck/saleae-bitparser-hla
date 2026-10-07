cmake_minimum_required(VERSION 3.16)
project(dslcap LANGUAGES C)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(DSVIEW_SRC "/home/wroberts/DSView-1.3.2" CACHE PATH "Read-only DSView 1.3.2 source tree")
if(NOT EXISTS "${DSVIEW_SRC}/libsigrok4DSL/lib_main.c")
    message(FATAL_ERROR "DSVIEW_SRC must point to the DSView 1.3.2 source tree")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(USB REQUIRED IMPORTED_TARGET libusb-1.0)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# Match DSView 1.3.2's libsigrok4DSL_SOURCES and common_SOURCES, without
# its Qt application or Python decoder library. Compile in the build tree.
set(driver_sources
    version.c strutil.c std.c session_driver.c session.c log.c hwdriver.c
    error.c backend.c output/output.c input/input.c hardware/demo/demo.c
    input/in_binary.c input/in_vcd.c input/in_wav.c output/csv.c
    output/gnuplot.c output/srzip.c output/vcd.c hardware/DSL/dslogic.c
    hardware/common/usb.c hardware/common/ezusb.c trigger.c dsdevice.c
    hardware/DSL/dscope.c hardware/DSL/command.c hardware/DSL/dsl.c lib_main.c)
list(TRANSFORM driver_sources PREPEND "${DSVIEW_SRC}/libsigrok4DSL/")
add_library(sigrok4dsl STATIC ${driver_sources}
    "${DSVIEW_SRC}/common/minizip/zip.c"
    "${DSVIEW_SRC}/common/minizip/unzip.c"
    "${DSVIEW_SRC}/common/minizip/ioapi.c"
    "${DSVIEW_SRC}/common/log/xlog.c")
target_include_directories(sigrok4dsl PUBLIC
    "${DSVIEW_SRC}/libsigrok4DSL" "${DSVIEW_SRC}/common")
target_link_libraries(sigrok4dsl PUBLIC PkgConfig::GLIB PkgConfig::USB
    ZLIB::ZLIB Threads::Threads m)

set(frontend_sources options.c convert.c ring.c control.c capture.c)
add_library(dslcap_core STATIC ${frontend_sources})
target_include_directories(dslcap_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    "${DSVIEW_SRC}/libsigrok4DSL" "${DSVIEW_SRC}/common")
target_compile_features(dslcap_core PUBLIC c_std_11)
target_compile_options(dslcap_core PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(dslcap_core PUBLIC PkgConfig::GLIB PkgConfig::USB Threads::Threads m)
add_library(dslcap_config STATIC driver_config.c)
target_link_libraries(dslcap_config PUBLIC dslcap_core)
add_executable(dslcap main.c)
target_compile_features(dslcap PRIVATE c_std_11)
target_compile_options(dslcap PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(dslcap PRIVATE dslcap_core dslcap_config sigrok4dsl)

option(DSLCAP_BUILD_USB_TESTS "Build explicit hardware-only volatile FPGA reload test" OFF)
if(DSLCAP_BUILD_USB_TESTS)
    add_executable(dslcap_reload_fpga main.c tests/reload_fpga.c)
    target_compile_definitions(dslcap_reload_fpga PRIVATE DSLCAP_TEST_RELOAD_FPGA=1)
    target_compile_features(dslcap_reload_fpga PRIVATE c_std_11)
    target_link_libraries(dslcap_reload_fpga PRIVATE dslcap_core dslcap_config sigrok4dsl)
    # Intentionally no add_test: this must never run automatically via CTest.
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(dslcap_core_tests tests/test_core.c)
    target_link_libraries(dslcap_core_tests PRIVATE dslcap_core)
    add_test(NAME capture_core COMMAND dslcap_core_tests)
    add_test(NAME driver_watchdog COMMAND ${CMAKE_COMMAND}
        "-DPROGRAM=$<TARGET_FILE:dslcap_core_tests>" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/control.cmake")
    # Exercise the real frontend with a fake driver, never touching USB.
    # Keep the actual DSView logger so non-NUL-terminated callbacks and
    # errors-only log filtering are tested too.
    add_executable(dslcap_fake main.c tests/fake_driver.c ${frontend_sources} driver_config.c
        "${DSVIEW_SRC}/common/log/xlog.c"
        "${DSVIEW_SRC}/libsigrok4DSL/log.c")
    target_include_directories(dslcap_fake PRIVATE
        "${DSVIEW_SRC}/libsigrok4DSL" "${DSVIEW_SRC}/common")
    target_include_directories(dslcap_fake PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(dslcap_fake PRIVATE DSLCAP_TEST_TIME_SCALE=0.02)
    target_compile_options(dslcap_fake PRIVATE -UNDEBUG -Wno-variadic-macros -Wall -Wextra -Werror -Wpedantic)
    target_link_libraries(dslcap_fake PRIVATE PkgConfig::GLIB PkgConfig::USB Threads::Threads m)
    target_compile_features(dslcap_fake PRIVATE c_std_11)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-firmware")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test-firmware/DSLogicPlus-pgl12-2.bin" "fake bitstream for offline tests only\n")
    add_test(NAME bringup_contract COMMAND ${CMAKE_COMMAND}
        "-DPROGRAM=$<TARGET_FILE:dslcap_fake>"
        "-DFIRMWARE=${CMAKE_CURRENT_BINARY_DIR}/test-firmware"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/bringup.cmake")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME trigger_contract COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trigger.py" "$<TARGET_FILE:dslcap_fake>"
        "${CMAKE_CURRENT_BINARY_DIR}/test-firmware")
    add_test(NAME capture_contract COMMAND ${CMAKE_COMMAND}
        "-DPROGRAM=$<TARGET_FILE:dslcap_fake>"
        "-DFIRMWARE=${CMAKE_CURRENT_BINARY_DIR}/test-firmware"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/capture.cmake")
endif()
