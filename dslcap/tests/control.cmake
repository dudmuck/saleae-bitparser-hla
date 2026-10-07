execute_process(COMMAND "${PROGRAM}" --watchdog
    RESULT_VARIABLE code ERROR_VARIABLE err TIMEOUT 5)
if(NOT code EQUAL 15 OR NOT err MATCHES "driver watchdog timed out")
    message(FATAL_ERROR "watchdog did not terminate stalled driver with exit15: ${code}\n${err}")
endif()
execute_process(COMMAND "${PROGRAM}" --signal-watchdog
    RESULT_VARIABLE code ERROR_VARIABLE err TIMEOUT 9)
if(NOT code EQUAL 143 OR NOT err MATCHES "interrupted driver did not stop")
    message(FATAL_ERROR "signal grace did not terminate stalled driver with exit143: ${code}\n${err}")
endif()
