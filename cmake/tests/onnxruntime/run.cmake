# SPDX-License-Identifier: BSD-3-Clause

# Reuse the build directory so the CUDA-off pass sees the cached CUDA library.
foreach(cuda_mode ON OFF)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -G "${TEST_GENERATOR}"
            -S "${CMAKE_CURRENT_LIST_DIR}"
            -B "${TEST_BINARY_DIR}"
            "-DCUDA_ENABLED=${cuda_mode}"
        RESULT_VARIABLE result)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "ONNX Runtime configure failed with CUDA_ENABLED=${cuda_mode}")
    endif()
endforeach()
