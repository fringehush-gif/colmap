# SPDX-License-Identifier: BSD-3-Clause

foreach(cuda_import ON OFF)
    # Reuse each build directory so CUDA-off sees the previous lookup cache.
    foreach(cuda_mode ON OFF)
        execute_process(
            COMMAND "${CMAKE_COMMAND}"
                -G "${TEST_GENERATOR}"
                -S "${CMAKE_CURRENT_LIST_DIR}"
                -B "${TEST_BINARY_DIR}/cuda-import-${cuda_import}"
                "-DTEST_CUDA_IMPORT=${cuda_import}"
                "-DCUDA_ENABLED=${cuda_mode}"
            RESULT_VARIABLE result)
        if(NOT result STREQUAL "0")
            message(FATAL_ERROR
                "ONNX Runtime configure failed with CUDA=${cuda_mode}, CUDA import library=${cuda_import}")
        endif()
    endforeach()
endforeach()
