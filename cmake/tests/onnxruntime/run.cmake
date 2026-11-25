# SPDX-License-Identifier: BSD-3-Clause

foreach(both_locations OFF ON)
    foreach(cuda_import ON OFF)
        # Reuse each build directory so CUDA-off sees the previous lookup cache.
        foreach(cuda_mode ON OFF)
            execute_process(
                COMMAND "${CMAKE_COMMAND}"
                    -G "${TEST_GENERATOR}"
                    -S "${CMAKE_CURRENT_LIST_DIR}"
                    -B "${TEST_BINARY_DIR}/both-${both_locations}-cuda-import-${cuda_import}"
                    "-DTEST_CUDA_IMPORT=${cuda_import}"
                    "-DTEST_BOTH_DLL_LOCATIONS=${both_locations}"
                    "-DCUDA_ENABLED=${cuda_mode}"
                RESULT_VARIABLE result)
            if(NOT result STREQUAL "0")
                message(FATAL_ERROR
                    "ONNX Runtime configure failed with CUDA=${cuda_mode}, CUDA import library=${cuda_import}, both DLL locations=${both_locations}")
            endif()
        endforeach()
    endforeach()
endforeach()
