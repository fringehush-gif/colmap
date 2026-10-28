# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "feature")

COLMAP_ADD_LIBRARY(
    NAME colmap_feature_types
    SRCS
        types.h types.cc
    PUBLIC_LINK_LIBS
        colmap_util
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME types_test
    SRCS types_test.cc
    LINK_LIBS colmap_feature_types
)

COLMAP_ADD_LIBRARY(
    NAME colmap_feature
    SRCS
        aliked.h aliked.cc
        extractor.h extractor.cc
        index.h index.cc
        lightglue.h lightglue.cc
        loma.h loma.cc
        matcher.h matcher.cc
        onnx_matchers.h onnx_matchers.cc
        sift.h sift.cc
        types.h types.cc
        utils.h utils.cc
    PUBLIC_LINK_LIBS
        colmap_feature_types
        colmap_scene_types
        colmap_geometry
        colmap_util
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_vlfeat
        Eigen3::Eigen
        faiss
)

if(GPU_ENABLED)
    target_link_libraries(colmap_feature PRIVATE colmap_sift_gpu)
    if(NOT GUI_ENABLED)
        target_link_libraries(colmap_feature PRIVATE GLEW::GLEW)
    endif()
endif()

if(OPENMP_FOUND)
    target_link_libraries(colmap_feature PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ONNX_ENABLED)
    target_link_libraries(colmap_feature PRIVATE colmap_onnx_util)
endif()

# Deliberately narrow first qualification target; all existing defaults stay off.
option(ALIKED_NATIVE_CUDA_ENABLED "Build experimental ALIKED CUDA custom ops" OFF)
if(ALIKED_NATIVE_CUDA_ENABLED)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CUDA_ENABLED OR NOT ONNX_ENABLED)
        message(FATAL_ERROR "ALIKED native ops require Linux, CUDA and ONNX")
    endif()
    if(NOT CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2" OR
       CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.3")
        message(FATAL_ERROR "ALIKED native ops currently require CUDA 13.2")
    endif()
    target_sources(colmap_feature PRIVATE
        aliked_native/custom_ops.h aliked_native/custom_ops.cc
        aliked_native/model.h aliked_native/model.cc
        aliked_native/ort_status_boundary.h
        aliked_native/fused_concat_op.cc aliked_native/fused_concat_kernel.h
        aliked_native/fused_concat_kernel.cu
        aliked_native/normalized_grid_sample_op.cc
        aliked_native/normalized_grid_sample_kernel.h
        aliked_native/normalized_grid_sample_kernel.cu)
    # Keep these per-source options last, after inherited CUDA math/stream flags.
    set_source_files_properties(
        aliked_native/fused_concat_kernel.cu
        aliked_native/normalized_grid_sample_kernel.cu
        PROPERTIES COMPILE_OPTIONS
            "--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=true;--default-stream=legacy")
    set_property(TARGET colmap_feature PROPERTY CUDA_ARCHITECTURES 86)
    target_compile_definitions(colmap_feature PRIVATE COLMAP_ALIKED_NATIVE_CUDA_ENABLED)
    target_link_libraries(colmap_feature PRIVATE CUDA::cudart)
    COLMAP_ADD_TEST(
        NAME aliked_native_test
        SRCS aliked_native/custom_ops_test.cc aliked_native/test_models.h
        LINK_LIBS
            colmap_feature
            colmap_onnx_util
            onnxruntime::onnxruntime
            CUDA::cudart
    )
endif()

if(CUDA_ENABLED OR HIP_ENABLED)
    # loma.cc calls CudaDeviceSupportsBf16 from colmap_util_cuda.
    target_link_libraries(colmap_feature PRIVATE colmap_util_cuda)
endif()

if(ONNX_ENABLED)
    COLMAP_ADD_TEST(
        NAME aliked_test
        SRCS aliked_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME onnx_matchers_test
        SRCS onnx_matchers_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME lightglue_test
        SRCS lightglue_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
    COLMAP_ADD_TEST(
        NAME loma_test
        SRCS loma_test.cc
        LINK_LIBS
            colmap_feature
            onnxruntime::onnxruntime
    )
endif()
COLMAP_ADD_TEST(
    NAME extractor_test
    SRCS extractor_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME index_test
    SRCS index_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME matcher_test
    SRCS matcher_test.cc
    LINK_LIBS colmap_feature
)
COLMAP_ADD_TEST(
    NAME utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_feature colmap_sensor
)
COLMAP_ADD_TEST(
    NAME sift_test
    SRCS sift_test.cc
    LINK_LIBS
        colmap_feature
        colmap_util
)
if(TESTS_ENABLED AND GUI_ENABLED)
    target_link_libraries(colmap_feature_sift_test Qt${QT_VERSION_MAJOR}::Widgets)
endif()
if(TESTS_ENABLED AND (CUDA_ENABLED OR HIP_ENABLED))
    # NOTE: plain signature to match COLMAP_ADD_TEST() above; mixing
    # keyword and plain signatures for one target is a CMake error.
    target_link_libraries(colmap_feature_sift_test colmap_util_cuda)
endif()
