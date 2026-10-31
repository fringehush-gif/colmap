# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "mvs")

COLMAP_ADD_LIBRARY(
    NAME colmap_mvs
    SRCS
        advancing_front_meshing.h advancing_front_meshing.cc
        bounded_workspace.h bounded_workspace.cc
        immutable_map_pool.h
        consistency_graph.h consistency_graph.cc
        delaunay_meshing.h delaunay_meshing.cc
        depth_map.h depth_map.cc
        fusion.h fusion.cc
        image.h image.cc
        mat.h mat.cc
        mesh_simplification.h mesh_simplification.cc
        model.h model.cc
        poisson_meshing.h poisson_meshing.cc
        normal_map.h normal_map.cc
        patch_match_options.h patch_match_options.cc
        texture_mapping.h texture_mapping.cc
        transient_pins.h
        workspace.h workspace.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_scene
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_image
        OpenImageIO::OpenImageIO
        colmap_poisson_recon
        Eigen3::Eigen
)
if(CGAL_ENABLED)
    target_link_libraries(colmap_mvs PRIVATE CGAL)
endif()
if(OPENMP_FOUND)
    target_link_libraries(colmap_mvs PRIVATE OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME consistency_graph_test
    SRCS consistency_graph_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME depth_map_test
    SRCS depth_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME exact_depth_state_test
    SRCS exact_depth_state_test.cc
    LINK_LIBS colmap_mvs
)

COLMAP_ADD_TEST(
    NAME fusion_test
    SRCS fusion_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mat_test
    SRCS mat_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mesh_simplification_test
    SRCS mesh_simplification_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME poisson_meshing_test
    SRCS poisson_meshing_test.cc
    LINK_LIBS colmap_mvs
)
# The test directly includes Poisson's OpenMP-dependent ThreadPool header.
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_mvs_poisson_meshing_test OpenMP::OpenMP_CXX)
endif()
COLMAP_ADD_TEST(
    NAME delaunay_meshing_test
    SRCS delaunay_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME advancing_front_meshing_test
    SRCS advancing_front_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME model_test
    SRCS model_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME normal_map_test
    SRCS normal_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME patch_match_io_test
    SRCS patch_match_io_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME texture_mapping_test
    SRCS texture_mapping_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME workspace_test
    SRCS workspace_test.cc
    LINK_LIBS colmap_mvs
)

if(CUDA_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu
            patch_match_source_array.h patch_match_source_array.cc
            patch_match_state_map.h
            patch_match_hmm_checkpoint.h
            patch_match_rng_perimeter.h
            packed_consistency_mask.h
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            CUDA::cudart
            CUDA::curand
    )

    if(NOT HIP_ENABLED)
        include(CheckCXXSourceCompiles)
        include(CMakePushCheckState)
        cmake_push_check_state(RESET)
        set(CMAKE_REQUIRED_LIBRARIES CUDA::cudart)
        set(_colmap_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        # Refresh on reconfiguration so a changed toolkit cannot retain admission.
        unset(COLMAP_CUDA_SPARSE_ARRAY_API CACHE)
        check_cxx_source_compiles([=[
            #define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
            #include <cuda_runtime_api.h>
            #include <cuda.h>
            #include <cudaTypedefs.h>
            static_assert(sizeof(PFN_cuMemCreate_v10020) +
                          sizeof(PFN_cuMemRelease_v10020) +
                          sizeof(PFN_cuMemGetAllocationGranularity_v10020) +
                          sizeof(PFN_cuMemMapArrayAsync_v11010) > 0, "Sparse APIs");
            int main() {
                CUmemAllocationProp allocation = {};
                allocation.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                allocation.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                allocation.location.id = 0;
                allocation.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
                CUarrayMapInfo mapping = {};
                mapping.resourceType = CU_RESOURCE_TYPE_ARRAY;
                mapping.resource.array = nullptr;
                mapping.subresourceType = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
                mapping.subresource.sparseLevel.layer = 0;
                mapping.subresource.sparseLevel.extentWidth = 1;
                mapping.subresource.sparseLevel.extentHeight = 1;
                mapping.subresource.sparseLevel.extentDepth = 1;
                mapping.memOperationType = CU_MEM_OPERATION_TYPE_MAP;
                mapping.memOperationType = CU_MEM_OPERATION_TYPE_UNMAP;
                mapping.memHandleType = CU_MEM_HANDLE_TYPE_GENERIC;
                mapping.memHandle.memHandle = 0;
                mapping.offset = 0;
                mapping.deviceBitMask = 1;
                (void)CU_MEM_ALLOC_GRANULARITY_MINIMUM;
                cudaDeviceProp device = {};
                (void)allocation;
                (void)mapping;
                (void)device.sparseCudaArraySupported;
                (void)(cudaArrayLayered | cudaArraySparse);
                cudaArraySparseProperties sparse = {};
                sparse.tileExtent.width = 1;
                sparse.tileExtent.height = 1;
                sparse.tileExtent.depth = 1;
                sparse.miptailSize = 0;
                void* address = nullptr;
                cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
                if (cudaGetDriverEntryPointByVersion("cuMemMapArrayAsync", &address,
                        11010, cudaEnableLegacyStream, &query) != cudaSuccess) return 1;
                return static_cast<int>(cudaArrayGetSparseProperties(&sparse, nullptr));
            }
        ]=] COLMAP_CUDA_SPARSE_ARRAY_API)
        if(_colmap_try_compile_type)
            set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_colmap_try_compile_type}")
        else()
            unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
        endif()
        unset(_colmap_try_compile_type)
        cmake_pop_check_state()
        if(COLMAP_CUDA_SPARSE_ARRAY_API)
            set_property(SOURCE patch_match_source_array.cc APPEND PROPERTY
                COMPILE_DEFINITIONS COLMAP_SPARSE_SOURCE_RUNTIME=1)
        endif()

        COLMAP_ADD_TEST(
            NAME patch_match_cuda_test
            SRCS patch_match_cuda_test.cc
            LINK_LIBS colmap_mvs_cuda
        )
        COLMAP_ADD_TEST(
            NAME patch_match_source_depth_lease_test
            SRCS patch_match_source_depth_lease_test.cc
            LINK_LIBS colmap_mvs_cuda
        )
        if(TESTS_ENABLED)
            set_tests_properties(mvs/patch_match_cuda_test PROPERTIES TIMEOUT 120)
        endif()

        # GNU --wrap must reach CudaSafeCall in the static utility library.
        if(TESTS_ENABLED AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT BUILD_SHARED_LIBS)
            COLMAP_ADD_TEST(
                NAME cuda_texture_constructor_test
                SRCS cuda_texture_constructor_test.cc
                LINK_LIBS colmap_util_cuda
            )
            target_link_options(colmap_mvs_cuda_texture_constructor_test PRIVATE
                "LINKER:--wrap=cudaCreateChannelDesc"
                "LINKER:--wrap=cudaMalloc3DArray"
                "LINKER:--wrap=cudaCreateTextureObject"
                "LINKER:--wrap=cudaFreeArray"
                "LINKER:--wrap=cudaDestroyTextureObject"
                "LINKER:--wrap=cudaGetErrorString"
            )
            set_tests_properties(mvs/cuda_texture_constructor_test
                PROPERTIES TIMEOUT 15)
        endif()
    endif()

    # Work around NVCC miscompilation on Blackwell GPUs (sm_100+).
    # The PatchMatch kernels are miscompiled at -O2/-O3 when generating native
    # SASS for sm_100+. Instead, we emit sm_90 PTX which Blackwell GPUs
    # JIT-compile at first launch, bypassing the buggy codegen path.
    # See: https://github.com/colmap/colmap/issues/3514
    #      https://github.com/ggml-org/llama.cpp/issues/18331
    set(_COLMAP_BLACKWELL_TARGET FALSE)

    # Classify architecture keywords. "all"/"all-major" expand based on the
    # CUDA toolkit version, while "native" expands based on installed GPUs.
    set(_HAS_ALL_KEYWORD FALSE)
    set(_HAS_NATIVE_KEYWORD FALSE)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_arch STREQUAL "all" OR _arch STREQUAL "all-major")
            set(_HAS_ALL_KEYWORD TRUE)
        elseif(_arch STREQUAL "native")
            set(_HAS_NATIVE_KEYWORD TRUE)
        endif()
    endforeach()

    # Query installed GPUs only for "native", since it targets present hardware.
    if(_HAS_NATIVE_KEYWORD)
        execute_process(
            COMMAND nvidia-smi
                --query-gpu=compute_cap
                --format=csv,noheader
            OUTPUT_VARIABLE _GPU_COMPUTE_CAPS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _NVIDIA_SMI_RESULT
        )
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            string(REPLACE "\n" ";" _GPU_COMPUTE_CAPS
                   "${_GPU_COMPUTE_CAPS}")
        endif()
    endif()

    # Detect Blackwell from explicit numeric architectures.
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
        if(_arch_num MATCHES "^[0-9]+$" AND _arch_num GREATER_EQUAL 100)
            set(_COLMAP_BLACKWELL_TARGET TRUE)
            break()
        endif()
    endforeach()

    # Detect Blackwell from "all"/"all-major": these expand based on the
    # toolkit, not installed GPUs. CUDA 12.8 added sm_100 support.
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_ALL_KEYWORD)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            set(_COLMAP_BLACKWELL_TARGET TRUE)
        endif()
    endif()

    # Detect Blackwell from "native" via nvidia-smi (installed GPU check).
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_NATIVE_KEYWORD)
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                string(STRIP "${_cap}" _cap)
                string(REGEX MATCH "^([0-9]+)" _major "${_cap}")
                if(_major AND _major GREATER_EQUAL 10)
                    set(_COLMAP_BLACKWELL_TARGET TRUE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(_COLMAP_BLACKWELL_TARGET)
        message(WARNING
            "Blackwell GPU (sm_100+) detected. Replacing sm_100+ architectures "
            "with 90-virtual (PTX) for colmap_mvs_cuda to work around an NVCC "
            "compiler bug. Blackwell GPUs will JIT-compile from PTX at first "
            "launch (see https://github.com/colmap/colmap/issues/3514).")

        # Build filtered architecture list: keep archs < 100, drop >= 100.
        set(_COLMAP_MVS_CUDA_ARCHS)
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                if(_arch_num LESS 100)
                    list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
                endif()
            elseif(_arch STREQUAL "all")
                # "all"/"all-major" keywords require CMake 3.24+, which
                # provides CMAKE_CUDA_ARCHITECTURES_ALL[_MAJOR] variables.
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "all-major")
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "native")
                # Replace with explicit non-Blackwell GPU architectures.
                if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
                    foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                        string(STRIP "${_cap}" _cap)
                        string(REGEX MATCH
                               "^([0-9]+)\.([0-9]+)" _match "${_cap}")
                        if(_match)
                            math(EXPR _sm
                                 "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
                            if(_sm LESS 100)
                                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_sm}")
                            endif()
                        endif()
                    endforeach()
                endif()
            else()
                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
            endif()
        endforeach()

        # Add PTX for the highest pre-Blackwell arch so Blackwell GPUs
        # can JIT-compile at runtime.
        list(APPEND _COLMAP_MVS_CUDA_ARCHS "90-virtual")
        list(REMOVE_DUPLICATES _COLMAP_MVS_CUDA_ARCHS)

        set_target_properties(colmap_mvs_cuda PROPERTIES
            CUDA_ARCHITECTURES "${_COLMAP_MVS_CUDA_ARCHS}"
        )
    endif()

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME cuda_texture_host_slices_test
        SRCS cuda_texture_host_slices_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
endif()

# ROCm HIP support for MVS (patch_match_stereo on AMD GPUs).
#
# enable_language(HIP) and find_package(hip/hiprand/rocrand) are handled
# centrally in cmake/FindDependencies.cmake. Here we just mark the three
# .cu translation units as LANGUAGE HIP so CMake hands them to the HIP
# toolchain (clang++ -x hip) instead of the CUDA one. The source itself
# is unmodified; symbol remapping happens in src/colmap/util/cuda_to_hip.h.
if(HIP_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu
            patch_match_source_array.h patch_match_source_array.cc
            patch_match_state_map.h
            patch_match_hmm_checkpoint.h
            patch_match_rng_perimeter.h
            packed_consistency_mask.h
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            hip::host
            hip::hiprand
            roc::rocrand
    )
    set_source_files_properties(
        gpu_mat_prng.cu
        gpu_mat_ref_image.cu
        patch_match_cuda.cu
        PROPERTIES LANGUAGE HIP
    )

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    set_source_files_properties(gpu_mat_test.cu PROPERTIES LANGUAGE HIP)
endif()
