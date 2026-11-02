if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_poisson_recon
    SRCS
        PoissonRecon.h PoissonRecon.cpp
        SurfaceTrimmer.h SurfaceTrimmer.cpp
        PolygonIndexChain.h
)
target_compile_definitions(colmap_poisson_recon PRIVATE RELEASE)
target_include_directories(colmap_poisson_recon PRIVATE .)
if(OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(colmap_poisson_recon PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

if(CUDA_ENABLED AND NOT HIP_ENABLED)
    # The optional CUDA backend requires the shared runtime-only allocation header.
    # Standalone CUDA-aux packaging must include this declared project dependency.
    target_sources(colmap_poisson_recon PRIVATE aux_gpu.h aux_gpu.cu native_aux.h.inc
        ../../colmap/util/cuda_allocation.h)
    target_include_directories(colmap_poisson_recon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    target_sources(colmap_poisson_recon PRIVATE trim_gpu.h trim_gpu.cu trim_gpu_adapter.h)
    target_sources(colmap_poisson_recon PRIVATE normal_gpu.h normal_gpu.cu normal_capture.h)
    target_compile_definitions(colmap_poisson_recon PRIVATE POISSON_AUX_NATIVE_GPU POISSON_TRIM_NATIVE_GPU)
    target_compile_definitions(colmap_poisson_recon PRIVATE POISSON_NORMAL_NATIVE_GPU)
    target_compile_options(colmap_poisson_recon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(colmap_poisson_recon PRIVATE CUDA::cudart)
endif()

# The bounded matrix is a private PoissonRecon type; keep its test isolated
# from the MVS library and its CUDA dependencies.
set(FOLDER_NAME "poisson_recon")
if(TESTS_ENABLED AND CUDA_ENABLED AND NOT HIP_ENABLED)
    COLMAP_ADD_TEST(
        NAME aux_gpu_failure_test
        SRCS aux_gpu_failure_test.cu
        LINK_LIBS CUDA::cudart
    )
    target_include_directories(colmap_poisson_recon_aux_gpu_failure_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    target_compile_options(colmap_poisson_recon_aux_gpu_failure_test
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()
COLMAP_ADD_TEST(
    NAME sparse_matrix_test
    SRCS sparse_matrix_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon_sparse_matrix_test
        OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME template_return_test
    SRCS template_return_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon_template_return_test
        OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME fem_tree_test
    SRCS fem_tree_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_poisson_recon_fem_tree_test
        PRIVATE _GLIBCXX_ASSERTIONS)
    if(OPENMP_FOUND)
        target_link_libraries(colmap_poisson_recon_fem_tree_test
            OpenMP::OpenMP_CXX)
    endif()
    if(OPENMP_FOUND AND CUDA_ENABLED AND NOT HIP_ENABLED)
        target_compile_definitions(colmap_poisson_recon_fem_tree_test PRIVATE
            RELEASE POISSON_AUX_NATIVE_GPU POISSON_TRIM_NATIVE_GPU
            POISSON_NORMAL_NATIVE_GPU)
        target_link_libraries(colmap_poisson_recon_fem_tree_test
            colmap_poisson_recon)
    endif()
endif()

# Exercise the private PLY writer independently of the Poisson executable.
COLMAP_ADD_TEST(
    NAME ply_writer_test
    SRCS ply_writer_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon_ply_writer_test
        OpenMP::OpenMP_CXX)
endif()
