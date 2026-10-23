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
    target_sources(colmap_poisson_recon PRIVATE aux_gpu.h aux_gpu.cu native_aux.h.inc)
    target_sources(colmap_poisson_recon PRIVATE trim_gpu.h trim_gpu.cu trim_gpu_adapter.h)
    target_sources(colmap_poisson_recon PRIVATE normal_gpu.h normal_gpu.cu normal_capture.h)
    target_compile_definitions(colmap_poisson_recon PRIVATE POISSON_AUX_NATIVE_GPU POISSON_TRIM_NATIVE_GPU)
    target_compile_definitions(colmap_poisson_recon PRIVATE POISSON_NORMAL_NATIVE_GPU)
    target_compile_options(colmap_poisson_recon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(colmap_poisson_recon PRIVATE CUDA::cudart)
endif()
