if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_poisson_recon
    SRCS
        PoissonRecon.h PoissonRecon.cpp
        SurfaceTrimmer.h SurfaceTrimmer.cpp
)
target_compile_definitions(colmap_poisson_recon PRIVATE RELEASE)
target_include_directories(colmap_poisson_recon PRIVATE .)
if(OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(colmap_poisson_recon PRIVATE /bigobj)
endif()

COLMAP_ADD_TEST(
    NAME vertex_factory_test
    SRCS vertex_factory_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_thirdparty_vertex_factory_test
        OpenMP::OpenMP_CXX)
endif()
if(TESTS_ENABLED AND COVERAGE_ENABLED)
    target_link_options(colmap_thirdparty_vertex_factory_test PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME ply_scalar_test
    SRCS ply_scalar_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_ply_scalar_test PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_ply_scalar_test OpenMP::OpenMP_CXX)
    endif()
endif()

if(TESTS_ENABLED AND COVERAGE_ENABLED)
    # These sibling-directory tests link instrumented COLMAP libraries.
    target_link_options(colmap_thirdparty_ply_scalar_test
        PRIVATE -coverage)
endif()
