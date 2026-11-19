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
    NAME regular_tree_neighbors_test
    SRCS regular_tree_neighbors_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(${COLMAP_ADD_TEST_TARGET} PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(${COLMAP_ADD_TEST_TARGET} OpenMP::OpenMP_CXX)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(${COLMAP_ADD_TEST_TARGET} PRIVATE -coverage)
    endif()
endif()
