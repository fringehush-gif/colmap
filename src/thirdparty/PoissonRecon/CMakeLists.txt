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
    NAME point_evaluation_test
    SRCS point_evaluation_test.cc
    LINK_LIBS colmap_poisson_recon
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_point_evaluation_test
        PRIVATE RELEASE)
    if(MSVC)
        target_compile_options(colmap_thirdparty_point_evaluation_test
            PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_point_evaluation_test
            OpenMP::OpenMP_CXX)
    endif()
endif()
