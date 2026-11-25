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
    NAME attribute_state_test
    SRCS attribute_state_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_attribute_state_test
        PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_attribute_state_test
            OpenMP::OpenMP_CXX)
    endif()
endif()

if(TESTS_ENABLED AND COVERAGE_ENABLED)
    # These sibling-directory tests link instrumented COLMAP libraries.
    target_link_options(colmap_thirdparty_attribute_state_test
        PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME geometry_test
    SRCS geometry_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_thirdparty_geometry_test
        OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME surface_trimmer_geometry_test
    SRCS surface_trimmer_geometry_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_surface_trimmer_geometry_test
        PRIVATE RELEASE)
    if(MSVC)
        target_compile_options(colmap_thirdparty_surface_trimmer_geometry_test
            PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_surface_trimmer_geometry_test
            OpenMP::OpenMP_CXX)
    endif()
endif()

if(TESTS_ENABLED AND COVERAGE_ENABLED)
    # These sibling-directory tests link instrumented COLMAP libraries.
    target_link_options(colmap_thirdparty_geometry_test PRIVATE -coverage)
    target_link_options(colmap_thirdparty_surface_trimmer_geometry_test
        PRIVATE -coverage)
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

COLMAP_ADD_TEST(
    NAME output_position_precision_test
    SRCS output_position_precision_test.cc
    LINK_LIBS colmap_poisson_recon
)
COLMAP_ADD_TEST(
    NAME level_set_position_precision_test
    SRCS level_set_position_precision_test.cc
)
if(TESTS_ENABLED)
    foreach(test_target
            colmap_thirdparty_output_position_precision_test
            colmap_thirdparty_level_set_position_precision_test)
        target_compile_definitions(${test_target} PRIVATE RELEASE)
        if(OPENMP_FOUND)
            target_link_libraries(${test_target} OpenMP::OpenMP_CXX)
        endif()
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /bigobj)
        endif()
        if(COVERAGE_ENABLED)
            target_link_options(${test_target} PRIVATE -coverage)
        endif()
    endforeach()
endif()
