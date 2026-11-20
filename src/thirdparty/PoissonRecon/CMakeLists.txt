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
    NAME level_set_barycenter_test
    SRCS level_set_barycenter_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_level_set_barycenter_test
        PRIVATE RELEASE)
    if(MSVC)
        target_compile_options(colmap_thirdparty_level_set_barycenter_test
            PRIVATE /bigobj /fp:precise)
    else()
        # The scalar regression deliberately uses a defined NaN default value.
        target_compile_options(colmap_thirdparty_level_set_barycenter_test
            PRIVATE -fno-fast-math)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_level_set_barycenter_test
            OpenMP::OpenMP_CXX)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(colmap_thirdparty_level_set_barycenter_test
            PRIVATE -coverage)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME level_set_color_owner_test
    SRCS level_set_color_owner_test.cc
)
if(TESTS_ENABLED)
    # Use atomic edge flags so the controlled contender schedule has no
    # unrelated volatile-read race in the original ownership implementation.
    target_compile_definitions(colmap_thirdparty_level_set_color_owner_test
        PRIVATE RELEASE SANITIZED_PR)
    if(MSVC)
        target_compile_options(colmap_thirdparty_level_set_color_owner_test
            PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_level_set_color_owner_test
            OpenMP::OpenMP_CXX)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(colmap_thirdparty_level_set_color_owner_test
            PRIVATE -coverage)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME ply_floating_properties_test
    SRCS ply_floating_properties_test.cc
)
COLMAP_ADD_TEST(
    NAME output_position_precision_test
    SRCS output_position_precision_test.cc
    LINK_LIBS colmap_poisson_recon
)
if(TESTS_ENABLED)
    foreach(test_name ply_floating_properties_test output_position_precision_test)
        target_compile_definitions(colmap_thirdparty_${test_name} PRIVATE RELEASE)
        if(MSVC)
            target_compile_options(colmap_thirdparty_${test_name} PRIVATE /bigobj)
        endif()
        if(OPENMP_FOUND)
            target_link_libraries(colmap_thirdparty_${test_name}
                OpenMP::OpenMP_CXX)
        endif()
        if(COVERAGE_ENABLED)
            target_link_options(colmap_thirdparty_${test_name} PRIVATE -coverage)
        endif()
    endforeach()
endif()
