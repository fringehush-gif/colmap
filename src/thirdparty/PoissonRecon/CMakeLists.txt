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
    NAME fem_tree_test
    SRCS fem_tree_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_fem_tree_test PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_fem_tree_test
            OpenMP::OpenMP_CXX)
    endif()
endif()

if(TESTS_ENABLED AND COVERAGE_ENABLED)
    # These sibling-directory tests link instrumented COLMAP libraries.
    target_link_options(colmap_thirdparty_fem_tree_test
        PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME fem_constraints_test
    SRCS fem_constraints_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(${COLMAP_ADD_TEST_TARGET} PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(${COLMAP_ADD_TEST_TARGET} OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(${COLMAP_ADD_TEST_TARGET} PRIVATE /bigobj)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(${COLMAP_ADD_TEST_TARGET} PRIVATE -coverage)
    endif()
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
    NAME surface_trimmer_clipping_test
    SRCS
        SurfaceTrimmer.cpp
        surface_trimmer_clipping_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_surface_trimmer_clipping_test
        PRIVATE RELEASE)
    if(MSVC)
        target_compile_options(colmap_thirdparty_surface_trimmer_clipping_test
            PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_surface_trimmer_clipping_test
            OpenMP::OpenMP_CXX)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(colmap_thirdparty_surface_trimmer_clipping_test
            PRIVATE -coverage)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME atomic_read_test
    SRCS atomic_read_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_thirdparty_atomic_read_test
        OpenMP::OpenMP_CXX)
endif()

if(TESTS_ENABLED AND COVERAGE_ENABLED)
    # This sibling-directory test links instrumented COLMAP libraries.
    target_link_options(colmap_thirdparty_atomic_read_test PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME point_evaluation_test
    SRCS point_evaluation_test.cc
)
if(TESTS_ENABLED)
    # Header-instantiated evaluator; deep nodes cover nudges across small cells.
    target_compile_definitions(colmap_thirdparty_point_evaluation_test
        PRIVATE RELEASE USE_DEEP_TREE_NODES)
    if(COVERAGE_ENABLED)
        target_link_options(colmap_thirdparty_point_evaluation_test
            PRIVATE -coverage)
    endif()
    if(MSVC)
        target_compile_options(colmap_thirdparty_point_evaluation_test
            PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_point_evaluation_test
            OpenMP::OpenMP_CXX)
    endif()
endif()

COLMAP_ADD_TEST(
    NAME sparse_matrix_test
    SRCS sparse_matrix_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(${COLMAP_ADD_TEST_TARGET} OpenMP::OpenMP_CXX)
endif()
if(TESTS_ENABLED AND COVERAGE_ENABLED)
    target_link_options(${COLMAP_ADD_TEST_TARGET} PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME weighted_samples_test
    SRCS weighted_samples_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(${COLMAP_ADD_TEST_TARGET} PRIVATE RELEASE)
    if(OPENMP_FOUND)
        target_link_libraries(${COLMAP_ADD_TEST_TARGET} OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(${COLMAP_ADD_TEST_TARGET} PRIVATE /bigobj)
    endif()
endif()
if(TESTS_ENABLED AND COVERAGE_ENABLED)
    target_link_options(${COLMAP_ADD_TEST_TARGET} PRIVATE -coverage)
endif()

COLMAP_ADD_TEST(
    NAME regular_tree_test
    SRCS regular_tree_test.cc
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
    NAME ply_read_test
    SRCS ply_read_test.cc
)
if(TESTS_ENABLED)
    target_compile_definitions(colmap_thirdparty_ply_read_test PRIVATE RELEASE)
    if(MSVC)
        target_compile_options(colmap_thirdparty_ply_read_test PRIVATE /bigobj)
    endif()
    if(OPENMP_FOUND)
        target_link_libraries(colmap_thirdparty_ply_read_test OpenMP::OpenMP_CXX)
    endif()
    if(COVERAGE_ENABLED)
        target_link_options(colmap_thirdparty_ply_read_test PRIVATE -coverage)
    endif()
endif()

set(FOLDER_NAME "poisson_recon")
COLMAP_ADD_TEST(
    NAME vertex_factory_test
    SRCS vertex_factory_test.cc
)
if(TESTS_ENABLED AND OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon_vertex_factory_test
        OpenMP::OpenMP_CXX)
endif()
if(TESTS_ENABLED AND COVERAGE_ENABLED)
    target_link_options(colmap_poisson_recon_vertex_factory_test PRIVATE -coverage)
endif()
