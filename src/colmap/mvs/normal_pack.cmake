# SPDX-License-Identifier: BSD-3-Clause
# Include after colmap_mvs is defined. No runtime library or download is needed.
# Preserve strict adapter/test arithmetic even when callers otherwise
# permit FMA or change FP modes.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/normal_pack.cc"
                      "${CMAKE_CURRENT_SOURCE_DIR}/normal_pack_test.cc" APPEND PROPERTY
    COMPILE_OPTIONS -fno-fast-math -ffp-contract=off -frounding-math -fno-math-errno)
endif()
option(ISPC_NORMAL_PACK_ENABLED "Optional AVX2 ISPC normal packing" OFF)
set(COLMAP_ISPC_NORMAL_PACK_TARGET "avx2-i32x4" CACHE STRING "Measured ISPC lane width")
set_property(CACHE COLMAP_ISPC_NORMAL_PACK_TARGET PROPERTY STRINGS avx2-i32x4 avx2-i32x8)
if(ISPC_NORMAL_PACK_ENABLED)
  find_program(COLMAP_ISPC_EXECUTABLE NAMES ispc DOC "Optional ISPC compiler")
  if(COLMAP_ISPC_EXECUTABLE AND NOT CMAKE_CROSSCOMPILING
      AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
      AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
      AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    if(NOT COLMAP_ISPC_NORMAL_PACK_TARGET MATCHES "^avx2-i32x(4|8)$")
      message(FATAL_ERROR "Only AVX2 x4/x8 ISPC targets are supported")
    endif()
    set(normal_pack_generated "${CMAKE_CURRENT_BINARY_DIR}/normal_pack_generated")
    file(MAKE_DIRECTORY "${normal_pack_generated}")
    set(normal_pack_header "${normal_pack_generated}/normal_pack_ispc.h")
    set(normal_pack_object "${normal_pack_generated}/normal_pack_ispc.o")
    add_custom_command(OUTPUT "${normal_pack_header}" "${normal_pack_object}"
      COMMAND "${CMAKE_COMMAND}" -E env --unset=ISPC_ARGS
        "${COLMAP_ISPC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/normal_pack.ispc"
        -O2 --arch=x86-64 --target=${COLMAP_ISPC_NORMAL_PACK_TARGET}
        --opt=disable-fma --pic -h "${normal_pack_header}" -o "${normal_pack_object}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/normal_pack.ispc" "${COLMAP_ISPC_EXECUTABLE}"
      VERBATIM)
    add_custom_target(colmap_normal_pack_ispc_codegen DEPENDS "${normal_pack_header}" "${normal_pack_object}")
    # The generated ABI is private to the adapter; public mat.h is unchanged.
    add_dependencies(colmap_mvs colmap_normal_pack_ispc_codegen)
    set_source_files_properties("${normal_pack_object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_sources(colmap_mvs PRIVATE "${normal_pack_object}")
    target_include_directories(colmap_mvs PRIVATE "$<BUILD_INTERFACE:${normal_pack_generated}>")
    target_compile_definitions(colmap_mvs PRIVATE COLMAP_HAVE_ISPC_NORMAL_PACK=1)
    message(STATUS "ISPC normal packing enabled: ${COLMAP_ISPC_NORMAL_PACK_TARGET}")
  else()
    message(STATUS "ISPC compiler/platform unavailable: normal packing keeps C++ fallback")
  endif()
endif()
