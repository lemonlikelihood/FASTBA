cmake_minimum_required(VERSION 3.15)
project(FASTBA VERSION 2.3.2.0 LANGUAGES CXX)

set(CMAKE_POLICY_DEFAULT_CMP0120 OLD)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")

set(THIRDPARTY_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty")
set(THIRDPARTY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps")


option(PROJECTION_BUILD_LOGGING "Should be off for release. Log code won't be compiled if off." ON)
message(STATUS "PROJECTION_BUILD_LOGGING: ${PROJECTION_BUILD_LOGGING}")
# 3rdparty libraries

## spdlog

FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.10.0.tar.gz
    URL_HASH MD5=effea813cd81cfb5588806c5754e14f1
    DOWNLOAD_NAME spdlog-1.10.0.tar.gz
    DOWNLOAD_DIR ${THIRDPARTY_SOURCE_DIR}
    # SOURCE_DIR ${THIRDPARTY_SOURCE_DIR}/spdlog-src
)
FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
endif()
add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
add_library(depends::spdlog INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::spdlog INTERFACE spdlog::spdlog_header_only)
set(spdlog-source-dir ${spdlog_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(spdlog-binary-dir ${spdlog_BINARY_DIR} CACHE INTERNAL "" FORCE)
mark_as_advanced(spdlog-source-dir)
mark_as_advanced(spdlog-binary-dir)

# eigen
add_library(depends::eigen INTERFACE IMPORTED GLOBAL)
target_compile_definitions(depends::eigen INTERFACE EIGEN_MAX_ALIGN_BYTES=16)
target_include_directories(
    depends::eigen INTERFACE ${THIRDPARTY_SOURCE_DIR}/eigen-3.4.0
)

# opencv 
find_package(OpenCV REQUIRED)
add_library(depends::opencv INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::opencv INTERFACE ${OpenCV_LIBS})

# # ceres-solver
find_package(Ceres REQUIRED)
add_library(depends::ceres INTERFACE IMPORTED GLOBAL)
target_compile_definitions(depends::ceres INTERFACE EIGEN_MAX_ALIGN_BYTES=16)
target_link_libraries(depends::ceres INTERFACE ceres)

# glfw (dependency of lightvis)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(
    ${THIRDPARTY_SOURCE_DIR}/glfw-3.3.4 ${THIRDPARTY_BINARY_DIR}/glfw-3.3.4
)
add_library(depends::glfw INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::glfw INTERFACE glfw)

# glbinding (dependency of lightvis)
set(OPTION_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(OPTION_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(
    ${THIRDPARTY_SOURCE_DIR}/glbinding-3.1.0
    ${THIRDPARTY_BINARY_DIR}/glbinding-3.1.0
)
find_package(
    glbinding
    REQUIRED
    PATHS
    "${THIRDPARTY_SOURCE_DIR}/glbinding-3.1.0"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
add_library(depends::glbinding INTERFACE IMPORTED GLOBAL)
target_link_libraries(
    depends::glbinding INTERFACE glbinding::glbinding
                                 glbinding::glbinding-aux
)

# nuklear (dependency of lightvis)
add_library(depends::nuklear INTERFACE IMPORTED GLOBAL)
target_compile_definitions(
    depends::nuklear
    INTERFACE NK_INCLUDE_FIXED_TYPES
              NK_INCLUDE_DEFAULT_ALLOCATOR
              NK_INCLUDE_STANDARD_IO
              NK_INCLUDE_STANDARD_VARARGS
              NK_INCLUDE_VERTEX_BUFFER_OUTPUT
              NK_INCLUDE_FONT_BAKING
              NK_INCLUDE_DEFAULT_FONT
              NK_KEYSTATE_BASED_INPUT
)
target_include_directories(
    depends::nuklear
    INTERFACE
        ${THIRDPARTY_SOURCE_DIR}/Nuklear-6322f53cb569bb989873ee8a1b71ae705b230708
)

# lightvis dependency
add_library(dep-lightvis)
target_sources(
    dep-lightvis
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/image.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/image.h
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/lightvis_font_roboto.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/lightvis_font_roboto.h
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/lightvis.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/lightvis.h
           ${CMAKE_CURRENT_SOURCE_DIR}/lightvis/shader.h
)
target_include_directories(
    dep-lightvis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(
    dep-lightvis PUBLIC -DGL_SILENCE_DEPRECATION
)
target_link_libraries(
    dep-lightvis
    PUBLIC depends::eigen
           depends::glfw
           depends::opencv
           depends::glbinding
           depends::nuklear
)

configure_file(src/cmake/version.h.in include/fastba/version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_executable(FASTBA main.cpp
               src/optimizer/local_parameterization.cc
               src/optimizer/ceres_bal.cpp
               src/optimizer/read_bal.cpp
               src/optimizer/map.cpp
               src/utils/read_file.cpp
               src/optimizer/problem.cpp
               src/optimizer/lm_optimizer.cpp
               src/optimizer/read_colmap.cpp
               src/optimizer/solver.cpp
               )
target_link_libraries(FASTBA PUBLIC depends::eigen depends::opencv depends::ceres)

add_library(dep-dataset)
target_sources(dep-dataset PUBLIC
               dataset/dataset_reader.cpp
               dataset/dior_dataset_reader.cpp
               dataset/euroc_dataset_reader.cpp
               dataset/libsensors.cpp
               dataset/sensors_dataset_reader.cpp
               )
target_link_libraries(dep-dataset depends::eigen depends::ceres depends::opencv)

add_executable(run_euroc demo/run_euroc.cpp)
target_link_libraries(run_euroc PUBLIC dep-dataset)

add_executable(run_lightvis demo/run_lightvis.cpp)
target_link_libraries(run_lightvis PUBLIC dep-dataset dep-lightvis)

add_executable(run_log_forensics demo/run_log_forensics.cpp src/utils/forensics.cpp)
target_link_libraries(run_log_forensics PUBLIC depends::spdlog)
message("version: " ${CMAKE_CURRENT_BINARY_DIR}/include/version.h)