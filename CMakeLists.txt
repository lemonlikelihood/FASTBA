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

## lyra
FetchContent_Declare(
    lyra
    URL https://github.com/bfgroup/Lyra/archive/refs/tags/1.6.tar.gz
    URL_HASH MD5=08b7b28aacb0bf07f28dbd5d0858c6ea
    DOWNLOAD_NAME lyra-1.6.tar.gz
    DOWNLOAD_DIR ${THIRDPARTY_SOURCE_DIR}
    # SOURCE_DIR ${THIRDPARTY_SOURCE_DIR}/lyra-src
)
FetchContent_GetProperties(lyra)
if(NOT lyra_POPULATED)
    FetchContent_Populate(lyra)
endif()
add_subdirectory(${lyra_SOURCE_DIR} ${lyra_BINARY_DIR})
add_library(depends::lyra INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::lyra INTERFACE bfg::lyra)
set(lyra-source-dir ${lyra_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(lyra-binary-dir ${lyra_BINARY_DIR} CACHE INTERNAL "" FORCE)
mark_as_advanced(lyra-source-dir)
mark_as_advanced(lyra-binary-dir)

## yaml-cpp
FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.tar.gz
    URL_HASH MD5=74d646a3cc1b5d519829441db96744f0
    DOWNLOAD_NAME yaml-cpp-0.7.0.tar.gz
    DOWNLOAD_DIR ${THIRDPARTY_SOURCE_DIR}
    # SOURCE_DIR ${THIRDPARTY_SOURCE_DIR}/yaml-cpp-src
)
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
    FetchContent_Populate(yaml-cpp)
endif()
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
add_library(depends::yaml-cpp INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::yaml-cpp INTERFACE yaml-cpp::yaml-cpp)
set(yaml-cpp-source-dir ${yaml-cpp_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(yaml-cpp-binary-dir ${yaml-cpp_BINARY_DIR} CACHE INTERNAL "" FORCE)
mark_as_advanced(yaml-cpp-source-dir)
mark_as_advanced(yaml-cpp-binary-dir)

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

include_directories(/usr/local/include/eigen3)
# eigen
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH MD5=4c527a9171d71a72a9d4186e65bea559
    DOWNLOAD_NAME eigen-3.4.0.tar.gz
    DOWNLOAD_DIR ${THIRDPARTY_SOURCE_DIR}
    # SOURCE_DIR ${THIRDPARTY_SOURCE_DIR}/eigen-src
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
file(READ "${eigen_SOURCE_DIR}/Eigen/src/Core/util/Macros.h" _eigen_version_header)
string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")
set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})

set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/eigen3" CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen header files are installed")
set(CMAKEPACKAGE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/eigen3/cmake" CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen3Config.cmake is installed")
# message(STATUS "${CMAKE_INSTALL_INCLUDEDIR} ${CMAKE_INSTALL_DATADIR}")
# message(STATUS "${INCLUDE_INSTALL_DIR} ${CMAKEPACKAGE_INSTALL_DIR}")

set(EIGEN_VERSION_STRING ${EIGEN_VERSION_NUMBER})
set(EIGEN_VERSION_MAJOR  ${EIGEN_WORLD_VERSION})
set(EIGEN_VERSION_MINOR  ${EIGEN_MAJOR_VERSION})
set(EIGEN_VERSION_PATCH  ${EIGEN_MINOR_VERSION})
set(EIGEN_DEFINITIONS "")
set(EIGEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}")
set(EIGEN_ROOT_DIR ${CMAKE_INSTALL_PREFIX})

add_library(eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS eigen)
target_include_directories(eigen INTERFACE
    $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)
### Export as title case Eigen
set_target_properties (eigen PROPERTIES EXPORT_NAME Eigen)
install(TARGETS eigen EXPORT Eigen3Targets)
configure_package_config_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/Eigen3Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
    PATH_VARS EIGEN_INCLUDE_DIR EIGEN_ROOT_DIR
    INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO # Eigen does not provide components
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake
    VERSION ${EIGEN_VERSION_NUMBER}
    COMPATIBILITY SameMajorVersion
)
export(TARGETS eigen NAMESPACE Eigen3:: FILE Eigen3Targets.cmake)
export(PACKAGE Eigen3)
install(
    EXPORT Eigen3Targets
    NAMESPACE Eigen3::
    DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/UseEigen3.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake
    DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
)

add_library(depends::eigen INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::eigen INTERFACE Eigen3::Eigen)
set(eigen-source-dir ${eigen_SOURCE_DIR} CACHE INTERNAL "" FORCE)
set(eigen-binary-dir ${eigen_BINARY_DIR} CACHE INTERNAL "" FORCE)
mark_as_advanced(eigen-source-dir)
mark_as_advanced(eigen-binary-dir)

# opencv 
find_package(OpenCV 4 REQUIRED)
add_library(depends::opencv INTERFACE IMPORTED GLOBAL)
target_link_libraries(depends::opencv INTERFACE ${OpenCV_LIBS})
message(STATUS "opencv include path: ${OpenCV_INCLUDE_DIRS}" )

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
            #    dataset/dior_dataset_reader.cpp
               dataset/euroc_dataset_reader.cpp
               dataset/opencv_image.cpp
            #    dataset/libsensors.cpp
            #    dataset/sensors_dataset_reader.cpp
               )
target_link_libraries(dep-dataset depends::eigen depends::ceres depends::opencv depends::yaml-cpp depends::spdlog)

add_executable(run_euroc 
                demo/run_euroc.cpp 
                src/fastba/fastba.cpp 
                src/map/frame.cpp 
                src/map/map.cpp
                src/map/sliding_window.cpp
                # src/map/tracker.cpp
                src/map/feature.cpp
                src/optimizer/ceres/preintegration_error_cost.h
                src/optimizer/ceres/reprojection_error_cost.h
                src/optimizer/factor.cpp
                src/optimizer/preintegrator.cpp
                src/optimizer/pnp.cpp
                src/optimizer/bundle_adjustor.cpp
                src/optimizer/initializer.cpp
                src/geometry/stereo.cpp
                src/geometry/essential.cpp
                src/geometry/homography.cpp
                src/geometry/lie_algebra.cpp
                src/utils/euler_angle.cpp
                )

target_link_libraries(run_euroc PUBLIC dep-dataset depends::spdlog depends::opencv depends::lyra)

# add_executable(run_lightvis demo/run_lightvis.cpp)
# target_link_libraries(run_lightvis PUBLIC dep-dataset dep-lightvis)

# add_executable(run_log_forensics demo/run_log_forensics.cpp src/utils/forensics.cpp)
# target_link_libraries(run_log_forensics PUBLIC depends::spdlog)
# message("version: " ${CMAKE_CURRENT_BINARY_DIR}/include/version.h)