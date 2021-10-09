cmake_minimum_required(VERSION 3.19)
project(FASTBA)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")
find_package(Ceres REQUIRED)

find_package(Eigen3)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})


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
target_link_libraries(FASTBA ${CERES_LIBRARIES} ${EIEGEN3_LIBS})