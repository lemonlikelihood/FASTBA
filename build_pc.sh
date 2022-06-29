#!/bin/bash -e

PROJ_DIR=$(
    cd "$(dirname "$0")"
    pwd
)
echo $PROJ_DIR

export CC=$(which clang)
export CXX=$(which clang++)

echo "Warning: 1. Run this script only when you are in ubuntu."
echo "         2. Ensure CC and CXX are set correctly."
echo "    CC =$CC"
echo "    CXX=$CXX"


CMAKE_BUILD_TYPE="RelWithDebInfo"

cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -G Ninja \
    -D PROJECTION_BUILD_LOGGING=ON \
    -D PROJECTION_ENABLE_FORENSICS=OFF

cmake --build build --config ${CMAKE_BUILD_TYPE}