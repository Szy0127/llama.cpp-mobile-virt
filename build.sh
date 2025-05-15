#!/bin/bash
#CC=gcc-12-aarch64-linux-gnu
#CXX=g++-12-aarch64-linux-gnu
cmake -Bbuild -H. \
    -DCMAKE_TOOLCHAIN_FILE=./aarch64-toolchain.cmake\
    -DCMAKE_BUILD_TYPE=Debug\
    -DGGML_STATIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF\
    -DCMAKE_C_FLAGS="-march=armv8-a"\
    -DCMAKE_CXX_FLAGS="-march=armv8-a"\
    -DLLAMA_CURL=OFF\
    -DGGML_CPU_AARCH64=OFF



cmake --build build -j20
