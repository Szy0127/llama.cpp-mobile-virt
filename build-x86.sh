#!/bin/bash
#CC=gcc-12-aarch64-linux-gnu
#CXX=g++-12-aarch64-linux-gnu
cmake -Bbuild-x86 -H. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_STATIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CPU_AARCH64=OFF\
    -DLLAMA_CURL=OFF\



cmake --build build-x86 -j20
