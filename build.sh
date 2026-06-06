#!/bin/bash
#CC=gcc-12-aarch64-linux-gnu
#CXX=g++-12-aarch64-linux-gnu
set -euo pipefail

DOCKER_IMAGE="${LLAMA_DOCKER_IMAGE:-llama-jammy-xbuild}"

if [[ ! -f /.dockerenv ]] && docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    mkdir -p "$HOME/.android"

    docker run --rm -it \
        --network host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "$PWD":/workspace \
        -v "$HOME/.android":/tmp/.android \
        -w /workspace \
        "$DOCKER_IMAGE" \
        bash build.sh "$@"
    exit 0
fi

cmake -Bbuild -H. \
    -DCMAKE_TOOLCHAIN_FILE=./aarch64-toolchain.cmake\
    -DCMAKE_BUILD_TYPE=Release\
    -DGGML_STATIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF\
    -DCMAKE_C_FLAGS="-march=armv8-a"\
    -DCMAKE_CXX_FLAGS="-march=armv8-a -std=c++17"\
    -DLLAMA_CURL=OFF\
    -DGGML_CPU_AARCH64=OFF \
    -DGGML_RKNPU_RE=ON


# cmake --build build -j10 --target llama-rknpu-prepack

cmake --build build -j20 --target llama-cli
adb push build/bin/llama-cli /data/data/com.termux/files/home/llama-benchmark