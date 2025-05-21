cmake -Bbuild -H.\
  -DCMAKE_TOOLCHAIN_FILE=../hm-sdk/native/build/cmake/ohos.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release\
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF\
  -DOHOS_PLATFORM=OHOS\
  -DGGML_STATIC=ON \
  -DBUILD_SHARED_LIBS=OFF\
  -DGGML_CPU_AARCH64=OFF\
  -DLLAMA_CURL=OFF

  #-DCMAKE_TOOLCHAIN_FILE=./toolchain.cmake \
  cmake --build build -j50
