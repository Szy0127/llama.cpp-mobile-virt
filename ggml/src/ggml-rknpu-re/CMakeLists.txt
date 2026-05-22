ggml_add_backend_library(ggml-rknpu-re
    ../ggml-rknpu-re.cpp
    matmul_cpu_check.c
    npu_interface.c
    npu_matmul.c)

target_compile_definitions(ggml-rknpu-re PRIVATE GGML_BACKEND_BUILD)
target_include_directories(ggml-rknpu-re PRIVATE . .. ../..)
find_package(OpenSSL REQUIRED)
target_link_libraries(ggml-rknpu-re PRIVATE OpenSSL::Crypto)
