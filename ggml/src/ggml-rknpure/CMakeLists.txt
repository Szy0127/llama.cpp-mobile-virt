ggml_add_backend_library(ggml-rknpure
    ggml-rknpure.cpp
    npu_matmul.c
    matmul_cpu_check.c)

target_compile_definitions(ggml-rknpure PRIVATE GGML_BACKEND_BUILD)
target_include_directories(ggml-rknpure PRIVATE . .. ../..)
