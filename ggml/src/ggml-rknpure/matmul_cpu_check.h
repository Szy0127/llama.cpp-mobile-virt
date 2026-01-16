#ifndef MATMUL_CPU_CHECK_H
#define MATMUL_CPU_CHECK_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_matmul_fp16_fp16_fp32(int m, int k, int n, const ggml_fp16_t *src0 , const ggml_fp16_t *src1, float* dst);

#ifdef __cplusplus
}
#endif
#endif // MATMUL_CPU_CHECK_H
