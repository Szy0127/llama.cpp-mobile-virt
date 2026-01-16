#include "matmul_cpu_check.h"

void cpu_matmul_fp16_fp16_fp32(int m, int k, int n, const ggml_fp16_t *src0 , const ggml_fp16_t *src1, float* dst) {
    // CPU fallback implementation for fp16 matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += GGML_FP16_TO_FP32(src0[i * k + p]) * GGML_FP16_TO_FP32(src1[p * n + j]);
            }
            dst[i * n + j] = sum;
        }
    }
}
