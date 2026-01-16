#include "matmul_cpu_check.h"
#include "ggml.h"

void cpu_matmul_fp16_fp16_fp32(int m, int k, int n, const ggml_fp16_t *src0 , const ggml_fp16_t *src1, float* dst) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0;
      for (int l = 0; l < k; l++) {
        sum += ggml_fp16_to_fp32(src0[i*k + l]) * ggml_fp16_to_fp32(src1[l*n + j]);
      }
     dst[i*n + j] = sum;
    }
  }
}
