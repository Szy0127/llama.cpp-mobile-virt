#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#define GGML_RKNPU2_NAME "RKNPURE"
#define GGML_RKNPU2_MAX_DEVICES 1
// backend API
GGML_API  ggml_backend_t ggml_backend_rknpure_init(int32_t device);

GGML_API  bool ggml_backend_is_rknpu2(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_API  void ggml_backend_rknpu2_set_n_threads(ggml_backend_t backend_rknpu2, int n_threads);
int ggml_rknpure_can_mul_mat_b(const ggml_tensor * tensor);
int ggml_rknpure_transform_tensor(const void * data, ggml_tensor * tensor, size_t offset, size_t size);
void ggml_rknpu2_transform_tensor_back(void * data, const ggml_tensor * tensor, size_t offset, size_t size);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API  ggml_backend_buffer_type_t ggml_backend_rknpu2_host_buffer_type(void);
GGML_API  ggml_backend_buffer_type_t ggml_backend_rknpure_buffer_type(int32_t dev_num);
GGML_API  ggml_backend_buffer_type_t ggml_backend_rknpu2_host_dma_buffer_type(int32_t dev_num);
GGML_API  int32_t ggml_backend_rknpu2_get_device_count(void);
GGML_API  void ggml_rknpu2_clear_matmul_cache(void);

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_rknpu2_reset_compute_used(void);
GGML_API int ggml_rknpu2_get_shinfo_phys_addr(uint64_t *phys_addr);
GGML_API int ggml_rknpu2_flush_payload_range(uint64_t payload_offset, uint64_t size);
GGML_API int ggml_rknpu2_flush_all_payload(void);
GGML_API bool ggml_rknpu2_pipeline_has_pending_work(void);
GGML_API int ggml_rknpu2_pipeline_wait_compute_ready(void);
GGML_API int ggml_rknpu2_pipeline_clear_legacy_reclaim_if_done(void);
GGML_API int ggml_rknpu2_pipeline_start_async_decrypt(void);
GGML_API void ggml_rknpu2_pipeline_stop_async_decrypt(void);

#ifdef __cplusplus
}
#endif

struct ggml_rknpu_prepack_meta {
    const char * tensor_name;
    const char * meta_tensor_name;
    const char * payload_tensor_name;
    const char * layout;
    uint32_t K;
    uint32_t N;
    uint32_t block_count;
    uint32_t weight_bytes_per_block;
    uint32_t scale_type;
    uint32_t scales_bytes_total;
    uint32_t packed_bytes_total;
};

GGML_API void ggml_rknpu2_clear_offline_prepack_registry(void);
GGML_API bool ggml_rknpu2_register_offline_prepack(const struct ggml_rknpu_prepack_meta * meta, const struct ggml_tensor * meta_tensor, const struct ggml_tensor * payload_tensor);
