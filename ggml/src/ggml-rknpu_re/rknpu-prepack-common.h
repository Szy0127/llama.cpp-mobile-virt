#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RKNN_TENSOR_INT8 0x9
#define RKNN_TENSOR_FLOAT16 0x16
#define RKNN_TENSOR_FLOAT32 0x33

typedef int rknn_tensor_type;

static const float    RKNPU_PREPACK_SCALE_MIN = 1e-9f;
static const int      RKNPU_PREPACK_NPU_CORE_NUM = 3;
static const uint32_t RKNPU_PREPACK_VERSION = 1;
static const uint32_t RKNPU_PREPACK_LAYOUT_FP16 = 1;
static const uint32_t RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE = 2;
static const uint32_t RKNPU_PREPACK_SCALE_TYPE_NONE = 0;
static const uint32_t RKNPU_PREPACK_SCALE_TYPE_F32 = 1;
static const uint32_t RKNPU_PREPACK_MAGIC = 0x31504b52u; // RKP1
static const char * const RKNPU_PREPACK_FORMAT = "split-v2";
static const char * const RKNPU_PREPACK_META_FORMAT = "split-suffix-v1";
static const char * const RKNPU_PREPACK_BACKEND = "rknpu2";
static const char * const RKNPU_PREPACK_META_SUFFIX = ".__rknpu_meta";
static const char * const RKNPU_PREPACK_PAYLOAD_SUFFIX = ".__rknpu_payload";
static const char * const RKNPU_PREPACK_VERSION_KEY = "rknpu.prepack.version";
static const char * const RKNPU_PREPACK_BACKEND_KEY = "rknpu.prepack.backend";
static const char * const RKNPU_PREPACK_FORMAT_KEY = "rknpu.prepack.format";
static const char * const RKNPU_PREPACK_META_FORMAT_KEY = "rknpu.prepack.meta_format";

struct rknpu_offline_blob_header {
    uint32_t magic;
    uint32_t version;
    uint32_t layout;
    uint32_t orig_type;
    uint32_t k;
    uint32_t n;
    uint32_t K;
    uint32_t N;
    uint32_t block_count;
    uint32_t weight_bytes_per_block;
    uint32_t scale_type;
    uint32_t scales_bytes_total;
    uint32_t packed_bytes_total;
};

#if defined(__cplusplus)
static_assert(sizeof(struct rknpu_offline_blob_header) == 52, "unexpected blob header size");
#else
_Static_assert(sizeof(struct rknpu_offline_blob_header) == 52, "unexpected blob header size");
#endif

static inline int rknpu_prepack_weight_fp16(int C, int k, int c) {
    const int kpg = (k - 1) / 16;
    const int cpg = (c - 1) / 32;
    int dst = ((cpg * 32) * 16) + (kpg * 16 * C);
    dst += ((c - 1) % 32) + (((k - 1) % 16) * 32);
    return dst;
}

static inline int rknpu_prepack_weight_int8(int C, int k, int c) {
    const int kpg = (k - 1) / 32;
    const int cpg = (c - 1) / 32;
    int dst = ((cpg * 32) * 32) + (kpg * 32 * C);
    dst += ((c - 1) % 32) + (((k - 1) % 32) * 32);
    return dst;
}

static inline int rknpu_prepack_partition(int num, int max, int align) {
    const int div = (num + max - 1) / max;
    const int part = ((num + div - 1) / div + align - 1) / align * align;
    GGML_ASSERT(part <= max && part % align == 0);
    return part;
}

static inline int rknpu_prepack_block_k(int k) {
    return rknpu_prepack_partition(k, 4096, 32);
}

static inline int rknpu_prepack_block_n(int n) {
    return rknpu_prepack_partition(n / RKNPU_PREPACK_NPU_CORE_NUM, 4096, 32);
}

static inline bool rknpu_prepack_is_supported_tensor_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

static inline rknn_tensor_type rknpu_prepack_tensor_type(enum ggml_type type) {
    if (type == GGML_TYPE_F16) {
        return RKNN_TENSOR_FLOAT32;
    }
    if (type == GGML_TYPE_Q8_0) {
        return RKNN_TENSOR_INT8;
    }
    GGML_ABORT("unsupported tensor type for RKNPU prepack: %s", ggml_type_name(type));
}

static inline size_t rknpu_prepack_weight_type_size(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) {
        return sizeof(ggml_fp16_t);
    }
    if (type == RKNN_TENSOR_INT8) {
        return sizeof(int8_t);
    }
    GGML_ABORT("unsupported RKNPU tensor type: %d", type);
}

static inline const char * rknpu_prepack_layout_name(uint32_t layout) {
    switch (layout) {
        case RKNPU_PREPACK_LAYOUT_FP16:
            return "fp16";
        case RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE:
            return "int8-block-scale";
        default:
            return NULL;
    }
}

static inline uint32_t rknpu_prepack_scales_offset(const struct rknpu_offline_blob_header * header) {
    GGML_UNUSED(header);
    return (uint32_t) sizeof(struct rknpu_offline_blob_header);
}

static inline size_t rknpu_prepack_meta_bytes(const struct rknpu_offline_blob_header * header) {
    return (size_t) rknpu_prepack_scales_offset(header) + (size_t) header->scales_bytes_total;
}

static inline size_t rknpu_prepack_payload_bytes(const struct rknpu_offline_blob_header * header) {
    return (size_t) header->packed_bytes_total;
}

static inline bool rknpu_prepack_header_is_valid(const struct rknpu_offline_blob_header * header) {
    if (header == NULL) {
        return false;
    }
    if (header->magic != RKNPU_PREPACK_MAGIC || header->version != RKNPU_PREPACK_VERSION) {
        return false;
    }
    if (!rknpu_prepack_is_supported_tensor_type((enum ggml_type) header->orig_type)) {
        return false;
    }
    if (header->k == 0 || header->n == 0 || header->K == 0 || header->N == 0 || header->block_count == 0) {
        return false;
    }
    if (rknpu_prepack_layout_name(header->layout) == NULL) {
        return false;
    }

    const uint64_t expected_block_count =
        ((uint64_t)(header->n + header->N - 1) / (uint64_t) header->N) *
        ((uint64_t)(header->k + header->K - 1) / (uint64_t) header->K);
    if (expected_block_count != header->block_count) {
        return false;
    }

    const uint64_t expected_packed_bytes = (uint64_t) header->block_count * (uint64_t) header->weight_bytes_per_block;
    if (expected_packed_bytes != header->packed_bytes_total) {
        return false;
    }

    switch (header->layout) {
        case RKNPU_PREPACK_LAYOUT_FP16:
            if (header->scale_type != RKNPU_PREPACK_SCALE_TYPE_NONE || header->scales_bytes_total != 0) {
                return false;
            }
            return header->weight_bytes_per_block == rknpu_prepack_weight_type_size(RKNN_TENSOR_FLOAT32) * (size_t) header->K * (size_t) header->N;
        case RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE:
            if (header->scale_type != RKNPU_PREPACK_SCALE_TYPE_F32) {
                return false;
            }
            if (header->scales_bytes_total != header->block_count * sizeof(float)) {
                return false;
            }
            return header->weight_bytes_per_block == rknpu_prepack_weight_type_size(RKNN_TENSOR_INT8) * (size_t) header->K * (size_t) header->N;
        default:
            return false;
    }
}

static inline int8_t rknpu_prepack_f32_to_i8(float x, float scale) {
    float value = x / scale;
    if (value < -127.0f) {
        value = -127.0f;
    } else if (value > 127.0f) {
        value = 127.0f;
    }
    return (int8_t) value;
}

static inline bool rknpu_prepack_is_candidate_name(const char * name) {
    if (name == NULL) {
        return false;
    }

    const size_t name_len = strlen(name);
    static const char * suffix = ".weight";
    static const size_t suffix_len = 7;
    if (name_len < suffix_len || strcmp(name + name_len - suffix_len, suffix) != 0) {
        return false;
    }
    if (strstr(name, RKNPU_PREPACK_META_SUFFIX) != NULL ||
        strstr(name, RKNPU_PREPACK_PAYLOAD_SUFFIX) != NULL) {
        return false;
    }
    if (strcmp(name, "token_embd.weight") == 0 || strcmp(name, "output.weight") == 0) {
        return false;
    }
    if (strstr(name, "norm") != NULL) {
        return false;
    }

    return true;
}

static inline bool rknpu_prepack_is_candidate_tensor(const struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        return false;
    }
    if (!rknpu_prepack_is_supported_tensor_type(tensor->type)) {
        return false;
    }
    if (ggml_n_dims(tensor) != 2) {
        return false;
    }
    if (!ggml_is_contiguous(tensor)) {
        return false;
    }
    return rknpu_prepack_is_candidate_name(tensor->name);
}
