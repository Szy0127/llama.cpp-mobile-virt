#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#define RKNN_TENSOR_INT8 0x9
#define RKNN_TENSOR_FLOAT32 0x33

using rknn_tensor_type = int;

static constexpr float RKNPU_PREPACK_SCALE_MIN = 1e-9f;
static constexpr int RKNPU_PREPACK_NPU_CORE_NUM = 3;

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

static inline int8_t rknpu_prepack_f32_to_i8(float x, float scale) {
    const float clamped = std::min(std::max(x / scale, -127.0f), 127.0f);
    return static_cast<int8_t>(clamped);
}

static inline bool rknpu_prepack_is_candidate_name(const char * name) {
    if (name == nullptr) {
        return false;
    }

    const std::string s(name);
    if (s.size() < 7 || s.rfind(".weight") != s.size() - 7) {
        return false;
    }
    if (s.find(".__rknpu_blob") != std::string::npos) {
        return false;
    }
    if (s == "token_embd.weight") {
        return false;
    }
    if (s.find("norm") != std::string::npos) {
        return false;
    }

    return true;
}

static inline bool rknpu_prepack_is_candidate_tensor(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_Q8_0) {
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
