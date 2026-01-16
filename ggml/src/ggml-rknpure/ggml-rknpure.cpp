#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "npu_matmul.h"
#include "matmul_cpu_check.h"

// RKNPURE backend implementation for GGML
// This is a simplified version for element-wise operations

#define GGML_RKNPURE_NAME "RKNPURE"

// Buffer type
struct ggml_backend_rknpure_buffer_type {
    std::string name;
};

struct ggml_backend_rknpure_buffer {
    void * data = nullptr;
    size_t size = 0;
    bool own_data = false;
};

// Context
struct ggml_backend_rknpure_context {
    std::string name;
    bool has_npu = false;
};

// Check if NPU is available
static bool ggml_backend_rknpure_is_available() {
    // In a real implementation, this would check for NPU hardware
    // For now, we assume NPU is always available when this backend is enabled
    return true;
}

// Buffer type operations
static const char * ggml_backend_rknpure_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_RKNPURE_NAME;
}

static ggml_backend_buffer_t ggml_backend_rknpure_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rknpure_buffer * buf = new ggml_backend_rknpure_buffer();
    buf->data = new char[size];
    buf->size = size;
    buf->own_data = true;

    return ggml_backend_buffer_init(buft, {}, buf, size);
}

static void ggml_backend_rknpure_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpure_buffer * buf = (ggml_backend_rknpure_buffer *)buffer->context;
    if (buf->own_data) {
        delete[] (char *)buf->data;
    }
    delete buf;
}

// Backend operations
static ggml_backend_buffer_type_t ggml_backend_rknpure_get_default_buffer_type(ggml_backend_t backend) {
    static ggml_backend_rknpure_buffer_type buffer_type = {GGML_RKNPURE_NAME};
    static ggml_backend_buffer_type buffer_type_desc = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_rknpure_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_rknpure_buffer_type_alloc_buffer,
            /* .get_alignment    = */ nullptr,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .context = */ &buffer_type,
    };

    return &buffer_type_desc;
}

static const char * ggml_backend_rknpure_get_name(ggml_backend_t backend) {
    return GGML_RKNPURE_NAME;
}

static void ggml_backend_rknpure_free(ggml_backend_t backend) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend->context;
    delete ctx;
}

// Graph compute - for now, only handle element-wise operations
static enum ggml_status ggml_backend_rknpure_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // Simplified implementation - just mark as success
    // In a full implementation, this would dispatch operations to NPU
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_rknpure_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    // Only support element-wise multiplication for now
    return op->op == GGML_OP_MUL;
}

static bool ggml_backend_rknpure_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return strcmp(ggml_backend_buft_name(buft), GGML_RKNPURE_NAME) == 0;
}
