#include "ggml.h"
#include "gguf.h"
#include "rknpu-prepack-common.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>

namespace {

struct blob_result {
    std::vector<uint8_t> meta_bytes;
    std::vector<uint8_t> payload_bytes;
    rknpu_offline_blob_header header;
    const char * layout_name;
};

struct params {
    std::string input;
    std::string output;
    std::unordered_set<std::string> include_tensors;
    // These two values are mirrored into plaintext GGUF KV so the standalone host reloader
    // can reject files generated for a different guest layout.
    uint64_t entry_size = 0;
    uint64_t compute_buffer_size = 0;
    bool list_only = false;
};
using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct selected_tensor_desc {
    std::string output_name;
    std::string source_name;
    const ggml_tensor * source_tensor = nullptr;
    bool synthesized_output = false;
};

struct prepared_tensor_blob {
    selected_tensor_desc desc;
    blob_result blob;
    // guest payload pool offset for this tensor's real payload bytes
    size_t payload_pool_offset = 0;
    // guest payload pool bytes consumed by this tensor after guest-side allocation alignment
    size_t payload_alloc_size = 0;
};

struct tensor_stats {
    size_t count = 0;
    size_t bytes = 0;
};

static bool should_include_name(const std::string & name, const params & p) {
    return p.include_tensors.empty() || p.include_tensors.count(name) > 0;
}

static void validate_synthesized_output_source(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        throw std::runtime_error("missing token_embd.weight required to synthesize output.weight");
    }
    if (!rknpu_prepack_is_supported_tensor_type(tensor->type)) {
        throw std::runtime_error("token_embd.weight has unsupported type for synthesized output.weight prepack");
    }
    if (ggml_n_dims(tensor) != 2) {
        throw std::runtime_error("token_embd.weight must be 2D to synthesize output.weight");
    }
    if (!ggml_is_contiguous(tensor)) {
        throw std::runtime_error("token_embd.weight must be contiguous to synthesize output.weight");
    }
}

static ggml_ptr make_output_ctx(size_t tensor_count) {
    const size_t mem_size = (tensor_count + 8) * ggml_tensor_overhead();
    ggml_init_params params = {
        /*.mem_size   = */ mem_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("failed to create output ggml context");
    }
    return ggml_ptr(ctx, ggml_free);
}

static void zeros(std::ofstream & file, size_t n) {
    static constexpr char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--list] [--tensor NAME ...] [--entry-size BYTES] [--compute-buffer-size BYTES] INPUT.gguf OUTPUT.gguf\n"
        "       %s --list INPUT.gguf\n",
        argv0, argv0);
}

static uint64_t parse_u64_arg(const char * name, const char * value) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || (end != nullptr && *end != '\0')) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

static size_t get_page_size() {
    const long value = sysconf(_SC_PAGESIZE);
    if (value <= 0) {
        return 4096;
    }
    return static_cast<size_t>(value);
}

static size_t page_align(size_t size, size_t page_size) {
    if ((size % page_size) == 0) {
        return size;
    }
    return size + (page_size - (size % page_size));
}

static params parse_args(int argc, char ** argv) {
    params p;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--list") {
            p.list_only = true;
            continue;
        }
        if (arg == "--tensor") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after --tensor");
            }
            p.include_tensors.insert(argv[++i]);
            continue;
        }
        if (arg == "--entry-size") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after --entry-size");
            }
            p.entry_size = parse_u64_arg("--entry-size", argv[++i]);
            continue;
        }
        if (arg == "--compute-buffer-size") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after --compute-buffer-size");
            }
            p.compute_buffer_size = parse_u64_arg("--compute-buffer-size", argv[++i]);
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown argument: " + arg);
        }
        if (p.input.empty()) {
            p.input = arg;
        } else if (p.output.empty()) {
            p.output = arg;
        } else {
            throw std::invalid_argument("too many positional arguments");
        }
    }

    if (p.input.empty()) {
        throw std::invalid_argument("missing input GGUF path");
    }
    if (!p.list_only && p.output.empty()) {
        throw std::invalid_argument("missing output GGUF path");
    }
    if (p.list_only && !p.output.empty()) {
        throw std::invalid_argument("--list takes only an input path");
    }
    if (!p.list_only && p.entry_size == 0) {
        throw std::invalid_argument("missing --entry-size");
    }
    if (!p.list_only && p.compute_buffer_size == 0) {
        throw std::invalid_argument("missing --compute-buffer-size");
    }
    if (!p.list_only && p.compute_buffer_size >= RKNPU_PREPACK_DOMAIN_BYTES) {
        throw std::invalid_argument("--compute-buffer-size must be smaller than one 4 GiB payload domain");
    }

    return p;
}

static bool is_selected_tensor(const ggml_tensor * tensor, const params & p) {
    if (!rknpu_prepack_is_candidate_tensor(tensor)) {
        return false;
    }
    return should_include_name(tensor->name, p);
}

static blob_result build_blob(const ggml_tensor * tensor) {
    const int64_t k64 = tensor->ne[0];
    const int64_t n64 = tensor->ne[1];
    GGML_ASSERT(k64 > 0 && n64 > 0);
    GGML_ASSERT(k64 <= UINT32_MAX && n64 <= UINT32_MAX);

    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const rknn_tensor_type tensor_type = rknpu_prepack_tensor_type(tensor->type);
    const uint32_t K = static_cast<uint32_t>(rknpu_prepack_block_k((int) k));
    const uint32_t N = static_cast<uint32_t>(rknpu_prepack_block_n((int) n));
    const uint32_t block_count = ((n + N - 1) / N) * ((k + K - 1) / K);
    const uint32_t weight_bytes_per_block = static_cast<uint32_t>(rknpu_prepack_weight_type_size(tensor_type) * K * N);

    blob_result result;
    result.header.magic = RKNPU_PREPACK_MAGIC;
    result.header.version = RKNPU_PREPACK_VERSION;
    result.header.orig_type = static_cast<uint32_t>(tensor->type);
    result.header.k = k;
    result.header.n = n;
    result.header.K = K;
    result.header.N = N;
    result.header.block_count = block_count;
    result.header.weight_bytes_per_block = weight_bytes_per_block;

    const uint8_t * src_bytes = static_cast<const uint8_t *>(tensor->data);

    if (tensor_type == RKNN_TENSOR_FLOAT32) {
        result.header.layout = RKNPU_PREPACK_LAYOUT_FP16;
        result.header.scale_type = RKNPU_PREPACK_SCALE_TYPE_NONE;
        result.header.scales_bytes_total = 0;
        result.header.packed_bytes_total = block_count * weight_bytes_per_block;
        result.layout_name = rknpu_prepack_layout_name(result.header.layout);

        result.meta_bytes.resize(rknpu_prepack_meta_bytes(&result.header));
        result.payload_bytes.resize(rknpu_prepack_payload_bytes(&result.header));
        std::memcpy(result.meta_bytes.data(), &result.header, sizeof(result.header));

        auto * packed_base = reinterpret_cast<ggml_fp16_t *>(result.payload_bytes.data());
        std::memset(packed_base, 0, result.header.packed_bytes_total);

        const auto * src_fp16 = reinterpret_cast<const ggml_fp16_t *>(src_bytes);
        uint32_t block_index = 0;
        for (uint32_t nn = 0; nn < n; nn += N) {
            for (uint32_t kk = 0; kk < k; kk += K) {
                ggml_fp16_t * block = packed_base + size_t(block_index) * K * N;
                for (uint32_t i = 0; i < N; ++i) {
                    for (uint32_t j = 0; j < K; ++j) {
                        const uint32_t ii = nn + i;
                        const uint32_t jj = kk + j;
                        if (ii >= n || jj >= k) {
                            continue;
                        }
                        block[rknpu_prepack_weight_fp16((int) K, (int) i + 1, (int) j + 1)] = src_fp16[size_t(ii) * k + jj];
                    }
                }
                ++block_index;
            }
        }
        return result;
    }

    GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
    result.header.layout = RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE;
    result.header.scale_type = RKNPU_PREPACK_SCALE_TYPE_F32;
    result.header.scales_bytes_total = block_count * sizeof(float);
    result.header.packed_bytes_total = block_count * weight_bytes_per_block;
    result.layout_name = rknpu_prepack_layout_name(result.header.layout);

    result.meta_bytes.resize(rknpu_prepack_meta_bytes(&result.header));
    result.payload_bytes.resize(rknpu_prepack_payload_bytes(&result.header));
    std::memcpy(result.meta_bytes.data(), &result.header, sizeof(result.header));

    float * scales = reinterpret_cast<float *>(result.meta_bytes.data() + rknpu_prepack_scales_offset(&result.header));
    int8_t * packed_base = reinterpret_cast<int8_t *>(result.payload_bytes.data());
    std::memset(scales, 0, result.header.scales_bytes_total);
    std::memset(packed_base, 0, result.header.packed_bytes_total);

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    GGML_ASSERT(traits != nullptr && traits->to_float != nullptr);
    std::vector<float> f32(size_t(k) * n);
    traits->to_float(src_bytes, f32.data(), static_cast<int64_t>(f32.size()));

    uint32_t block_index = 0;
    for (uint32_t nn = 0; nn < n; nn += N) {
        for (uint32_t kk = 0; kk < k; kk += K) {
            float scale = RKNPU_PREPACK_SCALE_MIN;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t ii = nn + i;
                    const uint32_t jj = kk + j;
                    if (ii >= n || jj >= k) {
                        continue;
                    }
                    scale = std::max(scale, std::abs(f32[size_t(ii) * k + jj]));
                }
            }
            scale /= 127.0f;
            scales[block_index] = scale;

            int8_t * block = packed_base + size_t(block_index) * K * N;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t ii = nn + i;
                    const uint32_t jj = kk + j;
                    if (ii >= n || jj >= k) {
                        continue;
                    }
                    block[rknpu_prepack_weight_int8((int) K, (int) i + 1, (int) j + 1)] =
                        rknpu_prepack_f32_to_i8(f32[size_t(ii) * k + jj], scale);
                }
            }
            ++block_index;
        }
    }

    return result;
}

static int run(const params & p) {
    ggml_context * meta_ctx_raw = nullptr;
    gguf_init_params init_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &meta_ctx_raw,
    };
    gguf_ptr ctx_in(gguf_init_from_file(p.input.c_str(), init_params), gguf_free);
    ggml_ptr meta_ctx(meta_ctx_raw, ggml_free);

    if (!ctx_in || !meta_ctx) {
        throw std::runtime_error("failed to load input GGUF");
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_in.get());
    std::vector<selected_tensor_desc> selected_tensors;
    selected_tensors.reserve(n_tensors + 1);
    std::unordered_set<std::string> selected_output_names;
    selected_output_names.reserve(n_tensors + 1);
    tensor_stats selected_input_stats;
    tensor_stats synthesized_output_stats;
    tensor_stats passthrough_stats;

    ggml_tensor * token_embd_tensor = nullptr;
    bool has_output_weight = false;

    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * tensor = ggml_get_tensor(meta_ctx.get(), gguf_get_tensor_name(ctx_in.get(), i));
        GGML_ASSERT(tensor != nullptr);

        if (std::strcmp(tensor->name, "token_embd.weight") == 0) {
            token_embd_tensor = tensor;
        }
        if (std::strcmp(tensor->name, "output.weight") == 0) {
            has_output_weight = true;
        }

        if (is_selected_tensor(tensor, p)) {
            selected_tensors.push_back({tensor->name, tensor->name, tensor, false});
            selected_output_names.insert(tensor->name);
            selected_input_stats.count += 1;
            selected_input_stats.bytes += ggml_nbytes(tensor);
        } else {
            passthrough_stats.count += 1;
            passthrough_stats.bytes += ggml_nbytes(tensor);
        }
    }

    if (!has_output_weight) {
        validate_synthesized_output_source(token_embd_tensor);
        selected_tensors.push_back({"output.weight", "token_embd.weight", token_embd_tensor, true});
        selected_output_names.insert("output.weight");
        synthesized_output_stats.count += 1;
        synthesized_output_stats.bytes += ggml_nbytes(token_embd_tensor);
        std::fprintf(stderr, "synthesizing output.weight from token_embd.weight for RKNPU prepack\n");
    }

    if (p.list_only) {
        for (const auto & desc : selected_tensors) {
            std::printf("%s\n", desc.output_name.c_str());
        }
        std::printf("found %zu candidate tensors (%zu bytes); synthesized output tensors: %zu (%zu bytes); passthrough tensors: %zu (%zu bytes)\n",
                selected_input_stats.count, selected_input_stats.bytes,
                synthesized_output_stats.count, synthesized_output_stats.bytes,
                passthrough_stats.count, passthrough_stats.bytes);
        return 0;
    }

    // Phase 1: build one prepack blob per selected tensor, then place the payloads in the
    // same global order that guest-side RKNPU allocation expects.
    std::vector<prepared_tensor_blob> prepared_tensors;
    prepared_tensors.reserve(selected_tensors.size());
    for (const auto & desc : selected_tensors) {
        if (desc.synthesized_output) {
            std::fprintf(stderr, "converting synthesized tensor '%s' from source '%s'...\n", desc.output_name.c_str(), desc.source_name.c_str());
        } else {
            std::fprintf(stderr, "converting tensor '%s'...\n", desc.output_name.c_str());
        }
        GGML_ASSERT(desc.source_tensor != nullptr);

        prepared_tensor_blob prepared;
        prepared.desc = desc;
        prepared.blob = build_blob(desc.source_tensor);
        prepared_tensors.push_back(std::move(prepared));
    }

    std::sort(prepared_tensors.begin(), prepared_tensors.end(), [](const auto & a, const auto & b) {
        return rknpu_prepack_tensor_name_less(a.desc.output_name, b.desc.output_name);
    });

    const size_t align = gguf_get_alignment(ctx_in.get());
    const size_t page_size = get_page_size();
    const size_t payload_window_bytes = static_cast<size_t>(RKNPU_PREPACK_DOMAIN_BYTES - p.compute_buffer_size);
    size_t payload_region_bytes = 0;
    size_t padding_tensor_count = 0;

    for (auto & prepared : prepared_tensors) {
        prepared.payload_alloc_size = page_align(prepared.blob.payload_bytes.size(), page_size);
        if (prepared.payload_alloc_size > payload_window_bytes) {
            throw std::runtime_error("payload tensor exceeds one payload domain");
        }

        const size_t used_in_window = payload_region_bytes % payload_window_bytes;
        if (used_in_window + prepared.payload_alloc_size > payload_window_bytes) {
            const size_t domain_padding_bytes = payload_window_bytes - used_in_window;
            if (domain_padding_bytes > 0) {
                ++padding_tensor_count;
                payload_region_bytes += domain_padding_bytes;
            }
        }

        prepared.payload_pool_offset = payload_region_bytes;

        const size_t payload_tensor_file_bytes = GGML_PAD(prepared.blob.payload_bytes.size(), align);
        GGML_ASSERT(payload_tensor_file_bytes <= prepared.payload_alloc_size);
        if (prepared.payload_alloc_size > payload_tensor_file_bytes) {
            ++padding_tensor_count;
        }

        payload_region_bytes += prepared.payload_alloc_size;
    }

    gguf_ptr ctx_out(gguf_init_empty(), gguf_free);
    gguf_set_kv(ctx_out.get(), ctx_in.get());
    gguf_set_val_u32(ctx_out.get(), RKNPU_PREPACK_VERSION_KEY, RKNPU_PREPACK_VERSION);
    gguf_set_val_str(ctx_out.get(), RKNPU_PREPACK_BACKEND_KEY, RKNPU_PREPACK_BACKEND);
    gguf_set_val_str(ctx_out.get(), RKNPU_PREPACK_FORMAT_KEY, RKNPU_PREPACK_FORMAT);
    gguf_set_val_str(ctx_out.get(), RKNPU_PREPACK_META_FORMAT_KEY, RKNPU_PREPACK_META_FORMAT);
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_PAYLOAD_START_KEY, 0);
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_PAYLOAD_BYTES_KEY, static_cast<uint64_t>(payload_region_bytes));
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_ENTRY_SIZE_KEY, p.entry_size);
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_COMPUTE_BUFFER_SIZE_KEY, p.compute_buffer_size);

    const size_t output_tensor_count = static_cast<size_t>(passthrough_stats.count + prepared_tensors.size() * 2 + padding_tensor_count);
    ggml_ptr out_ctx = make_output_ctx(output_tensor_count);
    std::vector<std::vector<uint8_t>> blob_parts;
    blob_parts.reserve(prepared_tensors.size() * 2 + padding_tensor_count);
    std::unordered_map<std::string, ggml_tensor *> out_tensors;

    // Phase 2: write passthrough tensors and prepack metadata before the host-visible payload data segment.
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * src_tensor = ggml_get_tensor(meta_ctx.get(), gguf_get_tensor_name(ctx_in.get(), i));
        GGML_ASSERT(src_tensor != nullptr);

        if (selected_output_names.count(src_tensor->name) > 0) {
            continue;
        }

        ggml_tensor * out_tensor = ggml_new_tensor(out_ctx.get(), src_tensor->type, ggml_n_dims(src_tensor), src_tensor->ne);
        GGML_ASSERT(out_tensor != nullptr);
        ggml_set_name(out_tensor, src_tensor->name);
        out_tensor->data = src_tensor->data;

        gguf_add_tensor(ctx_out.get(), out_tensor);
        out_tensors.emplace(out_tensor->name, out_tensor);
    }

    for (const auto & prepared : prepared_tensors) {
        blob_parts.push_back(prepared.blob.meta_bytes);
        auto & meta_storage = blob_parts.back();
        int64_t meta_ne[GGML_MAX_DIMS] = { static_cast<int64_t>(meta_storage.size()), 1, 1, 1 };
        ggml_tensor * meta_tensor = ggml_new_tensor(out_ctx.get(), GGML_TYPE_I8, 1, meta_ne);
        GGML_ASSERT(meta_tensor != nullptr);
        const std::string meta_name = prepared.desc.output_name + RKNPU_PREPACK_META_SUFFIX;
        ggml_set_name(meta_tensor, meta_name.c_str());
        meta_tensor->data = meta_storage.data();
        gguf_add_tensor(ctx_out.get(), meta_tensor);
        out_tensors[meta_name] = meta_tensor;
    }

    // Phase 3: materialize the NPU payload data segment. Each hole is emitted as a zero-filled
    // padding tensor so standard GGUF readers still see monotonically increasing tensor offsets.
    size_t payload_cursor = 0;
    size_t padding_index = 0;
    std::string first_payload_tensor_name;

    for (const auto & prepared : prepared_tensors) {
        if (payload_cursor < prepared.payload_pool_offset) {
            const size_t gap_bytes = prepared.payload_pool_offset - payload_cursor;
            GGML_ASSERT(GGML_PAD(gap_bytes, align) == gap_bytes);
            blob_parts.emplace_back(gap_bytes, 0);
            auto & padding_storage = blob_parts.back();
            int64_t padding_ne[GGML_MAX_DIMS] = { static_cast<int64_t>(padding_storage.size()), 1, 1, 1 };
            ggml_tensor * padding_tensor = ggml_new_tensor(out_ctx.get(), GGML_TYPE_I8, 1, padding_ne);
            GGML_ASSERT(padding_tensor != nullptr);
            const std::string padding_name = std::string(RKNPU_PREPACK_PADDING_PREFIX) + std::to_string(padding_index++);
            ggml_set_name(padding_tensor, padding_name.c_str());
            padding_tensor->data = padding_storage.data();
            gguf_add_tensor(ctx_out.get(), padding_tensor);
            out_tensors[padding_name] = padding_tensor;
            payload_cursor += gap_bytes;
        }

        blob_parts.push_back(prepared.blob.payload_bytes);
        auto & payload_storage = blob_parts.back();
        int64_t payload_ne[GGML_MAX_DIMS] = { static_cast<int64_t>(payload_storage.size()), 1, 1, 1 };
        ggml_tensor * payload_tensor = ggml_new_tensor(out_ctx.get(), GGML_TYPE_I8, 1, payload_ne);
        GGML_ASSERT(payload_tensor != nullptr);
        const std::string payload_name = prepared.desc.output_name + RKNPU_PREPACK_PAYLOAD_SUFFIX;
        ggml_set_name(payload_tensor, payload_name.c_str());
        payload_tensor->data = payload_storage.data();
        gguf_add_tensor(ctx_out.get(), payload_tensor);
        out_tensors[payload_name] = payload_tensor;

        if (first_payload_tensor_name.empty()) {
            first_payload_tensor_name = payload_name;
        }

        const size_t payload_tensor_file_bytes = GGML_PAD(payload_storage.size(), align);
        payload_cursor += payload_tensor_file_bytes;

        const size_t tail_padding_bytes = prepared.payload_alloc_size - payload_tensor_file_bytes;
        if (tail_padding_bytes > 0) {
            GGML_ASSERT(GGML_PAD(tail_padding_bytes, align) == tail_padding_bytes);
            blob_parts.emplace_back(tail_padding_bytes, 0);
            auto & padding_storage = blob_parts.back();
            int64_t padding_ne[GGML_MAX_DIMS] = { static_cast<int64_t>(padding_storage.size()), 1, 1, 1 };
            ggml_tensor * padding_tensor = ggml_new_tensor(out_ctx.get(), GGML_TYPE_I8, 1, padding_ne);
            GGML_ASSERT(padding_tensor != nullptr);
            const std::string padding_name = std::string(RKNPU_PREPACK_PADDING_PREFIX) + std::to_string(padding_index++);
            ggml_set_name(padding_tensor, padding_name.c_str());
            padding_tensor->data = padding_storage.data();
            gguf_add_tensor(ctx_out.get(), padding_tensor);
            out_tensors[padding_name] = padding_tensor;
            payload_cursor += tail_padding_bytes;
        }
    }

    GGML_ASSERT(payload_cursor == payload_region_bytes);

    uint64_t payload_start = 0;
    if (!first_payload_tensor_name.empty()) {
        const int64_t tid = gguf_find_tensor(ctx_out.get(), first_payload_tensor_name.c_str());
        GGML_ASSERT(tid >= 0);
        payload_start = static_cast<uint64_t>(gguf_get_meta_size(ctx_out.get()) + gguf_get_tensor_offset(ctx_out.get(), tid));
    }

    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_PAYLOAD_START_KEY, payload_start);
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_PAYLOAD_BYTES_KEY, static_cast<uint64_t>(payload_region_bytes));
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_ENTRY_SIZE_KEY, p.entry_size);
    gguf_set_val_u64(ctx_out.get(), RKNPU_HOST_LOAD_COMPUTE_BUFFER_SIZE_KEY, p.compute_buffer_size);

    std::ofstream out(p.output, std::ios::binary);
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    zeros(out, gguf_get_meta_size(ctx_out.get()));

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_out.get()); ++i) {
        const char * name = gguf_get_tensor_name(ctx_out.get(), i);
        auto it = out_tensors.find(name);
        GGML_ASSERT(it != out_tensors.end() && it->second != nullptr);
        ggml_tensor * tensor = it->second;
        const size_t nbytes = ggml_nbytes(tensor);
        out.write(static_cast<const char *>(tensor->data), nbytes);
        zeros(out, GGML_PAD(nbytes, align) - nbytes);
    }

    out.seekp(0);
    std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out.get()));
    gguf_get_meta_data(ctx_out.get(), meta.data());
    out.write(reinterpret_cast<const char *>(meta.data()), meta.size());
    out.close();

    std::printf(
            "wrote %s with %zu RKNPU prepacked tensors (%zu meta + %zu payload + %zu padding); "
            "payload_start=%" PRIu64 " payload_bytes=%zu; selected input tensors: %zu (%zu bytes); synthesized output tensors: %zu (%zu bytes); passthrough tensors: %zu (%zu bytes)\n",
            p.output.c_str(), prepared_tensors.size() * 2, prepared_tensors.size(), prepared_tensors.size(), padding_tensor_count,
            payload_start, payload_region_bytes,
            selected_input_stats.count, selected_input_stats.bytes,
            synthesized_output_stats.count, synthesized_output_stats.bytes,
            passthrough_stats.count, passthrough_stats.bytes);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        return run(p);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
