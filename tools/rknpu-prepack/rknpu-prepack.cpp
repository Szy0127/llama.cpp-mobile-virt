#include "ggml.h"
#include "gguf.h"
#include "rknpu-prepack-common.h"

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
#include <vector>

namespace {

static constexpr uint32_t RKNPU_PREPACK_VERSION = 1;
static constexpr uint32_t RKNPU_PREPACK_LAYOUT_FP16 = 1;
static constexpr uint32_t RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE = 2;
static constexpr uint32_t RKNPU_PREPACK_SCALE_TYPE_NONE = 0;
static constexpr uint32_t RKNPU_PREPACK_SCALE_TYPE_F32 = 1;
static constexpr uint32_t RKNPU_PREPACK_MAGIC = 0x31504b52u; // RKP1
static constexpr const char * RKNPU_PREPACK_FORMAT = "blob-v1";
static constexpr const char * RKNPU_PREPACK_BACKEND = "rknpu2";
static constexpr const char * RKNPU_PREPACK_BLOB_SUFFIX = ".__rknpu_blob";

struct blob_header {
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

static_assert(sizeof(blob_header) == 52, "unexpected blob header size");

struct blob_result {
    std::vector<uint8_t> bytes;
    blob_header header;
    uint32_t scales_offset;
    uint32_t packed_offset;
    const char * layout_name;
};

struct params {
    std::string input;
    std::string output;
    std::unordered_set<std::string> include_tensors;
    bool list_only = false;
};
using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

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
        "usage: %s [--list] [--tensor NAME ...] INPUT.gguf OUTPUT.gguf\n"
        "       %s --list INPUT.gguf\n",
        argv0, argv0);
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

    return p;
}

static bool is_selected_tensor(const ggml_tensor * tensor, const params & p) {
    if (!rknpu_prepack_is_candidate_tensor(tensor)) {
        return false;
    }
    if (p.include_tensors.empty()) {
        return true;
    }
    return p.include_tensors.count(tensor->name) > 0;
}

static void set_tensor_meta(gguf_context * ctx, const std::string & prefix, const ggml_tensor * tensor, const blob_result & blob, const char * blob_name) {
    // create NPU prepack metadata for this tensor.
    gguf_set_val_bool(ctx, (prefix + "enabled").c_str(), true);
    gguf_set_val_str (ctx, (prefix + "layout").c_str(), blob.layout_name);
    gguf_set_val_u32 (ctx, (prefix + "K").c_str(), blob.header.K);
    gguf_set_val_u32 (ctx, (prefix + "N").c_str(), blob.header.N);
    gguf_set_val_u32 (ctx, (prefix + "block_count").c_str(), blob.header.block_count);
    gguf_set_val_str (ctx, (prefix + "blob_tensor").c_str(), blob_name);
    gguf_set_val_u32 (ctx, (prefix + "weight_bytes_per_block").c_str(), blob.header.weight_bytes_per_block);
    gguf_set_val_u32 (ctx, (prefix + "scale_type").c_str(), blob.header.scale_type);
    gguf_set_val_u32 (ctx, (prefix + "scales_offset").c_str(), blob.scales_offset);
    gguf_set_val_u32 (ctx, (prefix + "packed_offset").c_str(), blob.packed_offset);
    gguf_set_val_u32 (ctx, (prefix + "scales_bytes_total").c_str(), blob.header.scales_bytes_total);
    gguf_set_val_u32 (ctx, (prefix + "packed_bytes_total").c_str(), blob.header.packed_bytes_total);
    
    gguf_set_val_u32(ctx, (prefix + "orig_type").c_str(), static_cast<uint32_t>(tensor->type));
    gguf_set_val_i64(ctx, (prefix + "ne0").c_str(), tensor->ne[0]);
    gguf_set_val_i64(ctx, (prefix + "ne1").c_str(), tensor->ne[1]);
    gguf_set_val_i64(ctx, (prefix + "ne2").c_str(), tensor->ne[2]);
    gguf_set_val_i64(ctx, (prefix + "ne3").c_str(), tensor->ne[3]);
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
        result.scales_offset = sizeof(blob_header);
        result.packed_offset = result.scales_offset;
        result.layout_name = "fp16";

        result.bytes.resize(sizeof(blob_header) + result.header.packed_bytes_total);
        std::memcpy(result.bytes.data(), &result.header, sizeof(blob_header));

        auto * packed_base = reinterpret_cast<ggml_fp16_t *>(result.bytes.data() + result.packed_offset);
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
    result.scales_offset = sizeof(blob_header);
    result.packed_offset = result.scales_offset + result.header.scales_bytes_total;
    result.layout_name = "int8-block-scale";

    result.bytes.resize(sizeof(blob_header) + result.header.scales_bytes_total + result.header.packed_bytes_total);
    std::memcpy(result.bytes.data(), &result.header, sizeof(blob_header));

    float * scales = reinterpret_cast<float *>(result.bytes.data() + result.scales_offset);
    int8_t * packed_base = reinterpret_cast<int8_t *>(result.bytes.data() + result.packed_offset);
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
    std::vector<std::string> selected_names;
    selected_names.reserve(n_tensors);

    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * tensor = ggml_get_tensor(meta_ctx.get(), gguf_get_tensor_name(ctx_in.get(), i));
        GGML_ASSERT(tensor != nullptr);
        if (is_selected_tensor(tensor, p)) {
            selected_names.emplace_back(tensor->name);
        }
    }

    if (p.list_only) {
        for (const auto & name : selected_names) {
            std::printf("%s\n", name.c_str());
        }
        std::printf("found %zu candidate tensors\n", selected_names.size());
        return 0;
    }

    gguf_ptr ctx_out(gguf_init_empty(), gguf_free);
    gguf_set_kv(ctx_out.get(), ctx_in.get());
    gguf_set_val_u32(ctx_out.get(), "rknpu.prepack.version", RKNPU_PREPACK_VERSION);
    gguf_set_val_str(ctx_out.get(), "rknpu.prepack.backend", RKNPU_PREPACK_BACKEND);
    gguf_set_val_str(ctx_out.get(), "rknpu.prepack.format", RKNPU_PREPACK_FORMAT);

    ggml_ptr out_ctx = make_output_ctx(static_cast<size_t>(n_tensors + selected_names.size()));
    std::vector<std::vector<uint8_t>> blobs;
    blobs.reserve(selected_names.size());
    std::unordered_map<std::string, ggml_tensor *> out_tensors;
    const std::unordered_set<std::string> selected_set(selected_names.begin(), selected_names.end());

    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * src_tensor = ggml_get_tensor(meta_ctx.get(), gguf_get_tensor_name(ctx_in.get(), i));
        GGML_ASSERT(src_tensor != nullptr);

        if (selected_set.count(src_tensor->name) > 0) {
            continue;
        }

        ggml_tensor * out_tensor = ggml_new_tensor(out_ctx.get(), src_tensor->type, ggml_n_dims(src_tensor), src_tensor->ne);
        GGML_ASSERT(out_tensor != nullptr);
        ggml_set_name(out_tensor, src_tensor->name);
        out_tensor->data = src_tensor->data;

        gguf_add_tensor(ctx_out.get(), out_tensor);
        out_tensors.emplace(out_tensor->name, out_tensor);
    }

    for (const auto & name : selected_names) {
        ggml_tensor * tensor = ggml_get_tensor(meta_ctx.get(), name.c_str());
        GGML_ASSERT(tensor != nullptr);

        blob_result blob = build_blob(tensor);
        const blob_header header = blob.header;
        const uint32_t scales_offset = blob.scales_offset;
        const uint32_t packed_offset = blob.packed_offset;
        const char * layout_name = blob.layout_name;

        blobs.push_back(std::move(blob.bytes));
        auto & blob_storage = blobs.back();

        int64_t ne[GGML_MAX_DIMS] = { static_cast<int64_t>(blob_storage.size()), 1, 1, 1 };
        ggml_tensor * blob_tensor = ggml_new_tensor(out_ctx.get(), GGML_TYPE_I8, 1, ne);
        GGML_ASSERT(blob_tensor != nullptr);

        const std::string blob_name = name + RKNPU_PREPACK_BLOB_SUFFIX;
        ggml_set_name(blob_tensor, blob_name.c_str());
        blob_tensor->data = blob_storage.data();
        gguf_add_tensor(ctx_out.get(), blob_tensor);
        out_tensors[blob_name] = blob_tensor;

        blob_result meta_blob = {
            {},
            header,
            scales_offset,
            packed_offset,
            layout_name,
        };
        const std::string prefix = "rknpu.tensor." + name + ".";
        set_tensor_meta(ctx_out.get(), prefix, tensor, meta_blob, blob_name.c_str());
    }

    std::ofstream out(p.output, std::ios::binary);
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    zeros(out, gguf_get_meta_size(ctx_out.get()));

    const size_t align = gguf_get_alignment(ctx_out.get());
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

    std::printf("wrote %s with %zu RKNPU blob tensors\n", p.output.c_str(), selected_names.size());
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
