#include "ggml.h"
#include "gguf.h"

#include <openssl/evp.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static constexpr size_t LLAMA_AES_BLOCK_SIZE = 16;
static const uint8_t LLAMA_AES128_ECB_KEY[LLAMA_AES_BLOCK_SIZE] = {
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f,
};
static const char * const RKNPU_PREPACK_META_SUFFIX = ".__rknpu_meta";

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

struct params {
    std::string input;
    std::string output;
};

struct tensor_range {
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct stats {
    uint64_t total_tensors = 0;
    uint64_t encrypted_tensors = 0;
    uint64_t skipped_meta_tensors = 0;
    uint64_t skipped_unaligned_tensors = 0;
};

static bool ends_with(const std::string & value, const char * suffix) {
    const size_t suffix_len = std::strlen(suffix);
    return value.size() >= suffix_len && value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

static int aes128_ecb_encrypt_inplace(uint8_t * data, size_t len) {
    if (data == nullptr || len % LLAMA_AES_BLOCK_SIZE != 0) {
        return -1;
    }

    EVP_CIPHER_CTX * raw = EVP_CIPHER_CTX_new();
    if (raw == nullptr) {
        return -1;
    }
    std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)> ctx(raw, EVP_CIPHER_CTX_free);

    if (EVP_EncryptInit_ex(ctx.get(), EVP_aes_128_ecb(), nullptr, LLAMA_AES128_ECB_KEY, nullptr) != 1) {
        return -1;
    }
    EVP_CIPHER_CTX_set_padding(ctx.get(), 0);

    size_t offset = 0;
    while (offset < len) {
        size_t chunk = len - offset;
        if (chunk > static_cast<size_t>(INT_MAX)) {
            chunk = static_cast<size_t>(INT_MAX);
            chunk -= chunk % LLAMA_AES_BLOCK_SIZE;
        }

        int outlen = 0;
        if (EVP_EncryptUpdate(ctx.get(), data + offset, &outlen, data + offset, static_cast<int>(chunk)) != 1) {
            return -1;
        }
        if (outlen != static_cast<int>(chunk)) {
            return -1;
        }
        offset += chunk;
    }

    int final_outlen = 0;
    if (EVP_EncryptFinal_ex(ctx.get(), data + len, &final_outlen) != 1) {
        return -1;
    }
    return final_outlen == 0 ? 0 : -1;
}

static params parse_args(int argc, char ** argv) {
    if (argc != 3) {
        throw std::invalid_argument("usage: llama-encrypt INPUT.gguf OUTPUT.gguf");
    }

    params p;
    p.input = argv[1];
    p.output = argv[2];
    return p;
}

static void copy_stream(std::ifstream & input, std::ofstream & output) {
    std::vector<char> buffer(1024 * 1024);
    input.clear();
    input.seekg(0, std::ios::beg);
    output.seekp(0, std::ios::beg);

    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = input.gcount();
        if (got > 0) {
            output.write(buffer.data(), got);
        }
    }

    if (!input.eof()) {
        throw std::runtime_error("failed while copying input model");
    }
}

static std::vector<tensor_range> collect_tensor_ranges(const gguf_context * ctx) {
    std::vector<tensor_range> ranges;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    ranges.reserve(static_cast<size_t>(n_tensors));

    const uint64_t data_offset = static_cast<uint64_t>(gguf_get_data_offset(ctx));
    for (int64_t i = 0; i < n_tensors; ++i) {
        tensor_range range;
        range.name = gguf_get_tensor_name(ctx, i);
        range.offset = data_offset + static_cast<uint64_t>(gguf_get_tensor_offset(ctx, i));
        range.size = static_cast<uint64_t>(gguf_get_tensor_size(ctx, i));
        ranges.push_back(std::move(range));
    }

    return ranges;
}

static stats encrypt_model(const params & p) {
    gguf_init_params init_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_ptr meta(gguf_init_from_file(p.input.c_str(), init_params), gguf_free);
    if (!meta) {
        throw std::runtime_error("failed to open input GGUF metadata");
    }

    std::ifstream input(p.input, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input GGUF data");
    }
    std::ofstream output(p.output, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open output GGUF path");
    }

    input.exceptions(std::ifstream::badbit);
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    copy_stream(input, output);

    const std::vector<tensor_range> ranges = collect_tensor_ranges(meta.get());
    stats result;
    result.total_tensors = static_cast<uint64_t>(ranges.size());

    for (const tensor_range & range : ranges) {
        if (ends_with(range.name, RKNPU_PREPACK_META_SUFFIX)) {
            ++result.skipped_meta_tensors;
            continue;
        }
        if (range.size == 0 || range.size % LLAMA_AES_BLOCK_SIZE != 0) {
            ++result.skipped_unaligned_tensors;
            continue;
        }
        if (range.size > static_cast<uint64_t>(SIZE_MAX)) {
            throw std::runtime_error("tensor too large to encrypt on this host");
        }

        std::vector<uint8_t> buffer(static_cast<size_t>(range.size));

        input.clear();
        input.seekg(static_cast<std::streamoff>(range.offset), std::ios::beg);
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        if (input.gcount() != static_cast<std::streamsize>(buffer.size())) {
            throw std::runtime_error("failed to read tensor bytes for encryption");
        }

        if (aes128_ecb_encrypt_inplace(buffer.data(), buffer.size()) != 0) {
            throw std::runtime_error("AES-128-ECB encryption failed");
        }

        output.seekp(static_cast<std::streamoff>(range.offset), std::ios::beg);
        output.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        ++result.encrypted_tensors;
    }

    output.flush();
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        const stats s = encrypt_model(p);
        std::printf(
                "encrypted %s -> %s: total=%" PRIu64 " encrypted=%" PRIu64 " skipped_meta=%" PRIu64 " skipped_unaligned=%" PRIu64 "\n",
                p.input.c_str(),
                p.output.c_str(),
                s.total_tensors,
                s.encrypted_tensors,
                s.skipped_meta_tensors,
                s.skipped_unaligned_tensors);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
