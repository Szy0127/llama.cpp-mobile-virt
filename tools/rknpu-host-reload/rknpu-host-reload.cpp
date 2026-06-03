#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gguf.h"
#include "npu_interface.h"
#include "rknpu-prepack-common.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

constexpr uint64_t direct_read_alignment = 4096;

struct params {
    std::string input;
};

struct host_load_header {
    // These four fields are the only plaintext contract that the standalone host reloader needs.
    uint64_t payload_start = 0;
    uint64_t payload_bytes = 0;
    uint64_t entry_size = 0;
    uint64_t compute_buffer_size = 0;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s INPUT.gguf\n", argv0);
}

params parse_args(int argc, char ** argv) {
    params p;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown argument: " + arg);
        }
        if (!p.input.empty()) {
            throw std::runtime_error("too many positional arguments");
        }
        p.input = arg;
    }

    if (p.input.empty()) {
        throw std::runtime_error("missing input GGUF path");
    }

    return p;
}

uint64_t checked_add_u64(uint64_t a, uint64_t b, const char * what) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error(std::string("overflow while computing ") + what);
    }
    return a + b;
}

uint64_t checked_to_u64(std::streamoff value, const char * what) {
    if (value < 0) {
        throw std::runtime_error(std::string("invalid ") + what);
    }
    return static_cast<uint64_t>(value);
}

size_t checked_to_size(uint64_t value, const char * what) {
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string(what) + " exceeds host size_t range");
    }
    return static_cast<size_t>(value);
}

uint64_t must_get_u64(const gguf_context * meta, const char * key) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) {
        throw std::runtime_error(std::string("missing GGUF key: ") + key);
    }
    if (gguf_get_kv_type(meta, kid) != GGUF_TYPE_UINT64) {
        throw std::runtime_error(std::string("GGUF key has wrong type: ") + key);
    }
    return gguf_get_val_u64(meta, kid);
}

host_load_header read_host_load_header(const std::string & path) {
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_ptr meta(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!meta) {
        throw std::runtime_error("failed to parse GGUF metadata");
    }

    host_load_header header;
    header.payload_start = must_get_u64(meta.get(), RKNPU_HOST_LOAD_PAYLOAD_START_KEY);
    header.payload_bytes = must_get_u64(meta.get(), RKNPU_HOST_LOAD_PAYLOAD_BYTES_KEY);
    header.entry_size = must_get_u64(meta.get(), RKNPU_HOST_LOAD_ENTRY_SIZE_KEY);
    header.compute_buffer_size = must_get_u64(meta.get(), RKNPU_HOST_LOAD_COMPUTE_BUFFER_SIZE_KEY);

    if (header.payload_bytes == 0 || header.entry_size == 0 || header.compute_buffer_size == 0) {
        throw std::runtime_error("host-load header contains zero-sized fields");
    }

    const uint64_t data_offset = static_cast<uint64_t>(gguf_get_data_offset(meta.get()));
    if (header.payload_start < data_offset) {
        throw std::runtime_error("payload_start points inside GGUF metadata");
    }

    return header;
}

uint64_t get_file_size(int input_fd) {
    return checked_to_u64(lseek(input_fd, 0, SEEK_END), "file size");
}

void require_direct_read_alignment(uint64_t file_offset, size_t size) {
    if ((file_offset % direct_read_alignment) == 0 &&
        (size % direct_read_alignment) == 0) {
        return;
    }

    char message[512];
    std::snprintf(
            message,
            sizeof(message),
            "direct-read alignment check failed: align=%llu file_offset=0x%llx size=0x%zx",
            (unsigned long long) direct_read_alignment,
            (unsigned long long) file_offset,
            size);
    throw std::runtime_error(message);
}

void validate_header_against_file(const host_load_header & header, uint64_t file_size) {
    if (header.payload_start >= file_size) {
        throw std::runtime_error("payload_start is outside the file");
    }

    const uint64_t payload_end = checked_add_u64(header.payload_start, header.payload_bytes, "payload file range");
    if (payload_end > file_size) {
        throw std::runtime_error("payload range exceeds file size");
    }
}

void print_host_load_header_hex(const char * source, const host_load_header & header) {
    std::printf(
            "RKNPU host-load header (%s): payload_start=0x%llx payload_bytes=0x%llx entry_size=0x%llx compute_buffer_size=0x%llx\n",
            source != nullptr ? source : "unknown",
            (unsigned long long) header.payload_start,
            (unsigned long long) header.payload_bytes,
            (unsigned long long) header.entry_size,
            (unsigned long long) header.compute_buffer_size);
}

void validate_header_against_layout(const host_load_header & header, const npu_layout_info & layout) {
    if (header.entry_size != layout.entry_size) {
        throw std::runtime_error("entry_size mismatch between GGUF header and ioctl layout");
    }
    if (header.compute_buffer_size != layout.rknpu.compute_buffer_size) {
        throw std::runtime_error("compute_buffer_size mismatch between GGUF header and ioctl layout");
    }

    if (layout.rknpu.donate_size <= layout.rknpu.compute_buffer_size) {
        throw std::runtime_error("invalid ioctl layout: donate_size must exceed compute_buffer_size");
    }

    const uint64_t payload_capacity = layout.rknpu.donate_size - layout.rknpu.compute_buffer_size;
    if (header.payload_bytes > payload_capacity) {
        throw std::runtime_error("payload_bytes exceeds current payload capacity");
    }

    if ((layout.payload_reload_offset % header.entry_size) != 0 ||
        (layout.payload_reload_bytes % header.entry_size) != 0) {
        throw std::runtime_error("reload range is not entry-aligned");
    }
}

void stream_reload(const params & p, const host_load_header & header, const npu_layout_info & layout) {
    const int input_fd = open(p.input.c_str(), O_RDONLY);

    const uint64_t file_size = get_file_size(input_fd);
    validate_header_against_file(header, file_size);
    validate_header_against_layout(header, layout);

    const size_t pool_size = checked_to_size(header.payload_bytes, "payload_bytes");
    if (mem_pool_prepare(pool_size) != 0) {
        throw std::runtime_error("mem_pool_prepare failed");
    }

    if (mem_pool_vaddr() == nullptr) {
        throw std::runtime_error("mem_pool_vaddr returned null");
    }

    const uint64_t reload_start = layout.payload_reload_offset;
    const uint64_t reload_end = checked_add_u64(layout.payload_reload_offset, layout.payload_reload_bytes, "reload range");
    const uint64_t readable_reload_end = reload_start >= header.payload_bytes
            ? reload_start
            : std::min(reload_end, header.payload_bytes);
    uint64_t read_offset = reload_start;
    uint64_t read_call_count = 0;
    uint64_t read_byte_count = 0;
    std::chrono::nanoseconds read_total(0);

    // The ioctl reload window describes which guest payload entries were reclaimed, not how many
    // bytes of model payload actually exist in the file. Only the overlap with the GGUF payload
    // image needs file I/O; trailing reclaimed entries beyond payload_bytes stay zero-filled.
    while (read_offset < readable_reload_end) {
        const uint64_t read_end = std::min(read_offset + header.entry_size, readable_reload_end);

        const uint64_t file_offset = checked_add_u64(header.payload_start, read_offset, "payload file offset");

        const size_t chunk_size = checked_to_size(read_end - read_offset, "reload chunk size");
        require_direct_read_alignment(file_offset, chunk_size);
        const auto read_start = std::chrono::steady_clock::now();
        if (mem_pool_direct_read_payload_entry(input_fd, read_offset, file_offset, chunk_size) != 0) {
            throw std::runtime_error("mem_pool_direct_read_payload_entry failed");
        }
        const auto read_finish = std::chrono::steady_clock::now();
        read_total += read_finish - read_start;
        read_call_count++;
        read_byte_count += static_cast<uint64_t>(chunk_size);

        if (mem_pool_finish_payload_until(read_end) != 0) {
            throw std::runtime_error("mem_pool_finish_payload_until failed");
        }
        read_offset = read_end;
    }

    if (mem_pool_finish_all_payload() != 0) {
        throw std::runtime_error("mem_pool_finish_all_payload failed");
    }

    close(input_fd);

    const double read_total_ms = std::chrono::duration<double, std::milli>(read_total).count();
    std::printf(
            "RKNPU read timing: calls=%" PRIu64 " bytes=%" PRIu64 " total=%.3f ms avg=%.3f ms\n",
            read_call_count,
            read_byte_count,
            read_total_ms,
            read_call_count ? read_total_ms / static_cast<double>(read_call_count) : 0.0);

    std::printf(
            "reloaded %s: payload_start=%" PRIu64 " payload_bytes=%" PRIu64
            " reload_offset=%" PRIu64 " reload_bytes=%" PRIu64 " entry_size=%" PRIu64 "\n",
            p.input.c_str(),
            header.payload_start,
            header.payload_bytes,
            layout.payload_reload_offset,
            layout.payload_reload_bytes,
            header.entry_size);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        const host_load_header header = read_host_load_header(p.input);
        print_host_load_header_hex("gguf", header);

        npu_layout_info layout = {};
        if (npu_get_layout_info(&layout) != 0) {
            throw std::runtime_error(std::string("npu_get_layout_info failed errno=") + std::to_string(errno));
        }

        stream_reload(p, header, layout);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
