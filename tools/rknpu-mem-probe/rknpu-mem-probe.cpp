#include "npu_interface.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct alloc_rec {
    void * ptr = nullptr;
    uint64_t dma = 0;
    uint64_t obj = 0;
    uint64_t handle = 0;
    size_t size = 0;
};

struct params {
    size_t chunk_size = 4 * 1024 * 1024;
    size_t alignment = 4096;
    size_t max_allocs = 0;
    bool touch = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s [--chunk-mib N] [--max-allocs N] [--touch]\n"
            "\n"
            "  --chunk-mib N   size of each mem_allocate call in MiB (default: 4)\n"
            "  --max-allocs N  stop after N successful allocations (default: unlimited)\n"
            "  --touch         memset each allocation to force CPU writes\n",
            argv0);
}

size_t parse_size_t(const char * value, const char * flag) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid value for ") + flag + ": " + value);
    }
    return static_cast<size_t>(parsed);
}

params parse_args(int argc, char ** argv) {
    params p;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--touch") {
            p.touch = true;
            continue;
        }
        if (arg == "--chunk-mib") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after --chunk-mib");
            }
            p.chunk_size = parse_size_t(argv[++i], "--chunk-mib") * 1024 * 1024;
            continue;
        }
        if (arg == "--max-allocs") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after --max-allocs");
            }
            p.max_allocs = parse_size_t(argv[++i], "--max-allocs");
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }

    if (p.chunk_size == 0) {
        throw std::runtime_error("--chunk-mib must be > 0");
    }

    const size_t rem = p.chunk_size % p.alignment;
    if (rem != 0) {
        p.chunk_size += p.alignment - rem;
    }

    return p;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const params p = parse_args(argc, argv);
        std::vector<alloc_rec> allocs;
        size_t total_bytes = 0;

        std::fprintf(stderr,
                "[RKNPU_MEM_PROBE] starting: chunk=%.2f MiB, touch=%s, max_allocs=%s\n",
                double(p.chunk_size) / (1 << 20),
                p.touch ? "yes" : "no",
                p.max_allocs == 0 ? "unlimited" : "set");

        while (p.max_allocs == 0 || allocs.size() < p.max_allocs) {
            alloc_rec rec;
            rec.size = p.chunk_size;
            rec.ptr = mem_allocate(rec.size, &rec.dma, &rec.obj, 0, &rec.handle);
            if (rec.ptr == nullptr) {
                std::fprintf(stderr,
                        "[RKNPU_MEM_PROBE] allocation failed after %zu chunks, total=%.2f MiB\n",
                        allocs.size(),
                        double(total_bytes) / (1 << 20));
                break;
            }

            if (p.touch) {
                std::memset(rec.ptr, 0, rec.size);
            }

            total_bytes += rec.size;
            allocs.push_back(rec);
            std::fprintf(stderr,
                    "[RKNPU_MEM_PROBE] allocated chunk=%zu size=%.2f MiB total=%.2f MiB dma=0x%" PRIx64 "\n",
                    allocs.size(),
                    double(rec.size) / (1 << 20),
                    double(total_bytes) / (1 << 20),
                    rec.dma);
        }

        std::fprintf(stderr,
                "[RKNPU_MEM_PROBE] done: chunks=%zu total=%.2f MiB\n",
                allocs.size(),
                double(total_bytes) / (1 << 20));

        for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
            mem_destroy(it->ptr, it->size, it->handle, it->obj);
        }

        return allocs.empty() ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[RKNPU_MEM_PROBE] error: %s\n", e.what());
        return 2;
    }
}
