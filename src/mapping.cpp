#include "mapping.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <optional>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <atomic>
#include "interface.h"

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define PAGE_SIZE 0x1000

cma_region::cma_region(int tzd_fd, size_t size, std::function<void(void)> destructor)
    : done(false), len(size), destructor(destructor) {
    (void) tzd_fd; // No longer used, kept for API compatibility
}

std::atomic<int64_t> cma_time;

void cma_region::ready(void) {
    if (done) return;
#ifdef TZ_LLM_MEASURE
    auto start = get_micro();
#endif

    // Simple memory allocation using anonymous mmap
    addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    GGML_ASSERT(addr != MAP_FAILED);

    done = true;

#ifdef TZ_LLM_MEASURE
    cma_time += get_micro() - start;
#endif
}

cma_region::~cma_region() {
    if (addr) {
        munmap(addr, len);
    }
    destructor();
}

mappings::mappings(void) {
    // No device needed, fd is not used anymore
    fd = -1;
}

mappings::~mappings(void) {
    while (!tensors.empty())
        tensors.pop_back();
    if (fd >= 0) {
        close(fd);
    }
}

void mappings::push(size_t offset, size_t size, std::function<void(void)> destructor) {
    std::shared_ptr<struct cma_region> cma_region;
    auto cma_iter = cma_regions.find(offset);
    if (cma_iter == cma_regions.end() || !(cma_region = cma_iter->second.lock())) {
        cma_region = std::make_shared<struct cma_region>(fd, size, destructor);
        cma_regions.emplace(offset, cma_region);
    }
    std::lock_guard<std::mutex> _(work_queue_mutex);
    work_queue.emplace(cma_region);
}

void mappings::step(void) {
    std::shared_ptr<cma_region> cma_region;
    {
        std::lock_guard<std::mutex> _(work_queue_mutex);
        if (!work_queue.empty()) {
            cma_region = work_queue.front();
            work_queue.pop();
        }
    }
    if (cma_region) {
        cma_region->ready();
        std::lock_guard<std::mutex> _(tensors_mutex);
        tensors.push_back(cma_region);
    }
}

void *mappings::try_get(size_t offset) {
    std::shared_ptr<struct cma_region> cma_region;
    auto cma_iter = cma_regions.find(offset);
    if (cma_iter == cma_regions.end())
        return NULL;
    cma_region = cma_iter->second.lock();
    if (!cma_region)
        return NULL;
    if (!cma_region->done)
        return NULL;
    return cma_region->addr;
}

void *mappings::get(size_t offset) {
    void *addr = try_get(offset);
    while (!addr) {
        step();
        addr = try_get(offset);
    }
    return addr;
}

void mappings::pop(void) {
    std::lock_guard<std::mutex> _(tensors_mutex);
    tensors.pop_back();
}