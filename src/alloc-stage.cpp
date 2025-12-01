#include "ggml.h"
#include "pipeline.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <atomic>
#include <io-frontend.h>
#include <stdlib.h>

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define PAGE_SIZE 0x1000

std::atomic<int64_t> cma_time;
std::atomic<size_t> cma_size;

std::mutex alloc_mtx;

class AllocTask : public Task {
public:
    size_t size;
    void *addr;
    int cma_index;
    int entry_index;

    AllocTask(size_t size): size(size), addr(NULL) {
    }
    void step(void) override {
        std::lock_guard<std::mutex> _(alloc_mtx);
#ifdef TZ_LLM_MEASURE
        auto start = get_micro();
#endif
        GGML_ASSERT(addr == NULL);

        // Simple memory allocation using anonymous mmap
        cma_index = entry_index = -1;
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        GGML_ASSERT(addr != MAP_FAILED);

#ifdef TZ_LLM_MEASURE
        cma_size += size;
        cma_time += get_micro() - start;
#endif
    }
};

AllocStage::AllocStage(size_t off, size_t len): addr(NULL) {
    size = io_align_up(off + len) - io_align_down(off);
}

void AllocStage::start(void *input)
{
    (void)input;
}

std::pair<std::shared_ptr<Task>, bool> AllocStage::get_task(void *)
{
    return { std::make_shared<AllocTask>(size), true };
}

bool AllocStage::submit(std::shared_ptr<Task> task)
{
    AllocTask *alloc_task = dynamic_cast<AllocTask *>(task.get());
    GGML_ASSERT(alloc_task);
    GGML_ASSERT(!addr);
    addr = alloc_task->addr;
    msg.buf = alloc_task->addr;
    msg.cma_indexes.push_back({alloc_task->cma_index, alloc_task->entry_index, 0, size});
    return true;
}

void *AllocStage::get_msg(void)
{
    GGML_ASSERT(addr);
    return &msg;
}

void AllocStage::rollback(void)
{
    if (addr) {
        int ret = munmap(addr, size);
        GGML_ASSERT(ret == 0);
    }
    addr = NULL;
}

