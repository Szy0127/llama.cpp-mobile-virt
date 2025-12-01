#include "io.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>
#include <queue>
#include "my_assert.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define IO_BLK_SIZE (2 << 20)
static int fd;
// static const char *model_path = "/data/ssd/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
#if DUMMY_WEIGHT
static void *global_read_buf;
static size_t global_read_buf_len;
#endif

static void *get_buf(int cma_index, int entry_index, size_t len) {
    (void) cma_index;
    (void) entry_index;
    if (cma_index == -1)
        return NULL;
    
    // Simple memory allocation using anonymous mmap
    void *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    GGML_ASSERT(addr != MAP_FAILED);

    return addr;
}

static void launch_io(void *dst, int fd, const io_seg &io_seg, void *pipeline) {
    (void) pipeline;

#if DUMMY_WEIGHT
    // When using dummy weights, we may want to use a temporary buffer for testing,
    // but for the simplest implementation just read directly into dst.
#endif

    ssize_t nread = pread(fd, dst, io_seg.len, io_seg.off);
    GGML_ASSERT(nread == (ssize_t) io_seg.len);
}

static void *wait_io(void) {
    // No asynchronous I/O anymore; nothing to wait for.
    return NULL;
}

#define IO_TEST_FILE "/data/ssd/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
#define IO_TEST_FILE_SIZE (8UL << 30)
#define IO_PRE_LAUNCH_CNT (16)

void io_init(const char *model_path) {
    printf("backend %s %d %s\n", __func__, __LINE__, model_path);

    fd = open(model_path, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        fprintf(stderr, "Failed to open model file %s: %s (errno=%d)\n", model_path, strerror(errno), errno);
        perror("open");
    }
    GGML_ASSERT(fd != -1);

#if DUMMY_WEIGHT
    global_read_buf = mmap(NULL, IO_BLK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    global_read_buf_len = IO_BLK_SIZE;
    if (0) {
        printf("begin io test\n");
        auto start = get_micro();
        fd = open(IO_TEST_FILE, O_RDONLY | O_DIRECT);
        int all = 0, wait = 0;
        for (size_t i = 0; i < IO_TEST_FILE_SIZE; i += IO_BLK_SIZE) {
            struct io_seg io_seg = {
                .off = i,
                .len = IO_BLK_SIZE,
            };
            launch_io(global_read_buf, fd, io_seg, (void *)1);
            if (++all >= IO_PRE_LAUNCH_CNT) {
                while (wait_io());
                ++wait;
            }
        }
        for (; wait < all; wait++) {
            while (wait_io());
        }
        printf("io test %ld us thpt %.2f GB/s\n", get_micro() - start, 0.001f * IO_TEST_FILE_SIZE / (get_micro() - start));
    }
#endif
}

static void write_measurement(const io_task &task) {
    // Use POSIX shared memory object under /dev/shm
    /*
    const char *shm_name = "/current_measure";  // results in /dev/shm/current_measure
    int fd = shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        perror("shm_open");
        return;
    }

    FILE *fp = fdopen(fd, "w");
    if (!fp) {
        perror("fdopen");
        close(fd);
        return;
    }
    */

    // Overwrite with the latest measurement (no append)
    // Exact format requested by user
    //fprintf(fp, "ttft: %.2f\ndecoding_thpt: %.2f\n", task.ttft, task.decoding_thpt);
    printf("ttft: %.2f\ndecoding_thpt: %.2f\n", task.ttft, task.decoding_thpt);
    //fflush(fp);
    //fsync(fd);
    //fclose(fp); // also closes fd
}

void io_step(all_ring_buffer *task_queue) {
    io_task task;
    while (task_queue->io_tasks.consume(&task) == 0) {
        if (task.is_measurement) {
            write_measurement(task);
            return;
        } else {
            void *buf = get_buf(task.cma_index, task.entry_index, task.len);
            if (buf != NULL && task.len > 0) {
                // Synchronous read into the CMA buffer.
                ssize_t nread = pread(fd, buf, task.io_seg.len, task.io_seg.off);
                GGML_ASSERT(nread == (ssize_t) task.io_seg.len);
            }

            io_result result = {
                .pipeline = task.pipeline,
            };
            task_queue->io_results.produce(&result);
        }
    }
}
