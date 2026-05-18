#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <atomic>
#include <vector>

// System headers for file operations
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <sched.h>

// Minimal RKNPURE backend implementation for element-wise operations

GGML_API int32_t ggml_backend_rknpu2_get_device_count() {
    return 1;
}

static inline void ggml_thread_cpu_relax_out(void) {
    // Simple CPU relaxation - just yield
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    // For other architectures, do nothing
#endif
}


static uint64_t npu_count = 0;
static uint64_t npu_total_count = 0;
static uint64_t npu_total_failed_count = 0;
 static bool ggml_backend_rknpure_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst = op;
    //src0->name
    // if(src0 && src1 && dst){
    // fprintf(stderr, "src0->name=%s\n", src0->name ? src0->name : "NULL");
    // fprintf(stderr, "src1->name=%s\n", src1->name ? src1->name : "NULL");
    // fprintf(stderr, "dst->name=%s\n", dst->name ? dst->name : "NULL");
    // }
    npu_total_count++;
    //return false;
    if (op->op != GGML_OP_MUL_MAT) {
        // printf("zzh: op is %d, not mul mat\n", op->op);
        npu_total_failed_count++;
        // fprintf(stderr, "NPU failed0! npu_total_failed_count=%llu/%llu\n", (unsigned long long)npu_total_failed_count, (unsigned long long)npu_total_count);
        return false;
    }

    // /* cpu computation when batch size == 1, i.e., decoding stage */
    // if (src1->ne[1] == 1) {
    //     return false;
    // }
    
    // comment this because now we use NPU to compute lm_head MatMul
    // {
    //     const int64_t m = src1->ne[1];
    //     const int64_t k = src0->ne[0];
    //     const int64_t n = dst->ne[0];
    //     /* can not allocate large B buffers for large vocab_size. just use cpu to perform these matmuls */
    //     if (k >= 50000 || n >= 50000){
    //         npu_total_failed_count++;
    //         // fprintf(stderr, "NPU failed1! npu_total_failed_count=%llu/%llu\n", (unsigned long long)npu_total_failed_count, (unsigned long long)npu_total_count);
    //         return false;
    //     }
    // }

    // printf("ggml_backend_rknpure_supports_op, %d, %d, %p\n", src1->type, dst->type, src0->extra);
    // return false; // DEBUG: first, never use this backend

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    if(!ggml_is_contiguous(src0)){
        // fprintf(stderr,"src0 is not contiguous: name=%s, type=%d, ne=[%ld,%ld,%ld,%ld], nb=[%ld,%ld,%ld,%ld], "
        //         "view_src=%p, op=%d, is_permuted=%d, is_transposed=%d\n",
        //         src0->name ? src0->name : "NULL",
        //         src0->type,
        //         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        //         src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        //         src0->view_src,
        //         src0->op,
        //         ggml_is_permuted(src0),
        //         ggml_is_transposed(src0));
        return false;
    }
    if(!ggml_is_contiguous(src1)){
        fprintf(stderr,"src1 is not contiguous\n");
        return false;
    }
    
    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        const int64_t k = src0->ne[0];
        const int64_t n = src0->ne[1];
        // return false;
        // fprintf(stderr, "NPU support!  npu failed count=%llu/%llu\n", (unsigned long long)npu_total_failed_count, (unsigned long long)npu_total_count);
        return true;
        if (npu_count < 1123) {
            npu_count++;
            //fprintf(stderr, "NPU support! npu_count=%llu\n", (unsigned long long)npu_count);
            // return false;
            return true;
        }
        // fprintf(stderr, "NPU support! npu_count=%llu\n", (unsigned long long)npu_count);
        return true;
        // k > 8192 时，B 会被分成 T 段，int T = std::ceil(K / 8192)，推荐使用 rknn_B_normal_layout_to_native_layout 接口直接进行数据转换
        if(k > 8192 || n > 4096) // RKNPU2 limit （原来是10240）
        {
            // printf("oversize: k=%ld, n=%ld\n", k, n);

            return 0;
        }

        // k and n size must align to 32 bytes
        if(k % 32 != 0 || n % 32 != 0)
        {
            printf("not align: k=%ld, n=%ld\n", k, n);
            return 0;
        }

        // make sure the tensor has assosiated data
        // printf("zzh: %s\n", src0->buffer->buft->iface.get_name(src0->buffer->buft));
        // if (strcmp(src0->buffer->buft->iface.get_name(src0->buffer->buft), "RKNPURE")) {
        //     return 0;
        // }

        if(src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_F16)
        {
            printf("zzh: tensor->type wrong\n");
            return 0;
        }

        /*printf("RKNPU2: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
            return true;

    }
    npu_total_failed_count++;
    // fprintf(stderr, "NPU failed2! npu_total_failed_count=%llu/%llu type:%d %d %d\n", (unsigned long long)npu_total_failed_count, (unsigned long long)npu_total_count, src0->type, src1->type, dst->type);
    // printf("rknpu2 not support this MUL_MAT\n");
    return false;

    GGML_UNUSED(backend);
}

extern "C" {
bool ggml_backend_rknpure_supports_op_out(const struct ggml_tensor *op) {
    // Force CPU computation for debugging - always return false to skip NPU
    ggml_backend_t backend;
    
    // // Print call stack for debugging with symbol resolution
    // void *buffer[32];
    // int nptrs = backtrace(buffer, 32);
    
    // int cpu_id = sched_getcpu();
    // fprintf(stderr, "ggml_backend_rknpure_supports_op_out, op->op=%d,cpuid=%d\n", op->op, cpu_id);
    // fprintf(stderr, "Call stack (%d frames):\n", nptrs);
    
    // for (int i = 0; i < nptrs && i < 10; i++) {  // Print first 10 frames
    //     void *addr = buffer[i];
    //     Dl_info info;
        
    //     if (dladdr(addr, &info) && info.dli_sname) {
    //         // Successfully resolved symbol name
    //         fprintf(stderr, "  [%d] %p %s", i, addr, info.dli_sname);
    //         if (info.dli_saddr) {
    //             void *offset = (void*)((char*)addr - (char*)info.dli_saddr);
    //             fprintf(stderr, "+%p", offset);
    //         }
    //         if (info.dli_fname) {
    //             fprintf(stderr, " (%s)", info.dli_fname);
    //         }
    //         fprintf(stderr, "\n");
    //     } else {
    //         // Fallback: print address only
    //         fprintf(stderr, "  [%d] %p <unresolved>\n", i, addr);
    //     }
    // }
    
    // if (nptrs > 10) {
    //     fprintf(stderr, "  ... (%d more frames)\n", nptrs - 10);
    // }
    
    return ggml_backend_rknpure_supports_op(backend, op);
}

uint64_t ggml_backend_rknpure_get_npu_count(void) {
    return npu_count;
}
}
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <map>

#include <chrono>
#include <queue>
#include <future>

static inline int64_t get_micro(void) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}

struct measure_point {
    const char *file;
    int line;
    int64_t micro;

    measure_point(const char *file, int line, int64_t micro)
        : file(file), line(line), micro(micro) {}
};

#define LINE_NR (3000)

struct measure_stack {
    std::vector<measure_point> stack;
    std::unordered_map<int, int> regions;
    static int last_lines[LINE_NR];
    static std::atomic<int64_t> micros[LINE_NR];

    void begin_measure(const char *file, int line, int64_t micro) {
        stack.emplace_back(file, line, micro);
    }
    void end_measure(const char *file, int line, int64_t micro) {
        assert(!stack.empty());
        auto last = stack.back();
        stack.pop_back();
        if (regions.find(line) != regions.end()) {
            assert(last.line == regions[line]);
        } else {
            regions[line] = last.line;
        }
        measure_stack::last_lines[line] = last.line;
        measure_stack::micros[line] += micro - last.micro;
    }
};
thread_local measure_stack mstack;

#define RKNPU_MEASURE 1

#if RKNPU_MEASURE

#define BEGIN_MEASURE mstack.begin_measure(__FILE__, __LINE__, get_micro())
#define END_MEASURE mstack.end_measure(__FILE__, __LINE__, get_micro())

#else

#define BEGIN_MEASURE
#define END_MEASURE

#endif

#define BEGIN_MEASURE_0 if (ith == 0) BEGIN_MEASURE
#define END_MEASURE_0 if (ith == 0) END_MEASURE

#ifdef GGML_USE_CHCORE
extern "C" void npu_dump_measure(bool clear);
#endif

void ggml_rknpu_dump_measure(void) {
#if RKNPU_MEASURE
    printf("********************************begin rknpu measure dump********************************\n");
    for (int i = 0; i < LINE_NR; i++) {
        if (measure_stack::micros[i].load()) {
            printf("%s:%d:(from %d) %ld ms\n", __FILE__, i, measure_stack::last_lines[i], measure_stack::micros[i].load() / 1000);
        }
        measure_stack::last_lines[i] = 0;
        measure_stack::micros[i] = 0;
    }
    printf("*********************************end rknpu measure dump*********************************\n");

#ifdef GGML_USE_CHCORE
    npu_dump_measure(true);
#endif

#endif
}

// #define USE_CPU_CHECK

#include <ggml-rknpu-re.h>
#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"
#include "rknpu-prepack-common.h"
#ifdef USE_CPU_CHECK
#include "matmul_cpu_check.h"
#endif
#ifdef __MUSL__
#include <sys/syscall.h>
#endif

#define GGML_RKNPU2_INPUT_SCALE 9.0f
typedef int rknn_core_mask;


#define MAT_COPY

#include <sys/ioctl.h>

#ifndef MAT_COPY
#ifndef GGML_USE_CHCORE
//#define FAKE_CACHE
#endif
#endif
struct dma_heap_allocation_data {
	uint64_t len;
	uint32_t fd;
	uint32_t fd_flags;
	uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC		'H'
#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE		'b'
#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE	(1024 * 1024)

int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    /* open dma_heap fd */
    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        printf("open %s fail!\n", path);
        return dma_heap_fd;
    }

    /* alloc buffer */
    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));

    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        printf("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        return ret;
    }

    /* mmap va */
    if (fcntl(buf_data.fd, F_GETFL) & O_RDWR)
        prot = PROT_READ | PROT_WRITE;
    else
        prot = PROT_READ;

    /* mmap contiguors buffer to user */
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        printf("mmap failed: %s\n", strerror(errno));
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;

    close(dma_heap_fd);

    return 0;
}
int dma_sync_device_to_cpu(int fd) {
    uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

int dma_sync_cpu_to_device(int fd) {
    uint64_t flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}
void dma_buf_free(size_t size, int *fd, void *va) {
    int len;

    len =  size;
    munmap(va, len);

    close(*fd);
    *fd = -1;
}

#ifdef GGML_USE_CHCORE
#define NPU_CORE_NUM (3)
#define THREAD_NR (1)
#else
#define THREAD_NR (RKNPU_PREPACK_NPU_CORE_NUM)
#define NPU_CORE_NUM THREAD_NR
#endif
#define NON_NPU_THREAD 0xdeadbeef

thread_local int ttid = NON_NPU_THREAD;


struct ggml_backend_rknpure_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
    int device;
    struct ggml_backend *         backend;
    char                          name[GGML_MAX_NAME];
};
struct ggml_rknpu2_data_pack
{
    int type;
    // save data used for mat mul
    void* ordered_data;
    int initialized;
};
struct rknpure_weight {
    void *weights; // B
    uint64_t weights_dma, weights_obj;
    uint64_t weights_handle;
    uint64_t buffer_size;
};
static const char * ggml_backend_rknpu2_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return GGML_RKNPU2_NAME;
}

// Stores the CPU pointer and optional DMA address for tensors allocated from the custom host/DMA buffer types.
struct ggml_backend_rknpu2_tensor_extra {
    void * cpu_ptr = nullptr;
    uint64_t dma = 0;
    uint32_t domain_id = 0;
    size_t size = 0;
    size_t offset = 0;
    bool is_blob = false;
};

// Owns one buffer allocation for a custom host or host-DMA buffer type and releases it when the buffer dies.
struct ggml_backend_rknpu2_buffer_context {
    ggml_backend_rknpu2_buffer_context(size_t device, std::string name)
            : device(device)
            , name(std::move(name)) {}

    ~ggml_backend_rknpu2_buffer_context() {
        for (auto * extra : tensor_extras) {
            delete extra;
        }
        if (buffer) {
            mem_destroy(buffer, alloc_size, handle, obj);
        }
    }

    void * buffer = nullptr;
    struct ggml_backend_rknpure_context * backend_ctx = nullptr;
    size_t buffer_size = 0;
    size_t alloc_size = 0;
    uint64_t dma = 0;
    uint64_t obj = 0;
    uint64_t handle = 0;
    uint32_t domain_id = 0;
    size_t device;
    std::string name;
    std::vector<ggml_backend_rknpu2_tensor_extra *> tensor_extras;
};

static const char * ggml_backend_rknpu2_tensor_name(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->name[0] != '\0' ? tensor->name : "<unnamed>";
}

static void ggml_backend_rknpu2_log_tensor_layout(const char * tag,
                                                  const ggml_tensor * tensor,
                                                  const void * cpu_ptr,
                                                  uint64_t dma,
                                                  size_t size,
                                                  uint32_t domain_id) {
    GGML_LOG_INFO("[RKNPU_TENSOR] %s tensor=%s cpu=%p dma=0x%llx size=%zu domain=%u\n",
                  tag,
                  ggml_backend_rknpu2_tensor_name(tensor),
                  cpu_ptr,
                  (unsigned long long) dma,
                  size,
                  domain_id);
}

// Frees the allocation backing one custom host or host-DMA buffer.
static void ggml_backend_rknpu2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    delete ctx;
}

// Returns the CPU-accessible base pointer for a custom host or host-DMA buffer.
static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    return ctx->buffer;
}

// Attaches CPU/DMA metadata to each tensor slice allocated from the custom host or host-DMA buffer.
static enum ggml_status ggml_backend_rknpu2_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor * tensor) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    if (tensor->view_src != nullptr) {
        if (tensor->view_src->extra != nullptr) {
            tensor->extra = tensor->view_src->extra;
            auto * extra = (ggml_backend_rknpu2_tensor_extra *) tensor->view_src->extra;
            const size_t offset = (size_t) ((const char *) tensor->data - (const char *) extra->cpu_ptr);
            /*
            ggml_backend_rknpu2_log_tensor_layout("view",
                                                  tensor,
                                                  tensor->data,
                                                  extra->dma + offset,
                                                  ggml_nbytes(tensor),
                                                  extra->domain_id);
                                                  */
        }
        return GGML_STATUS_SUCCESS;
    }

    auto * extra = new ggml_backend_rknpu2_tensor_extra();
    extra->cpu_ptr = tensor->data;
    extra->size = ggml_nbytes(tensor);
    extra->offset = (size_t) ((const char *) tensor->data - (const char *) ctx->buffer);
    extra->is_blob = std::strstr(tensor->name, RKNPU_PREPACK_META_SUFFIX) != nullptr ||
                     std::strstr(tensor->name, RKNPU_PREPACK_PAYLOAD_SUFFIX) != nullptr;
    extra->dma = ctx->dma + extra->offset;
    extra->domain_id = ctx->domain_id;

    tensor->extra = extra;
    ctx->tensor_extras.push_back(extra);
    /*
    ggml_backend_rknpu2_log_tensor_layout("alloc",
                                          tensor,
                                          extra->cpu_ptr,
                                          extra->dma,
                                          extra->size,
                                          extra->domain_id);
                                          */
    return GGML_STATUS_SUCCESS;
}

// in fact, this is never called when not using mmap
static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                       ggml_tensor * tensor, const void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    // TODO: how to handle offset and size?

    if (/*rknpu2_backend &&*/ ggml_rknpure_can_mul_mat_b(tensor) == false) { // we don't test this!
        printf("ggml_rknpure_can_mul_mat_b NOT OK\n");
        memcpy((char *) tensor->data + offset, data, size); // We must have this, otherwise will give meaningful output
        return;
    }
    // printf("ggml_backend_rknpu2_buffer_set_tensor, offset=%lu\n", offset);
    GGML_ASSERT(offset == 0);

    
    if (ggml_rknpure_transform_tensor(data, tensor, offset, size)) {
        printf("ggml_rknpure_transform_tensor failed\n");
    }
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                       const ggml_tensor * tensor, void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
#ifndef NDEBUG
    // printf("ggml_backend_rknpu2_buffer_get_tensor\n");
#endif
    GGML_ASSERT(offset == 0);
    memcpy(data, (char *) tensor->data + offset, size);
    return;
    abort();
    // memcpy(data, (const char *) tensor->data + offset, size);
    
        // We need to transform RKNPU2 tensor back
    ggml_rknpu2_transform_tensor_back(data, tensor, offset, size);
}

static bool ggml_backend_rknpu2_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                       const struct ggml_tensor * src,
                                       struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    printf("ggml_backend_rknpu2_buffer_cpy_tensor\n");
    abort();
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_backend_rknpu2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

// Describes one custom DMA-backed host buffer type.
struct ggml_backend_rknpu2_buffer_type_context {
    int32_t device;
    std::string name;
};

/**
 * @brief Retrieves the name associated with a CANN buffer type.
 */
static const char* ggml_backend_rknpu2_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_rknpu2_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_i ggml_backend_rknpu2_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rknpu2_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rknpu2_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rknpu2_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_rknpu2_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rknpu2_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rknpu2_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rknpu2_buffer_clear,
    /* .reset           = */ nullptr,
};

static struct ggml_backend_rknpure_context g_rknpu2_mgr[GGML_RKNPU2_MAX_DEVICES];

// Allocates one CPU-accessible DMA-backed buffer for the custom host-DMA buffer types.
static ggml_backend_buffer_t
ggml_backend_rknpu2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_rknpu2_buffer_type_context* buft_ctx =
        (ggml_backend_rknpu2_buffer_type_context*)buft->context;

    const size_t size_page = sysconf(_SC_PAGESIZE);
    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    ggml_backend_rknpu2_buffer_context* ctx =
        new ggml_backend_rknpu2_buffer_context(buft_ctx->device, buft_ctx->name);
    ctx->buffer_size = size;
    ctx->alloc_size = size_aligned;
    ctx->backend_ctx = &g_rknpu2_mgr[buft_ctx->device];
    ctx->buffer = mem_allocate_payload(size_aligned, &ctx->dma, &ctx->obj,
            RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT, &ctx->handle,
            &ctx->domain_id);

    if (ctx->buffer == nullptr) {
        printf("%s: failed to allocate %.2f MiB for %s\n", __func__, (double) size / (1 << 20), buft_ctx->name.c_str());
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_rknpu2_buffer_interface,
                                    ctx, size);
}

// Reports that the custom host and host-DMA buffer types expose CPU-accessible memory.
static bool ggml_backend_rknpu2_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

// Returns the tensor alignment used by the custom host and host-DMA buffer types.
static size_t ggml_backend_rknpu2_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

// TODO: this value is an experimental value, works fine with whisper/llm/minicpm-v inference on Android
// Returns the maximum allocation size accepted by the custom host and host-DMA buffer types.
static size_t ggml_backend_rknpu2_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (4095UL * 1024 * 1024); // original QNN: 96 * 1024 * 1024
}


/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
static ggml_backend_buffer_type_i ggml_backend_rknpu2_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rknpu2_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rknpu2_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rknpu2_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rknpu2_buffer_type_get_max_size,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ ggml_backend_rknpu2_buffer_is_host,
};
/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
// Returns the custom CPU-accessible DMA-backed buffer type used for offline prepack blobs.
static ggml_backend_buffer_type_t ggml_backend_rknpu2_get_host_dma_buffer_type_internal(int32_t device, const char * name_prefix) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_rknpu2_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type host_dma_buffer_types[GGML_RKNPU2_MAX_DEVICES];
    static bool host_dma_initialized = false;

    if (!host_dma_initialized) {
        for (int32_t i = 0; i < GGML_RKNPU2_MAX_DEVICES; i++) {
            host_dma_buffer_types[i].iface = ggml_backend_rknpu2_buffer_type_interface;
            host_dma_buffer_types[i].device = nullptr;
            host_dma_buffer_types[i].context = new ggml_backend_rknpu2_buffer_type_context{
                i, std::string(name_prefix) + std::to_string(i)};
        }
        host_dma_initialized = true;
    }

    return &host_dma_buffer_types[device];
}

// Returns the legacy RKNPURE buffer type used by the backend's existing tensor paths.
GGML_API ggml_backend_buffer_type_t
ggml_backend_rknpure_buffer_type(int32_t device) {
    return ggml_backend_rknpu2_get_host_dma_buffer_type_internal(device, "RKNPURE");
}

// Returns a CPU-accessible buffer type whose allocations are backed by DMA for offline prepack blobs.
GGML_API ggml_backend_buffer_type_t
ggml_backend_rknpu2_host_dma_buffer_type(int32_t device) {
    return ggml_backend_rknpu2_get_host_dma_buffer_type_internal(device, "HOST_DMA");
}

// g++ doesn't allow restrict? removed.
static void ggml_rknpu2_transposed_to_native_fp16(__fp16 * dst,
                                                  const float * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/16, K/32, 16, 32)
  const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

  // Block copy 32x16 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 16; i++) {
      for (size_t ii = 0; ii < 16; ii++) {
        size_t partial_src_idx = j * 32 + (i * 16 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

// native fp16 => float (32-bit)
static void ggml_rknpu2_transposed_from_native_fp16(float * dst,
                                                  const __fp16 * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/16, K/32, 16, 32)
  const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

  // Block copy 32x16 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 16; i++) {
      for (size_t ii = 0; ii < 16; ii++) {
        size_t partial_src_idx = j * 32 + (i * 16 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[src_idx] = src[dst_idx];
        }
      }
    }
  }
}

static void ggml_rknpu2_transposed_to_native_int8(int8_t * dst,
                                                  const float * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/32, K/32, 32, 32)
  const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

  // Block copy 32x32 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 32; i++) {
      for (size_t ii = 0; ii < 32; ii++) {
        size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          /*if (src[dst_idx] > 1.0f || src[dst_idx] < -1.0f) {
            printf("error: src[dst_idx]=%f exceeds 1.0f\n", src[dst_idx]);
          }*/
          dst[dst_idx] = roundf(fminf(fmaxf(src[src_idx], -1.0f), 1.0f) * 127.0f);
            // dst[dst_idx] = roundf(fminf(fmaxf(src[dst_idx], -1.0f), 1.0f) * 127.0f);
        }
      }
    }
  }
}

// RKNN native int8 -> float32
static void ggml_rknpu2_transposed_from_native_int8(float * dst,
                                                  const int8_t * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/32, K/32, 32, 32)
  const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

  // Block copy 32x32 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 32; i++) {
      for (size_t ii = 0; ii < 32; ii++) {
        size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          /*if (src[dst_idx] > 1.0f || src[dst_idx] < -1.0f) {
            printf("error: src[dst_idx]=%f exceeds 1.0f\n", src[dst_idx]);
          }*/
          dst[src_idx] = float(src[dst_idx]) / 127.0f;
            // dst[dst_idx] = roundf(fminf(fmaxf(src[dst_idx], -1.0f), 1.0f) * 127.0f);
        }
      }
    }
  }
}

// memcpy(data, (const char *) tensor->data + offset, size)
int ggml_rknpure_transform_tensor(const void * data, struct ggml_tensor * tensor, size_t offset, size_t size)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    // this is the original type of tensor
    const enum ggml_type type = tensor->type;
    int inference_type;
    // switch (type) {
    //     case GGML_TYPE_Q8_0:
    //         inference_type = RKNN_TENSOR_INT8;
    //         break;
    //     case GGML_TYPE_F16:
    //         inference_type = RKNN_TENSOR_FLOAT16;
    //         break;
    //     default:
    //         printf("ERROR: unsupported tensor transform\n");
    //         abort();
    //         return 1;
    // }
    inference_type = RKNN_TENSOR_FLOAT16;

    if (offset) {
        printf("Error: offset not zero!\n");
        abort();
        return -2;
    }
    GGML_ASSERT(ne2 == 1 && ne3 == 1 && ne1 > 0 && ne0 > 0);
    // GGML_ASSERT(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16);
    // if (type != GGML_TYPE_F16) {
    //     GGML_ASSERT(ggml_is_quantized(type));
    // }

    return 0;
}

// convert tensor back (tensor->extra->ordered_data -> float32 -> tensor-> type(F16/Q8)) (actually not called)
void ggml_rknpu2_transform_tensor_back(void * data, const struct ggml_tensor * tensor, size_t offset, size_t size)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    const enum ggml_type type = tensor->type;
    printf("ggml_rknpu2_transform_tensor_back\n");
    abort();

    GGML_ASSERT(ne2 == 1 && ne3 == 1 && ne1 > 0 && ne0 > 0);
    GGML_ASSERT(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16);
    // type is transformed target type
    GGML_ASSERT(ggml_is_quantized(type));

    int inference_type;
    switch (type) {
        case GGML_TYPE_Q8_0:
            inference_type = RKNN_TENSOR_INT8;
            break;
        case GGML_TYPE_F16:
            inference_type = RKNN_TENSOR_FLOAT16;
            break;
        default:
            printf("ERROR: unsupported tensor transform\n");
            abort();
            return;
    }
}

int ggml_rknpure_can_mul_mat_b(const struct ggml_tensor * tensor)
{
    // Force CPU computation - always return false to skip NPU data transformation
    GGML_ASSERT(0 && "ggml_rknpure_can_mul_mat_b should not be called");
    return 0;
    // return 1;
    const int64_t k = tensor->ne[0];
    const int64_t n = tensor->ne[1];
    if(k > 8192 || n > 4096) // RKNPU2 limit
    {
        printf("%s: exceed limit\n", __func__);
        return 0;
    }

    // k and n size must align to 32 bytes
    if(k % 32 != 0 || n % 32 != 0) {
        printf("%s: not align\n", __func__);
        return 0;
    }

    // make sure the tensor has assosiated data
    // zzh: deprecated.
    if(strcmp(tensor->buffer->buft->iface.get_name(tensor->buffer->buft), "RKNPURE")) {
    // if(tensor->backend != GGML_BACKEND_TYPE_GPU)
        printf("iface not RKNPURE\n");
        return 0;
    }

    if(tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_F16)
    {
        printf("zzh: tensor->type != GGML_TYPE_Q8_0!\n");
        return 0;
    }

    return 1;
}

#ifdef GGML_USE_CHCORE
extern "C" {
    int usys_set_prio(int thread_cap, int prio);
    int usys_cache_flush(unsigned long start, unsigned long size, int op_type);
    void usys_disable_local_irq(void);
    void usys_enable_local_irq(void);
    void usys_yield(void);
}

/* cache operations */
#define CACHE_CLEAN         1
#define CACHE_INVALIDATE    2
#define CACHE_CLEAN_AND_INV 3
#define SYNC_IDCACHE        4
#endif

inline size_t rknn_type_size_A(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(__fp16);
    if (type == RKNN_TENSOR_INT8) return sizeof(int8_t);
    GGML_ASSERT(false);
}
inline size_t rknn_type_size_B(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(__fp16);
    if (type == RKNN_TENSOR_INT8) return sizeof(int8_t);
    GGML_ASSERT(false);
}
inline size_t rknn_type_size_C(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(float);
    if (type == RKNN_TENSOR_INT8) return sizeof(int32_t);
    GGML_ASSERT(false);
}

static uint64_t total_allocated = 0;

static int ggml_rknpu2_ensure_payload_ready(const void * payload, size_t size);

enum class rknpu_mem_kind {
    payload,
    compute,
    placeholder,
};

struct rknn_mem {
    size_t size;
    void *ptr;
    int fd;
    void *dma_ptr;
    uint64_t dma, obj;
    uint64_t handle;
    uint32_t domain_id;
    float scale;
    pthread_mutex_t scale_lock;
    rknpu_mem_kind kind;

    std::atomic<int> pre_scale_cnt;
    std::atomic<int> pre1_cnt;
    std::atomic<int> post_cnt;


    rknn_mem(size_t size, rknpu_mem_kind kind): size(size), domain_id(UINT32_MAX), kind(kind) {
        scale = 1.0f;
        pthread_mutex_init(&scale_lock, 0);

        // In offline-prepack-only mode, B still needs an object for task/scale bookkeeping,
        // but pre1 will overwrite the real DMA/domain from the GGUF payload before submit.
        if (kind == rknpu_mem_kind::placeholder) {
            // init below fields to dummy values to avoid accidental use before overwrite
            domain_id = 0;
            ptr = nullptr;
            fd = -1;
            dma_ptr = nullptr;
            dma = 0;
            obj = 0;
            handle = 0;
            return;
        }

        // Use NON_CACHEABLE memory like rknpu-tests, so no cache sync needed after NPU computation
        if (kind == rknpu_mem_kind::compute) {
            dma_ptr = mem_allocate_compute(size, &dma, &obj,
                RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT, &handle);
        } else {
            dma_ptr = mem_allocate_payload(size, &dma, &obj,
                RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT, &handle, &domain_id);
        }
#ifdef FAKE_CACHE
        GGML_ASSERT(dma_alloc(size, &fd, &ptr) == 0);
#else
        ptr = dma_ptr;
#endif
        // GGML_ASSERT(dma_ptr);
        total_allocated += size;
        //fprintf(stderr, "rknn_mem allocated: %lu bytes, total allocated: %lu bytes\n", size, total_allocated);
    }
    ~rknn_mem(void) {
#ifdef FAKE_CACHE
        dma_buf_free(size, &fd, ptr);
#endif
        if (kind != rknpu_mem_kind::placeholder) {
            mem_destroy(dma_ptr, size, handle, obj);
        }
    }
    void init_scale(void) { scale = RKNPU_PREPACK_SCALE_MIN; }
    void commit_scale(float _scale) {
        pthread_mutex_lock(&scale_lock);
        scale = std::max(scale, _scale);
        pthread_mutex_unlock(&scale_lock);
    }
    void reset_cnt(void) {
        pre_scale_cnt = 0;
        pre1_cnt = 0;
        post_cnt = 0;
    }
};

#define A_buf_size (rknn_type_size_A(type) * M * K)
#define B_buf_size (rknn_type_size_B(type) * K * N)
#define C_buf_size (rknn_type_size_C(type) * M * N)
#define A_size (rknn_type_size_A(type) * m * k)
#define B_size (rknn_type_size_B(type) * k * n)
#define C_size (rknn_type_size_C(type) * m * n)

struct A_bufs {
    int M, K, m, k;
    rknn_tensor_type type;
    std::map<std::tuple<int, int>, std::shared_ptr<rknn_mem>> As;

    A_bufs(int M, int K, int m, int k, rknn_tensor_type type)
        : M(M), K(K), m(m), k(k), type(type) {
        for (int mm = 0; mm < m; mm += M)
            for (int kk = 0; kk < k; kk += K)
                As.emplace(std::make_tuple(mm, kk),
                           std::make_shared<rknn_mem>(A_buf_size,
                                                      rknpu_mem_kind::compute));
    }
};
struct B_bufs {
    int K, N, k, n;
    rknn_tensor_type type;
    std::map<std::tuple<int, int>, std::shared_ptr<rknn_mem>> Bs;

    // 由于使用prepack blob的方式来传递B，所以这里初始化的B_bufs只起到占位作用，并不分配真正的内存。
    B_bufs(int K, int N, int k, int n, rknn_tensor_type type)
        : K(K), N(N), k(k), n(n), type(type) {
        for (int nn = 0; nn < n; nn += N)
            for (int kk = 0; kk < k; kk += K)
                Bs.emplace(std::make_tuple(nn, kk),
                           std::make_shared<rknn_mem>(B_buf_size,
                                                      rknpu_mem_kind::placeholder));
    }
};
struct C_bufs {
    int M, K, N, m, k, n;
    rknn_tensor_type type;
    std::map<std::tuple<int, int, int>, std::shared_ptr<rknn_mem>> Cs;

    C_bufs(int M, int K, int N, int m, int k, int n, rknn_tensor_type type)
        : M(M), K(K), N(N), m(m), k(k), n(n), type(type) {
        for (int mm = 0; mm < m; mm += M)
            for (int nn = 0; nn < n; nn += N)
                for (int kk = 0; kk < k; kk += K)
                    Cs.emplace(std::make_tuple(mm, nn, kk),
                               std::make_shared<rknn_mem>(C_buf_size,
                                                          rknpu_mem_kind::compute));
    }
};

struct matmul_buffer_mgr {
    std::map<std::tuple<int, int, int, int, rknn_tensor_type>, std::shared_ptr<A_bufs>> A_map;
    std::map<std::tuple<int, int, int, int, rknn_tensor_type>, std::shared_ptr<B_bufs>> B_map;
    std::map<std::tuple<int, int, int, int, int, int, rknn_tensor_type>, std::shared_ptr<C_bufs>> C_map;

    std::shared_ptr<A_bufs> get_A_bufs(int M, int K, int m, int k, rknn_tensor_type type) {
        auto A_bufs_iter = A_map.find(std::make_tuple(M, K, m, k, type));
        std::shared_ptr<A_bufs> ret;
        if (A_bufs_iter == A_map.end()) {
            ret = std::make_shared<A_bufs>(M, K, m, k, type);
            A_map.emplace(std::make_tuple(M, K, m, k, type), ret);
        } else {
            ret = A_bufs_iter->second;
        }
        return ret;
    }
    std::shared_ptr<B_bufs> get_B_bufs(int K, int N, int k, int n, rknn_tensor_type type) {
        auto B_bufs_iter = B_map.find(std::make_tuple(K, N, k, n, type));
        std::shared_ptr<B_bufs> ret;
        if (B_bufs_iter == B_map.end()) {
            ret = std::make_shared<B_bufs>(K, N, k, n, type);
            B_map.emplace(std::make_tuple(K, N, k, n, type), ret);
        } else {
            ret = B_bufs_iter->second;
        }
        return ret;
    }
    std::shared_ptr<C_bufs> get_C_bufs(int M, int K, int N, int m, int k, int n, rknn_tensor_type type) {
        auto C_bufs_iter = C_map.find(std::make_tuple(M, K, N, m, k, n, type));
        std::shared_ptr<C_bufs> ret;
        if (C_bufs_iter == C_map.end()) {
            ret = std::make_shared<C_bufs>(M, K, N, m, k, n, type);
            C_map.emplace(std::make_tuple(M, K, N, m, k, n, type), ret);
        } else {
            ret = C_bufs_iter->second;
        }
        return ret;
    }
    void clear(void) {
        A_map.clear();
        B_map.clear();
        C_map.clear();
    }
};

static struct matmul_buffer_mgr matmul_buffer_mgr;

struct npu_task {
    int M, N, K;
    int nn, kk;
    uint64_t *regcmd;
    uint64_t regcmd_dma, regcmd_obj;
    uint64_t regcmd_handle;
    uint32_t domain_id;
#ifdef GGML_USE_CHCORE
    struct rknpu_task *tasks;
    uint64_t tasks_dma, tasks_obj;
    uint64_t tasks_handle;
#endif
    std::shared_ptr<rknn_mem> input; // A
    std::shared_ptr<rknn_mem> weight; // B
    std::shared_ptr<rknn_mem> output; // C
    uint64_t npu_regs[112];
    matmul_params_t params;
    rknn_tensor_type type;
    const void * weight_payload = nullptr;
    size_t weight_payload_size = 0;

    npu_task(int M, int N, int K, int nn, int kk, rknn_tensor_type type,
             std::shared_ptr<rknn_mem> input, std::shared_ptr<rknn_mem> weight, std::shared_ptr<rknn_mem> output)
        : M(M), N(N), K(K), nn(nn), kk(kk), domain_id(weight->domain_id),
          input(input), weight(weight), output(output), type(type) {
        regcmd = (uint64_t*)mem_allocate_compute(1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);
        GGML_ASSERT(regcmd);

#ifdef GGML_USE_CHCORE
        tasks = (rknpu_task *)mem_allocate_compute(1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
        GGML_ASSERT(tasks);
#endif
        
        // memset(input->ptr, 1, input->size);
        //memset(weight->ptr, 1, 1);
        // *((int32_t*)weight->ptr) = 1;
        params.m = M;
        params.k = K;
        params.n = N;
        params.input_dma = input->dma;
        params.weights_dma = weight->dma;
        params.output_dma = output->dma;
        params.tasks = (uint64_t *)&npu_regs;
        if (type == RKNN_TENSOR_FLOAT32) {
            params.fp32tofp16 = 0;
            int ret = gen_matmul_fp16(&params);
            GGML_ASSERT(ret == 0);
        } else if (type == RKNN_TENSOR_INT8) {
            int ret = gen_matmul_int8(&params);
            GGML_ASSERT(ret == 0);
        } else {
            GGML_ASSERT(false);
        }
        memcpy(regcmd, npu_regs,sizeof(npu_regs));

#ifdef GGML_USE_CHCORE
        tasks[0].flags  = 0;
        tasks[0].op_idx = 0;
        tasks[0].enable_mask = 0xd;
        tasks[0].int_mask = 0x300; // wait for DPU to finish
        tasks[0].int_clear = 0x1ffff;
        tasks[0].int_status = 0;
        tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
        tasks[0].regcfg_offset = 0;
        tasks[0].regcmd_addr = regcmd_dma;
#endif

        // memset(input->ptr, 1, input->size);
        // memset(weight->ptr, 1, weight->size);
    }

    ~npu_task(void) {
        mem_destroy(regcmd, 1024, regcmd_handle, regcmd_obj);
#ifdef GGML_USE_CHCORE
        mem_destroy(tasks, 1024, tasks_handle, tasks_obj);
#endif
    }

#ifdef GGML_USE_CHCORE
    void flush_cache(void) {
        usys_cache_flush((unsigned long)input->ptr, A_buf_size, CACHE_CLEAN);
#ifdef MAT_COPY
        usys_cache_flush((unsigned long)weight->ptr, B_buf_size, CACHE_CLEAN);
#endif
        usys_cache_flush((unsigned long)output->ptr, C_buf_size, CACHE_CLEAN_AND_INV);
    }
#else
    void flush_cache_before_submit(void) {
#ifdef FAKE_CACHE
        GGML_ASSERT(dma_sync_cpu_to_device(input->fd) == 0);
#ifdef MAT_COPY
        GGML_ASSERT(dma_sync_cpu_to_device(weight->fd) == 0);
#endif
        GGML_ASSERT(dma_sync_cpu_to_device(output->fd) == 0);
#endif
    }
    void flush_cache_after_submit(void) {
#ifdef FAKE_CACHE
        // Sync output from device to CPU after NPU computation completes
        GGML_ASSERT(dma_sync_device_to_cpu(output->fd) == 0);
#else
        // For NON_CACHEABLE memory (like rknpu-tests), no cache sync needed
        // NPU writes are directly visible to CPU after ioctl returns
#endif
    }
    void submit(int core_mask) {
        flush_cache_before_submit();
        // for(int i = 0; i < 4; i++){
        //     fprintf(stderr, "weight->ptr[%d] = %d\n", i, ((int32_t*)weight->ptr)[i]);
        // }
        // *((int32_t*)weight->ptr) = 1;
        //log_submit_weight("single", (__u32)core_mask);
        int ret = npu_submit(regcmd_dma, (__u32)core_mask, domain_id);
        if (ret) {
            printf("RKNPU_SUBMIT returned %d, submitted m=%hu, k=%hu, n=%hu, errno %d\n",
                ret, M, K, N, errno);
        }
        GGML_ASSERT(ret == 0);
        // After npu_submit returns (blocking call), sync output from device to CPU
        flush_cache_after_submit();
        
        // Log NPU computation result immediately after completion
        // Print first 5 8-byte values from output buffer
        const uint64_t * output_bytes = (const uint64_t *)output->ptr;
        const size_t output_size = M*N*sizeof(int32_t);
        const int num_8bytes = (output_size >= 5 * sizeof(uint64_t)) ? 5 : (int)(output_size / sizeof(uint64_t));
        // fprintf(stderr, "[NPU_TASK] NPU computation type=%d completed! M=%d, N=%d, K=%d, output first 5 8bytes: ",type, M, N, K);
        // for (int i = 0; i < num_8bytes; i++) {
        //     fprintf(stderr, "0x%016llx ", (unsigned long long)output_bytes[i]);
        // }
        // fprintf(stderr, "\n");
    }
#endif

    void apply_scale(void) {
        output->scale = input->scale * weight->scale;
    }

    void set_weight_location(uint64_t weights_dma, uint32_t weights_domain_id,
                             const void * payload, size_t payload_size) {
        const bool same_location =
            params.weights_dma == weights_dma && domain_id == weights_domain_id;
        weight_payload = payload;
        weight_payload_size = payload_size;

        if (same_location) {
            return;
        }

        params.weights_dma = weights_dma;
        domain_id = weights_domain_id;
        if (type == RKNN_TENSOR_FLOAT32) {
            params.fp32tofp16 = 0;
            int ret = gen_matmul_fp16(&params);
            GGML_ASSERT(ret == 0);
        } else if (type == RKNN_TENSOR_INT8) {
            int ret = gen_matmul_int8(&params);
            GGML_ASSERT(ret == 0);
        } else {
            GGML_ASSERT(false);
        }

        memcpy(regcmd, npu_regs, sizeof(npu_regs));
    }

    void log_submit_weight(const char * mode, uint32_t core_mask) const {
        const void * dma_ptr = weight_payload != nullptr ? weight_payload :
            (weight ? weight->dma_ptr : nullptr);
        std::fprintf(stderr,
                "[GUEST_NPU_SUBMIT] mode=%s M=%d N=%d K=%d nn=%d kk=%d weight_dma_ptr=%p weight_dma=0x%llx domain=%u payload_size=0x%zx regcmd_dma=0x%llx core_mask=0x%x\n",
                mode,
                M, N, K, nn, kk,
                dma_ptr,
                (unsigned long long) params.weights_dma,
                domain_id,
                weight_payload_size,
                (unsigned long long) regcmd_dma,
                core_mask);
    }

    int ensure_weight_ready(void) const {
        if (weight_payload == nullptr || weight_payload_size == 0) {
            return 0;
        }

        const int ret = ggml_rknpu2_ensure_payload_ready(weight_payload,
                                                        weight_payload_size);
        if (ret != 0) {
            GGML_LOG_ERROR("%s: weight payload not ready nn=%d kk=%d dma=0x%llx domain=%u size=%zu\n",
                    __func__, nn, kk,
                    (unsigned long long) params.weights_dma,
                    domain_id, weight_payload_size);
        }
        return ret;
    }
};

extern "C" {
    int npu_submit_multi(uint64_t task_obj_addr[], int task_num,
                         uint32_t domain_id, void *poll);
}

struct npu_task_multi_core {
    std::vector<std::shared_ptr<npu_task>> npu_tasks;

    npu_task_multi_core(std::shared_ptr<npu_task> task)
        : npu_tasks(1, task) {}
    npu_task_multi_core(const std::vector<std::shared_ptr<npu_task>> &tasks)
        : npu_tasks(tasks) {}

    void submit(void) {
        GGML_ASSERT(npu_tasks.size() <= NPU_CORE_NUM);
        for (auto npu_task: npu_tasks) {
            GGML_ASSERT(npu_task->ensure_weight_ready() == 0);
        }

        uint32_t domain_id = npu_tasks.front()->domain_id;
        bool same_domain = true;
        std::vector<uint64_t> tasks_objs;
        for (auto npu_task: npu_tasks) {
            same_domain = same_domain && (npu_task->domain_id == domain_id);
    #ifdef GGML_USE_CHCORE
            npu_task->flush_cache();
            tasks_objs.push_back((uint64_t) npu_task->tasks_obj);
    #else
            npu_task->flush_cache_before_submit();
            tasks_objs.push_back((uint64_t) npu_task->regcmd_dma);
    #endif
        }
        if (same_domain) {
            /*
            for (size_t idx = 0; idx < npu_tasks.size(); ++idx) {
                npu_tasks[idx]->log_submit_weight("multi", 1u << idx);
            }
            */
            int ret = npu_submit_multi(tasks_objs.data(), tasks_objs.size(), domain_id,
                                       (void *)ggml_thread_cpu_relax_out);
            GGML_ASSERT(ret == 0);
        } else {
            for (size_t idx = 0; idx < npu_tasks.size(); ++idx) {
                npu_tasks[idx]->submit(1u << idx);
            }
        }
    #ifndef GGML_USE_CHCORE
        if (same_domain) {
            for (auto npu_task: npu_tasks) {
                npu_task->flush_cache_after_submit();
            }
        }
    #endif
        
        // Log NPU computation result immediately after completion
        for (size_t idx = 0; idx < npu_tasks.size(); idx++) {
            auto npu_task = npu_tasks[idx];
            const uint64_t * output_bytes = (const uint64_t *)npu_task->output->ptr;
            const size_t output_size = npu_task->output->size;
            const int num_8bytes = (output_size >= 5 * sizeof(uint64_t)) ? 5 : (int)(output_size / sizeof(uint64_t));
            // fprintf(stderr, "[NPU_TASK_MULTI] Task %zu, M=%d, N=%d, K=%d, output first 5 8bytes: ", 
            //         idx, npu_task->M, npu_task->N, npu_task->K);
            // for (int i = 0; i < num_8bytes; i++) {
            //     fprintf(stderr, "0x%016llx ", (unsigned long long)output_bytes[i]);
            // }
            // fprintf(stderr, "\n");
        }
    }
    void apply_scale(void) {
        for (auto task: npu_tasks) {
            task->apply_scale();
        }
    }
};

struct matmul_kernel {
    const int MAX_N = 4096;
    const int ALIGN_N = 32;
    const int MAX_K = 4096;
    const int ALIGN_K = 32;
    int m, n, k;
    int M, N, K;
    rknn_tensor_type type;
    std::vector<std::shared_ptr<npu_task_multi_core>> npu_tasks;
    std::shared_ptr<A_bufs> inputs;
    std::shared_ptr<B_bufs> weights;
    std::shared_ptr<C_bufs> outputs;
    static inline int partition(int num, int max, int align) {
        return rknpu_prepack_partition(num, max, align);
    }
    matmul_kernel(int m, int n, int k, rknn_tensor_type type)
        : m(m), n(n), k(k), type(type) {
        // partition m, n, k into M, N, K;

        N = partition(n / NPU_CORE_NUM, MAX_N, ALIGN_N);
        K = partition(k, MAX_K, ALIGN_K);
        // N = std::min(n / THREAD_NR / 32 * 32 + 32, 4096); K = std::min(k / 32 * 32 + 32, 4096);
        // let N be 4096, experiments show the following limitations: 
        if (type == RKNN_TENSOR_FLOAT32) {
            if (K <= 416) {
                M = std::min(m, 384);
            } else if (K <= 448) {
                M = std::min(m, 364);
            } else if (K <= 480) {
                M = std::min(m, 340);
            } else if (K <= 512) {
                M = std::min(m, 352);
            } else if (K <= 544) {
                M = std::min(m, 300);
            } else if (K <= 576) {
                M = std::min(m, 284);
            } else if (K <= 608) {
                M = std::min(m, 268);
            } else if (K <= 640) {
                M = std::min(m, 256);
            } else if (K <= 736) {
                M = std::min(m, 220);
            } else if (K <= 768) {
                M = std::min(m, 212);
            } else if (K <= 800) {
                M = std::min(m, 204);
            } else if (K <= 832) {
                M = std::min(m, 196);
            } else if (K <= 864) {
                M = std::min(m, 188);
            } else if (K <= 896) {
                M = std::min(m, 180);
            } else if (K <= 928) {
                M = std::min(m, 176);
            } else if (K <= 960) {
                M = std::min(m, 168);
            } else if (K <= 992) {
                M = std::min(m, 164);
            } else if (K <= 1024) {
                M = std::min(m, 176);
            } else if (K <= 1056) {
                M = std::min(m, 152);
            } else if (K <= 1088) {
                M = std::min(m, 148);
            } else if (K <= 1120) {
                M = std::min(m, 128);
            } else if (K <= 1152) {
                M = std::min(m, 140);
            } else if (K <= 1184) {
                M = std::min(m, 124);
            } else if (K <= 1216) {
                M = std::min(m, 120);
            } else if (K <= 1248) {
                M = std::min(m, 116);
            } else if (K <= 1280) {
                M = std::min(m, 128);
            } else if (K <= 1312) {
                M = std::min(m, 112);
            } else if (K <= 1344) {
                M = std::min(m, 108);
            } else if (K <= 1408) {
                M = std::min(m, 104);
            } else if (K <= 1472) {
                M = std::min(m, 100);
            } else if (K <= 1504) {
                M = std::min(m, 96);
            } else if (K <= 1536) {
                M = std::min(m, 100);
            } else if (K <= 1600) {
                M = std::min(m, 92);
            } else if (K <= 1664) {
                M = std::min(m, 88);
            } else if (K <= 1728) {
                M = std::min(m, 84);
            } else if (K <= 1824) {
                M = std::min(m, 80);
            } else if (K <= 1920) {
                M = std::min(m, 76);
            } else if (K <= 2016) {
                M = std::min(m, 72);
            } else if (K <= 2048) {
                M = std::min(m, 80);
            } else if (K <= 2112) {
                M = std::min(m, 68);
            } else if (K <= 2144) {
                M = std::min(m, 60);
            } else if (K <= 2176) {
                M = std::min(m, 64);
            } else if (K <= 2272) {
                M = std::min(m, 56);
            } else if (K <= 2304) {
                M = std::min(m, 64);
            } else if (K <= 2336) {
                M = std::min(m, 56);
            } else if (K <= 2496) {
                M = std::min(m, 52);
            } else if (K <= 2528) {
                M = std::min(m, 48);
            } else if (K <= 2560) {
                M = std::min(m, 56);
            } else if (K <= 2720) {
                M = std::min(m, 48);
            } else if (K <= 2976) {
                M = std::min(m, 44);
            } else if (K <= 3040) {
                M = std::min(m, 40);
            } else if (K <= 3072) {
                M = std::min(m, 48);
            } else if (K <= 3136) {
                M = std::min(m, 40);
            } else if (K <= 3168) {
                M = std::min(m, 36);
            } else if (K <= 3200) {
                M = std::min(m, 40);
            } else if (K <= 3296) {
                M = std::min(m, 32);
            } else if (K <= 3328) {
                M = std::min(m, 36);
            } else if (K <= 3552) {
                M = std::min(m, 32);
            } else if (K <= 3584) {
                M = std::min(m, 36);
            } else if (K <= 4064) {
                M = std::min(m, 28);
            } else {
                M = std::min(m, 32);
            }
        } else if (type == RKNN_TENSOR_INT8) {
            if (K <= 576) {
                M = std::min(m, 544);
            } else if (K <= 768) {
                M = std::min(m, 424);
            } else if (K <= 960) {
                M = std::min(m, 340);
            } else if (K <= 992) {
                M = std::min(m, 320);
            } else if (K <= 1024) {
                M = std::min(m, 352);
            } else if (K <= 1280) {
                M = std::min(m, 256);
            } else if (K <= 1472) {
                M = std::min(m, 200);
            } else if (K <= 1504) {
                M = std::min(m, 192);
            } else if (K <= 1536) {
                M = std::min(m, 212);
            } else if (K <= 1984) {
                M = std::min(m, 148);
            } else if (K <= 2016) {
                M = std::min(m, 144);
            } else if (K <= 2048) {
                M = std::min(m, 160);
            } else if (K <= 2528) {
                M = std::min(m, 100);
            } else if (K <= 2560) {
                M = std::min(m, 112);
            } else if (K <= 3040) {
                M = std::min(m, 84);
            } else if (K <= 3072) {
                M = std::min(m, 96);
            } else if (K <= 3552) {
                M = std::min(m, 64);
            } else if (K <= 3584) {
                M = std::min(m, 72);
            } else if (K <= 4064) {
                M = std::min(m, 56);
            } else {
                M = std::min(m, 64);
            }
        }
        // Besides, the open source driver could only handle (m==1) or (m%4==0).
        // Otherwise the results are not accurate.
        if (M != 1 && M % 4 != 0) {
            // Change to next multiple of 4
            M = (M + 3) / 4 * 4;
        }

        // printf("partition (m, k, n) = (%d, %d, %d) to (M, K, N) = (%d, %d, %d)\n", m, k, n, M, K, N);
        
        inputs = matmul_buffer_mgr.get_A_bufs(M, K, m, k, type);
#ifdef MAT_COPY
        weights = matmul_buffer_mgr.get_B_bufs(K, N, k, n, type);
#else
        weights = nullptr;
#endif
        outputs = matmul_buffer_mgr.get_C_bufs(M, K, N, m, k, n, type);

        for (int kk = 0; kk < k; kk += K) {
            for (int mm = 0; mm < m; mm += M) {
                std::vector<std::shared_ptr<npu_task>> tmp_tasks;
                for (int nn = 0; nn < n; nn += N) {
                    auto input = inputs->As.find(std::make_tuple(mm, kk))->second;
#ifdef MAT_COPY
                    auto weight = weights->Bs.find(std::make_tuple(nn, kk))->second;
#else
                    static std::shared_ptr<rknn_mem> global_weight =
                        std::make_shared<rknn_mem>(4096 * 4096 * rknn_type_size_B(type),
                                                   rknpu_mem_kind::payload);
                    std::shared_ptr<rknn_mem> weight(global_weight);
#endif
                    auto output = outputs->Cs.find(std::make_tuple(mm, nn, kk))->second;
                    tmp_tasks.emplace_back(std::make_shared<npu_task>(M, N, K, nn, kk, type, input, weight, output));
                    if (tmp_tasks.size() == NPU_CORE_NUM || nn + N >= n) {
                        npu_tasks.emplace_back(std::make_shared<npu_task_multi_core>(tmp_tasks));
                        tmp_tasks.clear();
                    }
                }
                GGML_ASSERT(tmp_tasks.empty());
            }
        }
    }

    void for_all_inputs(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
        for (const auto &[shape, mem]: inputs->As) {
            auto [mm, kk] = shape;
            closure(mm, kk, M, K, mem);
        }
    }
    void for_all_weights(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
#ifdef MAT_COPY
        for (const auto &[shape, mem]: weights->Bs) {
            auto [nn, kk] = shape;
            closure(nn, kk, N, K, mem);
        }
#endif
    }
    void for_all_outputs(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
        for (const auto &pair: outputs->Cs) {
            const auto &shape = pair.first;
            const auto &mem = pair.second;
            int mm = std::get<0>(shape);
            int nn = std::get<1>(shape);
            // int _ = std::get<2>(shape); // unused
            closure(mm, nn, M, N, mem);
        }
    }

#ifndef GGML_USE_CHCORE
    std::vector<std::vector<std::shared_ptr<npu_task>>> to_multi_npu(int npu_nr) const {
        std::vector<std::vector<std::shared_ptr<npu_task>>> tasks;
        tasks.resize(npu_nr);
        for (auto task: npu_tasks) {
            GGML_ASSERT(task->npu_tasks.size() <= npu_nr);
            for (std::size_t i = 0; i < task->npu_tasks.size(); i++) {
                tasks[i].push_back(task->npu_tasks[i]);
            }
        }
        return tasks;
    }
#endif
};

std::vector<std::shared_ptr<matmul_kernel>> matmul_kernels;

static inline int8_t f32_to_i8(float x, float scale);

struct rknpu_weight_prepack_block {
    int nn;
    int kk;
    float scale;
    uint64_t packed_dma = 0; // weight_dma may legitimately be 0 when the packed payload starts at the beginning of an IOVA window
    uint32_t domain_id = 0;
    const void * payload = nullptr;
    size_t payload_size = 0;
};

struct rknpu_weight_prepack_key {
    const ggml_tensor * src0;
    int64_t k;
    int64_t n;
    int K;
    int N;
    rknn_tensor_type type;

    bool operator==(const rknpu_weight_prepack_key & other) const {
        return src0 == other.src0 &&
               k == other.k &&
               n == other.n &&
               K == other.K &&
               N == other.N &&
               type == other.type;
    }
};

struct rknpu_weight_prepack_key_hash {
    size_t operator()(const rknpu_weight_prepack_key & key) const {
        size_t h = std::hash<const void *>{}(key.src0);
        h ^= std::hash<int64_t>{}(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.type) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct rknpu_weight_prepack_cache {
    rknpu_weight_prepack_key key;
    std::unordered_map<uint64_t, rknpu_weight_prepack_block> blocks;
};

struct rknpu_offline_prepack_blob {
    std::string tensor_name;
    std::string meta_tensor_name;
    std::string payload_tensor_name;
    std::string layout;
    ggml_rknpu_prepack_meta meta;
    const ggml_tensor * meta_tensor = nullptr;
    const ggml_tensor * payload_tensor = nullptr;
    const uint8_t * meta_cpu_ptr = nullptr;
    uint64_t payload_dma = 0;
    uint32_t payload_domain_id = 0;
    size_t meta_size = 0;
    size_t payload_size = 0;
};

static std::mutex g_weight_prepack_mtx;
static std::mutex g_offline_prepack_mtx;
static std::unordered_map<std::string, rknpu_offline_prepack_blob> g_offline_prepack_registry;
static std::unordered_map<rknpu_weight_prepack_key, std::shared_ptr<rknpu_weight_prepack_cache>, rknpu_weight_prepack_key_hash> g_weight_prepack_cache;
static std::atomic<ggml_rknpu2_payload_ready_callback> g_payload_ready_callback{nullptr};

extern "C" void ggml_rknpu2_set_payload_ready_callback(ggml_rknpu2_payload_ready_callback cb) {
    g_payload_ready_callback.store(cb, std::memory_order_release);
}

static int ggml_rknpu2_ensure_payload_ready(const void * payload, size_t size) {
    if (payload == nullptr || size == 0) {
        return 0;
    }

    ggml_rknpu2_payload_ready_callback cb =
        g_payload_ready_callback.load(std::memory_order_acquire);
    if (cb == nullptr) {
        return 0;
    }

    return cb(payload, size);
}

void ggml_rknpu2_clear_offline_prepack_registry(void) {
    std::lock_guard<std::mutex> lock(g_offline_prepack_mtx);
    g_offline_prepack_registry.clear();
}

bool ggml_rknpu2_register_offline_prepack(const struct ggml_rknpu_prepack_meta * meta, const struct ggml_tensor * meta_tensor, const struct ggml_tensor * payload_tensor) {
    if (meta == nullptr || meta->tensor_name == nullptr || meta->meta_tensor_name == nullptr || meta->payload_tensor_name == nullptr ||
        meta_tensor == nullptr || payload_tensor == nullptr || meta_tensor->data == nullptr || payload_tensor->data == nullptr) {
        return false;
    }

    auto * payload_extra = (const ggml_backend_rknpu2_tensor_extra *) payload_tensor->extra;
    if (payload_extra == nullptr) {
        GGML_LOG_ERROR("%s: payload tensor %s is missing RKNPURE DMA metadata\n",
                __func__, ggml_get_name(payload_tensor));
        return false;
    }

    rknpu_offline_prepack_blob blob;
    blob.tensor_name = meta->tensor_name;
    blob.meta_tensor_name = meta->meta_tensor_name;
    blob.payload_tensor_name = meta->payload_tensor_name;
    if (meta->layout != nullptr) {
        blob.layout = meta->layout;
    }
    blob.meta = *meta;
    blob.meta.tensor_name = blob.tensor_name.c_str();
    blob.meta.meta_tensor_name = blob.meta_tensor_name.c_str();
    blob.meta.payload_tensor_name = blob.payload_tensor_name.c_str();
    blob.meta.layout = blob.layout.c_str();
    blob.meta_tensor = meta_tensor;
    blob.payload_tensor = payload_tensor;
    blob.meta_cpu_ptr = static_cast<const uint8_t *>(meta_tensor->data);
    blob.payload_dma = payload_extra->dma;
    blob.payload_domain_id = payload_extra->domain_id;
    blob.meta_size = ggml_nbytes(meta_tensor);
    blob.payload_size = payload_extra->size;

    std::fprintf(stderr,
            "[RKNPU_ALLOC][WEIGHT_PREPACK] tensor=%s payload_tensor=%s size=%zu dma=0x%llx domain=%u\n",
            blob.tensor_name.c_str(),
            blob.payload_tensor_name.c_str(),
            blob.payload_size,
            (unsigned long long) blob.payload_dma,
            blob.payload_domain_id);

    std::lock_guard<std::mutex> lock(g_offline_prepack_mtx);
    g_offline_prepack_registry[meta->tensor_name] = std::move(blob);
    return true;
}

static const rknpu_offline_prepack_blob * ggml_rknpu2_find_offline_prepack(const char * tensor_name) {
    if (tensor_name == nullptr || tensor_name[0] == '\0') {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_offline_prepack_mtx);
    auto it = g_offline_prepack_registry.find(tensor_name);
    if (it == g_offline_prepack_registry.end()) {
        return nullptr;
    }
    return &it->second;
}

int ggml_rknpu2_get_offline_prepack_payload(const char * tensor_name, const void ** payload, size_t * size) {
    if (payload == nullptr || size == nullptr) {
        return -1;
    }

    *payload = nullptr;
    *size = 0;
    if (tensor_name == nullptr || tensor_name[0] == '\0') {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_offline_prepack_mtx);
    auto it = g_offline_prepack_registry.find(tensor_name);
    if (it == g_offline_prepack_registry.end()) {
        return -1;
    }

    const rknpu_offline_prepack_blob & offline = it->second;
    if (offline.payload_tensor == nullptr || offline.payload_tensor->data == nullptr ||
        offline.payload_size == 0) {
        return -1;
    }

    *payload = offline.payload_tensor->data;
    *size = offline.payload_size;
    return 0;
}

static int ggml_rknpu2_ensure_offline_payload_ready(const rknpu_offline_prepack_blob * offline) {
    if (offline == nullptr || offline->payload_tensor == nullptr || offline->payload_tensor->data == nullptr) {
        return -1;
    }

    return ggml_rknpu2_ensure_payload_ready(offline->payload_tensor->data,
                                            offline->payload_size);
}

static std::atomic<uint64_t> g_weight_prepack_lookup_cnt{0};
static std::atomic<uint64_t> g_weight_prepack_hit_cnt{0};
static std::atomic<uint64_t> g_weight_prepack_build_cnt{0};
static std::atomic<uint64_t> g_weight_block_hit_prescale_cnt{0};
static std::atomic<uint64_t> g_weight_block_miss_prescale_cnt{0};
static std::atomic<uint64_t> g_weight_block_hit_pre1_cnt{0};
static std::atomic<uint64_t> g_weight_block_miss_pre1_cnt{0};
static std::atomic<uint64_t> g_decode_pre0_cnt{0};

static void ggml_rknpu2_dump_weight_prepack_stats(void) {
    const uint64_t lookup = g_weight_prepack_lookup_cnt.load();
    const uint64_t hit = g_weight_prepack_hit_cnt.load();
    const uint64_t build = g_weight_prepack_build_cnt.load();
    const uint64_t pre_scale_hit = g_weight_block_hit_prescale_cnt.load();
    const uint64_t pre_scale_miss = g_weight_block_miss_prescale_cnt.load();
    const uint64_t pre1_hit = g_weight_block_hit_pre1_cnt.load();
    const uint64_t pre1_miss = g_weight_block_miss_pre1_cnt.load();

    // fprintf(stderr,
    //         "[RKNPU_PREPACK] lookup=%llu hit=%llu build=%llu | pre_scale block hit=%llu miss=%llu | pre1 block hit=%llu miss=%llu\n",
    //         (unsigned long long)lookup,
    //         (unsigned long long)hit,
    //         (unsigned long long)build,
    //         (unsigned long long)pre_scale_hit,
    //         (unsigned long long)pre_scale_miss,
    //         (unsigned long long)pre1_hit,
    //         (unsigned long long)pre1_miss);
}

static inline uint64_t rknpu_block_key(int a, int b) {
    return (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
}

static std::shared_ptr<rknpu_weight_prepack_cache> ggml_rknpu2_try_load_weight_prepack_from_offline(
    const ggml_tensor * src0,
    int64_t k,
    int64_t n,
    int K,
    int N,
    rknn_tensor_type tensor_type) {

    const rknpu_offline_prepack_blob * offline = ggml_rknpu2_find_offline_prepack(src0->name);
    GGML_ASSERT(offline); // 我们假定所有prepacked权重都必须被预打包并注册到离线预打包注册表中, 不存在fallback到现场pack的情况

    // std::fprintf(stderr, "[RKNPU_OFFLINE] %s: loaded offline prepack for tensor %s via meta %s payload %s\n",
    //         __func__, src0->name, offline->meta_tensor_name.c_str(), offline->payload_tensor_name.c_str());

    if (offline->meta_size < sizeof(rknpu_offline_blob_header)) {
        return nullptr;
    }

    const auto * header = reinterpret_cast<const rknpu_offline_blob_header *>(offline->meta_cpu_ptr);
    if (!rknpu_prepack_header_is_valid(header) ||
        header->k != (uint32_t) k ||
        header->n != (uint32_t) n ||
        header->K != (uint32_t) K ||
        header->N != (uint32_t) N ||
        header->block_count != offline->meta.block_count ||
        header->weight_bytes_per_block != offline->meta.weight_bytes_per_block ||
        header->scale_type != offline->meta.scale_type ||
        header->scales_bytes_total != offline->meta.scales_bytes_total ||
        header->packed_bytes_total != offline->meta.packed_bytes_total ||
        header->orig_type != (uint32_t) src0->type) {
        return nullptr;
    }

    const uint32_t expected_block_count = ((uint32_t(n) + uint32_t(N) - 1) / uint32_t(N)) * ((uint32_t(k) + uint32_t(K) - 1) / uint32_t(K));
    if (expected_block_count != header->block_count) {
        return nullptr;
    }

    const uint32_t packed_size = (uint32_t)(rknpu_prepack_weight_type_size(tensor_type) * K * N);
    if (packed_size != header->weight_bytes_per_block) {
        return nullptr;
    }

    if (offline->meta.K != header->K ||
        offline->meta.N != header->N) {
        return nullptr;
    }

    if (rknpu_prepack_meta_bytes(header) != offline->meta_size ||
        rknpu_prepack_payload_bytes(header) != offline->payload_size) {
        return nullptr;
    }

    if (tensor_type == RKNN_TENSOR_FLOAT32) {
        if (header->layout != RKNPU_PREPACK_LAYOUT_FP16) {
            return nullptr;
        }
    } else if (tensor_type == RKNN_TENSOR_INT8) {
        if (header->layout != RKNPU_PREPACK_LAYOUT_INT8_BLOCK_SCALE) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    auto cache = std::make_shared<rknpu_weight_prepack_cache>();
    cache->key = { src0, k, n, K, N, tensor_type };

    const float * scales = header->scales_bytes_total == 0
        ? nullptr
        : reinterpret_cast<const float *>(offline->meta_cpu_ptr + rknpu_prepack_scales_offset(header));
    const uint64_t packed_dma_base = offline->payload_dma;

    // std::fprintf(stderr, "[RKNPU_OFFLINE] %s: tensor=%s meta=%s payload=%s payload_dma=0x%llx meta_size=%zu payload_size=%zu packed_size=%u\n",
    //         __func__, src0->name, offline->meta_tensor_name.c_str(), offline->payload_tensor_name.c_str(),
    //         (unsigned long long) offline->payload_dma,
    //         offline->meta_size,
    //         offline->payload_size,
    //         packed_size);

    uint32_t block_index = 0;
    for (int nn = 0; nn < n; nn += N) {
        for (int kk = 0; kk < k; kk += K) {
            const uint64_t payload_offset = uint64_t(block_index) * packed_size;
            rknpu_weight_prepack_block block;
            block.nn = nn;
            block.kk = kk;
            block.scale = scales ? scales[block_index] : 1.0f;
            block.packed_dma = packed_dma_base + payload_offset;
            block.domain_id = offline->payload_domain_id;
            block.payload = static_cast<const uint8_t *>(offline->payload_tensor->data) +
                payload_offset;
            block.payload_size = packed_size;
            cache->blocks.emplace(rknpu_block_key(nn, kk), block);
            ++block_index;
        }
    }

    return cache;
}

// 构建cache函数，按理来说不需要cache直接现场算偏移就行，但是暂时保留方便debug
static std::shared_ptr<rknpu_weight_prepack_cache> ggml_rknpu2_get_weight_prepack(
    const ggml_tensor * src0,
    int64_t k,
    int64_t n,
    int K,
    int N,
    rknn_tensor_type tensor_type) {

    g_weight_prepack_lookup_cnt.fetch_add(1);
    const rknpu_offline_prepack_blob * offline = ggml_rknpu2_find_offline_prepack(src0->name);
    GGML_ASSERT(offline);
    if (ggml_rknpu2_ensure_offline_payload_ready(offline) != 0) {
        GGML_LOG_ERROR("%s: payload is not ready for tensor %s\n", __func__, src0->name);
        GGML_ABORT("%s: payload is not ready for tensor %s", __func__, src0->name);
    }

    rknpu_weight_prepack_key cache_key = { src0, k, n, K, N, tensor_type };
    {
        std::lock_guard<std::mutex> lock(g_weight_prepack_mtx);
        auto it = g_weight_prepack_cache.find(cache_key);
        if (it != g_weight_prepack_cache.end()) {
            g_weight_prepack_hit_cnt.fetch_add(1);
            return it->second;
        }
    }

    // 在此处构建好所有的预打包块信息，目前保证所有权重都被cache到，不存在fallback的情况（可以理解为下面这个函数返回nullptr都是校验不通过的情况）
    auto cache = ggml_rknpu2_try_load_weight_prepack_from_offline(src0, k, n, K, N, tensor_type);
    if (!cache) {
        GGML_LOG_ERROR("%s: offline prepack is required for tensor %s\n", __func__, src0->name);
        GGML_ABORT("%s: offline prepack is required for tensor %s", __func__, src0->name);
    }

    std::lock_guard<std::mutex> lock(g_weight_prepack_mtx);
    auto it = g_weight_prepack_cache.find(cache_key);
    if (it != g_weight_prepack_cache.end()) {
        g_weight_prepack_hit_cnt.fetch_add(1);
        return it->second;
    }
    g_weight_prepack_cache.emplace(cache_key, cache);
    g_weight_prepack_build_cnt.fetch_add(1);
    return cache;
}

static std::shared_ptr<rknpu_weight_prepack_cache> ggml_rknpu2_find_weight_prepack_cache(
    const ggml_tensor * src0,
    int64_t k,
    int64_t n,
    int K,
    int N,
    rknn_tensor_type tensor_type) {

    rknpu_weight_prepack_key cache_key = { src0, k, n, K, N, tensor_type };
    std::lock_guard<std::mutex> lock(g_weight_prepack_mtx);
    auto it = g_weight_prepack_cache.find(cache_key);
    if (it == g_weight_prepack_cache.end()) {
        return nullptr;
    }
    return it->second;
}

static inline const rknpu_weight_prepack_block * ggml_rknpu2_find_weight_prepack_block(
    const std::shared_ptr<rknpu_weight_prepack_cache> & cache,
    int nn,
    int kk) {
    if (!cache) {
        return nullptr;
    }
    auto it = cache->blocks.find(rknpu_block_key(nn, kk));
    if (it == cache->blocks.end()) {
        return nullptr;
    }
    return &it->second;
}

static std::shared_ptr<matmul_kernel>
ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type) {
    for (const auto &kernel: matmul_kernels) {
        if (kernel->m == m && kernel->k == k && kernel->n == n && kernel->type == type) {
            return kernel;
        }
    }
    return nullptr;
}
// first find from buffer, then reuse them
static std::shared_ptr<matmul_kernel>
ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_tensor_type type)
{
    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if (kernel != NULL)
        return kernel;

    kernel = std::make_shared<matmul_kernel>(m, n, k, type);
    matmul_kernels.emplace_back(kernel);
    return kernel;
}

void ggml_rknpu2_clear_matmul_cache(void) {
    matmul_kernels.clear();
    matmul_buffer_mgr.clear();
}

static void ggml_backend_rknpu2_mul_mat_mul_npu(
    struct ggml_rknpu2_data_pack** packs,
    const int64_t m,
    const int64_t k,
    const int64_t base_n,
    const int64_t size_n,
    const int64_t n,
    const enum ggml_type type,
    float *A,
    float *B,
    float *C,
    /*const */struct ggml_tensor * src0,
    rknn_core_mask core_mask
);

std::mutex mtx;
std::condition_variable cv_worker;
std::condition_variable cv_master;
bool done;
bool ready[THREAD_NR];
std::atomic<int> finished_nr;
static std::once_flag once_flag;

#ifdef GGML_USE_CHCORE
std::vector<std::shared_ptr<npu_task_multi_core>> npu_tasks[THREAD_NR];
#else
std::vector<std::shared_ptr<npu_task>> npu_tasks[THREAD_NR];
#endif

void npu_worker(int tid) {
    int ith = tid;
#ifndef GGML_USE_CHCORE
    struct sched_param param;
    int policy = SCHED_FIFO;
    int priority = 40;
    param.sched_priority = priority;
    // GGML_ASSERT(pthread_setschedparam(pthread_self(), policy, &param) == 0); // removed: pthread sched config

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid + 4, &cpuset);
    printf("npu worker: bind to core %d\n", tid + 4);
    //GGML_ASSERT(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0);
#else
    usys_set_prio(0, 54);
#endif

    rknn_core_mask core_mask;
    if (tid == 0) core_mask = 0x1;
    if (tid == 1) core_mask = 0x2;
    if (tid == 2) core_mask = 0x4;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_worker.wait(lock, [tid] { return ready[tid] || done; });
        }
        BEGIN_MEASURE_0;

        if (done) break;

        for (auto task: npu_tasks[tid]) {
            task->apply_scale();
#ifdef GGML_USE_CHCORE
            task->submit();
#else
            task->submit(core_mask);
#endif
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            finished_nr++;
            ready[tid] = false;
            if (finished_nr == THREAD_NR) {
                cv_master.notify_one();
            }
        }
        END_MEASURE_0;
    }
}

std::vector<std::thread> npu_threads;

static void init_npu_thread(int thread_nr) {
    GGML_ASSERT(thread_nr <= 3);
    for (int i = 0; i < thread_nr; i++) {
        npu_threads.emplace_back(npu_worker, i);
    }
}

static inline rknn_tensor_type ggml_type_to_rknn_type(enum ggml_type type) {
    if (type == GGML_TYPE_F16) return RKNN_TENSOR_FLOAT32;
    if (type == GGML_TYPE_Q8_0) return RKNN_TENSOR_INT8;
    GGML_ASSERT(false);
}

static inline int8_t f32_to_i8(float x, float scale) {
    return (int8_t)std::min(std::max(x / scale, -127.f), 127.f);
}
static inline float i32_to_f32(int32_t x, float scale) {
    return (float)x * scale;
}

/*
void matmul_perf_test(int m, int k, int n) {
    rknn_tensor_type tensor_type = RKNN_TENSOR_INT8;
    auto kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    auto tasks = kernel->to_multi_npu(1);
    GGML_ASSERT(tasks.size() == 1 && tasks[0].size() == 1);

    std::queue<std::pair<int64_t, int64_t>> tss;
    int64_t sum = 0;

    while (1) {
        auto start = get_micro();

        tasks[0][0]->submit(1);
        
        auto end = get_micro();
        auto cur = end - start;
        tss.push({end, cur});
        sum += cur;
        while (!tss.empty()) {
            if (end - tss.front().first <= 10 * 1000000) break;
            sum -= tss.front().second;
            tss.pop();
        }
        printf("%s %d: (m, k, n) = (%d %d %d) cnt %ld average %ld us current %ld us\n", __func__, __LINE__, m, k, n, tss.size(), sum / tss.size(), cur);
    }
}
*/

extern "C" {

void rknpu2_matmul_begin_measure(int ith) {
    BEGIN_MEASURE_0;
}
void rknpu2_matmul_end_measure(int ith) {
    END_MEASURE_0;
}
void rknpu2_matmul_begin_measure_npu(int ith) {
    BEGIN_MEASURE_0;
}
void rknpu2_matmul_end_measure_npu(int ith) {
    END_MEASURE_0;
}

void rknpu2_matmul_pre0(struct ggml_tensor * dst, int nth, int ith) {
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne12 == 1 && ne13 == 1);
    GGML_ASSERT(ne2 == 1 && ne3 == 1);

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

// printf("zzh: ggml_rknpu2_mul_mat\n");
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    BEGIN_MEASURE_0;

    GGML_ASSERT(GGML_TYPE_F16 == type || GGML_TYPE_Q8_0 == type);

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(type);

    if (ith == 0) {
        auto kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, tensor_type);
        GGML_ASSERT(kernel);
        memset(dst->data, 0, m * n * sizeof(float));

        if (m == 1) {
            const uint64_t decode_step = g_decode_pre0_cnt.fetch_add(1) + 1;
            if (decode_step == 1 || decode_step % 64 == 0) {
                ggml_rknpu2_dump_weight_prepack_stats();
            }
        }

        // fprintf(stderr, "!!!ggml_rknpu2_matmul_pre0: (m, k, n) = (%lld, %lld, %lld) M = %d K = %d N = %d\n", m, k, n, kernel->M, kernel->K, kernel->N);
        // Build static weight prepack once for this tensor/layout and reuse across tokens.
        (void)ggml_rknpu2_get_weight_prepack(src0, k, n, kernel->K, kernel->N, tensor_type);

        float *A = (float*)src1->data;
        void *B = src0->data;

        kernel->for_all_inputs(
            [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
                if (tensor_type == RKNN_TENSOR_INT8) {
                    input_mem->init_scale();
                }
                input_mem->reset_cnt();
            }
        );
        kernel->for_all_weights(
            [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
                if (tensor_type == RKNN_TENSOR_INT8) {
                    weight_mem->init_scale();
                }
                weight_mem->reset_cnt();
            }
        );
        kernel->for_all_outputs(
            [&](int mm, int nn, int M, int N, std::shared_ptr<rknn_mem> output_mem) {
                output_mem->reset_cnt();
            }
        );
    }
    END_MEASURE_0;
}

void rknpu2_matmul_pre_scale(struct ggml_tensor * dst, int nth, int ith) {
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    // pre0 already built/prepared this cache; here we only reuse it.
    auto weight_prepack = ggml_rknpu2_find_weight_prepack_cache(src0, k, n, kernel->K, kernel->N, tensor_type);

    float *A = (float*)src1->data;
    void *B = src0->data;
    const float *fB = nullptr;
    std::unique_ptr<float[]> fB_storage;

    auto ensure_fB = [&]() -> const float * {
        if (fB != nullptr) {
            return fB;
        }
        GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
        const ggml_type_traits * traits = ggml_get_type_traits(src0->type);
        GGML_ASSERT(traits->to_float != NULL);
        const int nele = k * n;
        fB_storage.reset(new float[nele]);
        traits->to_float(B, fB_storage.get(), nele);
        fB = fB_storage.get();
        return fB;
    };

    kernel->for_all_inputs(
        [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
            if (tensor_type == RKNN_TENSOR_INT8) {
                float scale = RKNPU_PREPACK_SCALE_MIN;
                for (int i = input_mem->pre_scale_cnt.fetch_add(1); i < M; i = input_mem->pre_scale_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j++) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        scale = std::max(scale, std::abs(A[ii * k + jj]));
                    }
                }
                input_mem->commit_scale(scale / 127.f);
            }
        }
    );
    kernel->for_all_weights(
        [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
            if (tensor_type == RKNN_TENSOR_INT8) {
                const rknpu_weight_prepack_block * block = ggml_rknpu2_find_weight_prepack_block(weight_prepack, nn, kk);
                GGML_ASSERT(block && "cache missed at pre_scale, but it should have been built at pre0");

                if (ith == 0) {
                    g_weight_block_hit_prescale_cnt.fetch_add(1);
                }
                weight_mem->commit_scale(block->scale);

                // if (ith == 0) {
                //     g_weight_block_miss_prescale_cnt.fetch_add(1);
                // }
                // const float *fB_local = ensure_fB();
                // float scale = RKNPU_PREPACK_SCALE_MIN;
                // for (int i = weight_mem->pre_scale_cnt.fetch_add(1); i < N; i = weight_mem->pre_scale_cnt.fetch_add(1))
                //     for (int j = 0; j < K; j++) {
                //         int ii = nn + i;
                //         int jj = kk + j;
                //         if (ii >= n || jj >= k) continue;
                //         scale = std::max(scale, std::abs(fB_local[ii * k + jj]));
                //     }
                // weight_mem->commit_scale(scale / 127.f);
            }
        }
    );
    END_MEASURE_0;
}

std::atomic<bool> finish;

void rknpu2_matmul_pre1(struct ggml_tensor * dst, int nth, int ith) {
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];
// fprintf(stderr, "rknpu2_matmul_pre1\n");
    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    float *A = (float*)src1->data;

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    // pre0 already built/prepared this cache; here we only reuse it.
    auto weight_prepack = ggml_rknpu2_find_weight_prepack_cache(src0, k, n, kernel->K, kernel->N, tensor_type);

    kernel->for_all_inputs(
        [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
            auto input = input_mem->ptr;
            if (tensor_type == RKNN_TENSOR_FLOAT32) {
                GGML_ASSERT(k % 8 == 0);
                for (int i = input_mem->pre1_cnt.fetch_add(1); i < M; i = input_mem->pre1_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 8;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j += 8) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 8; t++) {
                            ((__fp16 *)input)[base_sub_mat + off_in_sub_mat + t] = A[ii * k + jj + t];
                        }
                    }
                }
            } else {
                GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
                GGML_ASSERT(k % 16 == 0);
                // fprintf(stderr, "set input->ptr[0] = %d\n", ((int32_t*)input)[0]);
                for (int i = input_mem->pre1_cnt.fetch_add(1); i < M; i = input_mem->pre1_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 16;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j += 16) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 16; t++) {
                            ((int8_t *)input)[base_sub_mat + off_in_sub_mat + t] = f32_to_i8(A[ii * k + jj + t], input_mem->scale);
                        }
                    }
                }
            }
        }
    );
    kernel->for_all_weights(
        [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
            // fprintf(stderr, "set weight->ptr[0] = %d\n", ((int32_t*)weight_mem->ptr)[0]);
            // fprintf(stderr, "n = %d, k = %d, pre1 weight: nn=%d kk=%d N=%d K=%d\n", n, k, nn, kk, N, K);
            const rknpu_weight_prepack_block * block = ggml_rknpu2_find_weight_prepack_block(weight_prepack, nn, kk);
            GGML_ASSERT(block && "cache missed at pre1, but it should have been built at pre0");
            if (ith == 0) {
                g_weight_block_hit_pre1_cnt.fetch_add(1);
            }
            if (ith == 0) {
                const uint64_t weight_dma = block->packed_dma;
                const uint32_t weight_domain_id = block->domain_id;
                if (tensor_type == RKNN_TENSOR_INT8) {
                    weight_mem->scale = block->scale;
                }

                for (const auto & task_group : kernel->npu_tasks) {
                    for (const auto & task : task_group->npu_tasks) {
                        if (task->nn == nn && task->kk == kk) {
                            task->set_weight_location(weight_dma, weight_domain_id,
                                                      block->payload,
                                                      block->payload_size);
                        }
                    }
                }
            }
        }
    );
    END_MEASURE_0;
    finish = false;
}

void rknpu2_matmul_submit(struct ggml_tensor * dst, int nth, int ith) {
    if (ith) {
        // while (!finish.load())
        //     ggml_thread_cpu_relax_out();
        return;
    }
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    for (auto task: kernel->npu_tasks) {
        task->apply_scale();
        task->submit();
    }
    finish = true;
    END_MEASURE_0;
}

void rknpu2_matmul_post(struct ggml_tensor * dst, int nth, int ith) {
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    float *C = (float*)dst->data;

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    BEGIN_MEASURE_0;
    kernel->for_all_outputs(
        [&](int mm, int nn, int M, int N, std::shared_ptr<rknn_mem> output_mem) {
            auto output = output_mem->ptr;
            
            // Ensure NPU computation results are visible to CPU
            // For FAKE_CACHE mode, sync DMA buffer from device to CPU
            // For NON_CACHEABLE memory (like rknpu-tests), no sync needed
#ifdef FAKE_CACHE
            if (output_mem->fd >= 0) {
                dma_sync_device_to_cpu(output_mem->fd);
            }
#else
            // For NON_CACHEABLE memory, NPU writes are directly visible to CPU
            // No cache sync needed, just like rknpu-tests/matmul_int8.c
#endif
            
            // Log NPU output result immediately after sync (closest to NPU completion)
            const uint64_t * output_bytes = (const uint64_t *)output;
            const size_t output_size = output_mem->size;
            const int num_8bytes = (output_size >= 5 * sizeof(uint64_t)) ? 5 : (int)(output_size / sizeof(uint64_t));
            // fprintf(stderr, "[NPU_POST] Output block mm=%d, nn=%d, M=%d, N=%d, first 5 8bytes: ", mm, nn, M, N);
            // for (int i = 0; i < num_8bytes; i++) {
            //     fprintf(stderr, "0x%016llx ", (unsigned long long)output_bytes[i]);
            // }
            // fprintf(stderr, "\n");
            
            if (tensor_type == RKNN_TENSOR_FLOAT32) {
                GGML_ASSERT(n % 4 == 0);
                for (int i = output_mem->post_cnt.fetch_add(1); i < M; i = output_mem->post_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 4;
                    if (ii >= m) break;
                    for (int j = 0; j < N; j += 4) {
                        int jj = nn + j;
                        if (jj >= n) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 4; t++) {
                            C[ii * n + jj + t] += ((float *)output)[base_sub_mat + off_in_sub_mat + t];
                        }
                    }
                }
            } else {
                GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
                GGML_ASSERT(n % 4 == 0);
                for (int i = output_mem->post_cnt.fetch_add(1); i < M; i = output_mem->post_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 4;
                    if (ii >= m) break;
                    for (int j = 0; j < N; j += 4) {
                        int jj = nn + j;
                        if (jj >= n) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 4; t++) {
                            C[ii * n + jj + t] += i32_to_f32(((int32_t *)output)[base_sub_mat + off_in_sub_mat + t], output_mem->scale);
                        }
                    }
                }
            }
        }
    );
    END_MEASURE_0;
}

static void ggml_backend_rknpu2_mul_mat(ggml_backend_rknpure_context * ctx, struct ggml_tensor * dst) {
    rknpu2_matmul_pre0(dst, 1, 0);
    rknpu2_matmul_pre1(dst, 1, 0);
    rknpu2_matmul_submit(dst, 1, 0);
    rknpu2_matmul_post(dst, 1, 0);
}

// backend interface

static const char * ggml_backend_rknpu2_name(ggml_backend_t backend) {
    return "RKNPURE";

    GGML_UNUSED(backend);
}

static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend->context;
    // delete ctx;
    // delete backend;


    // rknpu2_instance * instance = (rknpu2_instance *)g_rknpu2_mgr[ctx->device].instance;
    // if (instance != nullptr) {
    //     std::map<std::string,
    //              std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *,
    //                         Qnn_Tensor_t *>>::iterator graph_it;
    //     for (graph_it = instance->_qnn_graph_map.begin();
    //          graph_it != instance->_qnn_graph_map.end(); graph_it++) {
    //         auto & graph_item   = graph_it->second;
    //         Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
    //         GGML_UNUSED(graph_handle);
    //         QNN_LOG_INFO("graph type:%s", graph_it->first.c_str());
    //     }
    //     instance->_qnn_graph_map.clear();

    //     instance->qnn_finalize();
    //     delete instance;
    //     g_qnn_mgr[ctx->device].instance = nullptr;
    // }

    if (g_rknpu2_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_rknpu2_mgr[ctx->device].backend = nullptr;
    }
    matmul_kernels.clear();
    {
        std::lock_guard<std::mutex> lock(g_weight_prepack_mtx);
        g_weight_prepack_cache.clear();
    }
    ggml_rknpu2_dump_weight_prepack_stats();

    done = true;
    cv_worker.notify_all();
    for (auto &thread: npu_threads) {
        thread.join();
    }
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *) backend->context;
    return ggml_backend_rknpure_buffer_type(ctx->device);
}

static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_rknpu2_mul_mat(ctx, node);
                break;

            // case GGML_OP_OUT_PROD:
            //     // ggml_backend_rknpu2_out_prod(ctx, node); // we don't support currently
            //     break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            // case GGML_OP_PERMUTE:
            // case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

// Initialize g_rknpu2_mgr name fields
static void init_rknpu2_mgr_names() {
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < GGML_RKNPU2_MAX_DEVICES; i++) {
            snprintf(g_rknpu2_mgr[i].name, sizeof(g_rknpu2_mgr[i].name), "%s%d", GGML_RKNPU2_NAME, i);
            g_rknpu2_mgr[i].device = i;
        }
        initialized = true;
    }
}

 static bool ggml_backend_rknpu2_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(backend);
    return buft != nullptr && std::strcmp(ggml_backend_buft_name(buft), GGML_RKNPU2_NAME) == 0;
}

extern "C" {
void ggml_backend_rknpure_mul_mat_out(struct ggml_tensor * dst) {
    return ggml_backend_rknpu2_mul_mat(NULL, dst);
}
}

// Device context for backend registry
struct ggml_backend_rknpu2_device_context {
    int device;
    std::string name;
    std::string description;
};

// Device interface functions
static const char * ggml_backend_rknpu2_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rknpu2_device_context * ctx = (ggml_backend_rknpu2_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_rknpu2_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rknpu2_device_context * ctx = (ggml_backend_rknpu2_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_rknpu2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu2_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu2_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknpu2_device_get_name(dev);
    props->description = ggml_backend_rknpu2_device_get_description(dev);
    props->type        = ggml_backend_rknpu2_device_get_type(dev);
    ggml_backend_rknpu2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rknpu2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_rknpu2_device_context * ctx = (ggml_backend_rknpu2_device_context *)dev->context;
    return ggml_backend_rknpure_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rknpu2_device_context * ctx = (ggml_backend_rknpu2_device_context *)dev->context;
    return ggml_backend_rknpure_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return nullptr;
}

static bool ggml_backend_rknpu2_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // fprintf(stderr, "ggml_backend_rknpu2_device_supports_op, op->op=%d\n", op->op);
    return ggml_backend_rknpure_supports_op_out(op);
}

static bool ggml_backend_rknpu2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft != nullptr && std::strcmp(ggml_backend_buft_name(buft), GGML_RKNPU2_NAME) == 0;
}

static bool ggml_backend_rknpu2_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return op->op == GGML_OP_MUL_MAT;
}

static const ggml_backend_device_i ggml_backend_rknpu2_device_interface = {
    /* .get_name                = */ ggml_backend_rknpu2_device_get_name,
    /* .get_description         = */ ggml_backend_rknpu2_device_get_description,
    /* .get_memory              = */ ggml_backend_rknpu2_device_get_memory,
    /* .get_type                = */ ggml_backend_rknpu2_device_get_type,
    /* .get_props               = */ ggml_backend_rknpu2_device_get_props,
    /* .init_backend            = */ ggml_backend_rknpu2_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_rknpu2_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_rknpu2_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_rknpu2_device_supports_op,
    /* .supports_buft           = */ ggml_backend_rknpu2_device_supports_buft,
    /* .offload_op              = */ ggml_backend_rknpu2_device_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// Registry context
struct ggml_backend_rknpu2_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

// Registry interface functions
static const char * ggml_backend_rknpu2_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_RKNPU2_NAME;
}

static size_t ggml_backend_rknpu2_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rknpu2_reg_context * ctx = (ggml_backend_rknpu2_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_rknpu2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rknpu2_reg_context * ctx = (ggml_backend_rknpu2_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_rknpu2_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_rknpu2_reg_interface = {
    /* .get_name          = */ ggml_backend_rknpu2_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rknpu2_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rknpu2_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rknpu2_reg_get_proc_address,
};

// Backend registry function
ggml_backend_reg_t ggml_backend_rknpu2_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            init_rknpu2_mgr_names();
            ggml_backend_rknpu2_reg_context * ctx = new ggml_backend_rknpu2_reg_context;

            for (int i = 0; i < ggml_backend_rknpu2_get_device_count(); i++) {
                ggml_backend_rknpu2_device_context * dev_ctx = new ggml_backend_rknpu2_device_context;
                dev_ctx->device = i;
                dev_ctx->name = std::string(GGML_RKNPU2_NAME) + std::to_string(i);
                dev_ctx->description = "RKNPU2 Accelerator";

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_rknpu2_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_rknpu2_reg_interface,
                /* .context     = */ ctx
            };

            initialized = true;
        }
    }

    return &reg;
}

 static bool ggml_backend_rknpu2_offload_op(ggml_backend_t backend, const ggml_tensor *tensor) {
    // ggml_backend_rknpu2_context *ctx = (ggml_backend_rknpu2_context *)backend->context;
    // return ggml_rknpu2_compute_forward(ctx, nullptr, (ggml_tensor *)tensor);
    return tensor->op == GGML_OP_MUL_MAT;
}

static struct ggml_backend_i rknpu2_backend_i = {
    /* .get_name                = */ ggml_backend_rknpu2_name,
    /* .free                    = */ ggml_backend_rknpu2_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknpu2_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    static ggml_guid guid = { 0x3c, 0x29, 0xa9, 0xac, 0x54, 0x27, 0x40, 0x55, 0x89, 0xf2, 0xdc, 0x83, 0xf1, 0xba, 0x02, 0xc4 };
    return &guid;
}

// static ggml_backend_t ggml_backend_rknpu2_reg_init(const char *params, void * user_data) {
//     GGML_UNUSED(params);
//     ggml_backend_t rknpu2_backend = ggml_backend_rknpure_init((int) (unsigned long) user_data);
//     return rknpu2_backend;
// }

} // end extern "C" for rknpu2_matmul functions

// Backend API functions (C++ linkage, but declared as extern "C" in header)
ggml_backend_t ggml_backend_rknpure_init(int32_t device) {
    // g_rknpu2_mgr[0].interface 
    // ggml_backend_rknpu2_context * ctx = new ggml_backend_rknpu2_context;

    // Get the device from registry
    ggml_backend_reg_t reg = ggml_backend_rknpu2_reg();
    ggml_backend_dev_t dev = nullptr;
    if (reg && device < (int32_t)ggml_backend_rknpu2_reg_get_device_count(reg)) {
        dev = ggml_backend_rknpu2_reg_get_device(reg, device);
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rknpu2_guid(),
        /* .iface     = */ rknpu2_backend_i,
        /* .device    = */ dev,
        /* .context   = */ &g_rknpu2_mgr[device],
    };

    g_rknpu2_mgr[device].backend = backend;
    return backend;
}

bool ggml_backend_is_rknpu2(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rknpu2_guid());
}

void ggml_backend_rknpu2_set_n_threads(ggml_backend_t backend_rknpu2, int n_threads) {
    GGML_ASSERT(ggml_backend_is_rknpu2(backend_rknpu2));

    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend_rknpu2->context;
    ctx->n_threads = n_threads;
}

int ggml_backend_rknpure_reg_devices() {
    // ggml_rknpu2_instance_init();
    init_rknpu2_mgr_names();
    ggml_backend_reg_t reg = ggml_backend_rknpu2_reg();
    if (reg) {
        ggml_backend_register(reg);
        return ggml_backend_rknpu2_reg_get_device_count(reg);
    }
    return 0;
}

// Static member definitions (must be outside extern "C" block)
int measure_stack::last_lines[LINE_NR];
std::atomic<int64_t> measure_stack::micros[LINE_NR];
