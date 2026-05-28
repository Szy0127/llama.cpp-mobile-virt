#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle, int use_cache);
void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr);

int npu_reset(void);
int npu_submit(uint64_t task_obj_addr, uint32_t core_mask);

#ifdef __cplusplus
}
#endif
#endif // NPU_INTERFACE_H
