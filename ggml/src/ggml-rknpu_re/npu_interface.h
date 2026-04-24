#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle);
void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr);
int mem_pool_prepare(size_t pool_size);

int npu_reset(void);
int npu_submit(uint64_t regcfg_obj_addr, uint32_t core_mask);
int npu_submit_multi(uint64_t regcfg_obj_addr[], int task_num, void *polling);

#ifdef __cplusplus
}
#endif
#endif // NPU_INTERFACE_H
