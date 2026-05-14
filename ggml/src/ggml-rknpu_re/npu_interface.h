#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#include "rknpu-ioctl.h"

#ifdef __cplusplus
extern "C" {
#endif

struct npu_layout_info {
    struct rknpu_layout_info rknpu;
    uint64_t entry_size;
    uint64_t payload_reload_offset;
    uint64_t payload_reload_bytes;
    uint32_t payload_entry_count;
    uint32_t compute_entry_count;
    uint32_t payload_reload_start_entry;
    uint32_t payload_reload_entry_count;
};

void* mem_allocate_payload(size_t size, uint64_t *dma_addr, uint64_t *obj,
                           uint32_t flags, uint64_t *handle,
                           uint32_t *domain_id);
void* mem_allocate_compute(size_t size, uint64_t *dma_addr, uint64_t *obj,
                           uint32_t flags, uint64_t *handle);
void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle);
void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr);
int mem_pool_prepare(size_t pool_size);
void* mem_pool_vaddr(void);
int mem_payload_mapped_slice(const void *addr, size_t size,
                             size_t *slice_offset, size_t *slice_size);
uint64_t mem_pool_finished_payload_offset(void);
int mem_pool_finish_payload_until(uint64_t payload_end);
int mem_pool_finish_all_payload(void);

int npu_layout_info_init(void);
int npu_get_layout_info(struct npu_layout_info *info);
void npu_dump_layout_info(const char *source);

int npu_reset(void);
int npu_submit(uint64_t regcfg_obj_addr, uint32_t core_mask, uint32_t domain_id);
int npu_submit_multi(uint64_t regcfg_obj_addr[], int task_num,
                     uint32_t domain_id, void *polling);

#ifdef __cplusplus
}
#endif
#endif // NPU_INTERFACE_H
