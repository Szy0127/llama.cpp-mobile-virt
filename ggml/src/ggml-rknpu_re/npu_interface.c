/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <limits.h>
#include <assert.h>

#include "rknpu-ioctl.h"
#include "npu_hw.h"
#include "npu_interface.h"
#include "llm_ioctl.h"

#include <pthread.h>

#define NPU_DEVICE "/dev/dri/card0"
#define LLM_DEVICE "/dev/llm"
#define PAGE_SIZE_BYTES 0x1000UL

struct npu_prealloc_layout {
  uint64_t gpa_base;
  uint64_t donate_size;
  uint64_t iova_window_bytes;
  uint64_t compute_buffer_bytes;
  uint64_t reserve_size;
  uint64_t payload_window_bytes;
  uint64_t payload_total_bytes;
  uint64_t compute_gpa_offset;
  uint64_t entry_size;
  uint64_t payload_reload_offset;
  uint64_t payload_reload_bytes;
  uint32_t domain_count;
  uint32_t payload_entry_count;
  uint32_t compute_entry_count;
  uint32_t payload_reload_start_entry;
  uint32_t payload_reload_entry_count;
};

static int npu_fd = -1;
static int llm_fd = -1;
static int llm_window_began = 0;
static void *pool_vaddr = NULL;
static size_t pool_map_size = 0;
static size_t payload_used = 0;
static size_t compute_used = 0;
static struct npu_prealloc_layout g_prealloc_layout;
static pthread_once_t npu_fd_once = PTHREAD_ONCE_INIT;
static struct npu_layout_info g_npu_layout_info;
static int g_npu_layout_info_initialized = 0;

int npu_open(void);
static void npu_fd_init(void);

static void log_npu_layout_info(const char *source,
                                const struct npu_layout_info *info) {
  printf("RKNPU layout (%s): gpa_base=0x%llx donate=0x%llx compute=0x%llx iova_window=0x%llx reserve=0x%llx payload_window=0x%llx compute_gpa=0x%llx domains=%u version=%u reload=0x%llx+0x%llx entries=%u+%u entry_size=0x%llx\n",
         source ? source : "unknown",
         (unsigned long long) info->rknpu.gpa_base,
         (unsigned long long) info->rknpu.donate_size,
         (unsigned long long) info->rknpu.compute_buffer_size,
         (unsigned long long) info->rknpu.iova_window_size,
         (unsigned long long) info->rknpu.reserve_size,
         (unsigned long long) info->rknpu.payload_window_size,
         (unsigned long long) info->rknpu.compute_buffer_gpa,
         info->rknpu.domain_count,
         info->rknpu.layout_version,
         (unsigned long long) info->payload_reload_offset,
         (unsigned long long) info->payload_reload_bytes,
         info->payload_reload_start_entry,
         info->payload_reload_entry_count,
         (unsigned long long) info->entry_size);
}

static int npu_layout_info_is_valid(const struct npu_layout_info *info) {
  return info->rknpu.gpa_base != 0 &&
         info->rknpu.donate_size != 0 &&
         info->rknpu.compute_buffer_size != 0 &&
         info->rknpu.iova_window_size != 0 &&
         info->rknpu.domain_count != 0;
}

static int set_npu_layout_info(const struct npu_layout_info *info,
                               const char *source) {
  if (!npu_layout_info_is_valid(info)) {
    printf("Invalid RKNPU layout: gpa=0x%llx donate=0x%llx compute=0x%llx iova=0x%llx domains=%u\n",
           (unsigned long long) info->rknpu.gpa_base,
           (unsigned long long) info->rknpu.donate_size,
           (unsigned long long) info->rknpu.compute_buffer_size,
           (unsigned long long) info->rknpu.iova_window_size,
           info->rknpu.domain_count);
    errno = EINVAL;
    return -1;
  }

  if (g_npu_layout_info_initialized) {
    return 0;
  }
  g_npu_layout_info = *info;
  g_npu_layout_info_initialized = 1;
  log_npu_layout_info(source, &g_npu_layout_info);
  return 0;
}

static void npu_layout_info_from_llm(const struct llm_layout_info *llm,
                                     struct npu_layout_info *info) {
  memset(info, 0, sizeof(*info));
  info->rknpu.gpa_base = llm->gpa_base;
  info->rknpu.donate_size = llm->donate_size;
  info->rknpu.compute_buffer_size = llm->compute_buffer_size;
  info->rknpu.iova_window_size = llm->iova_window_size;
  info->rknpu.reserve_size = llm->reserve_size;
  info->rknpu.payload_window_size = llm->payload_window_size;
  info->rknpu.compute_buffer_gpa = llm->compute_buffer_gpa;
  info->rknpu.domain_count = llm->domain_count;
  info->rknpu.layout_version = llm->layout_version;
  info->entry_size = llm->entry_size;
  info->payload_reload_offset = llm->reload_offset;
  info->payload_reload_bytes = llm->reload_length;
  info->payload_entry_count = llm->payload_entry_count;
  info->compute_entry_count = llm->compute_entry_count;
  info->payload_reload_start_entry = llm->reload_start_entry;
  info->payload_reload_entry_count = llm->reload_entry_count;
}

static int set_npu_layout_info_from_llm(const struct llm_layout_info *llm,
                                        const char *source) {
  struct npu_layout_info info;
  npu_layout_info_from_llm(llm, &info);
  return set_npu_layout_info(&info, source);
}

int npu_layout_info_init(void) {
  int fd;
  struct llm_layout_info llm_info;

  if (g_npu_layout_info_initialized) {
    return 0;
  }

  fd = open(LLM_DEVICE, O_RDWR);
  if (fd < 0) {
    printf("Failed to open %s for layout errno=%d\n", LLM_DEVICE, errno);
    return -1;
  }

  memset(&llm_info, 0, sizeof(llm_info));
  if (ioctl(fd, LLM_IOC_GET_LAYOUT, &llm_info) < 0) {
    int saved_errno = errno;
    printf("LLM_IOC_GET_LAYOUT early failed errno=%d\n", saved_errno);
    close(fd);
    errno = saved_errno;
    return -1;
  }
  close(fd);

  return set_npu_layout_info_from_llm(&llm_info, "init");
}

int npu_get_layout_info(struct npu_layout_info *info) {
  if (info == NULL) {
    errno = EINVAL;
    return -1;
  }

  if (npu_layout_info_init() != 0) {
    return -1;
  }

  *info = g_npu_layout_info;
  return 0;
}

void npu_dump_layout_info(const char *source) {
  struct npu_layout_info info;
  if (npu_get_layout_info(&info) != 0) {
    printf("RKNPU layout (%s): unavailable errno=%d\n",
           source ? source : "unknown", errno);
    return;
  }
  log_npu_layout_info(source, &info);
}

static int finish_llm_entry(unsigned int entry_index) {
  struct llm_extend_info ext;
  memset(&ext, 0, sizeof(ext));
  ext.flags = LLM_EXTEND_FLAG_FINISH;
  ext.entry_index = entry_index;

  if (ioctl(llm_fd, LLM_IOC_EXTEND, &ext) < 0) {
    printf("LLM_IOC_EXTEND finish failed entry=%u errno=%d\n",
           entry_index, errno);
    return -1;
  }

  return 0;
}

static int finish_llm_window(void) {
  if (ioctl(llm_fd, LLM_IOC_FINISH) < 0) {
    printf("LLM_IOC_FINISH failed errno=%d\n", errno);
    return -1;
  }

  for (unsigned int i = 0; i < g_prealloc_layout.payload_reload_entry_count;
       i++) {
    unsigned int entry_index =
        g_prealloc_layout.payload_reload_start_entry + i;

    if (finish_llm_entry(entry_index) != 0) {
      return -1;
    }
  }

  printf("LLM_IOC_EXTEND complete %u entries\n",
         g_prealloc_layout.payload_reload_entry_count);
  return 0;
}

static void cleanup_pool_mapping(void) {
  if (pool_vaddr && pool_map_size) {
    munmap(pool_vaddr, pool_map_size);
  }

  if (llm_fd >= 0 && llm_window_began) {
    finish_llm_window();
  }

  pool_vaddr = NULL;
  pool_map_size = 0;
  payload_used = 0;
  compute_used = 0;
  llm_window_began = 0;

  if (llm_fd >= 0) {
    close(llm_fd);
    llm_fd = -1;
  }

  memset(&g_prealloc_layout, 0, sizeof(g_prealloc_layout));
}

static int validate_prealloc_layout(struct npu_prealloc_layout *layout,
                                    const char *source) {
  uint64_t domain_count;

  if (layout->gpa_base == 0 || layout->donate_size == 0 ||
      layout->iova_window_bytes == 0 || layout->compute_buffer_bytes == 0) {
    printf("Invalid %s layout: zero field gpa=0x%llx donate=0x%llx iova=0x%llx compute=0x%llx\n",
           source,
           (unsigned long long) layout->gpa_base,
           (unsigned long long) layout->donate_size,
           (unsigned long long) layout->iova_window_bytes,
           (unsigned long long) layout->compute_buffer_bytes);
    return -1;
  }

  if ((layout->gpa_base & (PAGE_SIZE_BYTES - 1)) != 0 ||
      (layout->donate_size & (PAGE_SIZE_BYTES - 1)) != 0 ||
      (layout->iova_window_bytes & (PAGE_SIZE_BYTES - 1)) != 0 ||
      (layout->compute_buffer_bytes & (PAGE_SIZE_BYTES - 1)) != 0) {
    printf("Invalid %s layout: values must be page aligned\n", source);
    return -1;
  }

  if (layout->donate_size <= layout->compute_buffer_bytes) {
    printf("Invalid %s layout: donate size 0x%llx must be larger than compute buffer 0x%llx\n",
           source,
           (unsigned long long) layout->donate_size,
           (unsigned long long) layout->compute_buffer_bytes);
    return -1;
  }

  if (layout->compute_buffer_bytes >= layout->iova_window_bytes) {
    printf("Invalid %s layout: compute buffer 0x%llx must be smaller than iova window 0x%llx\n",
           source,
           (unsigned long long) layout->compute_buffer_bytes,
           (unsigned long long) layout->iova_window_bytes);
    return -1;
  }

  if (UINT64_MAX - layout->gpa_base < layout->donate_size) {
    printf("Invalid %s layout: compute gpa overflow\n", source);
    return -1;
  }

  layout->payload_window_bytes =
      layout->iova_window_bytes - layout->compute_buffer_bytes;
  layout->reserve_size = layout->donate_size;
  layout->payload_total_bytes =
      layout->donate_size - layout->compute_buffer_bytes;
  layout->compute_gpa_offset =
      layout->reserve_size - layout->compute_buffer_bytes;

  domain_count =
      (layout->payload_total_bytes + layout->payload_window_bytes - 1) /
      layout->payload_window_bytes;
  if (domain_count == 0 || domain_count > UINT32_MAX) {
    printf("Invalid %s layout: domain count overflow (%llu)\n",
           source, (unsigned long long) domain_count);
    return -1;
  }

  layout->domain_count = (uint32_t) domain_count;
  return 0;
}

static void log_prealloc_layout(const char *source,
                                const struct npu_prealloc_layout *layout) {
  printf("RKNPU layout (%s): gpa_base=0x%llx donate=0x%llx compute=0x%llx iova_window=0x%llx reserve=0x%llx payload_window=0x%llx reload=0x%llx+0x%llx entries=%u+%u domains=%u\n",
         source,
         (unsigned long long) layout->gpa_base,
         (unsigned long long) layout->donate_size,
         (unsigned long long) layout->compute_buffer_bytes,
         (unsigned long long) layout->iova_window_bytes,
         (unsigned long long) layout->reserve_size,
         (unsigned long long) layout->payload_window_bytes,
         (unsigned long long) layout->payload_reload_offset,
         (unsigned long long) layout->payload_reload_bytes,
         layout->payload_reload_start_entry,
         layout->payload_reload_entry_count,
         layout->domain_count);
}

static int load_prealloc_layout_from_info(struct npu_prealloc_layout *layout) {
  struct npu_layout_info info;

  if (npu_get_layout_info(&info) != 0) {
    printf("Failed to get initialized RKNPU layout errno=%d\n", errno);
    return -1;
  }
  memset(layout, 0, sizeof(*layout));
  layout->gpa_base = info.rknpu.gpa_base;
  layout->donate_size = info.rknpu.donate_size;
  layout->iova_window_bytes = info.rknpu.iova_window_size;
  layout->compute_buffer_bytes = info.rknpu.compute_buffer_size;

  if (validate_prealloc_layout(layout, "llm") != 0) {
    return -1;
  }

  layout->payload_entry_count = info.payload_entry_count;
  layout->compute_entry_count = info.compute_entry_count;
  layout->payload_reload_offset = info.payload_reload_offset;
  layout->payload_reload_bytes = info.payload_reload_bytes;
  layout->entry_size = info.entry_size;
  layout->payload_reload_start_entry = info.payload_reload_start_entry;
  layout->payload_reload_entry_count = info.payload_reload_entry_count;

  if (layout->payload_reload_start_entry > layout->payload_entry_count ||
      layout->payload_reload_entry_count >
          layout->payload_entry_count - layout->payload_reload_start_entry) {
    printf("Invalid LLM reload entries: start=%u count=%u payload=%u\n",
           layout->payload_reload_start_entry,
           layout->payload_reload_entry_count,
           layout->payload_entry_count);
    return -1;
  }
  if (layout->payload_reload_offset !=
          (uint64_t) layout->payload_reload_start_entry * info.entry_size ||
      layout->payload_reload_bytes !=
          (uint64_t) layout->payload_reload_entry_count * info.entry_size) {
    printf("Invalid LLM reload range: off=0x%llx len=0x%llx entry_size=0x%llx start=%u count=%u\n",
           (unsigned long long) layout->payload_reload_offset,
           (unsigned long long) layout->payload_reload_bytes,
           (unsigned long long) info.entry_size,
           layout->payload_reload_start_entry,
           layout->payload_reload_entry_count);
    return -1;
  }
  //log_prealloc_layout("llm", layout);
  return 0;
}

void* mem_pool_vaddr(void) {
  assert(pool_vaddr);
  return pool_vaddr;
}

int mem_payload_mapped_slice(const void *addr, size_t size,
                             size_t *slice_offset, size_t *slice_size) {
  if (slice_offset) {
    *slice_offset = 0;
  }
  if (slice_size) {
    *slice_size = 0;
  }

  if (addr == NULL || size == 0 || pool_vaddr == NULL || pool_map_size == 0) {
    return 0;
  }

  uintptr_t base = (uintptr_t) pool_vaddr;
  uintptr_t start = (uintptr_t) addr;
  if (start < base) {
    return -1;
  }

  uint64_t req_start = (uint64_t) (start - base);
  if (req_start > pool_map_size || size > pool_map_size - req_start) {
    return -1;
  }

  uint64_t req_end = req_start + (uint64_t) size;
  uint64_t reload_start = g_prealloc_layout.payload_reload_offset;
  uint64_t reload_end = reload_start + g_prealloc_layout.payload_reload_bytes;
  if (reload_end > pool_map_size || reload_end < reload_start) {
    reload_end = pool_map_size;
  }

  uint64_t copy_start = req_start > reload_start ? req_start : reload_start;
  uint64_t copy_end = req_end < reload_end ? req_end : reload_end;
  if (copy_start >= copy_end) {
    return 0;
  }

  if (slice_offset) {
    *slice_offset = (size_t) (copy_start - req_start);
  }
  if (slice_size) {
    *slice_size = (size_t) (copy_end - copy_start);
  }
  return 1;
}

static void npu_fd_init(void) {
  npu_fd = npu_open();
  printf("%s %d: npu_fd %d\n", __func__, __LINE__, npu_fd);
}

static int extend_payload_entry(unsigned int entry_index) {
    struct llm_extend_info ext;
    memset(&ext, 0, sizeof(ext));
    ext.entry_index = entry_index;

    if (ioctl(llm_fd, LLM_IOC_EXTEND, &ext) < 0) {
      printf("LLM_IOC_EXTEND failed entry=%u errno=%d\n",
             entry_index, errno);
      return -1;
    }

    if (!(ext.flags & LLM_EXTEND_FLAG_SKIPPED)) {
      uint64_t expected_offset =
          (uint64_t) entry_index * g_prealloc_layout.entry_size;

      if (g_prealloc_layout.entry_size == 0 ||
          ext.length == 0 || ext.offset != expected_offset ||
          ext.offset + ext.length > pool_map_size) {
        printf("LLM_IOC_EXTEND invalid mapping entry=%u off=0x%llx len=0x%llx expected=0x%llx map=%zu flags=0x%x\n",
               entry_index,
               (unsigned long long) ext.offset,
               (unsigned long long) ext.length,
               (unsigned long long) expected_offset,
               pool_map_size, ext.flags);
        return -1;
      }
    }

  return 0;
}

static int ensure_payload_mapped(void) {
  for (unsigned int i = 0; i < g_prealloc_layout.payload_reload_entry_count;
       i++) {
    unsigned int entry_index =
        g_prealloc_layout.payload_reload_start_entry + i;

    if (extend_payload_entry( entry_index) != 0) {
      return -1;
    }
  }

  return 0;
}

int mem_pool_prepare(size_t pool_size) {
  if (pool_size == 0) {
    printf("Invalid pool size 0\n");
    return -1;
  }

  if (pool_vaddr != NULL) {
    if (pool_size > g_prealloc_layout.payload_total_bytes) {
      printf("Requested pool size %zu exceeds payload capacity %llu\n",
             pool_size,
             (unsigned long long) g_prealloc_layout.payload_total_bytes);
      return -1;
    }
    return 0;
  }

  llm_fd = open(LLM_DEVICE, O_RDWR);
  if (llm_fd < 0) {
    printf("Failed to open %s errno=%d\n", LLM_DEVICE, errno);
    return -1;
  }

  int ret = ioctl(llm_fd, LLM_IOC_BEGIN);
  if (ret < 0) {
    printf("LLM_IOC_BEGIN failed for pool size=%zu ret=%d errno=%d\n",
           pool_size, ret, errno);
    close(llm_fd);
    llm_fd = -1;
    return -1;
  }
  llm_window_began = 1;

  if (load_prealloc_layout_from_info(&g_prealloc_layout) != 0) {
    cleanup_pool_mapping();
    return -1;
  }

  if (pool_size > g_prealloc_layout.payload_total_bytes) {
    printf("Requested pool size %zu exceeds payload capacity %llu\n",
           pool_size,
           (unsigned long long) g_prealloc_layout.payload_total_bytes);
    cleanup_pool_mapping();
    return -1;
  }

  const size_t map_size = g_prealloc_layout.payload_total_bytes;
  void *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, llm_fd, 0);
  if (map == MAP_FAILED) {
    printf("Failed to mmap %s size=%zu errno=%d\n",
           LLM_DEVICE, map_size, errno);
    cleanup_pool_mapping();
    return -1;
  }

  struct llm_map_info info;
  memset(&info, 0, sizeof(info));
  ret = ioctl(llm_fd, LLM_IOC_GET_INFO, &info);
  if (ret < 0) {
    printf("LLM_IOC_GET_INFO failed ret=%d errno=%d\n", ret, errno);
    munmap(map, map_size);
    cleanup_pool_mapping();
    return -1;
  }

  if (!info.mapped || info.user_vaddr != (uint64_t) (uintptr_t) map ||
      info.length < map_size || info.reload_offset != g_prealloc_layout.payload_reload_offset ||
      info.reload_length != g_prealloc_layout.payload_reload_bytes) {
    printf("LLM mapping metadata mismatch: mapped=%u user_vaddr=0x%llx length=%llu reload=0x%llx+0x%llx expected_vaddr=0x%llx expected_len=%zu expected_reload=0x%llx+0x%llx\n",
           info.mapped,
           (unsigned long long) info.user_vaddr,
           (unsigned long long) info.length,
           (unsigned long long) info.reload_offset,
           (unsigned long long) info.reload_length,
           (unsigned long long) (uint64_t) (uintptr_t) map,
           map_size,
           (unsigned long long) g_prealloc_layout.payload_reload_offset,
           (unsigned long long) g_prealloc_layout.payload_reload_bytes);
    munmap(map, map_size);
    cleanup_pool_mapping();
    return -1;
  }

  pool_vaddr = map;
  pool_map_size = map_size;

  if (ensure_payload_mapped() != 0) {
    cleanup_pool_mapping();
    return -1;
  }

  memset(&info, 0, sizeof(info));
  ret = ioctl(llm_fd, LLM_IOC_GET_INFO, &info);
  if (ret < 0) {
    printf("LLM_IOC_GET_INFO after extend failed ret=%d errno=%d\n", ret, errno);
    cleanup_pool_mapping();
    return -1;
  }
  if (info.reload_start_entry != g_prealloc_layout.payload_reload_start_entry ||
      info.reload_entry_count != g_prealloc_layout.payload_reload_entry_count) {
    printf("LLM reload range mismatch after extend: info=%u+%u expected=%u+%u\n",
           info.reload_start_entry,
           info.reload_entry_count,
           g_prealloc_layout.payload_reload_start_entry,
           g_prealloc_layout.payload_reload_entry_count);
    cleanup_pool_mapping();
    return -1;
  }

  payload_used = 0;
  compute_used = 0;
  atexit(cleanup_pool_mapping);
  printf("Mapped LLM pool: gpa_base=0x%llx reserve=0x%llx payload_window=0x%llx compute=0x%llx vaddr=%p\n",
         (unsigned long long) g_prealloc_layout.gpa_base,
         (unsigned long long) g_prealloc_layout.reserve_size,
         (unsigned long long) g_prealloc_layout.payload_window_bytes,
         (unsigned long long) g_prealloc_layout.compute_buffer_bytes,
         map);
  return 0;
}


static void *mem_allocate_internal(size_t size, uint64_t *dma_addr, uint64_t *obj,
                                   uint32_t flags, uint64_t *handle,
                                   uint32_t *domain_id, int use_compute_pool) {
  (void) flags;

  if (size == 0) {
    return NULL;
  }

  if (pool_vaddr == NULL || pool_map_size == 0) {
    printf("mem_allocate called before mem_pool_prepare\n");
    return NULL;
  }

  const size_t alloc_size = size;
  uint64_t phys_addr;
  uint64_t iova;
  uint32_t alloc_domain_id = UINT32_MAX;
  void *map;

  if (use_compute_pool) {
    if (alloc_size > g_prealloc_layout.compute_buffer_bytes - compute_used) {
      printf("Out of compute buffer: need=%zu used=%zu total=%llu\n",
             alloc_size, compute_used,
             (unsigned long long) g_prealloc_layout.compute_buffer_bytes);
      return NULL;
    }

    phys_addr = g_prealloc_layout.gpa_base +
                g_prealloc_layout.compute_gpa_offset + (uint64_t) compute_used;
    iova = g_prealloc_layout.payload_window_bytes + (uint64_t) compute_used;
    map = (char *) pool_vaddr + g_prealloc_layout.compute_gpa_offset + compute_used;
    compute_used += alloc_size;
    if (domain_id) {
      *domain_id = UINT32_MAX;
    }
  } else {
    uint64_t local_payload_used = payload_used;

    if (alloc_size > g_prealloc_layout.payload_window_bytes) {
      printf("Allocation exceeds payload window: need=%zu window=%llu\n",
             alloc_size,
             (unsigned long long) g_prealloc_layout.payload_window_bytes);
      return NULL;
    }

    if ((local_payload_used % g_prealloc_layout.payload_window_bytes) + alloc_size >
        g_prealloc_layout.payload_window_bytes) {
      local_payload_used =
          ((local_payload_used / g_prealloc_layout.payload_window_bytes) + 1) *
          g_prealloc_layout.payload_window_bytes;
    }

    if (alloc_size > g_prealloc_layout.payload_total_bytes - local_payload_used) {
      printf("Out of payload window: need=%zu used=%llu total=%llu reload=0x%llx+0x%llx\n",
             alloc_size,
             (unsigned long long) local_payload_used,
             (unsigned long long) g_prealloc_layout.payload_total_bytes,
             (unsigned long long) g_prealloc_layout.payload_reload_offset,
             (unsigned long long) g_prealloc_layout.payload_reload_bytes);
      return NULL;
    }

    phys_addr = g_prealloc_layout.gpa_base + local_payload_used;
    iova = local_payload_used % g_prealloc_layout.payload_window_bytes;
    map = (char *) pool_vaddr + local_payload_used;
    alloc_domain_id = (uint32_t)
        (local_payload_used / g_prealloc_layout.payload_window_bytes);
    if (domain_id) {
      *domain_id = alloc_domain_id;
    }
    payload_used = local_payload_used + alloc_size;
  }

  if (dma_addr) {
    *dma_addr = iova;
  }
  if (obj) {
    *obj = phys_addr;
  }
  if (handle) {
    *handle = 0;
  }

  return map;
}

void* mem_allocate_payload(size_t size, uint64_t *dma_addr, uint64_t *obj,
                           uint32_t flags, uint64_t *handle,
                           uint32_t *domain_id) {
  return mem_allocate_internal(size, dma_addr, obj, flags, handle, domain_id, 0);
}

void* mem_allocate_compute(size_t size, uint64_t *dma_addr, uint64_t *obj,
                           uint32_t flags, uint64_t *handle) {
  return mem_allocate_internal(size, dma_addr, obj, flags, handle, NULL, 1);
}

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj,
                   uint32_t flags, uint64_t *handle) {
  return mem_allocate_payload(size, dma_addr, obj, flags, handle, NULL);
}

void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr) {
  (void) addr;
  (void) len;
  (void) handle;
  (void) obj_addr;
  // Allocations are slices of the process-wide /dev/llm mapping, so there is
  // no per-allocation munmap here.
}

int npu_open(void) {
  char buf1[256], buf2[256], buf3[256];

  memset(buf1, 0 ,sizeof(buf1));
  memset(buf2, 0 ,sizeof(buf2));
  memset(buf3, 0, sizeof(buf3));

  int local_fd = open(NPU_DEVICE, O_RDWR);
  if (local_fd < 0) {
    printf("Failed to open %s %d\n", NPU_DEVICE, errno);
    return local_fd;
  }

  struct drm_version dv;
  memset(&dv, 0, sizeof(dv));
  dv.name = buf1;
  dv.name_len = sizeof(buf1);
  dv.date = buf2;
  dv.date_len = sizeof(buf2);
  dv.desc = buf3;
  dv.desc_len = sizeof(buf3);

  int ret = ioctl(local_fd, DRM_IOCTL_VERSION, &dv);
  if (ret < 0) {
    printf("DRM_IOCTL_VERISON failed %d\n", ret);
    close(local_fd);
    return ret;
  }

  printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
  return local_fd;
}

int npu_reset(void) {
  pthread_once(&npu_fd_once, npu_fd_init);

  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
    0,
  };
  return ioctl(npu_fd, DRM_IOCTL_RKNPU_ACTION, &act);
}

int npu_submit(uint64_t regcfg_obj_addr, uint32_t core_mask, uint32_t domain_id) {
  pthread_once(&npu_fd_once, npu_fd_init);

  struct rknpu_subcore_task subcore_tasks[5] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
  struct rknpu_submit_extend submit_ext;
  for (int i = 0; i < 5; ++i) {
    if (core_mask & (1u << i)) {
      subcore_tasks[i].task_start = 0;
      subcore_tasks[i].task_number = 1;
    }
  }

  memset(&submit_ext, 0, sizeof(submit_ext));
  submit_ext.submit = (struct rknpu_submit) {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = 0,
    .regcfg_obj_addr = regcfg_obj_addr,
    .iommu_domain_id = domain_id,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = core_mask,
    .fence_fd = -1,
    .subcore_task = {
      subcore_tasks[0],
      subcore_tasks[1],
      subcore_tasks[2],
      subcore_tasks[3],
      subcore_tasks[4]
    },
  };
  return ioctl(npu_fd, DRM_IOCTL_RKNPU_SUBMIT, &submit_ext.submit);
}

int npu_submit_multi(uint64_t regcfg_obj_addr[], int task_num,
                     uint32_t domain_id, void *polling) {
  struct rknpu_submit_extend submit_ext;

  (void) polling;

  pthread_once(&npu_fd_once, npu_fd_init);
  if (task_num <= 0 || task_num > RKNPU_MAX_MULTI_CORE_TASKS) {
    errno = EINVAL;
    return -1;
  }

  memset(&submit_ext, 0, sizeof(submit_ext));
  submit_ext.submit.flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG;
  submit_ext.submit.timeout = 6000;
  submit_ext.submit.task_start = 0;
  submit_ext.submit.task_number = 1;
  submit_ext.submit.task_counter = 0;
  submit_ext.submit.priority = 0;
  submit_ext.submit.task_obj_addr = 0;
  submit_ext.submit.regcfg_obj_addr = 0;
  submit_ext.submit.iommu_domain_id = domain_id;
  submit_ext.submit.task_base_addr = 0;
  submit_ext.submit.user_data = 0;
  submit_ext.submit.core_mask = 0;
  submit_ext.submit.fence_fd = -1;
  submit_ext.submit.submit_task_num = (__u32) task_num;

  for (int core_index = 0; core_index < task_num; ++core_index) {
    submit_ext.tasks[core_index].regcfg_obj_addr = regcfg_obj_addr[core_index];
    submit_ext.tasks[core_index].task_base_addr = 0;
    submit_ext.tasks[core_index].regcfg_amount = 104;
    submit_ext.tasks[core_index].core_mask = (1u << core_index);
    submit_ext.submit.core_mask |= submit_ext.tasks[core_index].core_mask;
    submit_ext.submit.subcore_task[core_index].task_start = 0;
    submit_ext.submit.subcore_task[core_index].task_number = 1;
  }

  return ioctl(npu_fd, DRM_IOCTL_RKNPU_SUBMIT_EXT, &submit_ext);
}
