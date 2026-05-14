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
#include <time.h>

#include "rknpu-ioctl.h"
#include "npu_hw.h"
#include "npu_interface.h"

#include <pthread.h>

#define NPU_DEVICE "/dev/dri/card0"
#define PAGE_SIZE_BYTES                0x1000UL

struct npu_prealloc_layout {
  uint64_t gpa_base;
  uint64_t donate_size;
  uint64_t iova_window_bytes;
  uint64_t compute_buffer_bytes;
  uint64_t reserve_size;
  uint64_t payload_window_bytes;
  uint64_t payload_total_bytes;
  uint64_t compute_gpa_offset;
  uint32_t domain_count;
};

static int fd = -1;
static int payload_mem_fd = -1;
static int compute_mem_fd = -1;
static void *payload_vaddr = NULL;
static void *compute_vaddr = NULL;
static size_t payload_map_size = 0;
static size_t compute_map_size = 0;
static size_t payload_used = 0;
static size_t compute_used = 0;
static struct npu_prealloc_layout g_prealloc_layout;
static int g_prealloc_layout_ready = 0;
static pthread_once_t fd_once = PTHREAD_ONCE_INIT;
static pthread_once_t layout_once = PTHREAD_ONCE_INIT;

int npu_open(void);

static void log_prealloc_usage(const char *kind, size_t alloc_size,
                               uint64_t phys_addr, uint64_t iova,
                               uint32_t domain_id,
                               uint64_t prealloc_used_now,
                               uint64_t left_now,
                               uint64_t compute_used_now,
                               uint64_t compute_left_now) {
  printf("[RKNPU_PREALLOC] kind=%s size=%zu phys=0x%llx iova=0x%llx domain=%u prealloc_used=%llu left=%llu compute_used=%llu compute_left=%llu\n",
         kind, alloc_size,
         (unsigned long long) phys_addr,
         (unsigned long long) iova,
         domain_id,
         (unsigned long long) prealloc_used_now,
         (unsigned long long) left_now,
         (unsigned long long) compute_used_now,
         (unsigned long long) compute_left_now);
}

static void cleanup_dev_mem_mapping(void) {
  if (payload_vaddr && payload_map_size) {
    munmap(payload_vaddr, payload_map_size);
  }
  if (compute_vaddr && compute_map_size) {
    munmap(compute_vaddr, compute_map_size);
  }
  payload_vaddr = NULL;
  compute_vaddr = NULL;
  payload_map_size = 0;
  compute_map_size = 0;
  payload_used = 0;
  compute_used = 0;

  if (payload_mem_fd >= 0) {
    close(payload_mem_fd);
    payload_mem_fd = -1;
  }
  if (compute_mem_fd >= 0) {
    close(compute_mem_fd);
    compute_mem_fd = -1;
  }
}

static void fd_init(void) {
  fd = npu_open();
  printf("%s %d: fd %d\n", __func__, __LINE__, fd);
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
  printf("RKNPU layout (%s): gpa_base=0x%llx donate=0x%llx compute=0x%llx iova_window=0x%llx reserve=0x%llx payload_window=0x%llx domains=%u\n",
         source,
         (unsigned long long) layout->gpa_base,
         (unsigned long long) layout->donate_size,
         (unsigned long long) layout->compute_buffer_bytes,
         (unsigned long long) layout->iova_window_bytes,
         (unsigned long long) layout->reserve_size,
         (unsigned long long) layout->payload_window_bytes,
         layout->domain_count);
}

static int load_prealloc_layout_from_driver(struct npu_prealloc_layout *layout) {
  struct rknpu_layout_info info;

  memset(&info, 0, sizeof(info));
  pthread_once(&fd_once, fd_init);
  if (fd < 0) {
    return -1;
  }

  if (ioctl(fd, DRM_IOCTL_RKNPU_GET_LAYOUT, &info) < 0) {
    printf("DRM_IOCTL_RKNPU_GET_LAYOUT failed errno=%d\n", errno);
    return -1;
  }

  memset(layout, 0, sizeof(*layout));
  layout->gpa_base = info.gpa_base;
  layout->donate_size = info.donate_size;
  layout->iova_window_bytes = info.iova_window_size;
  layout->compute_buffer_bytes = info.compute_buffer_size;

  if (validate_prealloc_layout(layout, "driver") != 0) {
    return -1;
  }

  if (info.payload_window_size != 0 &&
      info.payload_window_size != layout->payload_window_bytes) {
    printf("Driver payload window mismatch: ioctl=0x%llx local=0x%llx\n",
           (unsigned long long) info.payload_window_size,
           (unsigned long long) layout->payload_window_bytes);
  }
  if (info.reserve_size != 0 && info.reserve_size != layout->reserve_size) {
    printf("Driver reserve size mismatch: ioctl=0x%llx local=0x%llx\n",
           (unsigned long long) info.reserve_size,
           (unsigned long long) layout->reserve_size);
  }

  //log_prealloc_layout("driver", layout);
  return 0;
}

static void init_prealloc_layout(void) {
  memset(&g_prealloc_layout, 0, sizeof(g_prealloc_layout));
  g_prealloc_layout_ready = 0;

  if (load_prealloc_layout_from_driver(&g_prealloc_layout) == 0) {
    g_prealloc_layout_ready = 1;
    return;
  }

  printf("Failed to load RKNPU prealloc layout from driver\n");
}

static int ensure_prealloc_layout_loaded(void) {
  pthread_once(&layout_once, init_prealloc_layout);
  if (!g_prealloc_layout_ready) {
    errno = ENODEV;
    return -1;
  }
  return 0;
}

static int ensure_dev_mem_open(void) {
  if (ensure_prealloc_layout_loaded() != 0) {
    return -1;
  }

  if (payload_mem_fd >= 0 && compute_mem_fd >= 0 &&
      payload_vaddr != NULL && compute_vaddr != NULL) {
    return 0;
  }

  if (g_prealloc_layout.payload_total_bytes > SIZE_MAX ||
      g_prealloc_layout.compute_buffer_bytes > SIZE_MAX) {
    printf("RKNPU /dev/mem map too large: payload=0x%llx compute=0x%llx\n",
           (unsigned long long) g_prealloc_layout.payload_total_bytes,
           (unsigned long long) g_prealloc_layout.compute_buffer_bytes);
    return -1;
  }

  payload_mem_fd = open("/dev/mem", O_RDWR);
  if (payload_mem_fd < 0) {
    printf("Failed to open /dev/mem for payload, errno=%d\n", errno);
    return -1;
  }

  compute_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (compute_mem_fd < 0) {
    printf("Failed to open /dev/mem for compute, errno=%d\n", errno);
    cleanup_dev_mem_mapping();
    return -1;
  }

  const uint64_t payload_phys_base = g_prealloc_layout.gpa_base;
  const size_t payload_size = (size_t) g_prealloc_layout.payload_total_bytes;
  void *payload_map = mmap(NULL, payload_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, payload_mem_fd,
                           (off_t) payload_phys_base);
  if (payload_map == MAP_FAILED) {
    printf("Failed to mmap /dev/mem payload at phys=0x%llx, size=%zu, errno=%d\n",
           (unsigned long long) payload_phys_base, payload_size, errno);
    cleanup_dev_mem_mapping();
    return -1;
  }

  const uint64_t compute_phys_base =
      g_prealloc_layout.gpa_base + g_prealloc_layout.compute_gpa_offset;
  const size_t compute_size = (size_t) g_prealloc_layout.compute_buffer_bytes;
  void *compute_map = mmap(NULL, compute_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, compute_mem_fd,
                           (off_t) compute_phys_base);
  if (compute_map == MAP_FAILED) {
    printf("Failed to mmap /dev/mem compute at phys=0x%llx, size=%zu, errno=%d\n",
           (unsigned long long) compute_phys_base, compute_size, errno);
    cleanup_dev_mem_mapping();
    return -1;
  }

  payload_vaddr = payload_map;
  compute_vaddr = compute_map;
  payload_map_size = payload_size;
  compute_map_size = compute_size;
  payload_used = 0;
  compute_used = 0;
  atexit(cleanup_dev_mem_mapping);
  printf("Mapped /dev/mem payload: phys=0x%llx size=%zu vaddr=%p\n",
         (unsigned long long) payload_phys_base, payload_size, payload_map);
  printf("Mapped /dev/mem compute: phys=0x%llx size=%zu vaddr=%p\n",
         (unsigned long long) compute_phys_base, compute_size, compute_map);
  return 0;
}

static void *mem_allocate_internal(size_t size, uint64_t *dma_addr, uint64_t *obj,
                                   uint32_t flags, uint64_t *handle,
                                   uint32_t *domain_id, int use_compute_pool) {
  (void)flags;

  pthread_once(&fd_once, fd_init);
  if (size == 0) {
    return NULL;
  }

  if (ensure_dev_mem_open() != 0) {
    return NULL;
  }

  const size_t alloc_size = size;
  uint64_t phys_addr;
  uint64_t iova;
  uint32_t alloc_domain_id = UINT32_MAX;
  uint64_t prealloc_used_now;
  uint64_t left_now;
  uint64_t compute_used_now;
  uint64_t compute_left_now;
  const char *alloc_kind;
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
    map = (char *) compute_vaddr + compute_used;
    compute_used += alloc_size;
    alloc_kind = "compute";
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
      printf("Out of payload window: need=%zu used=%llu total=%llu\n",
             alloc_size,
             (unsigned long long) local_payload_used,
             (unsigned long long) g_prealloc_layout.payload_total_bytes);
      return NULL;
    }

    phys_addr = g_prealloc_layout.gpa_base + local_payload_used;
    iova = local_payload_used % g_prealloc_layout.payload_window_bytes;
    map = (char *) payload_vaddr + local_payload_used;
    alloc_domain_id = (uint32_t)
        (local_payload_used / g_prealloc_layout.payload_window_bytes);
    if (domain_id) {
      *domain_id = alloc_domain_id;
    }
    payload_used = local_payload_used + alloc_size;
    alloc_kind = "payload";
  }
  prealloc_used_now = payload_used;
  left_now = g_prealloc_layout.payload_total_bytes - prealloc_used_now;
  compute_used_now = compute_used;
  compute_left_now = g_prealloc_layout.compute_buffer_bytes - compute_used_now;

  if (dma_addr) {
    *dma_addr = iova;
  }
  if (obj) {
    *obj = phys_addr;
  }
  if (handle) {
    *handle = 0;
  }

  /*
  log_prealloc_usage(alloc_kind, alloc_size, phys_addr, iova, alloc_domain_id,
                     prealloc_used_now, left_now,
                     compute_used_now, compute_left_now);
                     */

  //printf("mem allocate addr:0x%lx, size:%d\n", map, size);
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

void ggml_rknpu2_reset_compute_used(void) {
  compute_used = 0;
}

int ggml_rknpu2_payload_ptr_to_offset(const void *ptr, uint64_t *offset) {
  uintptr_t begin;
  uintptr_t end;
  uintptr_t value;

  if (ptr == NULL || offset == NULL || payload_vaddr == NULL) {
    errno = EINVAL;
    return -1;
  }

  begin = (uintptr_t) payload_vaddr;
  end = begin + payload_map_size;
  value = (uintptr_t) ptr;
  if (end < begin || value < begin || value >= end) {
    errno = EINVAL;
    return -1;
  }

  *offset = (uint64_t) (value - begin);
  return 0;
}

uint64_t ggml_rknpu2_get_payload_used(void) {
  return (uint64_t) payload_used;
}

uint64_t ggml_rknpu2_get_payload_total_bytes(void) {
  if (ensure_prealloc_layout_loaded() != 0) {
    return 0;
  }
  return g_prealloc_layout.payload_total_bytes;
}

int ggml_rknpu2_get_shinfo_phys_addr(uint64_t *phys_addr) {
  if (phys_addr == NULL) {
    errno = EINVAL;
    return -1;
  }

  if (ensure_prealloc_layout_loaded() != 0) {
    return -1;
  }

  if (g_prealloc_layout.gpa_base < PAGE_SIZE_BYTES) {
    errno = ERANGE;
    return -1;
  }

  *phys_addr = g_prealloc_layout.gpa_base - PAGE_SIZE_BYTES;
  return 0;
}

int ggml_rknpu2_flush_payload_range(uint64_t payload_offset, uint64_t size) {
  struct rknpu_mem_sync sync;

  pthread_once(&fd_once, fd_init);
  if (fd < 0) {
    return -1;
  }

  if (ensure_dev_mem_open() != 0) {
    return -1;
  }

  if (payload_offset > g_prealloc_layout.payload_total_bytes ||
      size > g_prealloc_layout.payload_total_bytes - payload_offset) {
    errno = EINVAL;
    printf("Invalid RKNPU payload flush range: offset=0x%llx size=0x%llx total=0x%llx\n",
           (unsigned long long) payload_offset,
           (unsigned long long) size,
           (unsigned long long) g_prealloc_layout.payload_total_bytes);
    return -1;
  }

  if (size == 0) {
    return 0;
  }

  memset(&sync, 0, sizeof(sync));
  sync.flags = RKNPU_MEM_SYNC_TO_DEVICE;
  sync.obj_addr = g_prealloc_layout.gpa_base;
  sync.offset = payload_offset;
  sync.size = size;

  int ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &sync);
  if (ret < 0) {
    printf("DRM_IOCTL_RKNPU_MEM_SYNC payload flush failed: errno=%d phys=0x%llx offset=0x%llx size=0x%llx\n",
           errno,
           (unsigned long long) g_prealloc_layout.gpa_base,
           (unsigned long long) payload_offset,
           (unsigned long long) size);
  }
  return ret;
}

int ggml_rknpu2_flush_all_payload(void) {
  struct timespec start_ts;
  struct timespec end_ts;
  long long elapsed_us = 0;
  int ret;

  clock_gettime(CLOCK_MONOTONIC, &start_ts);

  if (ensure_dev_mem_open() != 0) {
    return -1;
  }

  //ret = ggml_rknpu2_flush_payload_range(0, g_prealloc_layout.payload_total_bytes);
  for(uint64_t offset = 0 ; offset < g_prealloc_layout.payload_total_bytes ; offset += 64*1024*1024){
    ret = ggml_rknpu2_flush_payload_range(offset, 64*1024*1024);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_ts);
  elapsed_us =
      (long long) (end_ts.tv_sec - start_ts.tv_sec) * 1000000LL +
      (long long) (end_ts.tv_nsec - start_ts.tv_nsec) / 1000LL;
  printf("[RKNPU_FLUSH] flush_all total=%#llx ret=%d elapsed=%lld us\n",
         (unsigned long long) g_prealloc_layout.payload_total_bytes,
         ret, elapsed_us);
  return ret;
}

void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr) {
  (void)addr;
  (void)len;
  (void)handle;
  (void)obj_addr;
  // Allocations are slices of the process-wide /dev/mem mapping, so there is
  // no per-allocation munmap here.
}

int npu_open(void) {

  char buf1[256], buf2[256], buf3[256];

  memset(buf1, 0 ,sizeof(buf1));
  memset(buf2, 0 ,sizeof(buf2));
  memset(buf3, 0, sizeof(buf3));

  // Open DRI called "rknpu"
  int local_fd = open(NPU_DEVICE, O_RDWR);
  if(local_fd<0) {
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
  if (ret <0) {
    printf("DRM_IOCTL_VERISON failed %d\n",ret);
    close(local_fd);
    return ret;
  }
  printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
  return local_fd;
}

int npu_reset(void) {
  pthread_once(&fd_once, fd_init);

  // Reset the NPU
  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
    0,
  };
  return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);	
}

int npu_submit(uint64_t regcfg_obj_addr, uint32_t core_mask, uint32_t domain_id)
{
  pthread_once(&fd_once, fd_init);
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
  return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit_ext.submit);
}

int npu_submit_multi(uint64_t regcfg_obj_addr[], int task_num,
                     uint32_t domain_id, void *polling)
{
  struct rknpu_submit_extend submit_ext;

  (void)polling;

  pthread_once(&fd_once, fd_init);
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

  return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT_EXT, &submit_ext);
}
