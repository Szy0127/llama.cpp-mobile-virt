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

#include "rknpu-ioctl.h"
#include "npu_hw.h"
#include "npu_interface.h"

#include <pthread.h>

#define NPU_DEVICE "/dev/dri/card0"
#define PREALLOC_PFN_BASE_DEFAULT      0xa0000ULL
#define PAGE_SIZE_BYTES   0x1000UL
#define DOMAIN_WINDOW_BYTES            (4ULL << 30)
#define PREALLOC_MMAP_BYTES_DEFAULT    (8ULL << 30)
#define COMPUTE_BUFFER_BYTES_DEFAULT   (256ULL << 20)
//#define COMPUTE_BUFFER_BYTES_DEFAULT   (2048ULL << 20)

struct npu_prealloc_layout {
  uint64_t gpa_base;
  uint64_t total_window_bytes;
  uint64_t compute_buffer_bytes;
  uint64_t payload_window_bytes;
  uint64_t payload_total_bytes;
  uint64_t compute_gpa_offset;
};

static int fd = -1;
static int dev_mem_fd = -1;
static void *dev_mem_vaddr = NULL;
static size_t dev_mem_map_size = 0;
static size_t payload_used = 0;
static size_t compute_used = 0;
static struct npu_prealloc_layout g_prealloc_layout;
static pthread_once_t fd_once = PTHREAD_ONCE_INIT;
static pthread_once_t layout_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mem_lock = PTHREAD_MUTEX_INITIALIZER;

int npu_open(void);

static size_t page_align_up(size_t n) {
  return (n + PAGE_SIZE_BYTES - 1) & ~(size_t)(PAGE_SIZE_BYTES - 1);
}

static void cleanup_dev_mem_mapping(void) {
  if (dev_mem_vaddr && dev_mem_map_size) {
    munmap(dev_mem_vaddr, dev_mem_map_size);
  }
  dev_mem_vaddr = NULL;
  dev_mem_map_size = 0;
  payload_used = 0;
  compute_used = 0;

  if (dev_mem_fd >= 0) {
    close(dev_mem_fd);
    dev_mem_fd = -1;
  }
}

static uint64_t parse_env_u64(const char *name, uint64_t fallback) {
  const char *value = getenv(name);
  char *end = NULL;
  unsigned long long parsed;

  if (value == NULL || value[0] == '\0') {
    return fallback;
  }

  errno = 0;
  parsed = strtoull(value, &end, 0);
  if (errno != 0 || end == value || (end && *end != '\0')) {
    printf("Invalid value for %s: %s, fallback=0x%llx\n",
           name, value, (unsigned long long) fallback);
    return fallback;
  }

  return (uint64_t) parsed;
}

static void init_prealloc_layout(void) {
  uint64_t total_window_bytes;
  uint64_t compute_buffer_bytes;
  uint64_t gpa_base;

  gpa_base = parse_env_u64("RKNPU_GPA_BASE",
                           PREALLOC_PFN_BASE_DEFAULT * PAGE_SIZE_BYTES);
  total_window_bytes = parse_env_u64("RKNPU_TOTAL_WINDOW_BYTES",
                                     PREALLOC_MMAP_BYTES_DEFAULT);
  total_window_bytes = parse_env_u64("RKNPU_TOTAL_WINDOW_MB",
                                     total_window_bytes >> 20) << 20;
  compute_buffer_bytes = parse_env_u64("RKNPU_COMPUTE_BUFFER_BYTES",
                                       COMPUTE_BUFFER_BYTES_DEFAULT);
  compute_buffer_bytes = parse_env_u64("RKNPU_COMPUTE_BUFFER_MB",
                                       compute_buffer_bytes >> 20) << 20;

  total_window_bytes = page_align_up(total_window_bytes);
  compute_buffer_bytes = page_align_up(compute_buffer_bytes);

  if (compute_buffer_bytes == 0 || compute_buffer_bytes >= DOMAIN_WINDOW_BYTES ||
      total_window_bytes <= compute_buffer_bytes) {
    printf("Invalid RKNPU layout, fallback to defaults total=0x%llx compute=0x%llx\n",
           (unsigned long long) total_window_bytes,
           (unsigned long long) compute_buffer_bytes);
    gpa_base = PREALLOC_PFN_BASE_DEFAULT * PAGE_SIZE_BYTES;
    total_window_bytes = PREALLOC_MMAP_BYTES_DEFAULT;
    compute_buffer_bytes = COMPUTE_BUFFER_BYTES_DEFAULT;
  }

  g_prealloc_layout.gpa_base = gpa_base;
  g_prealloc_layout.total_window_bytes = total_window_bytes;
  g_prealloc_layout.compute_buffer_bytes = compute_buffer_bytes;
  g_prealloc_layout.payload_window_bytes = DOMAIN_WINDOW_BYTES - compute_buffer_bytes;
  g_prealloc_layout.payload_total_bytes = total_window_bytes - compute_buffer_bytes;
  g_prealloc_layout.compute_gpa_offset = total_window_bytes - compute_buffer_bytes;
}

static void fd_init(void) {
  fd = npu_open();
  printf("%s %d: fd %d\n", __func__, __LINE__, fd);
}

static int ensure_dev_mem_open(void) {
  pthread_once(&layout_once, init_prealloc_layout);

  if (dev_mem_fd >= 0 && dev_mem_vaddr != NULL) {
    return 0;
  }

  dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (dev_mem_fd < 0) {
    printf("Failed to open /dev/mem, errno=%d\n", errno);
    return -1;
  }

  const uint64_t phys_base = g_prealloc_layout.gpa_base;
  const size_t map_size = g_prealloc_layout.total_window_bytes;
  void *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   dev_mem_fd, (off_t)phys_base);
  if (map == MAP_FAILED) {
    printf("Failed to mmap /dev/mem pool at phys=0x%llx, size=%zu, errno=%d\n",
           (unsigned long long)phys_base, map_size, errno);
    cleanup_dev_mem_mapping();
    return -1;
  }

  dev_mem_vaddr = map;
  dev_mem_map_size = map_size;
  payload_used = 0;
  compute_used = 0;
  atexit(cleanup_dev_mem_mapping);
  printf("Mapped /dev/mem pool: phys=0x%llx size=%zu vaddr=%p\n",
         (unsigned long long)phys_base, map_size, map);
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

  pthread_mutex_lock(&mem_lock);
  if (ensure_dev_mem_open() != 0) {
    pthread_mutex_unlock(&mem_lock);
    return NULL;
  }

  const size_t alloc_size = size;
  uint64_t phys_addr;
  uint64_t iova;
  void *map;

  if (use_compute_pool) {
    if (alloc_size > g_prealloc_layout.compute_buffer_bytes - compute_used) {
      printf("Out of compute buffer: need=%zu used=%zu total=%llu\n",
             alloc_size, compute_used,
             (unsigned long long) g_prealloc_layout.compute_buffer_bytes);
      pthread_mutex_unlock(&mem_lock);
      return NULL;
    }

    phys_addr = g_prealloc_layout.gpa_base +
                g_prealloc_layout.compute_gpa_offset + (uint64_t) compute_used;
    iova = g_prealloc_layout.payload_window_bytes + (uint64_t) compute_used;
    map = (char *) dev_mem_vaddr + g_prealloc_layout.compute_gpa_offset + compute_used;
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
      pthread_mutex_unlock(&mem_lock);
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
      pthread_mutex_unlock(&mem_lock);
      return NULL;
    }

    phys_addr = g_prealloc_layout.gpa_base + local_payload_used;
    iova = local_payload_used % g_prealloc_layout.payload_window_bytes;
    map = (char *) dev_mem_vaddr + local_payload_used;
    if (domain_id) {
      *domain_id = (uint32_t)
          (local_payload_used / g_prealloc_layout.payload_window_bytes);
    }
    payload_used = local_payload_used + alloc_size;
  }
  pthread_mutex_unlock(&mem_lock);

  if (dma_addr) {
    *dma_addr = iova;
  }
  if (obj) {
    *obj = phys_addr;
  }
  if (handle) {
    *handle = 0;
  }

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
