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

#include "rknpu-ioctl.h"
#include "npu_hw.h"
#include "npu_interface.h"

#include <pthread.h>

#define NPU_DEVICE "/dev/dri/card0"
#define PREALLOC_PFN_BASE 0xa0000ULL
#define PAGE_SIZE_BYTES   0x1000UL
#define PREALLOC_MMAP_BYTES         (4ULL << 30)

static int fd = -1;
static int dev_mem_fd = -1;
static void *dev_mem_vaddr = NULL;
static size_t dev_mem_map_size = 0;
static size_t prealloc_used = 0;
static pthread_once_t fd_once = PTHREAD_ONCE_INIT;
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
  prealloc_used = 0;

  if (dev_mem_fd >= 0) {
    close(dev_mem_fd);
    dev_mem_fd = -1;
  }
}

static void fd_init(void) {
  fd = npu_open();
  printf("%s %d: fd %d\n", __func__, __LINE__, fd);
}

static int ensure_dev_mem_open(void) {
  if (dev_mem_fd >= 0 && dev_mem_vaddr != NULL) {
    return 0;
  }

  dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (dev_mem_fd < 0) {
    printf("Failed to open /dev/mem, errno=%d\n", errno);
    return -1;
  }

  const uint64_t phys_base = PREALLOC_PFN_BASE * PAGE_SIZE_BYTES;
  const size_t map_size = PREALLOC_MMAP_BYTES;
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
  prealloc_used = 0;
  atexit(cleanup_dev_mem_mapping);
  printf("Mapped /dev/mem pool: phys=0x%llx size=%zu vaddr=%p\n",
         (unsigned long long)phys_base, map_size, map);
  return 0;
}

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle) {
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

  const size_t alloc_size = size;//page_align_up(size);
  if (alloc_size > dev_mem_map_size - prealloc_used) {
    printf("Out of /dev/mem pool: need=%zu used=%zu total=%zu\n",
           alloc_size, prealloc_used, dev_mem_map_size);
    pthread_mutex_unlock(&mem_lock);
    return NULL;
  }

  const uint64_t phys_addr = PREALLOC_PFN_BASE * PAGE_SIZE_BYTES + (uint64_t)prealloc_used;
  void *map = (char *)dev_mem_vaddr + prealloc_used;
  prealloc_used += alloc_size;
  pthread_mutex_unlock(&mem_lock);

  if (dma_addr) {
    *dma_addr = phys_addr;
  }
  if (obj) {
    *obj = phys_addr;
  }
  if (handle) {
    *handle = 0;
  }

  printf("mem allocate addr:0x%lx, size:%d\n", map, size);
  return map;
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

int npu_submit(__u64 regcfg_obj_addr, __u32 core_mask)
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

int npu_submit_multi(__u64 regcfg_obj_addr[], int task_num, void *polling)
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
