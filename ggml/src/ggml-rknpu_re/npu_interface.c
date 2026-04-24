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
#include "llm_ioctl.h"

#include <pthread.h>

#define NPU_DEVICE "/dev/dri/card0"
#define LLM_DEVICE "/dev/llm"
#define POOL_ALIGN_BYTES (128ULL << 20)

static int npu_fd = -1;
static int llm_fd = -1;
static int llm_window_began = 0;
static void *pool_vaddr = NULL;
static size_t pool_map_size = 0;
static size_t pool_used = 0;
static uint64_t pool_dma_base = 0;
static uint64_t pool_obj_base = 0;
static uint64_t pool_handle = 0;
static pthread_once_t npu_fd_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mem_lock = PTHREAD_MUTEX_INITIALIZER;

int npu_open(void);

static size_t pool_align_up(size_t n) {
  return (n + POOL_ALIGN_BYTES - 1) & ~(size_t)(POOL_ALIGN_BYTES - 1);
}

static void cleanup_pool_mapping(void) {
  if (pool_vaddr && pool_map_size) {
    munmap(pool_vaddr, pool_map_size);
  }
  pool_vaddr = NULL;
  pool_map_size = 0;
  pool_used = 0;

  if (llm_fd >= 0 && llm_window_began) {
    const int ret = ioctl(llm_fd, LLM_IOC_FINISH);
    if (ret < 0) {
      printf("LLM_IOC_FINISH failed ret=%d errno=%d\n", ret, errno);
    }
    llm_window_began = 0;
  }

  if (llm_fd >= 0) {
    close(llm_fd);
    llm_fd = -1;
  }

  pool_dma_base = 0;
  pool_obj_base = 0;
  pool_handle = 0;
}

static void npu_fd_init(void) {
  npu_fd = npu_open();
  printf("%s %d: npu_fd %d\n", __func__, __LINE__, npu_fd);
}

static int mem_pool_prepare_locked(size_t pool_size) {
  pool_size = pool_align_up(pool_size);
  if (pool_size == 0) {
    printf("Invalid pool size 0\n");
    return -1;
  }

  if (pool_vaddr != NULL) {
    if (pool_size > pool_map_size) {
      printf("Requested pool size %zu exceeds existing mapped pool size %zu\n",
             pool_size, pool_map_size);
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

  void *map = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, llm_fd, 0);
  if (map == MAP_FAILED) {
    printf("Failed to mmap %s size=%zu errno=%d\n",
           LLM_DEVICE, pool_size, errno);
    (void) ioctl(llm_fd, LLM_IOC_FINISH);
    llm_window_began = 0;
    close(llm_fd);
    llm_fd = -1;
    return -1;
  }

  struct llm_map_info info;
  memset(&info, 0, sizeof(info));
  ret = ioctl(llm_fd, LLM_IOC_GET_INFO, &info);
  if (ret < 0) {
    printf("LLM_IOC_GET_INFO failed ret=%d errno=%d\n", ret, errno);
    munmap(map, pool_size);
    (void) ioctl(llm_fd, LLM_IOC_FINISH);
    llm_window_began = 0;
    close(llm_fd);
    llm_fd = -1;
    return -1;
  }

  if (!info.mapped || info.user_vaddr != (uint64_t)(uintptr_t) map || info.length < pool_size) {
    printf("LLM mapping metadata mismatch: mapped=%u user_vaddr=0x%llx length=%llu expected_vaddr=0x%llx expected_len=%zu\n",
           info.mapped,
           (unsigned long long) info.user_vaddr,
           (unsigned long long) info.length,
           (unsigned long long) (uint64_t)(uintptr_t) map,
           pool_size);
    munmap(map, pool_size);
    (void) ioctl(llm_fd, LLM_IOC_FINISH);
    llm_window_began = 0;
    close(llm_fd);
    llm_fd = -1;
    return -1;
  }

  pool_vaddr = map;
  pool_map_size = pool_size;
  pool_used = 0;
  pool_dma_base = (uint64_t)(uintptr_t) map;
  pool_obj_base = (uint64_t)(uintptr_t) map;
  pool_handle = 0;

  atexit(cleanup_pool_mapping);
  printf("Mapped LLM pool: dma=0x%llx obj=0x%llx handle=%llu size=%zu vaddr=%p\n",
         (unsigned long long) pool_dma_base,
         (unsigned long long) pool_obj_base,
         (unsigned long long) pool_handle,
         pool_map_size,
         map);
  return 0;
}

int mem_pool_prepare(size_t pool_size) {
  int ret;

  pthread_mutex_lock(&mem_lock);
  ret = mem_pool_prepare_locked(pool_size);
  pthread_mutex_unlock(&mem_lock);

  return ret;
}

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle) {
  (void)flags;

  if (size == 0) {
    return NULL;
  }

  pthread_mutex_lock(&mem_lock);
  if (pool_vaddr == NULL) {
    printf("mem_allocate called before mem_pool_prepare\n");
    pthread_mutex_unlock(&mem_lock);
    return NULL;
  }

  const size_t alloc_size = size;
  if (alloc_size > pool_map_size - pool_used) {
    printf("Out of LLM pool: need=%zu used=%zu total=%zu\n",
           alloc_size, pool_used, pool_map_size);
    pthread_mutex_unlock(&mem_lock);
    return NULL;
  }

  const uint64_t dma = pool_dma_base + (uint64_t) pool_used;
  const uint64_t obj_addr = pool_obj_base + (uint64_t) pool_used;
  void *map = (char *) pool_vaddr + pool_used;
  printf("addr:%lx, size:%ld, used:%ld\n", map, size, pool_used);
  pool_used += alloc_size;
  pthread_mutex_unlock(&mem_lock);

  if (dma_addr) {
    *dma_addr = dma;
  }
  if (obj) {
    *obj = obj_addr;
  }
  if (handle) {
    *handle = pool_handle;
  }

  return map;
}

void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr) {
  (void)addr;
  (void)len;
  (void)handle;
  (void)obj_addr;
  // Allocations are slices of the process-wide /dev/llm mapping, so there is
  // no per-allocation unmap here.
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
  pthread_once(&npu_fd_once, npu_fd_init);

  // Reset the NPU
  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
    0,
  };
  return ioctl(npu_fd, DRM_IOCTL_RKNPU_ACTION, &act);	
}

int npu_submit(uint64_t regcfg_obj_addr, uint32_t core_mask)
{
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

int npu_submit_multi(uint64_t regcfg_obj_addr[], int task_num, void *polling)
{
  struct rknpu_submit_extend submit_ext;

  (void)polling;

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
