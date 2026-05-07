#ifndef _LLM_IOCTL_H_
#define _LLM_IOCTL_H_

#include <linux/ioctl.h>
#include <linux/types.h>

struct llm_map_info {
    __u64 user_vaddr;
    __u64 length;
    __u64 next_page_offset;
    __u32 nr_pages;
    __u32 mapped;
};

struct llm_layout_info {
    __u64 gpa_base;
    __u64 donate_size;
    __u64 compute_buffer_size;
    __u64 iova_window_size;
    __u64 reserve_size;
    __u64 payload_window_size;
    __u64 compute_buffer_gpa;
    __u32 domain_count;
    __u32 layout_version;
};

#define LLM_IOC_MAGIC    'L'
#define LLM_LAYOUT_INFO_VERSION 1
#define LLM_IOC_GET_INFO _IOR(LLM_IOC_MAGIC, 0x00, struct llm_map_info)
#define LLM_IOC_BEGIN    _IO(LLM_IOC_MAGIC, 0x01)
#define LLM_IOC_FINISH   _IO(LLM_IOC_MAGIC, 0x02)
#define LLM_IOC_GET_LAYOUT _IOR(LLM_IOC_MAGIC, 0x03, struct llm_layout_info)

#endif /* _LLM_IOCTL_H_ */
