#ifndef _LLM_IOCTL_H_
#define _LLM_IOCTL_H_

#include <linux/ioctl.h>
#include <linux/types.h>

struct llm_map_info {
    __u64 user_vaddr;
    __u64 length;
    __u64 committed_length;
    __u64 next_page_offset;
    __u64 entry_size;
    __u32 nr_pages;
    __u32 committed_entries;
    __u32 finish_entry_count;
    __u32 mapped;
};

struct llm_layout_info {
    __u64 gpa_base;
    __u64 donate_size;
    __u64 compute_buffer_size;
    __u64 iova_window_size;
    __u64 reserve_size;
    __u64 payload_window_size;
    __u64 payload_total_size;
    __u64 compute_buffer_gpa;
    __u64 entry_size;
    __u32 domain_count;
    __u32 payload_entry_count;
    __u32 compute_entry_count;
    __u32 layout_version;
};

struct llm_extend_info {
    __u64 offset;
    __u64 length;
    __u64 committed_length;
    __u64 reserved_length;
    __u64 entry_size;
    __u32 entry_index;
    __u32 flags;
};

#define LLM_EXTEND_FLAG_FINISH (1U << 0)
#define LLM_EXTEND_FLAG_DONE   (1U << 1)

#define LLM_IOC_MAGIC    'L'
#define LLM_LAYOUT_INFO_VERSION 2
#define LLM_IOC_GET_INFO _IOR(LLM_IOC_MAGIC, 0x00, struct llm_map_info)
#define LLM_IOC_BEGIN    _IO(LLM_IOC_MAGIC, 0x01)
#define LLM_IOC_FINISH   _IO(LLM_IOC_MAGIC, 0x02)
#define LLM_IOC_GET_LAYOUT _IOR(LLM_IOC_MAGIC, 0x03, struct llm_layout_info)
#define LLM_IOC_EXTEND _IOWR(LLM_IOC_MAGIC, 0x04, struct llm_extend_info)

#endif /* _LLM_IOCTL_H_ */
