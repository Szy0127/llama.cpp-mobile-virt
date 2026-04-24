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

#define LLM_IOC_MAGIC    'L'
#define LLM_IOC_GET_INFO _IOR(LLM_IOC_MAGIC, 0x00, struct llm_map_info)
#define LLM_IOC_BEGIN    _IO(LLM_IOC_MAGIC, 0x01)
#define LLM_IOC_FINISH   _IO(LLM_IOC_MAGIC, 0x02)

#endif /* _LLM_IOCTL_H_ */
