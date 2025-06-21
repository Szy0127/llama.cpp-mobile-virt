#include <linux/module.h>
#include <linux/init.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/ktime.h>
#include <linux/mmzone.h>
#include <linux/highmem.h>
#include <linux/mm.h>
#include <asm/io.h>

/* Module parameters */
static int cmd = -1;
static int size = 0;
static int use_dma = 0;

module_param(cmd, int, 0644);
MODULE_PARM_DESC(cmd, "Command: 0=allocate with pmalloc, 1=free, 2=allocate with alloc_pages");

module_param(size, int, 0644);
MODULE_PARM_DESC(size, "Size in MB to allocate when cmd=0");

module_param(use_dma, int, 0644);
MODULE_PARM_DESC(use_dma, "Use DMA zone specifically (0=normal, 1=DMA zone)");

/* Structure to store regular allocated pages for cmd=2 */
struct regular_page_info {
	struct list_head list;
	struct page *page;
	unsigned int order;
	unsigned long size;
	unsigned long phys_addr;
};

/* Global storage for allocated pages */
static LIST_HEAD(global_page_list);
static LIST_HEAD(regular_page_list);  /* For cmd=2 regular allocations */
static DEFINE_MUTEX(page_list_mutex);
static bool pages_allocated = false;



/*
 * pmalloc_allocate - Allocate memory using pmalloc
 */
static int pmalloc_allocate(unsigned long size_mb)
{
	struct pmalloc_page_info *info;
	unsigned long total_pages = 0;
	unsigned long total_allocated = 0;
	int block_count = 0;
	ktime_t start_time, end_time;
	s64 duration_ns;
	
	mutex_lock(&page_list_mutex);
	
	/* Check if pages are already allocated */
	if (pages_allocated) {
		printk(KERN_WARNING "pmalloc_exp: Pages already allocated. Free them first with cmd=1\n");
		mutex_unlock(&page_list_mutex);
		return -EBUSY;
	}
	
	printk(KERN_INFO "pmalloc_exp: Allocating %lu MB using pmalloc (use_dma=%d)...\n", size_mb, use_dma);
	printk(KERN_INFO "pmalloc_exp: Before allocation - check /proc/buddyinfo\n");
	printk(KERN_INFO "pmalloc_exp: Available info structures: %d\n", pmalloc_get_available_info_count());
	
	/* Start timing */
	start_time = ktime_get();
	
	/* Allocate pages getting biggest orders first */
	if (use_dma) {
		printk(KERN_INFO "pmalloc_exp: Using DMA-specific allocation\n");
		total_pages = pmalloc_get_dma_pages(size_mb, &global_page_list);
	} else {
		printk(KERN_INFO "pmalloc_exp: Using normal allocation (includes DMA)\n");
		total_pages = pmalloc_get_size_with_info(size_mb, &global_page_list);
	}
	
	/* End timing */
	end_time = ktime_get();
	duration_ns = ktime_to_ns(ktime_sub(end_time, start_time));
	
	if (total_pages == 0) {
		printk(KERN_ERR "pmalloc_exp: Failed to allocate any pages\n");
		mutex_unlock(&page_list_mutex);
		return -ENOMEM;
	}
	
	printk(KERN_INFO "pmalloc_exp: Successfully allocated %lu total pages\n", total_pages);
	
	/* Display information about each allocated block */
	list_for_each_entry(info, &global_page_list, list) {
		void *memory = pmalloc_page_address(info);
		unsigned int order = pmalloc_page_order(info);
		unsigned long block_size = pmalloc_page_size(info);
		unsigned long phys_addr = info->phys_addr;
		unsigned long num_pages = 1UL << order;
		
		block_count++;
		
		printk(KERN_INFO "pmalloc_exp: Block %d - Order=%u, Pages=%lu, Size=%luKB (%luMB)\n",
		       block_count, order, num_pages, block_size/1024, block_size/(1024*1024));
		       
		printk(KERN_INFO "pmalloc_exp:          Virtual Addr:  %p\n", memory);
		printk(KERN_INFO "pmalloc_exp:          Physical Addr: 0x%lx\n", phys_addr);
		
		/* Test memory access */
		if (memory) {
			/* Write test pattern to verify memory is accessible */
			*((u32*)memory) = 0xDEADBEEF;
			if (*((u32*)memory) == 0xDEADBEEF) {
				printk(KERN_INFO "pmalloc_exp:          Memory test: PASSED\n");
			} else {
				printk(KERN_ERR "pmalloc_exp:          Memory test: FAILED\n");
			}
		}
		
		total_allocated += block_size;
	}
	
	printk(KERN_INFO "pmalloc_exp: Total allocated: %d blocks, %luMB\n", 
	       block_count, total_allocated / (1024 * 1024));
	
	/* Print timing results */
	printk(KERN_INFO "pmalloc_exp: TIMING RESULTS:\n");
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld nanoseconds\n", duration_ns);
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld microseconds\n", duration_ns / 1000);
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld milliseconds\n", duration_ns / 1000000);
	if (total_pages > 0) {
		printk(KERN_INFO "pmalloc_exp: Average time per page: %lld nanoseconds\n", duration_ns / total_pages);
	}
	
	pages_allocated = true;
	mutex_unlock(&page_list_mutex);
	
	return 0;
}

/*
 * regular_free - Free regularly allocated pages
 */
static int regular_free(void)
{
	struct regular_page_info *info, *tmp;
	
	mutex_lock(&page_list_mutex);
	
	if (!pages_allocated) {
		printk(KERN_WARNING "pmalloc_exp: No pages to free. Allocate first\n");
		mutex_unlock(&page_list_mutex);
		return -EINVAL;
	}
	
	printk(KERN_INFO "pmalloc_exp: Freeing all regular allocated pages...\n");
	
	/* Free all pages */
	list_for_each_entry_safe(info, tmp, &regular_page_list, list) {
		__free_pages(info->page, info->order);
		list_del(&info->list);
		kfree(info);
	}
	
	pages_allocated = false;
	
	printk(KERN_INFO "pmalloc_exp: All regular pages freed successfully\n");
	
	mutex_unlock(&page_list_mutex);
	
	return 0;
}

/*
 * pmalloc_free - Free previously allocated memory
 */
static int pmalloc_free(void)
{
	mutex_lock(&page_list_mutex);
	
	if (!pages_allocated) {
		printk(KERN_WARNING "pmalloc_exp: No pages to free. Allocate first with cmd=0 or cmd=2\n");
		mutex_unlock(&page_list_mutex);
		return -EINVAL;
	}
	
	printk(KERN_INFO "pmalloc_exp: Freeing all allocated pages...\n");
	
	/* Check which type of allocation was used and free accordingly */
	if (!list_empty(&global_page_list)) {
		/* Free pmalloc pages */
		pmalloc_free_pages_with_info(&global_page_list);
		printk(KERN_INFO "pmalloc_exp: pmalloc pages freed successfully\n");
	} else if (!list_empty(&regular_page_list)) {
		/* Free regular pages */
		regular_free();
		return 0;  /* regular_free already handles mutex unlock */
	}
	
	pages_allocated = false;
	
	printk(KERN_INFO "pmalloc_exp: All pages freed successfully\n");
	
	mutex_unlock(&page_list_mutex);
	
	return 0;
}

/*
 * regular_allocate - Allocate memory using regular alloc_pages for comparison
 */
static int regular_allocate(unsigned long size_mb)
{
	struct regular_page_info *info;
	unsigned long target_pages, allocated_pages = 0;
	unsigned long total_allocated = 0;
	int block_count = 0;
	ktime_t start_time, end_time;
	s64 duration_ns;
	int order;
	gfp_t gfp_flags;
	
	mutex_lock(&page_list_mutex);
	
	/* Check if pages are already allocated */
	if (pages_allocated) {
		printk(KERN_WARNING "pmalloc_exp: Pages already allocated. Free them first with cmd=1\n");
		mutex_unlock(&page_list_mutex);
		return -EBUSY;
	}
	
	target_pages = (size_mb * 1024 * 1024) / PAGE_SIZE;
	
	printk(KERN_INFO "pmalloc_exp: Allocating %lu MB (%lu pages) using regular alloc_pages (use_dma=%d)...\n", 
	       size_mb, target_pages, use_dma);
	printk(KERN_INFO "pmalloc_exp: Before allocation - check /proc/buddyinfo\n");
	
	/* Set GFP flags */
	if (use_dma) {
		gfp_flags = GFP_KERNEL | __GFP_DMA;
		printk(KERN_INFO "pmalloc_exp: Using DMA zone allocation\n");
	} else {
		gfp_flags = GFP_KERNEL;
		printk(KERN_INFO "pmalloc_exp: Using normal zone allocation\n");
	}
	
	/* Start timing */
	start_time = ktime_get();
	
	/* Allocate pages using regular alloc_pages - try biggest orders first like pmalloc */
	while (allocated_pages < target_pages) {
		//unsigned long remaining_pages = target_pages - allocated_pages;
		struct page *page;
		
        /*
		order = 0;
		while ((1UL << (order + 1)) <= remaining_pages && (order + 1) <= 10) {
			order++;
		}
		
		page = alloc_pages(gfp_flags, order);
		if (!page) {
			while (order > 0) {
				order--;
				page = alloc_pages(gfp_flags, order);
				if (page)
					break;
			}
		}
        */
		order = 0;
		page = alloc_pages(gfp_flags, order);
		
		if (!page) {
			printk(KERN_WARNING "pmalloc_exp: Failed to allocate order %d page, stopping\n", order);
			break;
		}
		
		/* Create info structure */
		info = kmalloc(sizeof(*info), GFP_KERNEL);
		if (!info) {
			__free_pages(page, order);
			printk(KERN_ERR "pmalloc_exp: Failed to allocate info structure\n");
			break;
		}
		
		/* Fill info structure */
		INIT_LIST_HEAD(&info->list);
		info->page = page;
		info->order = order;
		info->size = PAGE_SIZE << order;
		info->phys_addr = page_to_phys(page);
		
		/* Add to list */
		list_add_tail(&info->list, &regular_page_list);
		
		allocated_pages += (1UL << order);
		block_count++;
	}
	
	/* End timing */
	end_time = ktime_get();
	duration_ns = ktime_to_ns(ktime_sub(end_time, start_time));
	
	if (allocated_pages == 0) {
		printk(KERN_ERR "pmalloc_exp: Failed to allocate any pages\n");
		mutex_unlock(&page_list_mutex);
		return -ENOMEM;
	}
	
	printk(KERN_INFO "pmalloc_exp: Successfully allocated %lu total pages (%lu requested)\n", 
	       allocated_pages, target_pages);
	
	/* Display information about each allocated block */
	block_count = 0;
    /*
	list_for_each_entry(info, &regular_page_list, list) {
		void *memory = page_address(info->page);
		unsigned long num_pages = 1UL << info->order;
		
		block_count++;
		
		printk(KERN_INFO "pmalloc_exp: Block %d - Order=%u, Pages=%lu, Size=%luKB (%luMB)\n",
		       block_count, info->order, num_pages, info->size/1024, info->size/(1024*1024));
		       
		printk(KERN_INFO "pmalloc_exp:          Virtual Addr:  %p\n", memory);
		printk(KERN_INFO "pmalloc_exp:          Physical Addr: 0x%lx\n", info->phys_addr);
		
		if (memory) {
			*((u32*)memory) = 0xDEADBEEF;
			if (*((u32*)memory) == 0xDEADBEEF) {
				printk(KERN_INFO "pmalloc_exp:          Memory test: PASSED\n");
			} else {
				printk(KERN_ERR "pmalloc_exp:          Memory test: FAILED\n");
			}
		}
		
		total_allocated += info->size;
	}
*/
	
	printk(KERN_INFO "pmalloc_exp: Total allocated: %d blocks, %luMB\n", 
	       block_count, total_allocated / (1024 * 1024));
	
	/* Print timing results */
	printk(KERN_INFO "pmalloc_exp: TIMING RESULTS (alloc_pages):\n");
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld nanoseconds\n", duration_ns);
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld microseconds\n", duration_ns / 1000);
	printk(KERN_INFO "pmalloc_exp: Total allocation time: %lld milliseconds\n", duration_ns / 1000000);
	if (allocated_pages > 0) {
		printk(KERN_INFO "pmalloc_exp: Average time per page: %lld nanoseconds\n", duration_ns / allocated_pages);
	}
	
	pages_allocated = true;
	mutex_unlock(&page_list_mutex);
	
	return 0;
}



static int __init pmalloc_exp_init(void)
{
	int ret = 0;
	
	printk(KERN_INFO "pmalloc_exp: Module loaded with cmd=%d, size=%d\n", cmd, size);
	
	switch (cmd) {
	case 0:
		/* Allocate memory using pmalloc */
		if (size <= 0) {
			printk(KERN_ERR "pmalloc_exp: Invalid size=%d. Use: insmod exp.ko cmd=0 size=1024\n", size);
			return -EINVAL;
		}
		ret = pmalloc_allocate(size);
		if (ret) {
			printk(KERN_ERR "pmalloc_exp: pmalloc allocation failed with error %d\n", ret);
			return ret;
		}
		break;
		
	case 1:
		/* Free memory */
		ret = pmalloc_free();
		if (ret) {
			printk(KERN_ERR "pmalloc_exp: Free failed with error %d\n", ret);
			return ret;
		}
		break;
		
	case 2:
		/* Allocate memory using regular alloc_pages for comparison */
		if (size <= 0) {
			printk(KERN_ERR "pmalloc_exp: Invalid size=%d. Use: insmod exp.ko cmd=2 size=1024\n", size);
			return -EINVAL;
		}
		ret = regular_allocate(size);
		if (ret) {
			printk(KERN_ERR "pmalloc_exp: regular allocation failed with error %d\n", ret);
			return ret;
		}
		break;
		
	default:
		printk(KERN_INFO "pmalloc_exp: Usage:\n");
		printk(KERN_INFO "  Allocate with pmalloc: insmod exp.ko cmd=0 size=1024 [use_dma=0/1]\n");
		printk(KERN_INFO "  Allocate with alloc_pages: insmod exp.ko cmd=2 size=1024 [use_dma=0/1]\n");
		printk(KERN_INFO "  Free:     insmod exp.ko cmd=1\n");
		printk(KERN_INFO "  Current:  cmd=%d, size=%d, use_dma=%d\n", cmd, size, use_dma);
		break;
	}
	
	return 0;
}

static void __exit pmalloc_exp_exit(void)
{
	/* Clean up any remaining allocated pages on module exit */
	if (pages_allocated) {
		printk(KERN_INFO "pmalloc_exp: Cleanup: freeing remaining pages...\n");
		pmalloc_free();
	}
	
	printk(KERN_INFO "pmalloc_exp: Module unloaded\n");
}

module_init(pmalloc_exp_init);
module_exit(pmalloc_exp_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("pmalloc test module");
MODULE_DESCRIPTION("Test module for pmalloc interface with cmd/size parameters");
