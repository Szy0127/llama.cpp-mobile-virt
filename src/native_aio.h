#ifndef __NATIVE_AIO_H__
#define __NATIVE_AIO_H__

#define _GNU_SOURCE

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <linux/aio_abi.h>

static inline int io_setup(unsigned nr_events, aio_context_t* ctx_idp)
{
    return syscall(__NR_io_setup, nr_events, ctx_idp);
}

static inline int io_destroy(aio_context_t ctx)
{
    return syscall(__NR_io_destroy, ctx);
}

static inline int io_submit(aio_context_t ctx, long nr, struct iocb** iocbpp)
{
    return syscall(__NR_io_submit, ctx, nr, iocbpp);
}

static inline int io_getevents(aio_context_t ctx, long min_nr, long nr,
                               struct io_event* events, struct timespec* timeout)
{
    return syscall(__NR_io_getevents, ctx, min_nr, nr, events, timeout);
}

static inline int io_cancel(aio_context_t ctx, struct iocb* iocb,
                            struct io_event* result)
{
    return syscall(__NR_io_cancel, ctx, iocb, result);
}

#endif
