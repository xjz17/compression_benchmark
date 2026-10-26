#pragma once

// This header is force-included for GLZAcompress.c only.  Pull in the system
// declarations first, then remap the small pthread surface used by GLZA.  The
// other vendored translation units and the rest of baseline86 keep their
// normal libc/pthread behavior.
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

#include "glza_serial_scheduler.h"

struct glza_serial_task;
typedef struct glza_serial_task* glza_serial_pthread_t;

#ifdef __cplusplus
extern "C" {
#endif
int glza_serial_pthread_create(glza_serial_pthread_t* task,
                               const void* attributes,
                               void* (*entry)(void*), void* argument);
int glza_serial_pthread_join(glza_serial_pthread_t task, void** result);
int glza_serial_sched_yield(void);
#ifdef __cplusplus
}
#endif

#define pthread_t glza_serial_pthread_t
#define pthread_create glza_serial_pthread_create
#define pthread_join glza_serial_pthread_join
#define sched_yield glza_serial_sched_yield

// GLZA's producer/consumer waits use C11 atomics.  Yield before a load so a
// waiting consumer gives its producer a turn, and after a store so newly
// published work can run.  GCC/Clang atomic order constants match the C11
// memory_order values on the supported Linux build.
#undef atomic_load_explicit
#define atomic_load_explicit(object, order) \
  (__extension__({                            \
    glza_serial_scheduler_yield();            \
    __atomic_load_n((object), (int)(order));  \
  }))

#undef atomic_store_explicit
#define atomic_store_explicit(object, desired, order) \
  do {                                                  \
    __atomic_store_n((object), (desired), (int)(order)); \
    glza_serial_scheduler_yield();                      \
  } while (0)
