#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct glza_serial_scheduler_stats {
  uint64_t tasks_created;
  uint64_t context_switches;
  uint32_t maximum_live_tasks;
  size_t task_stack_bytes;
};

// One GLZA encode call owns one scheduler session.  All task contexts execute
// on the caller's OS thread.  finish returns nonzero if GLZA left a task
// unjoined or the scheduler observed an internal failure.
int glza_serial_scheduler_begin(void);
int glza_serial_scheduler_finish(struct glza_serial_scheduler_stats* stats);
int glza_serial_scheduler_last_stats(
    struct glza_serial_scheduler_stats* stats);
void glza_serial_scheduler_yield(void);

#ifdef __cplusplus
}
#endif
