#define _DEFAULT_SOURCE 1
#define _XOPEN_SOURCE 700

#include "glza_serial_scheduler.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

enum {
  GLZA_SERIAL_MAX_TASKS = 32,
  GLZA_SERIAL_STACK_BYTES = 2 * 1024 * 1024,
};

enum glza_serial_task_state {
  GLZA_SERIAL_TASK_FREE = 0,
  GLZA_SERIAL_TASK_RUNNABLE,
  GLZA_SERIAL_TASK_RUNNING,
  GLZA_SERIAL_TASK_DONE,
};

struct glza_serial_task {
  ucontext_t context;
  void* (*entry)(void*);
  void* argument;
  void* result;
  void* mapping;
  size_t mapping_bytes;
  unsigned slot;
  enum glza_serial_task_state state;
};

struct glza_serial_scheduler {
  int active;
  int failure;
  int current_task;
  unsigned next_slot;
  unsigned live_tasks;
  long page_bytes;
  ucontext_t main_context;
  struct glza_serial_task tasks[GLZA_SERIAL_MAX_TASKS];
  struct glza_serial_scheduler_stats stats;
};

static _Thread_local struct glza_serial_scheduler scheduler;
static _Thread_local struct glza_serial_scheduler_stats last_stats;
static _Thread_local int last_stats_valid;

static void destroy_stack(struct glza_serial_task* task) {
  if (task->mapping != NULL) {
    munmap(task->mapping, task->mapping_bytes);
    task->mapping = NULL;
    task->mapping_bytes = 0;
  }
}

static int make_stack(struct glza_serial_task* task) {
  const size_t guard = (size_t)scheduler.page_bytes;
  const size_t total = GLZA_SERIAL_STACK_BYTES + 2 * guard;
  void* mapping = mmap(NULL, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
  if (mapping == MAP_FAILED) return ENOMEM;
  void* stack = (unsigned char*)mapping + guard;
  if (mprotect(stack, GLZA_SERIAL_STACK_BYTES, PROT_READ | PROT_WRITE) != 0) {
    munmap(mapping, total);
    return ENOMEM;
  }
  task->mapping = mapping;
  task->mapping_bytes = total;
  return 0;
}

static void install_stack(struct glza_serial_task* task) {
  const size_t guard = (size_t)scheduler.page_bytes;
  task->context.uc_stack.ss_sp = (unsigned char*)task->mapping + guard;
  task->context.uc_stack.ss_size = GLZA_SERIAL_STACK_BYTES;
  task->context.uc_stack.ss_flags = 0;
}

static void task_trampoline(int slot) {
  struct glza_serial_task* task = &scheduler.tasks[(unsigned)slot];
  scheduler.current_task = slot;
  task->state = GLZA_SERIAL_TASK_RUNNING;
  task->result = task->entry(task->argument);
  task->state = GLZA_SERIAL_TASK_DONE;
  scheduler.current_task = -1;
  setcontext(&scheduler.main_context);
  abort();
}

int glza_serial_scheduler_begin(void) {
  if (scheduler.active) return EBUSY;
  memset(&scheduler, 0, sizeof(scheduler));
  scheduler.active = 1;
  scheduler.current_task = -1;
  scheduler.page_bytes = sysconf(_SC_PAGESIZE);
  if (scheduler.page_bytes <= 0) {
    scheduler.active = 0;
    return EINVAL;
  }
  scheduler.stats.task_stack_bytes = GLZA_SERIAL_STACK_BYTES;
  last_stats_valid = 0;
  for (unsigned slot = 0; slot < GLZA_SERIAL_MAX_TASKS; ++slot) {
    const int stack_status = make_stack(&scheduler.tasks[slot]);
    if (stack_status != 0) {
      for (unsigned prior = 0; prior < slot; ++prior)
        destroy_stack(&scheduler.tasks[prior]);
      scheduler.active = 0;
      return stack_status;
    }
  }
  return 0;
}

void glza_serial_scheduler_yield(void) {
  if (!scheduler.active || scheduler.failure) return;
  if (scheduler.current_task >= 0) {
    struct glza_serial_task* task =
        &scheduler.tasks[(unsigned)scheduler.current_task];
    task->state = GLZA_SERIAL_TASK_RUNNABLE;
    const int slot = scheduler.current_task;
    scheduler.current_task = -1;
    ++scheduler.stats.context_switches;
    if (swapcontext(&task->context, &scheduler.main_context) != 0) {
      scheduler.failure = errno ? errno : EFAULT;
      return;
    }
    scheduler.current_task = slot;
    task->state = GLZA_SERIAL_TASK_RUNNING;
    return;
  }

  for (unsigned offset = 0; offset < GLZA_SERIAL_MAX_TASKS; ++offset) {
    const unsigned slot =
        (scheduler.next_slot + offset) % GLZA_SERIAL_MAX_TASKS;
    struct glza_serial_task* task = &scheduler.tasks[slot];
    if (task->state != GLZA_SERIAL_TASK_RUNNABLE) continue;
    scheduler.next_slot = (slot + 1) % GLZA_SERIAL_MAX_TASKS;
    scheduler.current_task = (int)slot;
    task->state = GLZA_SERIAL_TASK_RUNNING;
    ++scheduler.stats.context_switches;
    if (swapcontext(&scheduler.main_context, &task->context) != 0)
      scheduler.failure = errno ? errno : EFAULT;
    scheduler.current_task = -1;
    return;
  }
}

int glza_serial_pthread_create(struct glza_serial_task** output,
                               const void* attributes,
                               void* (*entry)(void*), void* argument) {
  (void)attributes;
  if (!scheduler.active || output == NULL || entry == NULL) return EINVAL;
  for (unsigned slot = 0; slot < GLZA_SERIAL_MAX_TASKS; ++slot) {
    struct glza_serial_task* task = &scheduler.tasks[slot];
    if (task->state != GLZA_SERIAL_TASK_FREE) continue;
    void* mapping = task->mapping;
    const size_t mapping_bytes = task->mapping_bytes;
    memset(task, 0, sizeof(*task));
    task->mapping = mapping;
    task->mapping_bytes = mapping_bytes;
    task->slot = slot;
    task->entry = entry;
    task->argument = argument;
    if (getcontext(&task->context) != 0) return errno ? errno : EFAULT;
    install_stack(task);
    task->context.uc_link = &scheduler.main_context;
    makecontext(&task->context, (void (*)(void))task_trampoline, 1,
                (int)slot);
    task->state = GLZA_SERIAL_TASK_RUNNABLE;
    *output = task;
    ++scheduler.live_tasks;
    ++scheduler.stats.tasks_created;
    if (scheduler.live_tasks > scheduler.stats.maximum_live_tasks)
      scheduler.stats.maximum_live_tasks = scheduler.live_tasks;
    return 0;
  }
  scheduler.failure = EAGAIN;
  return EAGAIN;
}

int glza_serial_pthread_join(struct glza_serial_task* task, void** result) {
  if (!scheduler.active || task == NULL) return EINVAL;
  if (task < scheduler.tasks ||
      task >= scheduler.tasks + GLZA_SERIAL_MAX_TASKS)
    return ESRCH;
  while (task->state == GLZA_SERIAL_TASK_RUNNABLE ||
         task->state == GLZA_SERIAL_TASK_RUNNING) {
    glza_serial_scheduler_yield();
    if (scheduler.failure) return scheduler.failure;
  }
  if (task->state != GLZA_SERIAL_TASK_DONE) return EINVAL;
  if (result != NULL) *result = task->result;
  task->state = GLZA_SERIAL_TASK_FREE;
  --scheduler.live_tasks;
  return 0;
}

int glza_serial_sched_yield(void) {
  glza_serial_scheduler_yield();
  return scheduler.failure ? scheduler.failure : 0;
}

int glza_serial_scheduler_finish(struct glza_serial_scheduler_stats* stats) {
  if (!scheduler.active) return EINVAL;
  int status = scheduler.failure;
  if (scheduler.live_tasks != 0 && status == 0) status = EBUSY;
  last_stats = scheduler.stats;
  last_stats_valid = 1;
  if (stats != NULL) *stats = last_stats;
  for (unsigned slot = 0; slot < GLZA_SERIAL_MAX_TASKS; ++slot) {
    destroy_stack(&scheduler.tasks[slot]);
    scheduler.tasks[slot].state = GLZA_SERIAL_TASK_FREE;
  }
  scheduler.active = 0;
  scheduler.live_tasks = 0;
  scheduler.current_task = -1;
  return status;
}

int glza_serial_scheduler_last_stats(
    struct glza_serial_scheduler_stats* stats) {
  if (!last_stats_valid || stats == NULL) return EINVAL;
  *stats = last_stats;
  return 0;
}
