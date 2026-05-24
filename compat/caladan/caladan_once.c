/*
 * Race-safe pthread_once replacement for Caladan.
 *
 * 3-state CAS: UNINIT -> IN_PROGRESS -> DONE. Losers spin-yield until
 * the winner reaches DONE. We deliberately avoid futex/park because
 * once-init contention is bounded to a small number of concurrent
 * first-callers and the init routines themselves are fast.
 */

#include "compat/caladan/caladan_shim.h"

/* Forward-declare to dodge Caladan header / FFmpeg CFLAGS conflicts. */
extern void thread_yield(void);

#define ONCE_UNINIT      0u
#define ONCE_IN_PROGRESS 1u
#define ONCE_DONE        2u

int ff_caladan_once(ff_caladan_once_t *control, void (*routine)(void))
{
    uint32_t expected = ONCE_UNINIT;

    if (__atomic_load_n(control, __ATOMIC_ACQUIRE) == ONCE_DONE)
        return 0;

    if (__atomic_compare_exchange_n(control, &expected, ONCE_IN_PROGRESS,
                                    0, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE)) {
        routine();
        __atomic_store_n(control, ONCE_DONE, __ATOMIC_RELEASE);
        return 0;
    }

    while (__atomic_load_n(control, __ATOMIC_ACQUIRE) != ONCE_DONE)
        thread_yield();

    return 0;
}
