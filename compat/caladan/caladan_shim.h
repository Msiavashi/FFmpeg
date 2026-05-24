/*
 * FFmpeg Caladan compatibility layer.
 *
 * This header is force-included into every translation unit (via the
 * configure-injected -include) so that AV_ONCE_INIT and ff_thread_once
 * are redirected to a race-safe implementation. Caladan's pthread shim
 * does not implement pthread_once, and the AV_ONCE_INIT macro in
 * libavutil/thread.h is used in 100+ static initializers.
 */

#ifndef COMPAT_CALADAN_SHIM_H
#define COMPAT_CALADAN_SHIM_H

/*
 * Include this header only after config.h has been processed —
 * libavutil/thread.h does that for you. We don't gate the body on
 * CONFIG_CALADAN so that callers can rely on these symbols being
 * declared whenever they include the header.
 */

#include <stdint.h>

typedef uint32_t ff_caladan_once_t;
#define FF_CALADAN_ONCE_INIT 0u

int ff_caladan_once(ff_caladan_once_t *control, void (*routine)(void));

#endif /* COMPAT_CALADAN_SHIM_H */
