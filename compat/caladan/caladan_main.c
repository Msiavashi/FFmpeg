/*
 * FFmpeg main() wrapper for Caladan.
 *
 * Linked with -Wl,--wrap=main so the real ffmpeg main() becomes
 * __real_main, and execution starts here. We extract the Caladan
 * runtime config (from $CALADAN_CONFIG or argv[1] if it ends in
 * ".config"), call runtime_init(), and the real main() runs on a
 * Caladan uthread.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Avoid pulling in Caladan headers — they use GNU C extensions that
 * conflict with FFmpeg's strict warning flags. Forward-declare only
 * what we actually need.
 */
typedef void (*caladan_thread_fn_t)(void *arg);
extern int runtime_init(const char *cfgpath, caladan_thread_fn_t main_fn, void *arg);

extern int __real_main(int argc, char **argv);
int __wrap_main(int argc, char **argv);

struct trampoline_ctx {
    int    argc;
    char **argv;
    int    rc;
};

static void ffmpeg_trampoline(void *arg)
{
    struct trampoline_ctx *ctx = arg;
    ctx->rc = __real_main(ctx->argc, ctx->argv);
}

static int has_config_suffix(const char *s)
{
    size_t n;
    if (!s)
        return 0;
    n = strlen(s);
    return n >= 7 && strcmp(s + n - 7, ".config") == 0;
}

int __wrap_main(int argc, char **argv)
{
    struct trampoline_ctx ctx = { .argc = argc, .argv = argv, .rc = 0 };
    const char *cfg = getenv("CALADAN_CONFIG");
    int ret;

    if (!cfg && argc >= 2 && has_config_suffix(argv[1])) {
        cfg = argv[1];
        ctx.argc = argc - 1;
        ctx.argv = argv + 1;
        ctx.argv[0] = argv[0];
    }

    if (!cfg)
        cfg = "caladan.config";

    fprintf(stderr, "ffmpeg-caladan: runtime_init(%s)\n", cfg);

    ret = runtime_init(cfg, ffmpeg_trampoline, &ctx);
    if (ret) {
        fprintf(stderr, "ffmpeg-caladan: runtime_init failed: %d\n", ret);
        return ret;
    }

    return ctx.rc;
}
