# FFmpeg on Caladan

This branch ports FFmpeg's threading layer to the
[Caladan](https://github.com/shenango/caladan) runtime so that an `ffmpeg`
process can run as a Caladan-scheduled application.

## Why

The intended use is as an **antagonist workload** co-located with a
latency-sensitive Caladan application. FFmpeg's software pixel-format
conversion path in `libswscale` (`yuv2rgb32`, `yuv2rgb565`, RGB↔RGB
permutations) emits non-temporal store instructions (`MOVNTQ`,
`MOVNTDQ`, `MOVNTDQA`) in tight loops:

- `libswscale/x86/rgb2rgb.c` — 56 `MOVNTQ`
- `libswscale/x86/swscale_template.c` — 17 `MOVNTQ`
- `libavutil/x86/imgutils.asm` — 4 `MOVNTDQA`

This is the precise workload needed to exercise — and stress — Caladan's
memory-bandwidth contention management.

## Prerequisites

System packages (Ubuntu 24.04 names):

```
sudo apt-get install -y libnuma-dev libibverbs-dev nasm
```

Caladan must be built first. On a host with no Mellanox/MLX5 NIC (or
when only the antagonist runtime is needed), the bundled `rdma-core`
submodule and the directpath fast-path can be skipped:

```
cd /path/to/caladan

# Edit build/shared.mk: comment out the three lines that add
# MLX5_LIBS / MLX5_INC / -DDIRECTPATH to the build flags, and strip
# -flto=auto from CONFIG_OPTIMIZE FLAGS so the static libs stay link-
# compatible with non-LTO consumers (FFmpeg).

# Edit Makefile: comment out the two lines that add
# runtime/net/directpath/*.c and runtime/net/directpath/mlx5/*.c to
# runtime_src.

# Provide stubs for the three directpath_* symbols referenced by
# runtime/init.c. A 3-line C file in runtime/directpath_stub.c
# returning 0 from each is sufficient.

make libbase.a libnet.a libruntime.a
make -C shim
ar rcs libruntime.a runtime/directpath_stub.o
```

This produces `libbase.a`, `libnet.a`, `libruntime.a`, and
`shim/libshim.a` at the Caladan root.

To run the antagonist against a real Caladan workload you still need
`iokerneld` (which requires the full Mellanox stack). For build-only
verification or for running against a single-process Caladan workload
on loopback, the above is enough to link `ffmpeg_g` but the runtime
will refuse to start without an iokernel.

## Build FFmpeg

```
cd /path/to/FFmpeg
./configure \
    --enable-caladan --caladan-path=/path/to/caladan \
    --disable-network --disable-protocols \
    --enable-protocol=file --enable-protocol=pipe \
    --disable-jni \
    --disable-encoders --disable-muxers --enable-muxer=null \
    --disable-doc --disable-htmlpages \
    --disable-ffprobe --disable-ffplay
make -j$(nproc)
```

The configure flags beyond `--enable-caladan` are not cosmetic. Each
one removes a code path the Caladan shim cannot support:

| Flag | What it removes | Why it's needed |
|---|---|---|
| `--disable-network` | `libavformat/{tcp,udp,http,rtsp,rtmp}*.c` | Caladan's shim has no BSD socket wrapper |
| `--disable-network` (also) | `pthread_cancel` site in `udp.c` | Caladan's shim has no `pthread_cancel` |
| `--disable-jni` | `libavcodec/{jni,ffjni}.c` direct `PTHREAD_*_INIT` users | Eliminated source of static initialisers |
| `--disable-encoders --disable-muxers` | libx264/etc threading + TLS | Avoid `__thread` TLS surface from third-party encoders |
| `--enable-muxer=null` | (keeps `-f null -`) | Needed for the antagonist's discard output |

## Run

Drop the bundled config (or your own) where the process can find it:

```
cp compat/caladan/sample.config /tmp/caladan.config
```

Start `iokerneld` according to the Caladan README, then launch FFmpeg
as a Caladan application. Two argv styles are accepted:

```
# 1. Caladan config via environment variable:
CALADAN_CONFIG=/tmp/caladan.config ./ffmpeg_g \
    -f lavfi -i testsrc=size=1280x720:rate=30 -t 5 \
    -pix_fmt rgb32 -f null -

# 2. Caladan config as argv[1] (must end in .config):
./ffmpeg_g /tmp/caladan.config \
    -f lavfi -i testsrc=size=1280x720:rate=30 -t 5 \
    -pix_fmt rgb32 -f null -
```

For the heaviest non-temporal store path (the actual antagonist
scenario), feed a raw YUV source and convert to RGB32:

```
CALADAN_CONFIG=/tmp/caladan.config ./ffmpeg_g \
    -stream_loop -1 -f rawvideo -pix_fmt yuv420p -s 1920x1080 \
    -i big_buck_bunny_1080p.yuv \
    -pix_fmt rgb32 -f null -
```

This hits `yuv2rgb32` (the densest `MOVNTQ` block in `libswscale`).

If `runtime_init` fails with `failed to map iokernel info region`, the
iokernel daemon is not running — that is the only thing you should see
when running without it.

## Port mechanics

### `compat/caladan/`

- **`caladan_main.c`** — defines `__wrap_main`, the entry point
  installed by `-Wl,--wrap=main`. Extracts the Caladan config path
  from `$CALADAN_CONFIG` (or shifts it out of `argv[1]` if it ends in
  `.config`), calls `runtime_init(cfg, ffmpeg_trampoline, &ctx)`. The
  trampoline runs the original `main()` as `__real_main()` on a
  Caladan uthread.
- **`caladan_once.c`** — race-safe `pthread_once` replacement. Caladan's
  pthread shim does not implement `pthread_once`, yet `AV_ONCE_INIT`
  is used in 100+ static initialisers across `libavcodec/`. Implemented
  as a 3-state CAS: `UNINIT → IN_PROGRESS → DONE`. Losers yield via
  `thread_yield()` until the winner reaches `DONE`.
- **`caladan_shim.h`** — declares the symbols above. Included from
  `libavutil/thread.h` only when `CONFIG_CALADAN` is set.
- **`sample.config`** — drop-in Caladan runtime config with `runtime_priority
  be` (best-effort), so the antagonist is preemptible and yieldable.

### Patch sites in FFmpeg core

| File | Change |
|---|---|
| `configure` | Adds `--enable-caladan` and `--caladan-path=PATH`. When enabled, sets `-mxsavec -mxsave -mfsgsbase`, prepends `-T $caladan/base/base.ld` and `-Wl,--wrap=main`, links `libshim.a libruntime.a libnet.a libbase.a` with `-lnuma -lpthread -ldl`. |
| `libavutil/thread.h` | Under `CONFIG_CALADAN`, redefines `AVOnce`, `AV_ONCE_INIT`, and `ff_thread_once` to use `ff_caladan_once`. `AVMutex` continues to map to `pthread_mutex_t`, which the shim wraps to Caladan's native `mutex_t`. |
| `libavutil/log.c` | `av_log` mutex switched to GCC constructor init. |
| `libavcodec/avcodec.c` | `codec_mutex` switched to GCC constructor init. |
| `libavcodec/h274.c` | `init_slice` mutex hoisted from function-local to file-scope and constructor-initialised. |
| `fftools/resources/resman.c` | `resman` mutex switched to GCC constructor init. |

### Why constructors instead of static initialisers

Caladan's docs explicitly warn against using `PTHREAD_MUTEX_INITIALIZER`.
Inspection of `shim/sync.c:31-35` confirms the reason: the
`mutex_intialized_check` lazy-init fallback is **not race-free**. Two
threads hitting an uninitialised mutex concurrently can both observe
`magic != INIT_MAGIC` and both call `pthread_mutex_init`, with the
second clobbering the first's wait queue.

GCC constructors run before `main()`, i.e. before any uthread exists,
so initialisation is single-threaded by construction — race-free
without needing atomics.

### Why these specific build-time disables

The configure disables above remove every static-initialiser site the
shim cannot lazily resolve, beyond the four core sites patched
explicitly. After the disables, only `codec_mutex`, av_log `mutex`,
h274 `mutex`, and resman `mutex` remain live — all four patched.

## Verification

After `make`:

```
nm ffmpeg_g | grep -E '__wrap_main|__real_main|runtime_init|ff_caladan_once'
# should show: T __wrap_main, T main (= __real_main), T runtime_init, T ff_caladan_once

nm ffmpeg_g | grep -E ' (socket|connect|pthread_cancel)$' || echo OK
# should print OK (no forbidden symbols)

objdump -j .init_array -s ffmpeg_g | head
# should list .init_array entries including pointers to ff_init_codec_mutex,
# ff_init_av_log_mutex, ff_init_h274_slice_mutex, ff_init_resman_mutex
```

Smoke test (no iokernel — verifies the wrap path):

```
CALADAN_CONFIG=compat/caladan/sample.config ./ffmpeg_g \
    -hide_banner -f lavfi -i testsrc=size=320x240:rate=15 -t 1 \
    -pix_fmt rgb32 -f null -
```

Expected output:

```
ffmpeg-caladan: runtime_init(compat/caladan/sample.config)
CPU 00| <2> control_setup: failed to map iokernel info region
CPU 00| <2> Please make sure IOKernel is running
ffmpeg-caladan: runtime_init failed: -1
```

That is the correct failure mode without `iokerneld` running — it
proves the wrap path and config loading work end-to-end.

## Out of scope

- Async file I/O. Blocking `read`/`write`/`open` syscalls are accepted;
  for short transcodes the kthread blocking is tolerable.
- Network protocol porting. We disable network entirely rather than
  shimming `socket()`.
- Hardware acceleration paths (VAAPI, CUDA, NVENC). Software path only.
- Thread naming, CPU affinity, `pthread_cancel`. The shim silently
  no-ops these and FFmpeg ignores the return.
- Shimming `clock_gettime` / `gettimeofday`. On glibc 2.17+ they are
  vDSO-resolved (no syscall, no internal lock); no Caladan wrapper
  needed.
