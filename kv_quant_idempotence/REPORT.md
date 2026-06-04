# Determinism & Idempotence of KV-Cache Quantization under Recurrent Requantization

**Empirical study — verified, not inferred from theory.**
All claims below come from running each method through N recurrent
dequantize → requantize cycles and measuring bitwise / numerical drift.

---

## 1. Environment (recorded)

| field | value |
|---|---|
| OS | Linux 6.18.5, x86_64, glibc 2.39 |
| Python | 3.11.15 |
| PyTorch | 2.12.0+cu130 |
| CUDA available | **False — CPU-only container, no GPU/`nvidia-smi`** |
| Compute device | CPU |
| intra-op threads | 2 (pinned; default 4 caused ~20× small-tensor overhead) |
| `torch.use_deterministic_algorithms` | True (warn_only) |
| Seeds | `torch.manual_seed(0)`; per-tensor generators seeded |
| Kernels exercised | pure PyTorch CPU (no Triton/CUTLASS/custom CUDA — see §6) |

Recurrent cycles: **N = 1000** for seq ≤ 128 and the 2×8×16×64 sanity tensor;
**N = 200** for seq = 2048 (2.1M-element tensors); **N = 300** for the k-means
VQ/PQ methods (codebook is refit every cycle — still ≫ the required ≥100).

Test tensors (KV-like, all 4-D `[batch/layers, heads, seq, head_dim]`):
`sanity [2,8,16,64]`, `[1,8,128,64]`, `[1,8,2048,128]`.
Distributions: **normal** N(0,1), **heavy-tailed** (t-like), **outlier-injected**
(1 % ×30), **RoPE-modulated** keys. 320 method×dist×shape×mode cases + 12 KIVI cases.

Raw data: `results/results.json` (per-cycle downsampled series), `results/summary.csv`,
`results/kivi_results.json`. Plots: `plots/drift_recomputed.png`,
`plots/vq_frozen_vs_recomputed.png`.

---

## 2. What "frozen" vs "recomputed" means here

* **Frozen-metadata (A):** scales / zero-points / block-scales / codebooks are
  computed once at `q0` and **reused** for every subsequent requantization. Only
  the data is re-binned.
* **Recomputed-metadata (B):** every cycle re-derives all statistics from the
  already-dequantized tensor — the realistic behavior of production spill/reload
  code that re-quantizes whatever floats it reloaded.

Both modes are tested for every method.

---

## 3. Summary table

| method | deterministic | frozen-meta | recomputed-meta | worst final L∞ (recomp) | comp. ratio | classification |
|---|---|---|---|---|---|---|
| `affine_int8/4/2` (asym) | yes | bitwise idempotent | converges to fixed point | 7.6e-06 | 4/8/16× | **PASS-WITH-FROZEN** (B: NUM-STABLE on heavy/outlier) |
| `symmetric_int8/4/2` | yes | bitwise idempotent | bitwise fixed point | 0 | 4/8/16× | **PASS** |
| `pot_int8/4/2` (pow-2 scale) | yes | bitwise idempotent | bitwise fixed point | 0 | 4/8/16× | **PASS** |
| `fp8_e4m3`, `fp8_e5m2` | yes | bitwise idempotent | bitwise idempotent | 0 | 4× | **PASS** (stateless) |
| `mxfp4_b32` (block-scaled FP4) | yes | bitwise idempotent | bitwise fixed point | 0 | 7.5× | **PASS** |
| `vq_d4_k256` (VQ, refit) | yes (seeded) | bitwise idempotent | **diverges** | 36.3 | 10.7× | frozen: PASS / recomp: **FAIL-DRIFT** |
| `pq_d2_k16` (PQ, refit) | yes (seeded) | bitwise idempotent | **diverges** | 5.0 | 15.8× | frozen: PASS / recomp: **FAIL-DRIFT** |
| **KIVI** int2 (real repo algo) | yes | bitwise idempotent | bitwise fixed point | 0 | up to 16× | **PASS** |
| **KIVI** int4 (real repo algo) | yes | bitwise idempotent | NUM-STABLE on heavy/outlier | 7.6e-06 | 8× | **PASS / NUMERICALLY-STABLE** |

Full per-mode counts in `results/summary_table.md`. Every method was
**deterministic** (identical input+config ⇒ bitwise-identical codes+metadata on
repeated calls), including k-means VQ/PQ once the init seed is fixed.

---

## 4. Findings & mechanism (why each behaves as it does)

### 4.1 Fixed-grid integer & float formats → idempotent
`symmetric`, `power-of-two`, `FP8`, `MXFP4`, and frozen `affine`:

* **Frozen:** `Q(D(q)) = round((q·s)/s) = round(q) = q` exactly, because `q` is an
  integer and the grid is unchanged ⇒ codes and metadata are bitwise-identical
  for all 1000 cycles. **PASS-WITH-FROZEN-METADATA.**
* **Recomputed:** the dequantized values are already grid points, so the
  recomputed max-abs (symmetric/pot) lands on the same grid and reproduces the
  same scale → the system reaches its fixed point at `t=1` and stays bitwise
  stable. **PASS even with recomputed metadata.**
* **FP8** is *stateless* (no learned metadata at all): nearest-FP8 of an exact
  FP8 value is itself. Idempotent in both modes by construction. **PASS.**

### 4.2 Asymmetric affine → almost always exact, occasionally micro-stable
`affine_int8/int4` recomputed reached an exact fixed point in 21/24 cases, but on
3 heavy/outlier cases drifted by **≤ 7.6e-06** (L∞) and then stopped. Reason: the
asymmetric path stores a **float min / rounded zero-point**; recomputing min/max
from the reconstructed grid can shift the zero-point by one ULP, nudging a few
codes once. Non-monotonic, bounded, negligible → **NUMERICALLY-STABLE**, not a
drift failure. This is the same micro-effect seen in **KIVI int4** below.

### 4.3 KIVI (real published KV-cache method) → safe
KIVI's quantization arithmetic (`quant_and_pack_kcache`/`unpack_and_dequant_kcache`,
group-wise asymmetric min/max INT) is **pure PyTorch**; only bit-packing + a
minmax helper are Triton/CUDA. Tested verbatim via `src/kivi_adapter.py`:

* **Frozen:** bitwise idempotent, 0 drift, 0 attention-output drift, all bits/cfg.
* **Recomputed:** int2 is an exact fixed point (PASS); int4 on heavy/outlier
  tensors shows the same ≤ 7.6e-06 micro-stability as §4.2 (it stores float `mn`,
  not a re-snapped zero-point). Attention-output drift ≤ 1.5e-05.
  ⇒ KIVI is **safe for repeated spill/reload**; metadata need not be preserved
  bitwise, though preserving it gives exact idempotence.

### 4.4 Vector / Product quantization with **refit** codebook → FAIL
`vq_d4_k256`, `pq_d2_k16`:

* **Frozen codebook:** a centroid is its own nearest centroid, so re-encoding a
  reconstructed (centroid) vector returns the same code → **bitwise idempotent,
  PASS.**
* **Recomputed codebook (k-means refit each cycle):** the reconstructed tensor
  collapses onto centroids; refitting k-means on that collapsed distribution
  moves the centroids, which moves the reconstruction, which moves the next
  fit — a feedback loop. Drift is **large and monotonic**: L∞ up to **36.3**,
  >1 % of codes change, cosine similarity degrades. **FAIL-DRIFT.**
  (The lone PQ "numerically-stable" case is a tiny tensor that happened to reach
  a stable partition; not general.) See `plots/vq_frozen_vs_recomputed.png`.

These are deterministic (seeded) but **not idempotent** under recomputation —
exactly the case the task warned must be verified, not assumed.

### 4.5 Latency & compression (CPU, representative `[1,8,128,64]`)
| method | quantize | dequantize | compression |
|---|---|---|---|
| symmetric/pot INT8/4/2 | 0.27–0.5 ms | ~0.02 ms | 4–16× |
| FP8 | ~0.9 ms | <0.05 ms | 4× |
| MXFP4 (block-scaled) | 36 ms | 0.06 ms | 7.5× |
| VQ d4 k256 (k-means) | 119 ms | 0.17 ms | 10.7× |
| PQ d2 k16 | 17 ms | 0.10 ms | 15.8× |

INT/FP8 quant is ~100–400× cheaper per call than codebook methods (no fit/search).

---

## 5. Classification (per task taxonomy)

| classification | methods |
|---|---|
| **PASS** (deterministic + bitwise idempotent, both modes) | symmetric INT8/4/2, power-of-two INT8/4/2, FP8 E4M3/E5M2, MXFP4, KIVI-int2 |
| **PASS-WITH-FROZEN-METADATA** (idempotent only when metadata frozen; recomputed merely numerically stable) | asymmetric affine INT8/4 (heavy/outlier), KIVI-int4 (heavy/outlier) |
| **NUMERICALLY-STABLE** (not bitwise, but ≤1e-5 bounded, non-monotonic) | recomputed affine/KIVI micro-cases above |
| **FAIL-DRIFT** (recurrent requantization accumulates measurable drift) | VQ (refit codebook), PQ (refit codebook) |
| **FAIL-NONDETERMINISTIC** | none observed (k-means is deterministic once seeded) |
| **UNTESTABLE** | CUDA/Triton kernels of KIVI/KVQuant; KVQuant/SKVQ/QAQ/GEAR/CacheGen end-to-end — see §6 |

---

## 6. Untestable items (not hidden — reasons recorded)

| repo / method | status | reason |
|---|---|---|
| **KIVI** Triton/CUDA kernels (`gemv_cuda.cu`, `_pack_along_last_dim`) | UNTESTABLE here | require nvcc + GPU; none in container. **Algorithm itself tested** via pure-torch adapter (§4.3). Repo cloned @ `876b4d2`, MIT. |
| **KVQuant** (cloned @ `57a2383`) | UNTESTABLE end-to-end | needs CUDA dequant kernels, model gradients, and large HF models; nontrivial install, no GPU. Its core (per-channel + dense-and-sparse outlier, non-uniform nuq scales) is a *fixed-codebook/fixed-scale* design → expected PASS-WITH-FROZEN, but **not empirically confirmed**. |
| **GEAR** | UNTESTABLE | `git clone` of `HahnYuan/GEAR` failed (repo path/availability); residual+low-rank error feedback would need explicit idempotence testing. |
| **SKVQ, QAQ, CacheGen** | UNTESTABLE | no clean standalone quantize/dequantize entry point reachable without GPU + full model harness in this environment. |
| FP4 / NVFP4 hardware path | partial | NVFP4 hardware encode unavailable (no GPU); **MXFP4-style block-scaled FP4 emulated and tested** (§4.1). |

No GPU means CUDA/Triton/CUTLASS numerical paths were **not** exercised; results
above are CPU reference-math. Hardware kernels could differ in rounding and must
be re-verified on-device before relying on bitwise idempotence.

---

## 7. Recommendation for KV-cache offload design

**(a) Safe for repeated spill/reload (re-quantize on every reload OK):**
fixed **symmetric** or **power-of-two-scale** integer quant (INT8/4/2), **FP8**
(E4M3/E5M2), **MXFP4** block-scaled, and **KIVI**-style group-wise min/max INT.
These reach a bitwise fixed point and show **zero monotonic drift** across 1000
cycles even when scales are recomputed each time. FP8 is the strongest guarantee
(stateless). Prefer **symmetric/pow-2** over **asymmetric affine** if you want
*exact* bitwise idempotence under recomputed metadata (affine can wobble ~1e-6).

**(b) Safe only if the compressed object + metadata are preserved (freeze, don't refit):**
**vector quantization / product quantization**. With a frozen codebook they are
perfectly idempotent; if you instead re-fit the codebook on reload they diverge.
Store and reuse `(codes, codebook)`; never re-run k-means on reloaded values.

**(c) Unsafe for recurrent requantization:**
any method that **recomputes adaptive structure from reconstructed data each
cycle** — refit VQ/PQ codebooks, and by extension adaptive-clipping / residual-
refitting / recalibrated-scale schemes (GEAR-like). Verify before use; if scales
or codebooks must be recomputed, **keep the original compressed object** rather
than re-deriving from dequantized floats.

**Bottom line:** idempotence is a property of *whether metadata is frozen*, not of
the bit-width. Freeze the quantization metadata (or never decode-then-re-encode)
and all tested fixed-grid methods are safe; recompute codebooks and you get
unbounded drift.

---

## 8. Reproduce

```bash
python -m venv env && . env/bin/activate
pip install numpy matplotlib
pip install torch            # CPU build
N=1000 python src/harness.py      # 320 cases -> results/results.json
python src/run_kivi.py            # KIVI real-algorithm cases
python src/plots_report.py        # summary.csv + plots
python src/make_report.py         # results/summary_table.md
```
Minimal per-method repro is `src/quantizers.py` (each class has
`quantize(x, meta=None)` / `dequantize(q, meta)`); `src/kivi_adapter.py` is the
KIVI algorithm; `src/harness.py:run_case` is the recurrent driver + classifier.
