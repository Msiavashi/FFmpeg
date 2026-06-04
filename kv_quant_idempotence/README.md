# KV-Cache Quantization Idempotence & Determinism Study

Empirical test of whether KV-cache quantize/dequantize methods are deterministic
and idempotent under recurrent (spill/reload) requantization.

## Layout
- `src/quantizers.py`   — affine/symmetric/power-of-two INT, FP8, MXFP4, VQ/PQ
- `src/kivi_adapter.py` — KIVI repo algorithm (pure-torch path) adapter
- `src/harness.py`      — recurrent N-cycle driver + metrics + classification
- `src/run_kivi.py`     — KIVI runs
- `src/plots_report.py` — plots + summary.csv
- `src/make_report.py`  — assembles results/summary_table.md
- `results/`            — results.json, summary.csv, kivi_results.json
- `plots/`              — drift plots
- `REPORT.md`           — final findings + recommendation

## Reproduce
```
python -m venv env && . env/bin/activate
pip install numpy torch matplotlib   # CPU
N=1000 python src/harness.py
python src/run_kivi.py
python src/plots_report.py && python src/make_report.py
```
Environment used: CPU-only, no GPU. CUDA/Triton kernels untestable here; see REPORT.md.
