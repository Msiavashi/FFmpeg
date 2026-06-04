# Empirical Idempotence & Determinism of KV-Cache Quantization

## Environment

| field | value |
|---|---|
| os | Linux-6.18.5-x86_64-with-glibc2.39 |
| python | 3.11.15 |
| torch | 2.12.0+cu130 |
| cuda_available | False |
| device | cpu |
| num_threads | 2 |
| deterministic_algorithms | True |
| GPU | none (CPU-only container) |
| recurrent cycles N | 1000 (200 for seq=2048) |

## Summary table (aggregated over distributions & shapes)

| method | deterministic | frozen-meta result | recomputed-meta result | worst final L_inf (recomp) |
|---|---|---|---|---|
| `affine_int8` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×11, NUMERICALLY-STABLE×1 | 7.63e-06 |
| `symmetric_int8` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `pot_int8` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `affine_int4` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×10, NUMERICALLY-STABLE×2 | 7.63e-06 |
| `symmetric_int4` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `pot_int4` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `affine_int2` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `symmetric_int2` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `pot_int2` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `fp8_e4m3` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `fp8_e5m2` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `mxfp4_b32` | yes | PASS-WITH-FROZEN-METADATA×12 | PASS×12 | 0.00e+00 |
| `vq_d4_k256` | yes | PASS-WITH-FROZEN-METADATA×8 | FAIL-DRIFT×8 | 3.63e+01 |
| `pq_d2_k16` | yes | PASS-WITH-FROZEN-METADATA×8 | FAIL-DRIFT×7, NUMERICALLY-STABLE×1 | 5.04e+00 |

### KIVI (real repo algorithm, via adapter)

| config | mode | classification | final L_inf | code-change % | max attn drift |
|---|---|---|---|---|---|
| KIVI-int2 normal/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int2 normal/recomp | recomputed | PASS | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int2 outlier/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int2 outlier/recomp | recomputed | PASS | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int2 heavy/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int2 heavy/recomp | recomputed | PASS | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int4 normal/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int4 normal/recomp | recomputed | PASS | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int4 outlier/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int4 outlier/recomp | recomputed | NUMERICALLY-STABLE | 2.86e-06 | 0.000 | 1.53e-05 |
| KIVI-int4 heavy/frozen | frozen | PASS-WITH-FROZEN-METADATA | 0.00e+00 | 0.000 | 0.00e+00 |
| KIVI-int4 heavy/recomp | recomputed | NUMERICALLY-STABLE | 7.63e-06 | 0.000 | 9.54e-06 |

Full per-case data: `results/results.json`, `results/summary.csv`.
