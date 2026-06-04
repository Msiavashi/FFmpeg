"""Assemble REPORT.md from results.json + kivi_results.json."""
import json, os, collections

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "results")
d = json.load(open(os.path.join(R, "results.json")))
kivi = {}
kp = os.path.join(R, "kivi_results.json")
if os.path.exists(kp):
    kivi = json.load(open(kp))
cases = d["cases"]


def agg(method):
    """aggregate classification across dists/shapes for a method, per mode."""
    out = {"frozen": collections.Counter(), "recomputed": collections.Counter()}
    worst = {"frozen": 0.0, "recomputed": 0.0}
    det = True
    for k, v in cases.items():
        if k.split("|")[0] != method:
            continue
        mode = k.split("|")[-1]
        if mode not in out:
            continue
        out[mode][v.get("classification")] += 1
        if v.get("final_linf") is not None:
            worst[mode] = max(worst[mode], v["final_linf"])
        if v.get("deterministic") is False:
            det = False
    return out, worst, det


methods = []
for k in cases:
    m = k.split("|")[0]
    if m not in methods:
        methods.append(m)

lines = []
lines.append("# Empirical Idempotence & Determinism of KV-Cache Quantization\n")
e = d["env"]
lines.append("## Environment\n")
lines.append("| field | value |\n|---|---|")
for f in ("os", "python", "torch", "cuda_available", "device", "num_threads",
          "deterministic_algorithms"):
    lines.append(f"| {f} | {e.get(f)} |")
lines.append(f"| GPU | none (CPU-only container) |")
lines.append(f"| recurrent cycles N | {d['N']} (200 for seq=2048) |\n")

lines.append("## Summary table (aggregated over distributions & shapes)\n")
lines.append("| method | deterministic | frozen-meta result | recomputed-meta result | worst final L_inf (recomp) |")
lines.append("|---|---|---|---|---|")


def fmt(counter):
    if not counter:
        return "-"
    return ", ".join(f"{k}×{n}" for k, n in counter.most_common())


for m in methods:
    out, worst, det = agg(m)
    lines.append(f"| `{m}` | {'yes' if det else 'NO'} | {fmt(out['frozen'])} | "
                 f"{fmt(out['recomputed'])} | {worst['recomputed']:.2e} |")

# KIVI rows
if kivi:
    lines.append("\n### KIVI (real repo algorithm, via adapter)\n")
    lines.append("| config | mode | classification | final L_inf | code-change % | max attn drift |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in kivi.items():
        lines.append(f"| {k} | {v['mode']} | {v['classification']} | {v['final_linf']:.2e} | "
                     f"{v['final_code_change_pct']:.3f} | {v.get('attn',0):.2e} |")

lines.append("\nFull per-case data: `results/results.json`, `results/summary.csv`.\n")
open(os.path.join(R, "summary_table.md"), "w").write("\n".join(lines))
print("wrote results/summary_table.md")
