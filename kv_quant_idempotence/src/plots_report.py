"""Generate drift plots (matplotlib if available, else ASCII) and a summary
table CSV from results.json."""
import json, os, sys

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results", "results.json")
PLOTS = os.path.join(HERE, "..", "plots")
os.makedirs(PLOTS, exist_ok=True)

d = json.load(open(RES))
cases = d["cases"]

# ---- summary CSV ----
rows = [("method", "dist", "shape", "mode", "classification", "deterministic",
         "all_codes_eq_q0", "final_linf", "final_code_change_pct",
         "min_cos", "compression_ratio", "lat_q_ms", "lat_d_ms")]
for k, v in cases.items():
    parts = k.split("|")
    while len(parts) < 4:
        parts.append("")
    rows.append((parts[0], parts[1], parts[2], parts[3],
                 v.get("classification", ""), v.get("deterministic", ""),
                 v.get("all_codes_eq_q0", ""), v.get("final_linf", ""),
                 v.get("final_code_change_pct", ""), v.get("min_cos", ""),
                 v.get("compression_ratio", ""), v.get("latency_q_ms", ""),
                 v.get("latency_d_ms", "")))
with open(os.path.join(HERE, "..", "results", "summary.csv"), "w") as f:
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print("wrote summary.csv", len(rows) - 1, "rows")

# ---- plots ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False
    print("matplotlib unavailable:", e)

# representative drift plots: recomputed mode, normal dist, s128, key methods
sel = [k for k in cases if k.endswith("recomputed") and "normal" in k and "s128" in k]
if HAVE_MPL and sel:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k in sel:
        s = cases[k].get("series_ds")
        if not s:
            continue
        m = k.split("|")[0]
        xs = list(range(len(s["linf"])))
        ax[0].plot(xs, s["linf"], label=m)
        ax[1].plot(xs, s["code_changes"], label=m)
    ax[0].set_title("L_inf drift vs recurrent cycle (recomputed meta, normal, s128)")
    ax[0].set_xlabel("cycle (downsampled)"); ax[0].set_ylabel("max|x_t_hat - x0_hat|")
    ax[0].set_yscale("symlog", linthresh=1e-9)
    ax[1].set_title("changed quantized codes vs cycle")
    ax[1].set_xlabel("cycle (downsampled)"); ax[1].set_ylabel("# changed codes")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "drift_recomputed.png"), dpi=110)
    print("wrote plots/drift_recomputed.png")

    # frozen vs recomputed for VQ (the interesting failure)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("frozen", "recomputed"):
        k = f"vq_d4_k256|normal|llm_h8_s128_d64|{mode}"
        if k in cases and cases[k].get("series_ds"):
            s = cases[k]["series_ds"]
            ax.plot(s["linf"], label=f"VQ {mode}")
    ax.set_title("VQ: frozen vs recomputed codebook L_inf drift")
    ax.set_xlabel("cycle (downsampled)"); ax.set_ylabel("L_inf"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "vq_frozen_vs_recomputed.png"), dpi=110)
    print("wrote plots/vq_frozen_vs_recomputed.png")
else:
    # ASCII fallback summary of drift
    for k in sel:
        s = cases[k].get("series_ds", {})
        if s:
            print(k, "linf last:", s["linf"][-1])
