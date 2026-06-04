import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import torch
from kivi_adapter import KIVI
from harness import make_tensor, run_case
torch.manual_seed(0)

out = {}
for bits in (2, 4):
    for dist in ("normal", "outlier", "heavy"):
        x = make_tensor(dist, (1, 8, 128, 64), seed=1)
        for frozen in (True, False):
            m = KIVI(bits=bits, group_size=32, axis=-2)
            r = run_case(m, x, 1000, frozen, attn=True)
            attn = max(r["series"]["attn"]) if r["series"].get("attn") else 0.0
            key = f"KIVI-int{bits} {dist}"
            out[key + ("/frozen" if frozen else "/recomp")] = {
                "mode": "frozen" if frozen else "recomputed",
                "classification": r["classification"],
                "final_linf": r["final_linf"],
                "final_code_change_pct": r["final_code_change_pct"],
                "deterministic": r["deterministic"],
                "attn": attn,
            }
            print(key, "frozen" if frozen else "recomp", r["classification"],
                  f"linf={r['final_linf']:.2e} attn={attn:.2e}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "kivi_results.json"), "w"), indent=1)
print("WROTE kivi_results.json")
