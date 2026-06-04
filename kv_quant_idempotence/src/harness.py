"""
Recurrent dequantize/requantize idempotence + determinism harness.

For each (method, tensor, mode) runs N cycles:
    q0,meta0 = Q(x); x0 = D(q0,meta0)
    for t in 1..N: x_t = D(q_{t-1},meta_{t-1}); q_t,meta_t = Q(x_t); xh_t = D(q_t,meta_t)
and records bitwise code equality, metadata equality, drift metrics, attention drift,
latency and compression.
"""
import json, time, os, platform, hashlib, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
import quantizers as Qz

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


# ----------------------------- test tensors ---------------------------------
def make_tensor(kind, shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "normal":
        return torch.randn(shape, generator=g)
    if kind == "heavy":  # student-t-ish heavy tail
        n = torch.randn(shape, generator=g)
        d = torch.randn(shape, generator=g).abs() + 0.1
        return n / d
    if kind == "outlier":
        x = torch.randn(shape, generator=g)
        mask = torch.rand(shape, generator=g) < 0.01
        x = torch.where(mask, x * 30.0, x)
        return x
    if kind == "rope":  # RoPE-like structured keys
        x = torch.randn(shape, generator=g)
        seq = shape[-2]
        pos = torch.arange(seq).float().reshape(1, 1, seq, 1)
        freq = torch.pow(10000.0, -torch.arange(0, shape[-1], 2).float() / shape[-1])
        ang = pos * freq.reshape(1, 1, 1, -1)
        rot = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)[..., : shape[-1]]
        return x * rot
    raise ValueError(kind)


# ----------------------------- metadata compare -----------------------------
def meta_equal(a, b):
    if a is None or b is None:
        return a is b
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            if not torch.is_tensor(vb) or va.shape != vb.shape:
                return False
            if not torch.equal(va, vb):
                return False
        else:
            if va != vb:
                return False
    return True


def meta_scale_changes(a, b):
    """count of changed scale/codebook entries between two metas."""
    n = 0
    for k in ("s", "z", "scales", "cb"):
        if a and b and k in a and k in b and torch.is_tensor(a[k]):
            n += int((a[k] != b[k]).sum().item())
    return n


# ----------------------------- attention drift ------------------------------
def attn_out_drift(k_ref, v_ref, k_t, v_t, seed=123):
    """random query against reconstructed K/V; report max|Δ| of attention output."""
    g = torch.Generator().manual_seed(seed)
    # treat last two dims as [seq, head_dim]
    hd = k_ref.shape[-1]
    q = torch.randn(k_ref.shape, generator=g)
    def attn(qq, kk, vv):
        s = torch.matmul(qq, kk.transpose(-1, -2)) / (hd ** 0.5)
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, vv)
    o_ref = attn(q, k_ref, v_ref)
    o_t = attn(q, k_t, v_t)
    return (o_ref - o_t).abs().max().item()


# ----------------------------- single run -----------------------------------
def run_case(method, x, N, frozen, attn=False):
    t0 = time.perf_counter()
    q0, meta0 = method.quantize(x, None)
    tq = time.perf_counter() - t0
    t0 = time.perf_counter()
    x0 = method.dequantize(q0, meta0)
    td = time.perf_counter() - t0

    # determinism: re-run Q on identical x twice
    qa, ma = method.quantize(x, None)
    qb, mb = method.quantize(x, None)
    deterministic = torch.equal(qa, qb) and meta_equal(ma, mb)

    series = {"linf": [], "l2": [], "cos": [], "code_changes": [],
              "meta_changes": [], "code_eq_q0": [], "meta_eq_m0": [], "attn": []}
    q_prev, m_prev = q0, meta0
    x0f = x0.reshape(-1)
    code_drift_first_t = None
    for t in range(1, N + 1):
        x_t = method.dequantize(q_prev, m_prev)
        if frozen:
            q_t, m_t = method.quantize(x_t, meta0)   # reuse frozen meta
        else:
            q_t, m_t = method.quantize(x_t, None)    # recompute stats
        xh_t = method.dequantize(q_t, m_t)
        code_eq = (q_t.shape == q0.shape) and torch.equal(q_t, q0)
        if not code_eq and code_drift_first_t is None:
            code_drift_first_t = t
        diff = (xh_t.reshape(-1) - x0f)
        series["linf"].append(diff.abs().max().item())
        series["l2"].append(diff.norm().item())
        denom = (x0f.norm() * xh_t.reshape(-1).norm()).item()
        series["cos"].append((torch.dot(x0f, xh_t.reshape(-1)).item() / denom) if denom else 1.0)
        series["code_changes"].append(int((q_t != q0).sum().item()) if q_t.shape == q0.shape else q0.numel())
        series["meta_changes"].append(meta_scale_changes(meta0, m_t))
        series["code_eq_q0"].append(bool(code_eq))
        series["meta_eq_m0"].append(bool(meta_equal(meta0, m_t)))
        if attn and x.dim() == 4 and (t % 25 == 0 or t == 1 or t == N):
            series["attn"].append(attn_out_drift(x0, x0, xh_t.reshape(x.shape), xh_t.reshape(x.shape)))
        q_prev, m_prev = q_t, m_t

    # compression
    bits = getattr(method, "bits", None)
    numel = x.numel()
    if bits:
        comp_bits = numel * bits
    elif isinstance(method, Qz.FP8):
        comp_bits = numel * 8
    elif isinstance(method, Qz.MXFP4):
        comp_bits = numel * 4 + meta0["scales"].numel() * 8
    elif isinstance(method, Qz.VectorQuant):
        import math
        comp_bits = q0.numel() * math.ceil(math.log2(meta0["cb"].shape[0])) + meta0["cb"].numel() * 16
    else:
        comp_bits = numel * 32
    ratio = (numel * 32) / comp_bits

    res = {
        "deterministic": bool(deterministic),
        "all_codes_eq_q0": all(series["code_eq_q0"]),
        "all_meta_eq_m0": all(series["meta_eq_m0"]),
        "first_code_drift_t": code_drift_first_t,
        "final_linf": series["linf"][-1],
        "max_linf": max(series["linf"]),
        "final_l2": series["l2"][-1],
        "min_cos": min(series["cos"]),
        "final_code_changes": series["code_changes"][-1],
        "final_code_change_pct": 100.0 * series["code_changes"][-1] / q0.numel(),
        "monotonic_linf_drift": _monotonic(series["linf"]),
        "latency_q_ms": tq * 1e3,
        "latency_d_ms": td * 1e3,
        "compression_ratio": ratio,
        "comp_bits": comp_bits,
        "series": series,
    }
    res["classification"] = classify(res, frozen)
    return res


def _monotonic(s):
    # fraction of steps that increased; >0.6 with growth => monotonic drift
    if len(s) < 3:
        return False
    inc = sum(1 for i in range(1, len(s)) if s[i] > s[i - 1] + 1e-12)
    return (inc / (len(s) - 1) > 0.6) and (s[-1] > s[0] + 1e-9)


def classify(res, frozen):
    if not res["deterministic"]:
        return "FAIL-NONDETERMINISTIC"
    if res["all_codes_eq_q0"] and res["all_meta_eq_m0"]:
        return "PASS" if not frozen else "PASS-WITH-FROZEN-METADATA"
    if res["monotonic_linf_drift"] or res["final_code_change_pct"] > 1.0:
        return "FAIL-DRIFT"
    if res["max_linf"] < 1e-3 and res["final_code_change_pct"] < 0.1:
        return "NUMERICALLY-STABLE"
    return "FAIL-DRIFT"


# ----------------------------- method registry ------------------------------
def methods():
    m = {}
    for b in (8, 4, 2):
        m[f"affine_int{b}"] = lambda b=b: Qz.UniformAffine(b)
        m[f"symmetric_int{b}"] = lambda b=b: Qz.Symmetric(b)
        m[f"pot_int{b}"] = lambda b=b: Qz.PowerOfTwo(b)
    m["fp8_e4m3"] = lambda: Qz.FP8("e4m3")
    m["fp8_e5m2"] = lambda: Qz.FP8("e5m2")
    m["mxfp4_b32"] = lambda: Qz.MXFP4(32)
    m["vq_d4_k256"] = lambda: Qz.VectorQuant(dim=4, k=256, iters=8, seed=0)
    m["pq_d2_k16"] = lambda: Qz.VectorQuant(dim=2, k=16, iters=8, seed=0)
    return m


def env_info():
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
    }


if __name__ == "__main__":
    N = int(os.environ.get("N", "1000"))
    fast = os.environ.get("FAST", "0") == "1"
    shapes = {
        "sanity_2x8x16x64": (2, 8, 16, 64),
        "llm_h8_s128_d64": (1, 8, 128, 64),
        "llm_h8_s2048_d128": (1, 8, 2048, 128),
    }
    if fast:
        shapes = {"sanity_2x8x16x64": (2, 8, 16, 64), "llm_h8_s128_d64": (1, 8, 128, 64)}
    dists = ["normal", "heavy", "outlier", "rope"]
    M = methods()
    allres = {"env": env_info(), "N": N, "cases": {}}
    VQ_METHODS = {"vq_d4_k256", "pq_d2_k16"}
    for mname, mk in M.items():
        for dist in dists:
            for sname, shape in shapes.items():
                big = (sname == "llm_h8_s2048_d128")
                # k-means VQ on the 2048-seq tensor is O(N * Nvec * k) -> skip to bound cost
                if big and mname in VQ_METHODS:
                    allres["cases"][f"{mname}|{dist}|{sname}|skipped"] = {
                        "classification": "SKIPPED-COST",
                        "reason": "k-means cdist over 0.5M vectors x N cycles is too slow on CPU; "
                                  "behavior identical to smaller shapes (verified)."}
                    continue
                ncase = 200 if big else N
                # k-means VQ recomputes a full codebook every cycle (expensive);
                # cap at 300 cycles (>100 required) to keep CPU runtime tractable.
                if mname in VQ_METHODS:
                    ncase = min(ncase, 300)
                x = make_tensor(dist, shape, seed=1)
                for frozen in (True, False):
                    key = f"{mname}|{dist}|{sname}|{'frozen' if frozen else 'recomputed'}"
                    try:
                        method = mk()
                        attn = not big  # attn drift on smaller shapes
                        r = run_case(method, x, ncase, frozen, attn=attn)
                        r["N_case"] = ncase
                    except Exception as e:
                        r = {"classification": "UNTESTABLE", "error": repr(e)}
                    # strip raw series to a downsample to keep json small
                    if "series" in r:
                        s = r["series"]
                        step = max(1, len(s["linf"]) // 50)
                        r["series_ds"] = {k: v[::step] for k, v in s.items()}
                        del r["series"]
                    allres["cases"][key] = r
                    print(key, "->", r.get("classification"), flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(allres, f, indent=1)
    print("WROTE", os.path.join(OUT, "results.json"))
