"""
Quantizer / Dequantizer implementations for KV-cache idempotence testing.

Every method exposes:
    quantize(x, meta=None)   -> (q, meta)     # if meta given => FROZEN mode (reuse stats)
    dequantize(q, meta)      -> x_hat

Contract:
    * `meta` is a plain dict of tensors / python scalars (so it can be compared
      bitwise / structurally and serialized).
    * In FROZEN mode the caller passes the *previous* meta back in; the method
      MUST reuse all stats (scale / zero-point / codebook / block-scales) and
      only re-bin the data. In RECOMPUTED mode meta is None and the method
      derives fresh stats from the (already-dequantized) input.

All math is done on CPU torch tensors with fixed dtype paths. No randomness
except codebook init, which is seeded.
"""
import torch

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _qmin_qmax(bits, signed):
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def _round_half_to_even(x):
    # torch.round is round-half-to-even (banker's), deterministic on CPU.
    return torch.round(x)


# ----------------------------------------------------------------------------
# 1. Uniform AFFINE (asymmetric) integer quantization
# ----------------------------------------------------------------------------
class UniformAffine:
    """q = clamp(round(x/s) + z).  Per-tensor stats (min/max)."""
    def __init__(self, bits, per_channel_dim=None):
        self.bits = bits
        self.qmin, self.qmax = _qmin_qmax(bits, signed=False)
        self.pcd = per_channel_dim  # axis to keep (per-channel) or None=per-tensor

    def _stats(self, x):
        if self.pcd is None:
            xmin = x.min()
            xmax = x.max()
        else:
            dims = [d for d in range(x.dim()) if d != self.pcd]
            xmin = x.amin(dim=dims, keepdim=True)
            xmax = x.amax(dim=dims, keepdim=True)
        xmin = torch.minimum(xmin, torch.zeros_like(xmin))
        xmax = torch.maximum(xmax, torch.zeros_like(xmax))
        s = (xmax - xmin) / (self.qmax - self.qmin)
        s = torch.where(s == 0, torch.ones_like(s), s)
        z = _round_half_to_even(self.qmin - xmin / s)
        return s, z

    def quantize(self, x, meta=None):
        if meta is None:
            s, z = self._stats(x)
            meta = {"s": s, "z": z, "bits": self.bits}
        else:
            s, z = meta["s"], meta["z"]
        q = _round_half_to_even(x / s) + z
        q = torch.clamp(q, self.qmin, self.qmax).to(torch.int32)
        return q, meta

    def dequantize(self, q, meta):
        return (q.to(torch.float32) - meta["z"]) * meta["s"]


# ----------------------------------------------------------------------------
# 2. Symmetric integer quantization
# ----------------------------------------------------------------------------
class Symmetric:
    """q = clamp(round(x/s)), z=0.  Scale from max-abs."""
    def __init__(self, bits, per_channel_dim=None):
        self.bits = bits
        self.qmin, self.qmax = _qmin_qmax(bits, signed=True)
        self.pcd = per_channel_dim

    def _stats(self, x):
        if self.pcd is None:
            amax = x.abs().max()
        else:
            dims = [d for d in range(x.dim()) if d != self.pcd]
            amax = x.abs().amax(dim=dims, keepdim=True)
        s = amax / self.qmax
        s = torch.where(s == 0, torch.ones_like(s), s)
        return s

    def quantize(self, x, meta=None):
        if meta is None:
            s = self._stats(x)
            meta = {"s": s, "bits": self.bits}
        else:
            s = meta["s"]
        q = torch.clamp(_round_half_to_even(x / s), self.qmin, self.qmax).to(torch.int32)
        return q, meta

    def dequantize(self, q, meta):
        return q.to(torch.float32) * meta["s"]


# ----------------------------------------------------------------------------
# 3. Power-of-two scale symmetric quantization
# ----------------------------------------------------------------------------
class PowerOfTwo:
    def __init__(self, bits, per_channel_dim=None):
        self.bits = bits
        self.qmin, self.qmax = _qmin_qmax(bits, signed=True)
        self.pcd = per_channel_dim

    def _stats(self, x):
        if self.pcd is None:
            amax = x.abs().max()
        else:
            dims = [d for d in range(x.dim()) if d != self.pcd]
            amax = x.abs().amax(dim=dims, keepdim=True)
        amax = torch.where(amax == 0, torch.ones_like(amax), amax)
        s_real = amax / self.qmax
        exp = torch.ceil(torch.log2(s_real))
        s = torch.pow(2.0, exp)
        return s

    def quantize(self, x, meta=None):
        if meta is None:
            s = self._stats(x)
            meta = {"s": s, "bits": self.bits}
        else:
            s = meta["s"]
        q = torch.clamp(_round_half_to_even(x / s), self.qmin, self.qmax).to(torch.int32)
        return q, meta

    def dequantize(self, q, meta):
        return q.to(torch.float32) * meta["s"]


# ----------------------------------------------------------------------------
# 4. FP8 (E4M3 / E5M2) -- stateless round-to-nearest, no metadata
# ----------------------------------------------------------------------------
class FP8:
    """Emulated FP8 round-to-nearest-even. Stateless (no learned metadata)."""
    def __init__(self, fmt="e4m3"):
        assert fmt in ("e4m3", "e5m2")
        self.fmt = fmt
        if fmt == "e4m3":
            self.ebits, self.mbits, self.bias, self.maxval = 4, 3, 7, 448.0
        else:
            self.ebits, self.mbits, self.bias, self.maxval = 5, 2, 15, 57344.0
        self.emin = 1 - self.bias  # min normal exponent

    def _round(self, x):
        x = x.to(torch.float64)
        sign = torch.sign(x)
        ax = x.abs()
        out = torch.zeros_like(ax)
        nz = ax > 0
        # exponent of each value (clamped to subnormal floor)
        e = torch.floor(torch.log2(torch.where(nz, ax, torch.ones_like(ax))))
        e = torch.clamp(e, min=self.emin)
        scale = torch.pow(2.0, e - self.mbits)
        q = torch.round(ax / scale) * scale   # round-half-to-even mantissa
        q = torch.clamp(q, max=self.maxval)
        out = torch.where(nz, q, out)
        return (sign * out).to(torch.float32)

    def quantize(self, x, meta=None):
        q = self._round(x)
        return q, {"fmt": self.fmt}

    def dequantize(self, q, meta):
        return q.to(torch.float32)


# ----------------------------------------------------------------------------
# 5. FP4 / MXFP4-style block-scaled quantization
# ----------------------------------------------------------------------------
# FP4 E2M1 representable magnitudes: {0,0.5,1,1.5,2,3,4,6}
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


class MXFP4:
    """Block-scaled FP4 (E2M1) with power-of-two (E8M0) shared block scale.
    block_size elements share one 2^k scale derived from block amax."""
    def __init__(self, block_size=32):
        self.block_size = block_size
        self.levels = _FP4_LEVELS

    def _blockify(self, x):
        flat = x.reshape(-1)
        n = flat.numel()
        pad = (-n) % self.block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        return flat.reshape(-1, self.block_size), n, x.shape

    def _block_scales(self, blocks):
        amax = blocks.abs().amax(dim=1, keepdim=True)
        amax = torch.where(amax == 0, torch.ones_like(amax), amax)
        # scale so block max maps near top fp4 level (6.0); power-of-two (MX style)
        exp = torch.floor(torch.log2(amax / 6.0))
        return torch.pow(2.0, exp)

    def _to_levels(self, vals):
        sign = torch.sign(vals)
        a = vals.abs()
        # nearest representable level (round-half-up via midpoints)
        lv = self.levels
        idx = torch.bucketize(a, (lv[1:] + lv[:-1]) / 2.0)
        idx = torch.clamp(idx, 0, len(lv) - 1)
        return sign * lv[idx]

    def quantize(self, x, meta=None):
        blocks, n, shape = self._blockify(x)
        if meta is None:
            scales = self._block_scales(blocks)
            meta = {"scales": scales, "n": n, "shape": tuple(shape), "bs": self.block_size}
        else:
            scales = meta["scales"]
        codes = self._to_levels(blocks / scales)  # representable fp4 magnitudes
        return codes.to(torch.float32), meta

    def dequantize(self, q, meta):
        x = (q * meta["scales"]).reshape(-1)[: meta["n"]]
        return x.reshape(meta["shape"])


# ----------------------------------------------------------------------------
# 6. Fixed-codebook Vector Quantization (and Product Quantization)
# ----------------------------------------------------------------------------
class VectorQuant:
    """k-means codebook over sub-vectors of dimension `dim`.
    FROZEN: reuse codebook from meta. RECOMPUTED: refit (seeded) each call."""
    def __init__(self, dim=4, k=256, iters=10, seed=0, subspaces=1):
        self.dim = dim
        self.k = k
        self.iters = iters
        self.seed = seed
        self.subspaces = subspaces  # >1 => product quantization

    def _vectors(self, x):
        flat = x.reshape(-1)
        n = flat.numel()
        pad = (-n) % self.dim
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        return flat.reshape(-1, self.dim), n, x.shape

    def _kmeans(self, vecs):
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(vecs.shape[0], generator=g)[: self.k]
        cb = vecs[perm].clone()
        if cb.shape[0] < self.k:  # pad codebook if too few vectors
            cb = torch.cat([cb, cb[: self.k - cb.shape[0]]])
        for _ in range(self.iters):
            d = torch.cdist(vecs, cb)
            a = d.argmin(dim=1)
            for j in range(self.k):
                m = a == j
                if m.any():
                    cb[j] = vecs[m].mean(dim=0)
        return cb

    def quantize(self, x, meta=None):
        vecs, n, shape = self._vectors(x)
        if meta is None:
            cb = self._kmeans(vecs)
            meta = {"cb": cb, "n": n, "shape": tuple(shape), "dim": self.dim}
        else:
            cb = meta["cb"]
        d = torch.cdist(vecs, cb)
        codes = d.argmin(dim=1).to(torch.int32)
        return codes, meta

    def dequantize(self, codes, meta):
        cb = meta["cb"]
        vecs = cb[codes.long()]
        flat = vecs.reshape(-1)[: meta["n"]]
        return flat.reshape(meta["shape"])
