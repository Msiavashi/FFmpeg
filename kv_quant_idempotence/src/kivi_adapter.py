"""
Thin adapter exposing KIVI's quant/dequant math (repo: github.com/jy-yuan/KIVI,
commit 876b4d2, MIT). KIVI's quantization arithmetic in quant/new_pack.py
(`quant_and_pack_kcache` / `unpack_and_dequant_kcache`) is pure PyTorch; only the
*bit-packing* and a minmax helper are Triton/CUDA. We replicate the arithmetic
VERBATIM (group-wise asymmetric uniform min/max quant) and skip only packing,
which does not change numerical idempotence. Documented modification: packing
omitted; codes kept as int32. No algorithm change.

KIVI math (from new_pack.py lines 19-27, 65):
    mn = min(data, group axis); mx = max(...)
    scale = (mx-mn)/max_int
    code = clamp(round((data-mn)/scale), 0, max_int)
    dequant: data = code*scale + mn
"""
import torch


class KIVI:
    def __init__(self, bits=2, group_size=32, axis=-2):
        self.bits = bits
        self.group_size = group_size
        self.axis = axis            # KIVI quantizes K per-channel(-2), V per-token(-1)
        self.max_int = 2 ** bits - 1

    def quantize(self, x, meta=None):
        if meta is None:
            mn = torch.amin(x, dim=self.axis, keepdim=True)
            mx = torch.amax(x, dim=self.axis, keepdim=True)
            scale = (mx - mn) / self.max_int
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            meta = {"scale": scale, "mn": mn, "bits": self.bits}
        else:
            scale, mn = meta["scale"], meta["mn"]
        code = torch.clamp(torch.round((x - mn) / scale), 0, self.max_int).to(torch.int32)
        return code, meta

    def dequantize(self, code, meta):
        return code.to(torch.float32) * meta["scale"] + meta["mn"]
