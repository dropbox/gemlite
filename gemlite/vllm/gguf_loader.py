# SPDX-License-Identifier: Apache-2.0
"""GGUF block decoders for gemlite's A16Wn_HQQ_INT path.

Supports affine block types that map onto gemlite's HQQ convention
  W = (W_q - zeros) * scales :

  Legacy (super=32, 1 sub-block):
    Q4_0 -> A16W4 gs=32      Q4_1 -> A16W4 gs=32      Q8_0 -> A16W8 gs=32

  K-quants (super=256):
    Q4_K -> A16W4 gs=32 (8 sub-blocks, 6-bit packed sub-scales)
    Q2_K -> A16W2 gs=16 (16 sub-blocks, 4-bit packed sub-scales)

Q3_K/Q5_K/Q6_K need upcast (not done here); IQ_* are LUT-based and out of scope.
"""

from __future__ import annotations

from typing import Callable

import torch

# GGUF block format constants (ggml-common.h: block_qX / GGML_QUANT_SIZES).
_Q4_0_TYPESIZE = 18   # 2B d + 16B qs
_Q4_1_TYPESIZE = 20   # 2B d + 2B m + 16B qs
_Q8_0_TYPESIZE = 34   # 2B d + 32B qs(int8)

_QK_K = 256
_Q4_K_TYPESIZE = 144  # 2B d + 2B dmin + 12B sc/mn(6-bit) + 128B qs(4-bit)
_Q2_K_TYPESIZE = 84   # 16B sc/mn(4-bit) + 64B qs(2-bit) + 2B d + 2B dmin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_fp16_le(buf: torch.Tensor) -> torch.Tensor:
    """[..., 2] uint8 LE bytes -> [...] fp16. `.contiguous()` makes the last
    stride 1 byte so `view(float16)` reinterprets each pair as one fp16."""
    return buf.contiguous().view(torch.float16).squeeze(-1)


def _check_blob(blob: torch.Tensor, K: int, type_size: int,
                super_size: int = 32) -> int:
    assert blob.dtype == torch.uint8 and blob.ndim == 2, blob.shape
    assert K % super_size == 0, f"K={K} not divisible by {super_size}"
    G_sb = K // super_size
    expected = G_sb * type_size
    assert blob.shape[1] == expected, (
        f"blob dim1 {blob.shape[1]} != {G_sb}*{type_size}={expected}"
    )
    return G_sb


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """num / den with 0 wherever den == 0."""
    return torch.where(den == 0, torch.zeros_like(num),
                       num / torch.where(den == 0, torch.ones_like(den), den))


def _unpack_nibbles(qs: torch.Tensor, layout: str) -> torch.Tensor:
    """Unpack 4-bit pairs.
      "gguf-legacy": [..., 16] -> [..., 32], byte i = q[i] | (q[i+16] << 4).
                     Output order: [low0..low15, high0..high15].
      "interleaved": [..., B] -> [..., B, 2], pair = [low, high]. Caller
                     reshapes. Used by Q4_K qs.
    """
    if layout == "gguf-legacy":
        return torch.cat([qs & 0x0F, (qs >> 4) & 0x0F], dim=-1)
    if layout == "interleaved":
        return torch.stack([qs & 0x0F, (qs >> 4) & 0x0F], dim=-1)
    raise ValueError(layout)


# ---------------------------------------------------------------------------
# legacy (block=32)
# ---------------------------------------------------------------------------

def decode_q4_0(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q4_0 sym: w = d*(q - 8). HQQ: W_q=q, scales=d, zeros=8."""
    G = _check_blob(blob, K, _Q4_0_TYPESIZE)
    N = blob.shape[0]
    b = blob.view(N, G, _Q4_0_TYPESIZE)
    d = _read_fp16_le(b[:, :, 0:2]).to(dtype)
    W_q = _unpack_nibbles(b[:, :, 2:18], "gguf-legacy").view(N, K)
    return W_q, d, torch.full_like(d, 8.0)


def decode_q4_1(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q4_1 asym: w = d*q + m. HQQ: W_q=q, scales=d, zeros=-m/d."""
    G = _check_blob(blob, K, _Q4_1_TYPESIZE)
    N = blob.shape[0]
    b = blob.view(N, G, _Q4_1_TYPESIZE)
    d = _read_fp16_le(b[:, :, 0:2]).to(torch.float32)
    m = _read_fp16_le(b[:, :, 2:4]).to(torch.float32)
    W_q = _unpack_nibbles(b[:, :, 4:20], "gguf-legacy").view(N, K)
    return W_q, d.to(dtype), _safe_div(-m, d).to(dtype)


def decode_q8_0(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q8_0 sym: w = d*q, q int8. HQQ expects uint8: W_q=q+128, zeros=128."""
    G = _check_blob(blob, K, _Q8_0_TYPESIZE)
    N = blob.shape[0]
    b = blob.view(N, G, _Q8_0_TYPESIZE)
    d = _read_fp16_le(b[:, :, 0:2]).to(dtype)
    qs_i8 = b[:, :, 2:34].contiguous().view(torch.int8)
    W_q = (qs_i8.to(torch.int16) + 128).to(torch.uint8).view(N, K)
    return W_q, d, torch.full_like(d, 128.0)


# ---------------------------------------------------------------------------
# K-quants (super=256)
# ---------------------------------------------------------------------------

def _q4_k_unpack_sub_scales(q12: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., 12] uint8 -> (sc, mn) each [..., 8] uint8, per ggml get_scale_min_k4:
        j<4:  sc=q[j]&63,            mn=q[j+4]&63
        j>=4: sc=(q[j+4]&0xF) | ((q[j-4]>>6)<<4)
              mn=(q[j+4]>>4)  | ((q[j  ]>>6)<<4)
    """
    q0, q4, q8 = q12[..., 0:4], q12[..., 4:8], q12[..., 8:12]
    sc = torch.cat([q0 & 63,                  (q8 & 0xF) | ((q0 >> 6) << 4)], dim=-1)
    mn = torch.cat([q4 & 63,                  (q8 >> 4)  | ((q4 >> 6) << 4)], dim=-1)
    return sc, mn


def decode_q4_k(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q4_K: 4-bit, super=256, 8x32 sub-blocks. sc/mn 6-bit each.
    w(is) = d*sc_is*q - dmin*mn_is. HQQ gs=32: scales=d*sc, zeros=(dmin*mn)/(d*sc).

    qs layout: 4 chunks of 32 bytes; chunk c encodes sub-blocks (2c, 2c+1) as
    low / high nibbles of the shared byte.
    """
    G_sb = _check_blob(blob, K, _Q4_K_TYPESIZE, super_size=_QK_K)
    N = blob.shape[0]
    b = blob.view(N, G_sb, _Q4_K_TYPESIZE)

    d = _read_fp16_le(b[:, :, 0:2]).to(torch.float32)       # [N, G_sb]
    dmin = _read_fp16_le(b[:, :, 2:4]).to(torch.float32)
    sc_u, mn_u = _q4_k_unpack_sub_scales(b[:, :, 4:16])     # [N, G_sb, 8]

    sc_f = d.unsqueeze(-1) * sc_u.to(torch.float32)
    mn_f = dmin.unsqueeze(-1) * mn_u.to(torch.float32)
    scales = sc_f.reshape(N, G_sb * 8).to(dtype)
    zeros = _safe_div(mn_f, sc_f).reshape(N, G_sb * 8).to(dtype)

    # qs [N, G_sb, 128] = [N, G_sb, 4 chunks, 32 bytes]. Output ordering:
    # sb_index = 2*chunk + nib. Broadcast a [0,4] shift over unsqueezed nib dim
    # so the reshape lays out (chunk, nib, byte) directly — no permute needed.
    qs = b[:, :, 16:144].view(N, G_sb, 4, 1, 32).to(torch.int32)
    shifts = torch.tensor([0, 4], device=qs.device, dtype=torch.int32).view(1, 1, 1, 2, 1)
    W_q = ((qs >> shifts) & 0xF).to(torch.uint8).reshape(N, K)
    return W_q, scales, zeros


def decode_q2_k(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q2_K: 2-bit, super=256, 16x16 sub-blocks. sc/mn 4-bit each packed as
    {sc:4, mn:4} per sub-block. w(is) = d*sc_is*q - dmin*mn_is.
    HQQ gs=16: scales=d*sc, zeros=(dmin*mn)/(d*sc).

    qs layout: [half=2, g16=2, l=16] bytes; byte packs 4 shifts of 2 bits.
      is = 8*half + 2*shift + g16.
    """
    G_sb = _check_blob(blob, K, _Q2_K_TYPESIZE, super_size=_QK_K)
    N = blob.shape[0]
    b = blob.view(N, G_sb, _Q2_K_TYPESIZE)

    scmn = b[:, :, 0:16]
    sc_u, mn_u = scmn & 0x0F, scmn >> 4                      # [N, G_sb, 16]
    d = _read_fp16_le(b[:, :, 80:82]).to(torch.float32)
    dmin = _read_fp16_le(b[:, :, 82:84]).to(torch.float32)

    sc_f = d.unsqueeze(-1) * sc_u.to(torch.float32)
    mn_f = dmin.unsqueeze(-1) * mn_u.to(torch.float32)
    scales = sc_f.reshape(N, G_sb * 16).to(dtype)
    zeros = _safe_div(mn_f, sc_f).reshape(N, G_sb * 16).to(dtype)

    # qs [N, G_sb, 64] = [N, G_sb, half=2, g16=2, l=16]. Unsqueeze a shift dim
    # between half and g16 so the reshape lays out (half, shift, g16, l) —
    # exactly is = 8*half + 2*shift + g16.
    qs = b[:, :, 16:80].view(N, G_sb, 2, 1, 2, 16).to(torch.int32)
    shifts = torch.arange(0, 8, 2, device=qs.device, dtype=torch.int32).view(1, 1, 1, 4, 1, 1)
    W_q = ((qs >> shifts) & 3).to(torch.uint8).reshape(N, K)
    return W_q, scales, zeros


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_DECODERS: dict[int, tuple[int, Callable]] | None = None


def _decoders() -> dict[int, tuple[int, Callable]]:
    global _DECODERS
    if _DECODERS is None:
        from gguf import GGMLQuantizationType as T
        _DECODERS = {
            int(T.Q4_0): (4, decode_q4_0),
            int(T.Q4_1): (4, decode_q4_1),
            int(T.Q8_0): (8, decode_q8_0),
            int(T.Q4_K): (4, decode_q4_k),
            int(T.Q2_K): (2, decode_q2_k),
        }
    return _DECODERS


def supported_ggml_types() -> set[int]:
    return set(_decoders().keys())


def decode(ggml_type: int, blob: torch.Tensor, K: int,
           dtype: torch.dtype = torch.float16):
    """Returns (W_q [N,K] uint8, scales [N,G], zeros [N,G], nbits)."""
    nbits, fn = _decoders()[int(ggml_type)]
    W_q, scales, zeros = fn(blob, K, dtype=dtype)
    return W_q, scales, zeros, nbits
