# SPDX-License-Identifier: Apache-2.0
"""GGUF block decoders for gemlite's A16Wn_HQQ_INT path.

Supports the legacy affine block types (Q4_0, Q4_1, Q8_0) that map directly
onto gemlite's HQQ convention  W = (W_q - zeros) * scales  with group_size=32.

K-quants and I-quants (Q4_K, Q6_K, IQ4_*, ...) are NOT handled here.
"""

from __future__ import annotations

from typing import Callable

import torch

# GGUF block format constants (see gguf-py GGML_QUANT_SIZES).
_Q4_0_BLOCK = 32
_Q4_0_TYPESIZE = 18   # 2B d (fp16) + 16B qs (32 x 4-bit)
_Q4_1_BLOCK = 32
_Q4_1_TYPESIZE = 20   # 2B d + 2B m + 16B qs
_Q8_0_BLOCK = 32
_Q8_0_TYPESIZE = 34   # 2B d + 32B qs (int8)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_fp16_le(bytes_tensor: torch.Tensor) -> torch.Tensor:
    """[..., 2] uint8 (little-endian fp16 bytes) -> [...] fp16."""
    # `.contiguous()` ensures the final dim stride is 1 byte, so a raw dtype
    # view treats the pair as one fp16. Works on all torch versions.
    buf = bytes_tensor.contiguous()
    fp16 = buf.view(torch.float16)
    # view collapses the inner dim-of-2 to dim-of-1; squeeze it.
    return fp16.squeeze(-1)


def _unpack_4bit_pairs(qs: torch.Tensor) -> torch.Tensor:
    """[..., 16] uint8 -> [..., 32] uint8 in [0,15].
    GGUF Q4_0/Q4_1 pack: byte i = qs[i] | (qs[i+16] << 4)."""
    low = qs & 0x0F                  # elements [0..16)
    high = (qs >> 4) & 0x0F          # elements [16..32)
    return torch.cat([low, high], dim=-1)


def _check_blob(blob: torch.Tensor, K: int, type_size: int) -> int:
    assert blob.dtype == torch.uint8, f"GGUF blob must be uint8, got {blob.dtype}"
    assert blob.ndim == 2, f"GGUF blob must be 2D [N, bytes], got {tuple(blob.shape)}"
    assert K % 32 == 0, f"K={K} not divisible by GGUF block size 32"
    G = K // 32
    expected = G * type_size
    assert blob.shape[1] == expected, (
        f"blob dim1 = {blob.shape[1]} != G*type_size = {G}*{type_size} = {expected}"
    )
    return G


# ---------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------

def decode_q4_0(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q4_0: symmetric, 4-bit, block=32. w = d * (q_raw - 8), q_raw in [0..15].

    HQQ mapping: W_q = q_raw (uint8), scales = d, zeros = 8.

    Args:
        blob: [N, G*18] uint8. Raw GGUF byte layout.
        K:    in_features.
        dtype: output scales/zeros dtype.

    Returns:
        W_q:    [N, K]  uint8
        scales: [N, G]  dtype
        zeros:  [N, G]  dtype (all = 8)
    """
    G = _check_blob(blob, K, _Q4_0_TYPESIZE)
    N = blob.shape[0]
    blocks = blob.view(N, G, _Q4_0_TYPESIZE)

    d = _read_fp16_le(blocks[:, :, 0:2]).to(dtype)          # [N, G]
    W_q = _unpack_4bit_pairs(blocks[:, :, 2:18]).view(N, K) # [N, K]

    zeros = torch.full_like(d, 8.0)
    return W_q, d, zeros


def decode_q4_1(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q4_1: asymmetric, 4-bit, block=32. w = d*q + m, q in [0..15].

    HQQ mapping: W_q = q, scales = d, zeros = -m/d.
    (gemlite computes (W_q - zeros) * scales = q*d - (-m/d)*d = q*d + m.)
    """
    G = _check_blob(blob, K, _Q4_1_TYPESIZE)
    N = blob.shape[0]
    blocks = blob.view(N, G, _Q4_1_TYPESIZE)

    d = _read_fp16_le(blocks[:, :, 0:2]).to(torch.float32)  # [N, G], fp32 for div
    m = _read_fp16_le(blocks[:, :, 2:4]).to(torch.float32)
    W_q = _unpack_4bit_pairs(blocks[:, :, 4:20]).view(N, K)

    # Degenerate blocks (d == 0) would mean weights are identically m; leave
    # zeros=0 there, scales=0; gemlite will produce 0 regardless of W_q.
    safe_d = torch.where(d == 0, torch.ones_like(d), d)
    zeros = torch.where(d == 0, torch.zeros_like(d), -m / safe_d).to(dtype)
    return W_q, d.to(dtype), zeros


def decode_q8_0(blob: torch.Tensor, K: int, dtype: torch.dtype = torch.float16):
    """Q8_0: symmetric, 8-bit, block=32. w = d * q, q int8 in [-128..127].

    HQQ mapping (A16W8_HQQ_INT expects uint8): W_q = q + 128, zeros = 128,
    scales = d.
    """
    G = _check_blob(blob, K, _Q8_0_TYPESIZE)
    N = blob.shape[0]
    blocks = blob.view(N, G, _Q8_0_TYPESIZE)

    d = _read_fp16_le(blocks[:, :, 0:2]).to(dtype)  # [N, G]

    qs_u8 = blocks[:, :, 2:34].contiguous()                    # [N, G, 32] uint8
    # reinterpret the bytes as int8 for the (q + 128) upshift.
    qs_i8 = qs_u8.view(torch.int8)
    W_q = (qs_i8.to(torch.int16) + 128).to(torch.uint8).view(N, K)

    zeros = torch.full_like(d, 128.0)
    return W_q, d, zeros


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

# ggml_type_int -> (nbits, decoder).
# Keeps enum-import-free at top-level; populated lazily.
_DECODERS: dict[int, tuple[int, Callable]] | None = None


def _decoders() -> dict[int, tuple[int, Callable]]:
    global _DECODERS
    if _DECODERS is None:
        from gguf import GGMLQuantizationType as T
        _DECODERS = {
            int(T.Q4_0): (4, decode_q4_0),
            int(T.Q4_1): (4, decode_q4_1),
            int(T.Q8_0): (8, decode_q8_0),
        }
    return _DECODERS


def supported_ggml_types() -> set[int]:
    return set(_decoders().keys())


def decode(ggml_type: int, blob: torch.Tensor, K: int,
           dtype: torch.dtype = torch.float16):
    """Decode one tensor. Returns (W_q [N,K] uint8, scales [N,G], zeros [N,G],
    nbits). Raises KeyError if ggml_type is not in supported_ggml_types()."""
    nbits, fn = _decoders()[int(ggml_type)]
    W_q, scales, zeros = fn(blob, K, dtype=dtype)
    return W_q, scales, zeros, nbits
