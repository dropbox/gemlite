# SPDX-License-Identifier: Apache-2.0
"""gemlite LinearMethod schemes: FP8 block / per-tensor, NVFP4, MXFP4,
INT8 (A16W8, A8W8), A16Wn GPTQ/HQQ, A16Wn AWQ, GGUF, and compressed-tensors
wrappers for NVFP4/MXFP4."""

from __future__ import annotations

import logging

import torch

from gemlite.helper import (
    A4W4_NVFP_dynamic, A8W8_fp8_dynamic, A8W8_int8_dynamic,
    A16W2_HQQ_INT, A16W4_HQQ_INT, A16W4_MXFP, A16W4_NVFP,
    A16W8_HQQ_INT, A16W8_INT8,
)
from gemlite.triton_kernels.config import BLOCK_QUANT_SIZE

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4, CompressedTensorsW4A16Fp4,
    CompressedTensorsW4A16Mxfp4, CompressedTensorsWNA16,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.gguf import GGUFLinearMethod

from .common import (
    GemliteApplyMixin, GemliteCTApplyMixin, StockWrappedGemliteMethod,
    clear_layer_attrs, save_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

HQQ_INT_BY_BITS = {2: A16W2_HQQ_INT, 4: A16W4_HQQ_INT, 8: A16W8_HQQ_INT}


def _scalar_max_fp32(t):
    return (t.max() if t.ndim else t).to(torch.float32)


def _recip_max(t):
    """1 / max(t) in fp32 — the ModelOpt meta_scale convention."""
    return 1.0 / _scalar_max_fp32(t)


def _pick_dtype(obj, *, attr: str = "orig_dtype", default=torch.bfloat16):
    """Prefer obj.<attr>, fall back to default. Handles the None sentinel too."""
    return getattr(obj, attr, None) or default


def _resolve_group_size(cfg, K: int) -> int:
    """GPTQ/AWQ/CT group_size convention: -1 (or missing/None) = per-channel."""
    gs = getattr(cfg, "group_size", -1)
    return K if not gs or gs == -1 else gs


def _attach(layer, gl, cleanup):
    """Set gemlite_linear, drop the stock tensors, empty cache, save autotune."""
    layer.gemlite_linear = gl
    clear_layer_attrs(layer, cleanup)
    torch.cuda.empty_cache()
    save_cache()


def _build_hqq_int(*, bits: int, W_q, scales, zeros, device, dtype):
    """Dispatch to A16W{bits}_HQQ_INT and build a GemLiteLinear."""
    return HQQ_INT_BY_BITS[bits](device=device, dtype=dtype).from_weights(
        W_q=W_q, scales=scales, zeros=zeros, bias=None,
    )


# AWQ packs 8 nibbles per int32 with the permutation [0,4,1,5,2,6,3,7]; stock
# (GPTQ / CT / HQQ) uses the identity order. The `awq=True` flag on
# unpack_int32 selects between them.
_AWQ_SHIFTS = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32)


def unpack_int32(packed: torch.Tensor, bits: int, pack_dim: int,
                 awq: bool = False) -> torch.Tensor:
    """int32 [R, C] -> uint8, bit-unpacked along pack_dim."""
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    if awq:
        assert bits == 4, "AWQ nibble interleave is 4-bit only"
        shifts = _AWQ_SHIFTS.to(packed.device) * bits
    else:
        shifts = torch.arange(pack_factor, device=packed.device,
                              dtype=torch.int32) * bits
    out = ((packed.unsqueeze(-1) >> shifts) & mask).to(torch.uint8)
    R, C = packed.shape
    if pack_dim == 1:
        return out.reshape(R, C * pack_factor)
    return out.permute(0, 2, 1).reshape(R * pack_factor, C)


# ---------------------------------------------------------------------------
# FP8 (inherits weight-loading from Fp8LinearMethod; not StockWrapped)
# ---------------------------------------------------------------------------

_FP8_CLEANUP = ("weight", "weight_scale", "weight_scale_inv", "input_scale")


class _GemliteFp8Base(GemliteApplyMixin, Fp8LinearMethod):
    pass


class GemliteFp8BlockLinearMethod(_GemliteFp8Base):
    """DeepSeek-style block-FP8 (weight_scale_inv [N//B, K//B])."""

    def process_weights_after_loading(self, layer) -> None:
        assert self.block_quant and self.weight_block_size is not None
        bn, bk = self.weight_block_size
        assert bn == BLOCK_QUANT_SIZE and bk == BLOCK_QUANT_SIZE, (
            f"gemlite block FP8 requires {BLOCK_QUANT_SIZE}x{BLOCK_QUANT_SIZE}, "
            f"got {self.weight_block_size}"
        )
        w = layer.weight.data
        gl = A8W8_fp8_dynamic(
            device=w.device, dtype=layer.orig_dtype, block_quant=True,
        ).from_weights(w, bias=None, scales=layer.weight_scale_inv.data)
        _attach(layer, gl, _FP8_CLEANUP)


class GemliteFp8PerTensorLinearMethod(_GemliteFp8Base):
    """Per-tensor / per-channel FP8, dynamic activations."""

    def process_weights_after_loading(self, layer) -> None:
        assert not self.block_quant
        w = layer.weight.data
        scale = layer.weight_scale.data.view(-1, 1)
        if scale.numel() == 1:
            scale = scale.expand(w.shape[0], 1).contiguous()
        gl = A8W8_fp8_dynamic(
            device=w.device, dtype=layer.orig_dtype, block_quant=False,
        ).from_weights(w, bias=None, scales=scale)
        _attach(layer, gl, _FP8_CLEANUP)


# ---------------------------------------------------------------------------
# NVFP4 (ModelOpt)
# ---------------------------------------------------------------------------

_NVFP4_CLEANUP = ("weight", "weight_scale", "weight_scale_2",
                  "input_scale", "input_scale_inv", "alpha")


class GemliteNvFp4LinearMethod(StockWrappedGemliteMethod):
    """ModelOpt's weight_scale_2 = global_amax/(448*6); gemlite's meta_scale
    is the reciprocal."""

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight.data
        meta_scale = _recip_max(layer.weight_scale_2.data)

        input_scale = None
        _is = getattr(layer, "input_scale", None)
        if _is is not None:
            input_scale = _scalar_max_fp32(_is.data if hasattr(_is, "data") else _is)

        gl = A4W4_NVFP_dynamic(
            device=w.device, dtype=_pick_dtype(layer),
        ).from_weights(
            weight=w, scales=layer.weight_scale.data,
            meta_scale=meta_scale, input_scale=input_scale, packed=True,
        )
        _attach(layer, gl, _NVFP4_CLEANUP)


# ---------------------------------------------------------------------------
# MXFP4 weight-only
# ---------------------------------------------------------------------------

_MXFP4_CLEANUP = ("weight", "weight_packed", "weight_scale",
                  "weight_global_scale", "weight_scale_2")


class GemliteMxfp4WeightOnlyLinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        w = (getattr(layer, "weight_packed", None) or layer.weight).data
        gl = A16W4_MXFP(
            device=w.device, dtype=_pick_dtype(layer),
        ).from_packed_weights(
            W_q_packed=w, scales=layer.weight_scale.data, bias=None,
        )
        _attach(layer, gl, _MXFP4_CLEANUP)


# ---------------------------------------------------------------------------
# INT8 (A16W8 weight-only, A8W8 dynamic)
# ---------------------------------------------------------------------------

_INT8_CLEANUP = ("weight", "weight_scale", "input_scale")


def _pack_channelwise_int8(layer, helper_cls):
    w = layer.weight.data
    scale = layer.weight_scale.data.view(-1, 1)
    return helper_cls(device=w.device, dtype=_pick_dtype(layer)).from_weights(
        weight=w, bias=None, scales=scale,
    )


class GemliteA16W8Int8LinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        _attach(layer, _pack_channelwise_int8(layer, A16W8_INT8), _INT8_CLEANUP)


class GemliteA8W8Int8DynamicLinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        _attach(layer, _pack_channelwise_int8(layer, A8W8_int8_dynamic),
                _INT8_CLEANUP)


# ---------------------------------------------------------------------------
# A16Wn INT (HQQ / GPTQ / GPTQMarlin)
# ---------------------------------------------------------------------------

_INT4_CLEANUP = ("qweight", "qzeros", "scales", "g_idx", "exllama_state")


def _normalize_zeros(qzeros, *, bits, packed_dim, N, G, out_dtype,
                     gptq_v1_plus_one=False):
    """int32-packed qzeros -> [N, G] out_dtype. Returns None if empty."""
    if qzeros is None or not isinstance(qzeros, torch.Tensor) or qzeros.numel() == 0:
        return None

    if qzeros.dtype == torch.int32:
        z = unpack_int32(qzeros, bits, pack_dim=packed_dim)
        if gptq_v1_plus_one:
            z = (z.to(torch.int32) + 1).to(torch.uint8)
        z = z.to(out_dtype)
    else:
        z = qzeros.to(out_dtype)

    if z.shape == (N, G):
        return z.contiguous()
    if z.shape == (G, N):
        return z.t().contiguous()
    raise ValueError(f"qzeros shape {tuple(z.shape)}; want [N={N}, G={G}] or [G, N]")


def _gptq_perm_from_g_idx(g_idx, K, group_size):
    """[K] permutation that sorts g_idx so GPTQ groups are contiguous; None if
    g_idx is trivial (k // group_size)."""
    if g_idx is None:
        return None
    g = g_idx.data if hasattr(g_idx, "data") else g_idx
    if not isinstance(g, torch.Tensor) or g.numel() != K:
        return None
    g = g.to(torch.int64)
    trivial = torch.arange(K, device=g.device, dtype=torch.int64) // group_size
    if torch.equal(g, trivial):
        return None
    return torch.argsort(g, stable=True)


class GemliteA16W4GroupLinearMethod(StockWrappedGemliteMethod):
    """HQQ / GPTQ int weight-only (4 or 8 bit) via A16W{4,8}_HQQ_INT."""

    def __init__(self, quant_config, stock_method,
                 weight_bits=4, qweight_pack_dim=0, qzeros_pack_dim=1,
                 gptq_v1_plus_one=True):
        super().__init__(quant_config, stock_method)
        assert weight_bits in (4, 8), f"A16W{weight_bits} unsupported"
        self.weight_bits = weight_bits
        self.qweight_pack_dim = qweight_pack_dim
        self.qzeros_pack_dim = qzeros_pack_dim
        self.gptq_v1_plus_one = gptq_v1_plus_one

    def apply(self, layer, x, bias=None):
        perm = getattr(layer, "gemlite_act_perm", None)
        if perm is not None:
            x = x.index_select(-1, perm)
        out = layer.gemlite_linear(x)
        return out if bias is None else out + bias

    def process_weights_after_loading(self, layer) -> None:
        qweight = layer.qweight.data
        scales = layer.scales.data

        W_q_kn = unpack_int32(qweight, self.weight_bits,
                              pack_dim=self.qweight_pack_dim)
        K, N = W_q_kn.shape
        group_size = _resolve_group_size(self.quant_config, K)
        G = K // group_size

        # GPTQ desc_act=True: sort W_q rows by argsort(g_idx) so groups become
        # contiguous; each HQQ group of group_size rows maps to one GPTQ group
        # and shares scales/zeros exactly. apply() gathers x by the same perm.
        perm = _gptq_perm_from_g_idx(getattr(layer, "g_idx", None),
                                     K=K, group_size=group_size)
        if perm is not None:
            W_q_kn = W_q_kn.index_select(0, perm).contiguous()
        W_q = W_q_kn.t().contiguous()

        scales_ng = scales.t().contiguous()
        zeros = _normalize_zeros(
            getattr(layer, "qzeros", None),
            bits=self.weight_bits, packed_dim=self.qzeros_pack_dim,
            N=N, G=G, out_dtype=scales_ng.dtype,
            gptq_v1_plus_one=self.gptq_v1_plus_one,
        )
        if zeros is None:
            zeros = torch.zeros_like(scales_ng)

        gl = _build_hqq_int(
            bits=self.weight_bits, W_q=W_q, scales=scales_ng, zeros=zeros,
            device=qweight.device, dtype=scales.dtype,
        )
        if perm is not None:
            layer.gemlite_act_perm = perm.to(torch.int32)
        _attach(layer, gl, _INT4_CLEANUP)


# ---------------------------------------------------------------------------
# A16Wn AWQ / AWQMarlin
# ---------------------------------------------------------------------------

_AWQ_CLEANUP = ("qweight", "qzeros", "scales", "g_idx", "workspace")


class GemliteAwqLinearMethod(StockWrappedGemliteMethod):
    """AWQ via A16W{4,8}_HQQ_INT. Handles both `awq` and `awq_marlin` — the
    raw tensor layout is identical."""

    def __init__(self, quant_config, stock_method,
                 weight_bits=4, zero_point=True):
        super().__init__(quant_config, stock_method)
        assert weight_bits in (4, 8), f"AWQ A16W{weight_bits} unsupported"
        self.weight_bits = weight_bits
        self.zero_point = zero_point

    def process_weights_after_loading(self, layer) -> None:
        qweight = layer.qweight.data   # int32 [K, N//8]
        scales = layer.scales.data     # fp16  [G, N]

        W_q_kn = unpack_int32(qweight, self.weight_bits, pack_dim=1, awq=True)
        K, N = W_q_kn.shape
        W_q = W_q_kn.t().contiguous()

        group_size = _resolve_group_size(self.quant_config, K)
        G = K // group_size
        assert scales.shape == (G, N), (
            f"scales shape {tuple(scales.shape)} != [G={G}, N={N}]"
        )

        scales_ng = scales.t().contiguous()
        if self.zero_point:
            z = unpack_int32(layer.qzeros.data, self.weight_bits,
                             pack_dim=1, awq=True)
            zeros = z.t().contiguous().to(scales.dtype)
        else:
            zeros = torch.zeros_like(scales_ng)

        gl = _build_hqq_int(
            bits=self.weight_bits, W_q=W_q, scales=scales_ng, zeros=zeros,
            device=qweight.device, dtype=scales.dtype,
        )
        _attach(layer, gl, _AWQ_CLEANUP)


# ---------------------------------------------------------------------------
# compressed-tensors wrappers
# ---------------------------------------------------------------------------

class GemliteCTW4A4Fp4(GemliteCTApplyMixin, CompressedTensorsW4A4Fp4):
    """W4A4 NVFP4. CT stores global scales as 448*6/amax; ModelOpt stores the
    reciprocal. Gemlite inverts internally, so pre-invert to feed it ModelOpt."""

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight_packed.data
        gl = A4W4_NVFP_dynamic(
            device=w.device, dtype=_pick_dtype(layer, attr="params_dtype"),
        ).from_weights(
            weight=w, scales=layer.weight_scale.data,
            meta_scale=_recip_max(layer.weight_global_scale),
            input_scale=_recip_max(layer.input_global_scale),
            packed=True,
        )
        _attach(layer, gl, ("weight_packed", "weight_scale",
                            "weight_global_scale", "input_global_scale", "alpha"))


class GemliteCTW4A16Fp4(GemliteCTApplyMixin, CompressedTensorsW4A16Fp4):
    """W4A16 NVFP4 — weight-only NVFP4."""

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight_packed.data
        wg = getattr(layer, "weight_global_scale", None)
        gl = A16W4_NVFP(
            device=w.device, dtype=_pick_dtype(layer, attr="params_dtype"),
        ).from_packed_weights(
            weight_packed=w, scales=layer.weight_scale.data,
            meta_scale=_recip_max(wg) if wg is not None else None,
        )
        _attach(layer, gl, ("weight_packed", "weight_scale", "weight_global_scale"))


class GemliteCTW4A16Mxfp4(GemliteCTApplyMixin, CompressedTensorsW4A16Mxfp4):
    """W4A16 MXFP4 — e8m0 scales, no global scale."""

    def process_weights_after_loading(self, layer) -> None:
        w = layer.weight_packed.data
        gl = A16W4_MXFP(
            device=w.device, dtype=_pick_dtype(layer, attr="params_dtype"),
        ).from_packed_weights(
            W_q_packed=w, scales=layer.weight_scale.data, bias=None,
        )
        _attach(layer, gl, ("weight_packed", "weight_scale"))


# ---------------------------------------------------------------------------
# compressed-tensors WNA16 int (pack_quantized, type=int) -> A16W{4,8}_HQQ_INT
# ---------------------------------------------------------------------------

_CT_WNA16_CLEANUP = ("weight_packed", "weight_scale", "weight_zero_point",
                     "weight_shape", "weight_g_idx")


class GemliteCTWNA16Int(GemliteCTApplyMixin, CompressedTensorsWNA16):
    """CT pack_quantized W{4,8}A16 int. Expects weight_packed [N, K//8] int32,
    weight_scale [N, K/gs]; asym adds weight_zero_point [N//8, K/gs] int32.
    Does not handle actorder='group' (caller routes to stock)."""

    def __init__(self, num_bits: int, *args, **kwargs) -> None:
        super().__init__(num_bits=num_bits, *args, **kwargs)
        self.num_bits = num_bits  # parent takes it but doesn't store it

    def process_weights_after_loading(self, layer) -> None:
        wp = layer.weight_packed.data
        scales = layer.weight_scale.data

        W_q = unpack_int32(wp, self.num_bits, pack_dim=1)       # [N, K]
        N, K = W_q.shape
        G = K // _resolve_group_size(self, K)
        assert scales.shape == (N, G), (
            f"scales shape {tuple(scales.shape)} != [N={N}, G={G}]"
        )

        if self.symmetric:
            zeros = torch.full((N, G), 1 << (self.num_bits - 1),
                               dtype=scales.dtype, device=scales.device)
        else:
            zeros = unpack_int32(layer.weight_zero_point.data,
                                 self.num_bits, pack_dim=0).to(scales.dtype)
            assert zeros.shape == (N, G), zeros.shape

        gl = _build_hqq_int(
            bits=self.num_bits, W_q=W_q, scales=scales, zeros=zeros,
            device=wp.device,
            dtype=_pick_dtype(layer, attr="params_dtype", default=scales.dtype),
        )
        _attach(layer, gl, _CT_WNA16_CLEANUP)


# ---------------------------------------------------------------------------
# GGUF (Q4_0 / Q4_1 / Q4_K / Q8_0 / Q2_K -> A16W{2,4,8}_HQQ_INT)
# ---------------------------------------------------------------------------

class GemliteGGUFLinearMethod(GemliteApplyMixin, GGUFLinearMethod):
    """GGUF weight-only via A16W{2,4,8}_HQQ_INT for supported block types.

    Inherits create_weights from stock GGUFLinearMethod, runs the stock
    padded-weight materialization, then decodes each shard and attaches a
    GemLiteLinear. apply() always routes through gemlite_linear so
    torch.compile never traces stock's ggml_mul_mat_a8 path."""

    def process_weights_after_loading(self, layer):
        # Stock pads shards across N, sets up qweight.shard_offset_map /
        # shard_weight_type, re-registers qweight as a Parameter. Need all
        # of that before decoding.
        super().process_weights_after_loading(layer)
        prefix = getattr(layer, "prefix", "") or layer.__class__.__name__

        try:
            gl = self._build_gemlite(layer)
        except Exception as e:
            logger.warning("Failed to use gemlite at %s (%s: %s), reverting to stock",
                           prefix, type(e).__name__, e)
            layer.quant_method = GGUFLinearMethod(self.quant_config)
            return
        if gl is None:
            logger.warning("Unsupported GGUF ggml_type at %s, reverting to stock",
                           prefix)
            layer.quant_method = GGUFLinearMethod(self.quant_config)
            return

        logger.warning("gemlite GGUF attached at %s", prefix)
        layer.gemlite_linear = gl
        # NOTE: do NOT clear layer.qweight / layer.qweight_type. GGUF's qweight
        # is a Parameter that torch.compile captures as a graph input when
        # tracing the embedding layer; setting it to None triggers
        # copy_misaligned_inputs -> AssertionError during profile_run. Costs
        # ~weight_size VRAM until model init completes.
        torch.cuda.empty_cache()
        save_cache()

    @staticmethod
    def _collect_shards(qweight, qw_type):
        """[(ggml_type, bytes_blob)] in apply() order; unpads fused shards."""
        shard_offset_map = getattr(qweight, "shard_offset_map", None)
        if shard_offset_map is None:
            return [(int(qw_type.weight_type), qweight.data.contiguous())]

        shard_ids = list(qweight.shard_id)
        if "q" in shard_ids:
            shard_ids = ["q", "k", "v"]  # canonicalize QKV
        out = []
        for i in shard_ids:
            s, e, nb = shard_offset_map[i]
            out.append((int(qw_type.shard_weight_type[i]),
                        qweight.data[s:e, :nb].contiguous()))
        return out

    def _build_gemlite(self, layer):
        from .gguf_loader import _decoders, decode, supported_ggml_types

        qweight = layer.qweight
        shards = self._collect_shards(qweight, layer.qweight_type)

        supported = supported_ggml_types()
        if any(t not in supported for t, _ in shards):
            return None
        nbits_set = {_decoders()[t][0] for t, _ in shards}
        if len(nbits_set) != 1:
            return None  # mixed bitwidths: fall back
        nbits = nbits_set.pop()

        # tensor_shape = (out_total, in_per_partition), set in create_weights.
        K = qweight.tensor_shape[1] if hasattr(qweight, "tensor_shape") else None
        if K is None:
            return None

        dtype = _pick_dtype(self, attr="params_dtype", default=torch.float16)
        parts = [decode(t, blob, K, dtype=dtype) for t, blob in shards]
        W_q = torch.cat([p[0] for p in parts], dim=0).contiguous()
        scales = torch.cat([p[1] for p in parts], dim=0).contiguous()
        zeros = torch.cat([p[2] for p in parts], dim=0).contiguous()

        return _build_hqq_int(
            bits=nbits, W_q=W_q, scales=scales, zeros=zeros,
            device=W_q.device, dtype=dtype,
        )
