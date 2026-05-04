# SPDX-License-Identifier: Apache-2.0
"""All gemlite LinearMethod schemes: FP8 block / per-tensor, NVFP4,
MXFP4, INT8 (A16W8, A8W8), A16W4 GPTQ/HQQ, A16W4 AWQ,
GGUF (Q4_0/Q4_1/Q8_0/Q4_K/Q2_K), and the compressed-tensors wrappers for
NVFP4/MXFP4."""

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
    CompressedTensorsW4A16Mxfp4,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.gguf import GGUFLinearMethod

from .common import (
    GemliteApplyMixin, GemliteCTApplyMixin, StockWrappedGemliteMethod,
    clear_layer_attrs, save_cache,
)

logger = logging.getLogger(__name__)


def _scalar_max_fp32(t):
    return (t.max() if t.ndim else t).to(torch.float32)


# ---------------------------------------------------------------------------
# FP8 (inherits weight-loading from Fp8LinearMethod; not StockWrapped)
# ---------------------------------------------------------------------------

_FP8_CLEANUP = ("weight", "weight_scale", "weight_scale_inv", "input_scale")


class _GemliteFp8Base(GemliteApplyMixin, Fp8LinearMethod):
    def _attach(self, layer, gl):
        layer.gemlite_linear = gl
        clear_layer_attrs(layer, _FP8_CLEANUP)
        torch.cuda.empty_cache()
        save_cache()


class GemliteFp8BlockLinearMethod(_GemliteFp8Base):
    """DeepSeek-style block-FP8 (weight_scale_inv [N//B, K//B])."""

    def process_weights_after_loading(self, layer) -> None:
        assert self.block_quant and self.weight_block_size is not None
        bn, bk = self.weight_block_size
        assert bn == BLOCK_QUANT_SIZE and bk == BLOCK_QUANT_SIZE, (
            f"gemlite block FP8 requires {BLOCK_QUANT_SIZE}x{BLOCK_QUANT_SIZE}, "
            f"got {self.weight_block_size}"
        )
        weight = layer.weight.data
        gl = A8W8_fp8_dynamic(
            device=weight.device, dtype=layer.orig_dtype, block_quant=True,
        ).from_weights(weight, bias=None, scales=layer.weight_scale_inv.data)
        self._attach(layer, gl)


class GemliteFp8PerTensorLinearMethod(_GemliteFp8Base):
    """Per-tensor / per-channel FP8, dynamic activations."""

    def process_weights_after_loading(self, layer) -> None:
        assert not self.block_quant
        weight = layer.weight.data
        scale = layer.weight_scale.data.view(-1, 1)
        if scale.numel() == 1:
            scale = scale.expand(weight.shape[0], 1).contiguous()
        gl = A8W8_fp8_dynamic(
            device=weight.device, dtype=layer.orig_dtype, block_quant=False,
        ).from_weights(weight, bias=None, scales=scale)
        self._attach(layer, gl)


# ---------------------------------------------------------------------------
# NVFP4 (ModelOpt)
# ---------------------------------------------------------------------------

_NVFP4_CLEANUP = ("weight", "weight_scale", "weight_scale_2",
                  "input_scale", "input_scale_inv", "alpha")


class GemliteNvFp4LinearMethod(StockWrappedGemliteMethod):
    """ModelOpt's weight_scale_2 = global_amax/(448*6); gemlite's meta_scale
    is the reciprocal."""

    def process_weights_after_loading(self, layer) -> None:
        weight = layer.weight.data
        meta_scale = 1.0 / _scalar_max_fp32(layer.weight_scale_2.data)

        input_scale = None
        if getattr(layer, "input_scale", None) is not None:
            _s = layer.input_scale
            _s = _s.data if hasattr(_s, "data") else _s
            input_scale = _scalar_max_fp32(_s)

        dtype = getattr(layer, "orig_dtype", None) or torch.bfloat16

        gl = A4W4_NVFP_dynamic(
            device=weight.device, dtype=dtype,
        ).from_weights(
            weight=weight, scales=layer.weight_scale.data,
            meta_scale=meta_scale, input_scale=input_scale, packed=True,
        )
        self._finalize(layer, gl)
        clear_layer_attrs(layer, _NVFP4_CLEANUP)


# ---------------------------------------------------------------------------
# MXFP4 weight-only
# ---------------------------------------------------------------------------

_MXFP4_CLEANUP = ("weight", "weight_packed", "weight_scale",
                  "weight_global_scale", "weight_scale_2")


class GemliteMxfp4WeightOnlyLinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        weight = (getattr(layer, "weight_packed", None) or layer.weight).data
        dtype = getattr(layer, "orig_dtype", torch.bfloat16)
        gl = A16W4_MXFP(
            device=weight.device, dtype=dtype,
        ).from_packed_weights(
            W_q_packed=weight, scales=layer.weight_scale.data, bias=None,
        )
        self._finalize(layer, gl)
        clear_layer_attrs(layer, _MXFP4_CLEANUP)


# ---------------------------------------------------------------------------
# INT8 (A16W8 weight-only, A8W8 dynamic)
# ---------------------------------------------------------------------------

_INT8_CLEANUP = ("weight", "weight_scale", "input_scale")


def _pack_channelwise_int8(layer, helper_cls):
    weight = layer.weight.data
    scale = layer.weight_scale.data.view(-1, 1)
    dtype = getattr(layer, "orig_dtype", torch.bfloat16)
    return helper_cls(
        device=weight.device, dtype=dtype,
    ).from_weights(weight=weight, bias=None, scales=scale)


class GemliteA16W8Int8LinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        self._finalize(layer, _pack_channelwise_int8(layer, A16W8_INT8))
        clear_layer_attrs(layer, _INT8_CLEANUP)


class GemliteA8W8Int8DynamicLinearMethod(StockWrappedGemliteMethod):
    def process_weights_after_loading(self, layer) -> None:
        self._finalize(layer, _pack_channelwise_int8(layer, A8W8_int8_dynamic))
        clear_layer_attrs(layer, _INT8_CLEANUP)


# ---------------------------------------------------------------------------
# A16W4 INT4 (HQQ / GPTQ / GPTQMarlin)
# ---------------------------------------------------------------------------

_INT4_CLEANUP = ("qweight", "qzeros", "scales", "g_idx", "exllama_state")


def unpack_int32(packed: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Little-endian bit-unpack int32 along pack_dim into uint8."""
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(pack_factor, device=packed.device,
                          dtype=torch.int32) * bits
    out = ((packed.unsqueeze(-1) >> shifts) & mask).to(torch.uint8)
    if pack_dim == 0:
        R, C = packed.shape
        return out.permute(0, 2, 1).reshape(R * pack_factor, C)
    R, C = packed.shape
    return out.reshape(R, C * pack_factor)


def _normalize_zeros(qzeros, *, bits, packed_dim, N, G, out_dtype,
                     gptq_v1_plus_one=False):
    if qzeros is None or (isinstance(qzeros, int) and qzeros == 0):
        return None
    if isinstance(qzeros, int):
        return qzeros
    if not isinstance(qzeros, torch.Tensor) or qzeros.numel() == 0:
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
    raise ValueError(
        f"qzeros shape {tuple(z.shape)}; want [N={N}, G={G}] or [G, N]"
    )


def _gptq_perm_from_g_idx(g_idx, K, group_size):
    """Return a [K] int64 permutation that sorts g_idx so input columns
    belonging to the same GPTQ group are contiguous. Returns None if g_idx
    is missing or already trivial (k//group_size)."""
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
    """HQQ / GPTQ int4 weight-only via A16W4_HQQ_INT."""

    def __init__(self, quant_config, stock_method,
                 weight_bits=4, qweight_pack_dim=0, qzeros_pack_dim=1,
                 gptq_v1_plus_one=True):
        super().__init__(quant_config, stock_method)
        assert weight_bits in (4, 8), f"A16W{weight_bits} unsupported; want 4 or 8."
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

        group_size = getattr(self.quant_config, "group_size", -1) or -1
        if group_size == -1:
            group_size = K
        G = K // group_size

        # GPTQ desc_act=True: columns (input features) are re-ordered across
        # quant groups. Sort rows of unpacked W_q by argsort(g_idx) so groups
        # become contiguous; each HQQ group of `group_size` rows now comes
        # from a single GPTQ group and shares scales[n,g]/zeros[n,g] exactly.
        # No re-quantization. apply() gathers x by the same perm to keep the
        # math equivalent to the original GPTQ forward.
        perm = _gptq_perm_from_g_idx(
            getattr(layer, "g_idx", None), K=K, group_size=group_size,
        )
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

        HelperCls = A16W4_HQQ_INT if self.weight_bits == 4 else A16W8_HQQ_INT
        gl = HelperCls(
            device=qweight.device, dtype=scales.dtype,
        ).from_weights(W_q=W_q, scales=scales_ng, zeros=zeros, bias=None)

        self._finalize(layer, gl)
        if perm is not None:
            layer.gemlite_act_perm = perm.to(torch.int32)
        clear_layer_attrs(layer, _INT4_CLEANUP)


# ---------------------------------------------------------------------------
# A16W4 AWQ / AWQMarlin
# ---------------------------------------------------------------------------

_AWQ_CLEANUP = ("qweight", "qzeros", "scales", "g_idx", "workspace")
_REVERSE_AWQ_ORDER = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32)


def unpack_awq_int32(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """[R, C/8] int32 -> [R, C] uint8, undoing the AWQ interleave."""
    assert bits == 4, "AWQ uses 4-bit packing."
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    shifts = _REVERSE_AWQ_ORDER.to(packed.device) * bits
    out = ((packed.unsqueeze(-1) >> shifts) & mask).to(torch.uint8)
    R, C = packed.shape
    return out.reshape(R, C * pack_factor)


class GemliteAwqLinearMethod(StockWrappedGemliteMethod):
    """AWQ int4 via A16W4_HQQ_INT. Handles both `awq` and `awq_marlin`
    dispatch — the raw tensor layout is identical."""

    def __init__(self, quant_config, stock_method,
                 weight_bits=4, zero_point=True):
        super().__init__(quant_config, stock_method)
        assert weight_bits in (4, 8), f"AWQ A16W{weight_bits} unsupported; want 4 or 8."
        self.weight_bits = weight_bits
        self.zero_point = zero_point

    def process_weights_after_loading(self, layer) -> None:
        qweight = layer.qweight.data   # int32 [K, N//8]
        scales = layer.scales.data     # fp16  [G, N]
        qzeros = layer.qzeros.data     # int32 [G, N//8]

        W_q_kn = unpack_awq_int32(qweight, bits=self.weight_bits)
        K, N = W_q_kn.shape
        W_q = W_q_kn.t().contiguous()

        group_size = getattr(self.quant_config, "group_size", -1) or -1
        if group_size == -1:
            group_size = K
        G = K // group_size
        assert scales.shape == (G, N), (
            f"scales shape {tuple(scales.shape)} != [G={G}, N={N}]"
        )

        scales_ng = scales.t().contiguous()
        if self.zero_point:
            z = unpack_awq_int32(qzeros, bits=self.weight_bits)
            zeros = z.t().contiguous().to(scales.dtype)
        else:
            zeros = torch.zeros_like(scales_ng)

        HelperCls = A16W4_HQQ_INT if self.weight_bits == 4 else A16W8_HQQ_INT
        gl = HelperCls(
            device=qweight.device, dtype=scales.dtype,
        ).from_weights(W_q=W_q, scales=scales_ng, zeros=zeros, bias=None)

        self._finalize(layer, gl)
        clear_layer_attrs(layer, _AWQ_CLEANUP)


# ---------------------------------------------------------------------------
# compressed-tensors wrappers
# ---------------------------------------------------------------------------

def _ct_attach(layer, gl, cleanup):
    layer.gemlite_linear = gl
    clear_layer_attrs(layer, cleanup)
    torch.cuda.empty_cache()
    save_cache()


class GemliteCTW4A4Fp4(GemliteCTApplyMixin, CompressedTensorsW4A4Fp4):
    """W4A4 NVFP4 — activations + weights in NVFP4. CT stores global scales as
    448*6/amax (kernel-ready); ModelOpt is the reciprocal. Gemlite inverts
    internally, so pre-invert to feed it the ModelOpt convention."""

    def process_weights_after_loading(self, layer) -> None:
        meta_scale = 1.0 / _scalar_max_fp32(layer.weight_global_scale)
        modelopt_input_scale = 1.0 / _scalar_max_fp32(layer.input_global_scale)

        weight = layer.weight_packed.data
        dtype = getattr(layer, "params_dtype", torch.bfloat16)

        gl = A4W4_NVFP_dynamic(
            device=weight.device, dtype=dtype,
        ).from_weights(
            weight=weight, scales=layer.weight_scale.data,
            meta_scale=meta_scale, input_scale=modelopt_input_scale,
            packed=True,
        )
        _ct_attach(layer, gl, ("weight_packed", "weight_scale",
                               "weight_global_scale", "input_global_scale",
                               "alpha"))


class GemliteCTW4A16Fp4(GemliteCTApplyMixin, CompressedTensorsW4A16Fp4):
    """W4A16 NVFP4 — weight-only NVFP4."""

    def process_weights_after_loading(self, layer) -> None:
        weight = layer.weight_packed.data
        dtype = getattr(layer, "params_dtype", torch.bfloat16)

        meta_scale = None
        wg = getattr(layer, "weight_global_scale", None)
        if wg is not None:
            meta_scale = 1.0 / _scalar_max_fp32(wg)

        gl = A16W4_NVFP(
            device=weight.device, dtype=dtype,
        ).from_packed_weights(
            weight_packed=weight, scales=layer.weight_scale.data,
            meta_scale=meta_scale,
        )
        _ct_attach(layer, gl, ("weight_packed", "weight_scale",
                               "weight_global_scale"))


class GemliteCTW4A16Mxfp4(GemliteCTApplyMixin, CompressedTensorsW4A16Mxfp4):
    """W4A16 MXFP4 — e8m0 scales, no global scale."""

    def process_weights_after_loading(self, layer) -> None:
        weight = layer.weight_packed.data
        dtype = getattr(layer, "params_dtype", torch.bfloat16)
        gl = A16W4_MXFP(
            device=weight.device, dtype=dtype,
        ).from_packed_weights(
            W_q_packed=weight, scales=layer.weight_scale.data, bias=None,
        )
        _ct_attach(layer, gl, ("weight_packed", "weight_scale"))


# ---------------------------------------------------------------------------
# compressed-tensors WNA16 int4 (pack_quantized, type=int)
# ---------------------------------------------------------------------------

_CT_WNA16_CLEANUP = ("weight_packed", "weight_scale", "weight_zero_point",
                     "weight_shape", "weight_g_idx")


class GemliteCTWNA16Int(GemliteCTApplyMixin,
                       __import__("vllm.model_executor.layers.quantization."
                                  "compressed_tensors.schemes",
                                  fromlist=["CompressedTensorsWNA16"])
                       .CompressedTensorsWNA16):
    """CT pack_quantized W4A16 int -> gemlite A16W4_HQQ_INT.

    Expects layer.weight_packed [N, K//8] int32, layer.weight_scale [N, K/gs]
    and (asym) layer.weight_zero_point [N//8, K/gs] int32. No g_idx path here
    (caller should route to stock vLLM if actorder == 'group')."""

    def __init__(self, num_bits: int, *args, **kwargs) -> None:
        super().__init__(num_bits=num_bits, *args, **kwargs)
        self.num_bits = num_bits  # parent takes num_bits as arg but doesn't store it

    def process_weights_after_loading(self, layer) -> None:
        weight_packed = layer.weight_packed.data
        scales = layer.weight_scale.data

        # Unpack [N, K//8] int32 -> [N, K] uint8 along packed dim=1.
        W_q_nk = unpack_int32(weight_packed, bits=self.num_bits, pack_dim=1)
        N, K = W_q_nk.shape

        group_size = self.group_size if self.group_size != -1 else K
        G = K // group_size
        assert scales.shape == (N, G), (
            f"scales shape {tuple(scales.shape)} != [N={N}, G={G}]"
        )

        if self.symmetric:
            zeros = torch.full((N, G), 1 << (self.num_bits - 1), dtype=scales.dtype, device=scales.device)
        else:
            qzeros_packed = layer.weight_zero_point.data   # [N//8, G] int32
            # Unpack along packed dim=0 -> [N, G] uint8; cast to scales dtype.
            zeros = unpack_int32(qzeros_packed, bits=self.num_bits, pack_dim=0).to(scales.dtype)
            assert zeros.shape == (N, G), zeros.shape

        dtype = getattr(layer, "params_dtype", scales.dtype)
        HelperCls = A16W4_HQQ_INT if self.num_bits == 4 else A16W8_HQQ_INT
        gl = HelperCls(
            device=weight_packed.device, dtype=dtype,
        ).from_weights(W_q=W_q_nk, scales=scales, zeros=zeros, bias=None)

        _ct_attach(layer, gl, _CT_WNA16_CLEANUP)


# ---------------------------------------------------------------------------
# GGUF (Q4_0 / Q4_1 / Q4_K / Q8_0 / Q2_K -> A16W{2,4,8}_HQQ_INT)
# ---------------------------------------------------------------------------

_GGUF_CLEANUP = ("qweight", "qweight_type")


class GemliteGGUFLinearMethod(GemliteApplyMixin, GGUFLinearMethod):
    """GGUF weight-only via A16W{2,4,8}_HQQ_INT for supported block types.

    Inherits create_weights from vLLM's GGUFLinearMethod, runs the stock
    padded-weight materialization, then decodes each shard of the padded
    qweight into (W_q, scales, zeros) and attaches a GemLiteLinear. apply()
    always routes through gemlite_linear (via GemliteApplyMixin) — we never
    fall back at forward time, so torch.compile doesn't trace stock's
    ggml_mul_mat_a8 path (which breaks when we clear qweight). If a layer
    can't be decoded we raise; the caller in vLLM will surface it."""

    def process_weights_after_loading(self, layer):
        # Stock GGUFLinearMethod.process_weights_after_loading() pads shards
        # across N, sets up layer.qweight.shard_offset_map / shard_weight_type,
        # and re-registers layer.qweight as a real Parameter. We need all of
        # that before decoding.
        super().process_weights_after_loading(layer)

        prefix = getattr(layer, "prefix", "") or layer.__class__.__name__

        try:
            gl = self._build_gemlite(layer)
        except Exception as e:
            logger.warning(
                "Failed to use gemlite at %s (%s: %s), reverting to stock GGUF",
                prefix, type(e).__name__, e,
            )
            # Fallback: re-install stock apply for this layer only, so future
            # forward passes (and torch.compile tracing) skip gemlite.
            layer.quant_method = GGUFLinearMethod(self.quant_config)
            return

        if gl is None:
            # Unsupported ggml type — same fallback as above.
            logger.warning(
                "Unsupported GGUF ggml_type at %s, reverting to stock GGUF", prefix,
            )
            layer.quant_method = GGUFLinearMethod(self.quant_config)
            return

        logger.warning("gemlite GGUF attached at %s", prefix)
        layer.gemlite_linear = gl
        # NOTE: do NOT clear layer.qweight / layer.qweight_type here.
        # Unlike AWQ/GPTQ (where clear_layer_attrs works), GGUF's qweight is
        # registered as a Parameter that torch.compile's AOT pipeline captures
        # as a graph input when tracing the embedding layer. Setting it to
        # None triggers `copy_misaligned_inputs` -> AssertionError "Expected
        # tensors only, but got NoneType" during profile_run. Keeping the
        # real GGUF bytes around costs ~weight_size extra VRAM until model
        # init completes; gemlite still owns the forward path via apply().
        torch.cuda.empty_cache()
        save_cache()

    def _build_gemlite(self, layer):
        from .gguf_loader import _decoders, decode, supported_ggml_types

        qweight = layer.qweight
        qw_type = layer.qweight_type
        supported = supported_ggml_types()

        # Collect (ggml_type, blob [N_i, bytes_i]) per shard. For fused
        # qkv/gate_up layers the padded qweight holds each shard row-stacked
        # (in load order) with dim-1 padded to max bytes; shard_offset_map
        # gives each shard's real row range and real byte width. Iteration
        # order must match stock apply(): q/k/v canonicalized for QKV, else
        # shard_id order.
        shard_offset_map = getattr(qweight, "shard_offset_map", None)
        if shard_offset_map:
            shard_ids = list(qweight.shard_id)
            if "q" in shard_ids:
                shard_ids = ["q", "k", "v"]
            shards = []
            for idx in shard_ids:
                start, end, bytes_i = shard_offset_map[idx]
                shards.append((
                    int(qw_type.shard_weight_type[idx]),
                    qweight.data[start:end, :bytes_i].contiguous(),
                ))
        else:
            shards = [(int(qw_type.weight_type), qweight.data.contiguous())]

        if any(t not in supported for t, _ in shards):
            return None

        decoders = _decoders()
        nbits_set = {decoders[t][0] for t, _ in shards}
        if len(nbits_set) != 1:
            return None  # mixed 4-bit + 8-bit shards: fall back
        nbits = nbits_set.pop()

        # K is shared across shards (input dim). tensor_shape is
        # (out_total, in_per_partition) set in create_weights.
        K = qweight.tensor_shape[1] if hasattr(qweight, "tensor_shape") else None
        if K is None:
            return None

        dtype = getattr(self, "params_dtype", None) or torch.float16

        W_q_parts, scales_parts, zeros_parts = [], [], []
        for t, blob in shards:
            W_q_i, scales_i, zeros_i, _ = decode(t, blob, K, dtype=dtype)
            W_q_parts.append(W_q_i)
            scales_parts.append(scales_i)
            zeros_parts.append(zeros_i)

        W_q = torch.cat(W_q_parts, dim=0).contiguous()
        scales = torch.cat(scales_parts, dim=0).contiguous()
        zeros = torch.cat(zeros_parts, dim=0).contiguous()

        HelperCls = {2: A16W2_HQQ_INT, 4: A16W4_HQQ_INT, 8: A16W8_HQQ_INT}[nbits]
        return HelperCls(
            device=W_q.device, dtype=dtype,
        ).from_weights(W_q=W_q, scales=scales, zeros=zeros, bias=None)

