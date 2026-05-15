# SPDX-License-Identifier: Apache-2.0
"""Shared helpers, base classes, and the on-the-fly mode<->helper dispatcher."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import torch

from gemlite import helper as _helper
from gemlite.triton_kernels.utils import IS_HIP

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

CACHE_FILE = "/tmp/gemlite_cache.json"
DEFAULT_FP8 = torch.float8_e4m3fnuz if IS_HIP else torch.float8_e4m3fn


# --- cache ------------------------------------------------------------------

def load_cache() -> None:
    import gemlite
    try:
        gemlite.load_config(CACHE_FILE)
    except Exception:
        pass


def save_cache() -> None:
    import gemlite
    try:
        gemlite.cache_config(CACHE_FILE)
    except Exception:
        pass


# --- layer helpers ----------------------------------------------------------

def is_layer_skipped(prefix: str, skip_modules: Optional[Sequence[str]]) -> bool:
    if not skip_modules:
        return False
    components = prefix.split(".")
    return any(m in components for m in skip_modules)


def clear_layer_attrs(layer: torch.nn.Module, names: Iterable[str]) -> None:
    for n in names:
        if hasattr(layer, n):
            setattr(layer, n, None)


def gemlite_forward(layer, x, bias=None):
    out = layer.gemlite_linear(x)
    return out if bias is None else out + bias


# --- base classes -----------------------------------------------------------

class GemliteApplyMixin:
    """Provides `apply()` for LinearMethodBase subclasses."""

    def apply(self, layer, x, bias=None):
        return gemlite_forward(layer, x, bias)


class GemliteCTApplyMixin:
    """Same body under compressed-tensors' `apply_weights` name."""

    def apply_weights(self, layer, x, bias=None):
        return gemlite_forward(layer, x, bias)


class GemliteLinearMethod(GemliteApplyMixin, LinearMethodBase):
    """Base. Schemes attach `layer.gemlite_linear` and call `_finalize`."""

    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def _finalize(self, layer, gl) -> None:
        layer.gemlite_linear = gl
        torch.cuda.empty_cache()
        save_cache()


class StockWrappedGemliteMethod(GemliteLinearMethod):
    """Delegates create_weights to a stock vLLM LinearMethod; schemes
    override process_weights_after_loading to rebuild via gemlite."""

    def __init__(self, quant_config, stock_method: LinearMethodBase):
        super().__init__(quant_config)
        self._stock = stock_method

    def create_weights(self, *args, **kwargs):
        return self._stock.create_weights(*args, **kwargs)


# --- on-the-fly mode dispatcher --------------------------------------------

_MODE_TO_HELPER = {
    "int8_weightonly":  lambda **kw: _helper.A16W8_INT8(**kw),
    "fp8_weightonly":   lambda **kw: _helper.A16W8_FP8(**kw),
    "int8_dynamic":     lambda block_quant=False, **kw: _helper.A8W8_int8_dynamic(
        block_quant=block_quant, **kw),
    "fp8_dynamic":      lambda block_quant=False, **kw: _helper.A8W8_fp8_dynamic(
        block_quant=block_quant, **kw),
    "mxfp4_weightonly": lambda **kw: _helper.A16W4_MXFP(**kw),
    "mxfp4_dynamic":    lambda **kw: _helper.A4W4_MXFP_dynamic(**kw),
    "nvfp4_dynamic":    lambda **kw: _helper.A4W4_NVFP_dynamic(**kw),
}


def resolve_processor(*, weight_bits, quant_mode,
                      group_size=None, block_quant=False,
                      device="cuda:0", dtype: Optional[torch.dtype] = None):
    args = {"device": device, "dtype": dtype}

    if quant_mode == "int4_weightonly":
        try:
            from hqq.core.quantize import HQQLinear  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "int4_weightonly requires the hqq package (`pip install hqq`)."
            ) from e
        return _helper.A16W4_HQQ_INT(**args)

    if quant_mode == "mxfp8_dynamic":
        if weight_bits == 8:
            return _helper.A8W8_MXFP_dynamic(
                post_scale=(group_size is None), **args)
        if weight_bits == 4:
            return _helper.A8W4_MXFP_dynamic(**args)

    factory = _MODE_TO_HELPER.get(quant_mode)
    if factory is not None:
        if quant_mode in ("int8_dynamic", "fp8_dynamic"):
            return factory(block_quant=block_quant, **args)
        return factory(**args)

    raise ValueError(
        f"Unsupported (weight_bits={weight_bits}, quant_mode={quant_mode!r})"
    )
