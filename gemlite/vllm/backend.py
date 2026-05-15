# SPDX-License-Identifier: Apache-2.0
"""Route pre-quantized checkpoints through gemlite by subclassing vLLM's
QuantizationConfigs and overriding get_quant_method. Also the env-driven
entry point (`patch_vllm`) and the vLLM plugin hook (`register`).

Subclasses are module-scoped so they pickle into worker subprocesses."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.awq import (
    AWQConfig, AWQLinearMethod,
)
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig, GGUFLinearMethod,
)
from vllm.model_executor.layers.quantization.gptq import (
    GPTQConfig, GPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config, ModelOptNvFp4LinearMethod,
)

from gemlite.triton_kernels.config import BLOCK_QUANT_SIZE

from .common import load_cache
from .schemes import (
    GemliteAwqLinearMethod, GemliteCTW4A4Fp4, GemliteCTW4A16Fp4,
    GemliteCTW4A16Mxfp4, GemliteCTWNA16Int, GemliteFp8BlockLinearMethod,
    GemliteFp8PerTensorLinearMethod, GemliteA16W4GroupLinearMethod,
    GemliteGGUFLinearMethod, GemliteNvFp4LinearMethod,
)


logger = logging.getLogger(__name__)


SUPPORTED = {
    "A8W8_FP8_DYNAMIC",     # FP8 dynamic (block + per-tensor/channel)
    "A16W8_FP8",            # FP8 weight-only per-channel
    "A16W8_INT8",           # INT8 weight-only
    "A8W8_INT8_DYNAMIC",    # INT8 dynamic
    "A4W4_NVFP_DYNAMIC",    # NVFP4 (ModelOpt + compressed-tensors)
    "A4W4_MXFP_DYNAMIC",    # MXFP4 dynamic
    "A16W4_MXFP",           # MXFP4 weight-only
    "A16W2_HQQ_INT",        # GGUF Q2_K int2 weight-only
    "A16W4_HQQ_INT",        # HQQ/GPTQ/AWQ int4 weight-only
    "A16W8_HQQ_INT",        # HQQ/GPTQ/AWQ int8 weight-only
}

_ALIASES = {"A16W4_INT": "A16W4_HQQ_INT", "A16W8_INT": "A16W8_HQQ_INT"}
_HQQ_TOGGLE = {4: "A16W4_HQQ_INT", 8: "A16W8_HQQ_INT"}
_PATCHED = False
_ENABLED: set[str] = set()


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------

class GemliteFp8Config(Fp8Config):
    @classmethod
    def get_name(cls):
        return "fp8"

    def _block_ok(self) -> bool:
        if self.weight_block_size is None:
            return False
        bn, bk = self.weight_block_size
        return bn == BLOCK_QUANT_SIZE and bk == BLOCK_QUANT_SIZE

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if not (isinstance(method, Fp8LinearMethod)
                and self.is_checkpoint_fp8_serialized):
            return method
        if self._block_ok() and "A8W8_FP8_DYNAMIC" in _ENABLED:
            return GemliteFp8BlockLinearMethod(self)
        if (self.weight_block_size is None
                and self.activation_scheme == "dynamic"
                and "A16W8_FP8" in _ENABLED):
            return GemliteFp8PerTensorLinearMethod(self)
        return method


# ---------------------------------------------------------------------------
# NVFP4 (ModelOpt) + compressed-tensors
# ---------------------------------------------------------------------------

class GemliteModelOptNvFp4Config(ModelOptNvFp4Config):
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if (isinstance(method, ModelOptNvFp4LinearMethod)
                and "A4W4_NVFP_DYNAMIC" in _ENABLED):
            return GemliteNvFp4LinearMethod(self, method)
        return method


class GemliteCompressedTensorsConfig(CompressedTensorsConfig):
    @classmethod
    def get_name(cls):
        return "compressed-tensors"

    def _get_scheme_from_parts(self, weight_quant, input_quant,
                               format=None, layer_name=None):
        if self._is_nvfp4_format(weight_quant) and input_quant is None:
            if _ENABLED & {"A16W4_MXFP", "A4W4_NVFP_DYNAMIC"}:
                return GemliteCTW4A16Fp4()
        if self._is_mxfp4(weight_quant) and "A16W4_MXFP" in _ENABLED:
            return GemliteCTW4A16Mxfp4()

        # compressed_tensors moved this helper between versions; vLLM also
        # re-exports it from its CT utils module. Try all known locations so
        # we never silently fall back to act_fmt=False (which hid NVFP4 W4A4
        # routing in earlier revisions).
        is_act = None
        for _modname, _attr in (
            ("compressed_tensors.quantization", "is_activation_quantization_format"),
            ("compressed_tensors.compressors.format", "is_activation_quantization_format"),
            ("vllm.model_executor.layers.quantization.compressed_tensors.utils",
             "is_activation_quantization_format"),
        ):
            try:
                _m = __import__(_modname, fromlist=[_attr])
                is_act = getattr(_m, _attr, None)
                if is_act is not None:
                    break
            except Exception:
                continue
        if is_act is None:
            raise RuntimeError(
                "gemlite.vllm: could not locate is_activation_quantization_format "
                "in compressed_tensors or vllm; update the gemlite backend."
            )
        act_fmt = is_act(format or self.quant_format)
        if (act_fmt
                and self._is_nvfp4_format(weight_quant)
                and self._is_nvfp4_format(input_quant)
                and "A4W4_NVFP_DYNAMIC" in _ENABLED):
            return GemliteCTW4A4Fp4()

        # CT pack_quantized WNA16 int4/int8 (weight-only) without actorder.
        # Shares the A16W{4,8}_HQQ_INT toggle with GPTQ/AWQ.
        if (input_quant is None
                and getattr(weight_quant, "type", None) == "int"
                and weight_quant.num_bits in (4, 8)
                and weight_quant.strategy in ("group", "channel")
                and not weight_quant.dynamic
                and getattr(weight_quant, "actorder", None) in (None, "static")
                and _HQQ_TOGGLE[weight_quant.num_bits] in _ENABLED
                and (format or self.quant_format) == "pack-quantized"):
            return GemliteCTWNA16Int(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                symmetric=weight_quant.symmetric,
                group_size=weight_quant.group_size,
                actorder=weight_quant.actorder,
                layer_name=layer_name,
            )

        return super()._get_scheme_from_parts(
            weight_quant, input_quant, format=format, layer_name=layer_name,
        )


# ---------------------------------------------------------------------------
# GPTQ + AWQ (and their marlin auto-upgrades)
# ---------------------------------------------------------------------------

def _gptq_method(cfg, stock, is_v1):
    return GemliteA16W4GroupLinearMethod(
        cfg, stock, weight_bits=cfg.weight_bits,
        qweight_pack_dim=0, qzeros_pack_dim=1, gptq_v1_plus_one=is_v1,
    )


def _awq_method(cfg, stock):
    return GemliteAwqLinearMethod(
        cfg, stock, weight_bits=cfg.weight_bits, zero_point=cfg.zero_point,
    )


def _gptq_enabled(cfg) -> bool:
    return (cfg.weight_bits in (4, 8)
            and _HQQ_TOGGLE[cfg.weight_bits] in _ENABLED)


class GemliteGptqConfig(GPTQConfig):
    @classmethod
    def get_name(cls):
        return "gptq"

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, GPTQLinearMethod) and _gptq_enabled(self):
            is_v1 = getattr(self, "checkpoint_format", "") != "gptq_v2"
            return _gptq_method(self, method, is_v1)
        return method


class GemliteGptqMarlinConfig(GPTQMarlinConfig):
    """Catches vLLM's auto-upgrade `gptq` -> `gptq_marlin`."""

    @classmethod
    def get_name(cls):
        return "gptq_marlin"

    def get_quant_method(self, layer, prefix):
        if (_gptq_enabled(self) and isinstance(layer, LinearBase)
                and self.is_sym):
            stock = GPTQLinearMethod(GPTQConfig(
                weight_bits=self.weight_bits, group_size=self.group_size,
                desc_act=self.desc_act,
                lm_head_quantized=self.lm_head_quantized,
                dynamic=self.dynamic,
            ))
            is_v1 = self.full_config.get("checkpoint_format", "") != "gptq_v2"
            return _gptq_method(self, stock, is_v1)
        return super().get_quant_method(layer, prefix)


class GemliteAwqConfig(AWQConfig):
    @classmethod
    def get_name(cls):
        return "awq"

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, AWQLinearMethod) and _gptq_enabled(self):
            return _awq_method(self, method)
        return method


class GemliteAwqMarlinConfig(AWQMarlinConfig):
    """Catches vLLM's auto-upgrade `awq` -> `awq_marlin`."""

    @classmethod
    def get_name(cls):
        return "awq_marlin"

    def get_quant_method(self, layer, prefix):
        if _gptq_enabled(self) and isinstance(layer, LinearBase):
            stock = AWQLinearMethod(AWQConfig(
                weight_bits=self.weight_bits, group_size=self.group_size,
                zero_point=self.zero_point,
                modules_to_not_convert=self.modules_to_not_convert,
            ))
            return _awq_method(self, stock)
        return super().get_quant_method(layer, prefix)


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------

_GGUF_TOGGLES = frozenset({"A16W2_HQQ_INT", "A16W4_HQQ_INT", "A16W8_HQQ_INT"})


class GemliteGGUFConfig(GGUFConfig):
    """Wrap stock GGUFLinearMethod with GemliteGGUFLinearMethod for supported
    affine block types:
        Q4_0 / Q4_1 / Q4_K -> A16W4_HQQ_INT
        Q8_0               -> A16W8_HQQ_INT
        Q2_K               -> A16W2_HQQ_INT
    Unsupported types (Q3_K/Q5_K/Q6_K, IQ_*, MoE, embedding) fall through."""

    @classmethod
    def get_name(cls):
        return "gguf"

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if (type(method) is GGUFLinearMethod
                and _ENABLED & _GGUF_TOGGLES):
            return GemliteGGUFLinearMethod(self)
        return method


# ---------------------------------------------------------------------------
# enable_gemlite
# ---------------------------------------------------------------------------

_V2_SUPPORTED_METHODS = (
    "GemliteNvFp4LinearMethod",
    "GemliteFp8BlockLinearMethod",
    "GemliteFp8PerTensorLinearMethod",
    "GemliteA16W4GroupLinearMethod",
    "GemliteAwqLinearMethod",
    "GemliteA16W8Int8LinearMethod",
    "GemliteA8W8Int8DynamicLinearMethod",
    # GGUF uses the legacy weight loader (shard_id / data_container) — keep it
    # OFF the v2 list or fused qkv/gate_up loading routes to load_qkv_weight
    # which isn't set up on plain UninitializedParameters.
)


def _register_v2_methods() -> None:
    # Fused MergedColumn/QKV loaders only dispatch to the v2 path for classes
    # in WEIGHT_LOADER_V2_SUPPORTED; without this they fall back to naive copy
    # and crash on shape mismatch.
    import vllm.model_executor.layers.linear as _vl
    for name in _V2_SUPPORTED_METHODS:
        if name not in _vl.WEIGHT_LOADER_V2_SUPPORTED:
            _vl.WEIGHT_LOADER_V2_SUPPORTED.append(name)


def _build_overrides() -> dict[str, type]:
    o: dict[str, type] = {}
    if _ENABLED & {"A8W8_FP8_DYNAMIC", "A16W8_FP8"}:
        o["fp8"] = GemliteFp8Config
    if "A4W4_NVFP_DYNAMIC" in _ENABLED:
        o["modelopt_fp4"] = GemliteModelOptNvFp4Config
    if _ENABLED & {"A4W4_NVFP_DYNAMIC", "A4W4_MXFP_DYNAMIC",
                   "A16W4_MXFP", "A16W4_HQQ_INT", "A16W8_HQQ_INT"}:
        o["compressed-tensors"] = GemliteCompressedTensorsConfig
    if _ENABLED & {"A16W4_HQQ_INT", "A16W8_HQQ_INT"}:
        o["gptq"] = GemliteGptqConfig
        o["gptq_marlin"] = GemliteGptqMarlinConfig
        o["awq"] = GemliteAwqConfig
        o["awq_marlin"] = GemliteAwqMarlinConfig
    if _ENABLED & _GGUF_TOGGLES:
        o["gguf"] = GemliteGGUFConfig
    return o


def enable_gemlite(names: Optional[Iterable[str]] = None) -> None:
    """Swap vLLM's quant-config registry so requested types load through
    gemlite. `names=None` enables every type in `SUPPORTED`."""
    global _PATCHED, _ENABLED
    import vllm.model_executor.layers.quantization as _vq

    requested = {_ALIASES.get(n, n)
                 for n in (set(names) if names is not None else SUPPORTED)}
    unknown = requested - SUPPORTED
    if unknown:
        raise ValueError(f"Unknown gemlite quant names: {sorted(unknown)}. "
                         f"Supported: {sorted(SUPPORTED)}")
    _ENABLED = requested

    load_cache()
    _register_v2_methods()
    overrides = _build_overrides()

    if not _PATCHED:
        orig = _vq.get_quantization_config

        def _patched(quantization: str):
            return overrides.get(quantization) or orig(quantization)

        _vq.get_quantization_config = _patched
        # Other modules `from ...quantization import get_quantization_config`,
        # which captures the original as a local name — patch those bindings
        # too or they keep returning the stock classes.
        for modname in (
            "vllm.model_executor.model_loader.weight_utils",
            "vllm.model_executor.models.llama4_eagle",
            "vllm.model_executor.models.mistral_large_3_eagle",
        ):
            try:
                _mod = __import__(modname, fromlist=["get_quantization_config"])
            except Exception:
                continue
            if getattr(_mod, "get_quantization_config", None) is orig:
                _mod.get_quantization_config = _patched
        _PATCHED = True

    registry = getattr(_vq, "QUANTIZATION_METHODS", None)
    if registry is not None:
        for name, cls in overrides.items():
            try:
                registry[name] = cls
            except Exception:
                pass

    logger.warning("gemlite enabled for: %s", sorted(_ENABLED))


# ---------------------------------------------------------------------------
# Env-driven entry point + vLLM plugin hook
#
# Env vars:
#   VLLM_GEMLITE_ENABLE         -- "0"/"1" (default "0"). Pre-quantized path.
#   VLLM_GEMLITE_ENABLE_LIST    -- comma-separated subset of SUPPORTED names.
#   VLLM_GEMLITE_ONTHEFLY_QUANT -- preset name; on-the-fly quantization.
#   VLLM_GEMLITE_SKIP_MODULES   -- comma-separated skip list overrides.
# ---------------------------------------------------------------------------

_DEFAULT_SKIP = ["lm_head", "visual", "vision"]

_ONTHEFLY_PRESETS = {
    "A16W8_INT8":             dict(weight_bits=8, group_size=None, quant_mode="int8_weightonly"),
    "A16W8_FP8":              dict(weight_bits=8, group_size=None, quant_mode="fp8_weightonly"),
    "A16W4_INT4_HQQ":         dict(weight_bits=4, group_size=64,   quant_mode="int4_weightonly"),
    "A8W8_INT8_DYNAMIC":      dict(weight_bits=8, group_size=None, quant_mode="int8_dynamic"),
    "A8W8_FP8_DYNAMIC":       dict(weight_bits=8, group_size=None, quant_mode="fp8_dynamic"),
    "A8W8_FP8_DYNAMIC_BLOCK": dict(weight_bits=8, quant_mode="fp8_dynamic", block_quant=True),
    "MXFP8_DYNAMIC":          dict(weight_bits=8, group_size=32,   quant_mode="mxfp8_dynamic"),
    "MXFP4_WEIGHTONLY":       dict(weight_bits=4, quant_mode="mxfp4_weightonly"),
    "MXFP4_DYNAMIC":          dict(weight_bits=4, quant_mode="mxfp4_dynamic"),
    "A8W4_MXFP_DYNAMIC":      dict(weight_bits=4, quant_mode="mxfp8_dynamic"),
    "NVFP4_DYNAMIC":          dict(weight_bits=4, quant_mode="nvfp4_dynamic"),
}

_APPLIED = False


def _split_csv(val: str) -> list[str]:
    return [s.strip() for s in val.split(",") if s.strip()]


def patch_vllm() -> None:
    """Idempotent. Called from vllm/engine/__init__.py at import time."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    skip_env = os.environ.get("VLLM_GEMLITE_SKIP_MODULES")
    skip_modules = _split_csv(skip_env) if skip_env else list(_DEFAULT_SKIP)

    if os.environ.get("VLLM_GEMLITE_ENABLE", "0") != "0":
        names_env = os.environ.get("VLLM_GEMLITE_ENABLE_LIST")
        try:
            enable_gemlite(_split_csv(names_env) if names_env else None)
        except Exception as e:
            logger.warning("enable_gemlite failed: %s", e)

    mode = os.environ.get("VLLM_GEMLITE_ONTHEFLY_QUANT")
    if mode:
        preset = _ONTHEFLY_PRESETS.get(mode)
        if preset is None:
            logger.warning("VLLM_GEMLITE_ONTHEFLY_QUANT=%r unknown; valid: %s",
                           mode, sorted(_ONTHEFLY_PRESETS))
        else:
            from .onthefly import set_onthefly_quant
            set_onthefly_quant(skip_modules=skip_modules, **preset)

    logger.warning("gemlite vLLM patch applied (mode=%s)", mode)


def register() -> None:
    """vLLM plugin entry point. Install in setup.py:
        entry_points={"vllm.general_plugins":
                      ["gemlite = gemlite.vllm.backend:register"]}"""
    try:
        from . import onthefly  # noqa: F401  (registers gemlite_onthefly)
        patch_vllm()
    except Exception as e:
        logger.warning("gemlite plugin register() failed: %s", e)


# v1 spawn children don't run the engine/__init__ autopatch; re-run here so
# _ENABLED is populated when a Gemlite*Config unpickles and imports us.
if os.environ.get("VLLM_GEMLITE_ENABLE", "0") != "0" or os.environ.get(
    "VLLM_GEMLITE_ONTHEFLY_QUANT"
):
    try:
        patch_vllm()
    except Exception as _e:
        logger.warning("gemlite backend import-time autopatch failed: %s", _e)
