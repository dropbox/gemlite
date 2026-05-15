# SPDX-License-Identifier: Apache-2.0
"""On-the-fly gemlite quantization: quantize fp16/bf16 checkpoints at load
time. Mirrors hqq.utils.vllm.set_vllm_onthefly_hqq_quant()."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from vllm.model_executor.layers.linear import (
    LinearBase, UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter
import vllm.model_executor.layers.linear as _vllm_linear

from .common import (
    GemliteLinearMethod, is_layer_skipped, load_cache, resolve_processor,
)

logger = logging.getLogger(__name__)


def _error_loader(param, loaded_weight):
    raise ValueError("No loader provided for gemlite weight parameter")


if "GemliteOnTheFlyMethod" not in _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
    _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append("GemliteOnTheFlyMethod")


@register_quantization_config("gemlite_onthefly")
class GemliteOnTheFlyConfig(QuantizationConfig):
    def __init__(self, weight_bits, group_size, quant_mode,
                 block_quant=False, skip_modules=None):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.quant_mode = quant_mode
        self.block_quant = block_quant
        self.skip_modules = skip_modules or []

    def __repr__(self):
        return (f"GemliteOnTheFlyConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, quant_mode={self.quant_mode!r})")

    @classmethod
    def get_name(cls): return "gemlite_onthefly"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls): return 75

    @classmethod
    def get_config_filenames(cls): return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GemliteOnTheFlyConfig":
        return cls(
            weight_bits=config["weight_bits"],
            group_size=config.get("group_size"),
            quant_mode=config["quant_mode"],
            block_quant=config.get("block_quant", False),
            skip_modules=config.get("skip_modules"),
        )

    def get_quant_method(self, layer, prefix) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.skip_modules):
            return UnquantizedLinearMethod()
        return GemliteOnTheFlyMethod(self)


class GemliteOnTheFlyMethod(GemliteLinearMethod):
    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        self.output_size_per_partition = sum(output_partition_sizes)
        self.input_size_per_partition = input_size_per_partition
        self.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                self.output_size_per_partition,
                self.input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1, output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader", _error_loader),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer) -> None:
        load_cache()
        device = layer.weight.device
        cfg = self.quant_config

        processor = resolve_processor(
            weight_bits=cfg.weight_bits, quant_mode=cfg.quant_mode,
            group_size=cfg.group_size, block_quant=cfg.block_quant,
            device=device, dtype=self.params_dtype,
        )

        tmp = torch.nn.Linear(1, 1, bias=False)
        tmp.weight.data = layer.weight.data
        tmp.in_features = layer.weight.shape[1]
        tmp.out_features = layer.weight.shape[0]

        if cfg.quant_mode == "int4_weightonly":
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
            hqq_linear = HQQLinear(
                tmp,
                quant_config=BaseQuantizeConfig(
                    nbits=cfg.weight_bits, group_size=cfg.group_size, axis=1,
                ),
                compute_dtype=self.params_dtype, device=device,
            )
            gl = processor.from_hqqlinear(hqq_linear)
        else:
            gl = processor.from_linear(tmp)

        del layer.weight
        self._finalize(layer, gl)


def set_onthefly_quant(weight_bits=4, group_size=None,
                       quant_mode="int4_weightonly",
                       block_quant=False, skip_modules=None):
    """Patch LinearBase.__init__ to attach a GemliteOnTheFlyConfig to every
    linear that doesn't already have a quant_config."""
    from vllm.model_executor.layers import linear as _linear

    skip = list(skip_modules or [])
    original_init = _linear.LinearBase.__init__

    def patched_init(self, input_size, output_size, skip_bias_add=False,
                     params_dtype=None, quant_config=None, *args, **kwargs):
        if quant_config is None:
            quant_config = GemliteOnTheFlyConfig(
                weight_bits=weight_bits, group_size=group_size,
                quant_mode=quant_mode, block_quant=block_quant,
                skip_modules=skip,
            )
        original_init(self, input_size, output_size, skip_bias_add,
                      params_dtype, quant_config, *args, **kwargs)

    _linear.LinearBase.__init__ = patched_init
    logger.info("gemlite on-the-fly: weight_bits=%d mode=%s group=%s block=%s",
                weight_bits, quant_mode, group_size, block_quant)
