# SPDX-License-Identifier: Apache-2.0
"""gemlite <-> vLLM integration.

Entry points:
  enable_gemlite(...)     -- route pre-quantized checkpoints through gemlite
  set_onthefly_quant(...) -- quantize fp16/bf16 checkpoints at load time
  patch_vllm()            -- env-var driven; called by the autopatch hook
"""

try:
    import vllm  # noqa: F401
except ImportError as e:
    raise ImportError(
        "gemlite.vllm requires vLLM (`pip install vllm`)."
    ) from e

from .onthefly import (  # noqa: F401  (registers gemlite_onthefly)
    GemliteOnTheFlyConfig, GemliteOnTheFlyMethod, set_onthefly_quant,
)
from .backend import SUPPORTED, enable_gemlite, patch_vllm, register  # noqa: F401

__all__ = [
    "set_onthefly_quant", "enable_gemlite", "patch_vllm", "register",
    "SUPPORTED", "GemliteOnTheFlyConfig", "GemliteOnTheFlyMethod",
]
