# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch
from torch import Tensor
import numpy as np
from enum import Enum
import math, json, os
import warnings, random
from typing import List, Union, Tuple, Callable
import logging
from functools import partial
    
from .dtypes import *
from .triton_kernels import *
from .triton_kernels.utils import gpu_supports_float16_acc, IS_HIP
from .triton_kernels import utils

from .bitpack import (
    pack_weights_over_cols_triton,
    pack_weights_over_cols_torch,
)

from .quant_utils import (
    scale_activations_per_token,
    scale_activations_per_block,
    scale_activations_mxfp8,
    scale_activations_mxfp4,
    scale_activations_nvfp4,
)

import threading
FILE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

###################################################################################################################################
# Global variables
###################################################################################################################################
GEMLITE_ACC_DTYPE = {
    DType.FP16: DType.FP32,
    DType.BF16: DType.FP32,
    DType.FP32: DType.FP32,
    DType.FP8: DType.FP32,
    DType.FP8e4: DType.FP32,
    DType.FP8e4nuz: DType.FP32,
    DType.FP8e5: DType.FP32, 
    DType.FP8e5nuz: DType.FP32,
    DType.INT8: DType.INT32,
    DType.MXFP16: DType.FP32,
    DType.MXBF16: DType.FP32,
    DType.MXFP8: DType.FP32,
    DType.MXFP4: DType.FP32,
    DType.NVFP4: DType.FP32,
}

GEMLITE_TRITON_KERNELS = [
    gemv,
    gemv_revsplitK,
    gemv_splitK,
    gemm_splitK, #gemm_splitK / gemm_splitK_persistent
    gemm,
]

GEMLITE_TRITON_MAPPING       = {kernel.matmul_type : kernel for kernel in GEMLITE_TRITON_KERNELS}
GEMLITE_MATMUL_TYPES         = [kernel.matmul_type for kernel in GEMLITE_TRITON_KERNELS]
GEMLITE_MATMUL_TYPES_MAPPING = {GEMLITE_MATMUL_TYPES[i]: i for i in range(len(GEMLITE_MATMUL_TYPES))}
GEMLITE_TRITON_CONFIG_CACHE  = {} #Global config cache for all the kernels
_GROUP_SIZE_WARNED           = False
GEMLITE_USE_TMA              = False # Set to False for faster MXFP8 on sm_120
GEMLITE_ENABLE_PTX_FP4_PACK  = False # Set to True for hardware e2m1x2 FP4 packing (requires CUDA 13.0+ ptxas)
GEMLITE_FAST_NVFP4           = False
GEMLITE_NVFP4_META_SCALES    = []  # Pre-allocated per-GPU meta_scale tensors
GEMLITE_NVFP4_INPUT_SCALES: dict = {}  # per-layer calibrated scale, keyed by W_q.data_ptr()

###################################################################################
#Utils

#Main function to cache kernel autotune config
def cache_kernel_config(kernel, num_keys):
    kernel_cache = kernel.cache
    k_config = {}
    if(len(kernel_cache) > 0):
        for k in kernel_cache:
            key    = list(k[:num_keys])
            key[0] = utils.get_closest_m(key[0]) #restrict batch-size
            key    = tuple(key)
            k_config[str(key)] = kernel_cache[k].all_kwargs()
    return k_config

#Set M autotune logic
def set_autotune_setting(fct): #fct = lambda M: M for max-autotune
    utils.get_closest_m = fct 

#set default packing format
def set_packing_bitwidth(packing_bitwidth : int):
    GemLiteLinearTriton.PACKING_BITWIDTH = packing_bitwidth

#Set accumulation dtype
def set_acc_dtype(dtype):
    global GEMLITE_ACC_DTYPE
    assert dtype in [DType.FP16, DType.FP32], "Invalid dtype (should be DType.FP16 or DType.FP32)."
    GEMLITE_ACC_DTYPE[DType.FP16] = dtype

#Enable/disable TMA for MX kernel data loading
def enable_tma(enabled: bool = True):
    global GEMLITE_USE_TMA
    GEMLITE_USE_TMA = enabled

#Enable/disable Triton warp specialization.
def enable_warp_specialize(enabled: bool = True):
    from .triton_kernels import config as _cfg
    _cfg.WARP_SPECIALIZE = enabled

#Enable/disable BF16 native atomic add (disable for higher FP8/INT8 accuracy)
def set_native_atomic_bfp16(enabled: bool = True):
    from .triton_kernels import utils as _utils
    _utils.GEMLITE_USE_NATIVE_ATOMIC_BFP16 = enabled

#Enable/disable using the input dtype as GEMV accumulator (faster on some GPUs, lower accuracy)
def set_fast_gemv_acc(enabled: bool = True):
    from .triton_kernels import config as _cfg
    _cfg.GEMLITE_FAST_GEMV_ACC = enabled

#Enable/disable hardware PTX FP4 packing in activation quantization (requires CUDA 13.0+ ptxas)
def set_ptx_fp4_pack(enabled: bool = True):
    global GEMLITE_ENABLE_PTX_FP4_PACK
    GEMLITE_ENABLE_PTX_FP4_PACK = enabled
    from .quant_utils import set_ptx_fp4_pack_flag
    set_ptx_fp4_pack_flag(enabled)


# Auto-detect ptxas-blackwell version for hardware FP4 packing support
def auto_detect_ptx_fp4_pack():
    #TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas for example.
    try:
        from triton.knobs import nvidia
        major, minor = (int(x) for x in nvidia.ptxas_blackwell.version.split('.'))
        if (major, minor) >= (13, 0):
            set_ptx_fp4_pack(True)
            return True
    except Exception:
        pass
    return False


#Enable/disable CUDA graph-based autotuning (more accurate but slower)
#Enable/disable fast NVFP4 mode (pre-allocated static meta_scale, skips dynamic computation)
def set_fast_nvfp4(enabled: bool = True, default_value: float = 448.0):
    global GEMLITE_FAST_NVFP4, GEMLITE_NVFP4_META_SCALES
    GEMLITE_FAST_NVFP4 = enabled
    if enabled and len(GEMLITE_NVFP4_META_SCALES) == 0:
        num_gpus = torch.cuda.device_count()
        GEMLITE_NVFP4_META_SCALES = [
            torch.full((1,), fill_value=default_value, device=f"cuda:{i}", dtype=torch.float32)
            for i in range(num_gpus)
        ]

def enable_cudagraph_autotune(enabled: bool = True):
    set_autotune("fast", use_cuda_graph=enabled)

#Return the default gemv kernel to use for M==1
def get_default_gemv(W_nbits: int, mx_dtype: bool = False) -> str:
    #TODO: adapt mx for IS_HIP = True
    if mx_dtype: 
        return 'GEMM_SPLITK' #TODO:'GEMV' if (W_nbits < 8) else 'GEMM_SPLITK' -> Revisit NVFP4 failing test.
    else:
        return 'GEMV_REVSPLITK' if (W_nbits < 8) else 'GEMV_SPLITK'

#matmul type selection logic
def get_matmul_type(batch_size: int, W_nbits: int, mx_dtype: bool = False):
    gemm_limit = 64
    if batch_size >= gemm_limit:
        return "GEMM"
    gemv_limit = 4 if (W_nbits < 8 and not mx_dtype) else 2 # previous 1
    if batch_size > gemv_limit:
        return "GEMM_SPLITK"
    else:
        return get_default_gemv(W_nbits, mx_dtype)

#######################################################################################################################
def enable_activation_scaling(batch_size):
    """
    This functions enables adaptive activation quantization based on the batch-size. 
    For example, it would only do dynamic activation in more compute-bound settings (batch_size >=32 / 64). 
    Only works with the MXFP format - use with A8W4_MXFP/A4W4_MXFP.
    """
    return True
    #return batch_size >= 2 #TODO: Needs Triton fix https://github.com/triton-lang/triton/pull/9577


#Main functional forward call
@torch.library.custom_op("gemlite::forward_functional", mutates_args=())
def forward_functional(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1, #-1: auto, >=0: manual
) -> Tensor:
    
    data_contiguous = bool(meta_args[1])
    W_nbits = meta_args[1]
    out_features = tensor_args[0].shape[1]

    #Get type_id for autotune: use same autotune for float16/bfloat16
    input_dtype = meta_args[5]
    input_dtype = DType.FP16.value if (input_dtype == DType.BF16.value) else input_dtype
    input_dtype = DType.MXFP16.value if (input_dtype == DType.MXBF16.value) else input_dtype
    type_id = input_dtype * 100 + W_nbits

    if not x.is_contiguous():
        x = x.contiguous()

    batch_size = x.numel() // x.shape[-1]
    orig_shape = x.shape
    out_shape = x.shape[:-1] + (out_features,)
    x_dtype = x.dtype

    scaled_activations = bool(meta_args[0]) and enable_activation_scaling(batch_size)
    #Dynamic activation quantization
    scales_x = None
    meta_scale = 0.0
    if(scaled_activations):
        input_dtype = DType(meta_args[5])
        channel_scale_mode = meta_args[9]

        if(input_dtype in FP8_INT8_DTYPES and channel_scale_mode == 5): #INT8 / FP8 block quantization
            x, scales_x = scale_activations_per_block(x, w_dtype=DTYPE_TO_TORCH[input_dtype.value])

        elif(input_dtype in FP8_INT8_DTYPES): #INT8 / FP8
            x, scales_x = scale_activations_per_token(x, w_dtype=DTYPE_TO_TORCH[input_dtype.value])

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 4): #MXPF8
            x, scales_x = scale_activations_mxfp8(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 2): #MXPF8 with post-scale for the activations
            x, scales_x = scale_activations_per_token(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP4] and channel_scale_mode == 4): #MXPF4
            x, scales_x = scale_activations_mxfp4(x)

        elif(input_dtype in [DType.NVFP4] and channel_scale_mode == 4): #NVPF4: TODO
            meta_scale = tensor_args[3]
            _static_meta = GEMLITE_NVFP4_INPUT_SCALES.get(tensor_args[0].data_ptr()) \
                if GEMLITE_NVFP4_INPUT_SCALES else None
            if _static_meta is None and GEMLITE_FAST_NVFP4:
                _static_meta = GEMLITE_NVFP4_META_SCALES[x.device.index]
            x, scales_x, meta_scale_a = scale_activations_nvfp4(x, meta_scale=_static_meta)
            meta_scale = 1.0 / (meta_scale * meta_scale_a)

    # For weight-only NVFP4 (group_size=16): pass stored meta_scale to the kernel
    if not scaled_activations and meta_args[2] == 16:
        meta_scale = tensor_args[3]

    x = x.view(-1, x.shape[-1])

    if(matmul_type >= 0):
        matmul_type_str = GEMLITE_MATMUL_TYPES[matmul_type]
    else:
        matmul_type_str = get_matmul_type(x.shape[0], W_nbits, is_mx_dtype(input_dtype)) #batch_size, W_nbits, is_mx_dtype

    out = (
        GEMLITE_TRITON_MAPPING[matmul_type_str]
        .forward(
            x, *tensor_args[:3], scales_x, *meta_args[1:-1], data_contiguous, type_id, meta_scale=meta_scale
        )
        .view(out_shape)
    )

    if bias is not None:
        out += bias

    return out

@torch.library.register_fake("gemlite::forward_functional")
def forward_functional_fake(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1,
) -> Tensor:
    out_features = tensor_args[0].shape[1]
    return torch.empty(x.shape[:-1] + (out_features,), device=x.device, dtype=x.dtype)

#######################################################################################################################
#Main class
class GemLiteLinearTriton(torch.nn.Module):
    SUPPORTED_BITS_TRITON = [1, 2, 4, 8, 16]
    SUPPORTED_DTYPES = [
        DType.FP16,
        DType.BF16,
        DType.FP32,
        DType.FP8,
        DType.FP8e4,
        DType.FP8e4nuz,
        DType.FP8e5,
        DType.FP8e5nuz,
        DType.INT8,
        DType.MXFP16,
        DType.MXBF16,
        DType.MXFP8,
        DType.MXFP4,
        DType.NVFP4,
    ]
    MIN_SIZE = 32
    PACKING_BITWIDTH = 32  # Default packing bitwidth

    def __init__(
        self,
        W_nbits = 4,
        group_size = 64,
        in_features = None,
        out_features = None,
        input_dtype = DType.FP16,
        output_dtype = DType.FP16,
        acc_dtype = None,
        scaled_activations = False,
    ):
        global _GROUP_SIZE_WARNED

        super().__init__()

        if W_nbits not in GemLiteLinearTriton.SUPPORTED_BITS_TRITON:
            raise NotImplementedError(
                "Only "
                + str(GemLiteLinearTriton.SUPPORTED_BITS_TRITON)
                + " W_nbits are supported."
            )

        if in_features is not None and out_features is not None:
            if (in_features % GemLiteLinearTriton.MIN_SIZE != 0) or (
                (in_features % group_size != 0) if (group_size is not None and W_nbits < 16) else False
            ):
                raise NotImplementedError(
                    "Invalid input shapes: "
                    + str(in_features)
                    + " , "
                    + str(out_features)
                    + ". in_features should be divisible by 32 or the group_size"
                )

        #Warning: Input dtype should be the same as dequantize() weights dtype.
        if input_dtype not in GemLiteLinearTriton.SUPPORTED_DTYPES:
            raise NotImplementedError("Unsupport input dtype: " + str(input_dtype))

        if(group_size is not None):
            if(group_size < 16):
                raise NotImplementedError("Only group_size >= 16 is supported.")

        group_size = 1 if (group_size is None) else group_size

        self.in_features  = in_features
        self.out_features = out_features
        self.W_nbits      = W_nbits
        self.group_size   = group_size
        self.unpack_mask  = 2**self.W_nbits - 1
        self.elements_per_sample = None
        self.signature = (in_features, out_features, W_nbits, group_size)

        self.input_dtype   = input_dtype
        self.output_dtype  = output_dtype
        self.compute_dtype = DTYPE_TO_TORCH[self.input_dtype.value]
        self.meta_dtype    = input_dtype
        
        #Accumulation
        self.acc_dtype = GEMLITE_ACC_DTYPE[self.input_dtype] if(acc_dtype is None) else acc_dtype

        #Scales activations
        if(self.compute_dtype in [torch.float16, torch.bfloat16, torch.float32]):
            self.scaled_activations = False
        else:
            self.scaled_activations = scaled_activations

        #Default forward        
        self.forward = self.forward_auto_no_warmup

        # Register empty placeholders so fresh modules advertise GemLite checkpoint keys.
        self.register_buffer("W_q", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("bias", torch.empty(0, dtype=self.compute_dtype))
        self.register_buffer("scales", torch.empty(0, dtype=torch.int32))
        self.register_buffer("zeros", torch.empty(0, dtype=torch.int32))
        self.register_buffer("metadata", torch.empty(0, dtype=torch.int32))
        self.register_buffer("orig_shape", torch.empty(0, dtype=torch.int32))
        self.register_buffer("meta_scale", torch.tensor(0.0, dtype=torch.float32))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Rebuild metadata from live attributes to ensure consistency
        # (helpers may override channel_scale_mode/W_group_mode after pack())
        if hasattr(self, 'metadata') and self.metadata is not None and hasattr(self, 'W_nbits') and self.W_q.numel() > 0:
            self.metadata = torch.nn.Parameter(
                torch.tensor(self.get_meta_args(), device=self.W_q.device if isinstance(self.W_q, (torch.Tensor, torch.nn.Parameter)) else 'cpu', dtype=torch.int32),
                requires_grad=False,
            )
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.W_q        = state_dict.pop("W_q", None)
        self.bias       = state_dict.pop("bias", None)
        self.scales     = state_dict.pop("scales", None)
        self.zeros      = state_dict.pop("zeros", None)
        self.metadata   = state_dict.pop("metadata", None)
        orig_shape      = state_dict.pop("orig_shape", None)
        _meta_scale     = state_dict.pop("meta_scale", None)

        metadata        = [v.item() for v in self.metadata]
        orig_shape      = tuple(v.item() for v in orig_shape)

        (self.scaled_activations,
        self.W_nbits,
        self.group_size,
        self.unpack_mask,
        self.elements_per_sample,
        self.input_dtype,
        self.output_dtype,
        self.acc_dtype,
        self.meta_dtype,
        self.channel_scale_mode,
        self.W_group_mode,
        self.data_contiguous) = metadata

        self.input_dtype  = DType(self.input_dtype)
        self.output_dtype = DType(self.output_dtype)
        self.acc_dtype    = DType(self.acc_dtype)
        self.meta_dtype   = DType(self.meta_dtype)

        # Restore meta_scale with backward compat for old checkpoints
        if _meta_scale is not None:
            meta_scale = _meta_scale.float()
        else:
            meta_scale = 0.05 if self.input_dtype == DType.NVFP4 else 0.0  # backward compat default for old checkpoints

        self.out_features, self.in_features = orig_shape
        self.compute_dtype = DTYPE_TO_TORCH[self.input_dtype.value]
        self.scaled_activations = bool(self.scaled_activations)
        self.data_contiguous = bool(self.data_contiguous)
        
        # Backward compat: pop stale scales_5d from old saves
        state_dict.pop("scales_5d", None)
        # Convert 2D scales to 5D TMA layout for MX dtypes
        if is_mx_dtype(self.input_dtype) and self.scales is not None:
            s = self.scales.data if isinstance(self.scales, torch.nn.Parameter) else self.scales
            if s.ndim == 2:
                s_2d = s.contiguous()  # [N, K_S] contiguous (matches pack's self.scales.T.contiguous())
                N_dim, K_S = s_2d.shape[0], s_2d.shape[1]
                if GEMLITE_USE_TMA and self.elements_per_sample > 1 and N_dim % 128 == 0 and K_S % 4 == 0:
                    self.scales = s_2d.reshape(N_dim // 128, 4, 32, K_S // 4, 4).permute(0, 3, 2, 1, 4).reshape(1, N_dim // 128, K_S // 4, 2, 256).contiguous()

        # Re-register tensors as nn.Parameter for proper state_dict / .to() / .parameters() support
        _device = self.W_q.device
        self.W_q = torch.nn.Parameter(self.W_q, requires_grad=False)
        self.bias = torch.nn.Parameter(self.bias, requires_grad=False) if self.bias is not None else None
        self.scales = torch.nn.Parameter(self.scales, requires_grad=False)
        self.zeros = torch.nn.Parameter(self.zeros, requires_grad=False)
        self.metadata = torch.nn.Parameter(
            torch.tensor(self.get_meta_args(), device=_device, dtype=torch.int32),
            requires_grad=False,
        )
        self.orig_shape = torch.nn.Parameter(
            torch.tensor([self.out_features, self.in_features], device=_device, dtype=torch.int32),
            requires_grad=False,
        )
        if not isinstance(self.meta_scale, torch.nn.Parameter):
            _ms = meta_scale.item() if isinstance(meta_scale, torch.Tensor) else float(meta_scale)
            self.meta_scale = torch.nn.Parameter(
                torch.tensor(_ms, device=_device, dtype=torch.float32),
                requires_grad=False,
            )



    #Make sure to feed UINT8 W_q for packing
    def pack(
        self,
        W_q: Tensor,
        scales: Tensor,
        zeros: Union[Tensor, int],
        bias: Union[Tensor, None] = None,
        fma_mode: bool = True,
        contiguous: Union[int, None] = None,
        packing_bitwidth: Union[int, None] = None,
        packed: bool = False,
    ):  

        #Check inputs
        raise_unsupported = False
        if zeros is not None and self.input_dtype == DType.INT8:
            if isinstance(zeros, Tensor):
                if zeros.mean() != zeros.int().float().mean():
                    raise_unsupported = True
            elif isinstance(zeros, float):
                raise_unsupported = True

        if raise_unsupported:
            raise Exception("INT8 inputs is not compatible with floating-point zeros.")

        #Set packing bitwidth
        if(packing_bitwidth is None):
            packing_bitwidth = GemLiteLinearTriton.PACKING_BITWIDTH

        #Only 8-bit packing is supported for microscaling
        if(is_mx_dtype(self.input_dtype)):
            packing_bitwidth = 8

        #Non-packed weights
        self.W_q = None
        if(W_q.dtype in [torch.int8] or W_q.is_floating_point()):
            if(W_q.dtype in [torch.float32]):
                assert self.W_nbits == 32, "Invalid fp32 weights."
            elif(W_q.dtype in [torch.float16, torch.bfloat16]): 
                assert self.W_nbits == 16, "Invalid fp16 weights."
            else: 
                assert self.W_nbits == 8, "Invalid 8-bit weights."

            self.W_q = W_q.t()
            self.elements_per_sample = 1

            if(contiguous is None): 
                contiguous = False

        # Pre-packed weights (already nibble-packed, shape [N, K//elements_per_sample])
        if packed and W_q.dtype == torch.uint8:
            if packing_bitwidth is None:
                packing_bitwidth = 8 if is_mx_dtype(self.input_dtype) else GemLiteLinearTriton.PACKING_BITWIDTH
            self.elements_per_sample = packing_bitwidth // self.W_nbits
            self.W_q = W_q.view(self.out_features, self.in_features // self.elements_per_sample).t()

            if contiguous is None:
                contiguous = not is_mx_dtype(self.input_dtype)

        # Packed weigths
        elif W_q.dtype == torch.uint8:  
            _pack_weights_over_cols = pack_weights_over_cols_triton if (W_q.device.type == "cuda") else pack_weights_over_cols_torch

            self.W_q, self.elements_per_sample = _pack_weights_over_cols(
                W_q.view((self.out_features, self.in_features)),
                W_nbits=self.W_nbits,
                packing_bitwidth=packing_bitwidth,
                transpose=True,
            )  # Over-K
            
            if contiguous is None:
                if(is_mx_dtype(self.input_dtype)):
                    contiguous = False
                else:
                    contiguous = True

        if(self.W_q is None):
            raise Exception('Weights were not packed, please check your W_q.dtype')
        
        #Bias / device
        self.device = self.W_q.device
        self.bias = None if (bias is None) else bias.to(device=self.device)
        
        #initial values
        self.W_group_mode = -1
        self.channel_scale_mode = 0

        #FP16 x FP16 / FP8 x FP8 / INT8 x INT8 - no meta-data case 
        if((scales is None) and (zeros is None)):
            self.zeros  = None
            self.scales = None
            self.W_group_mode = 0
            self.channel_scale_mode = 2 if self.scaled_activations else 0 
            
        #The rest of the use-cases require some kind of meta-data
        if(scales is not None):
            self.scales = scales.view((self.out_features, -1)).t()
        else:
            self.scales = None

        #channel-wise scaling 
        self.meta_is_channelwise = False if(self.scales is None) else self.scales.numel() == self.out_features 

        #Symmetric no shift
        if(zeros is None):  
            self.zeros = None
            self.W_group_mode = 2 if(self.scales is not None) else 0
        else:
            #Asymmetric or Symmetric with shift
            if(isinstance(zeros, torch.Tensor)):
                if(fma_mode and (self.meta_is_channelwise is False)): #W ~ Wq * scales + zeros
                    self.zeros = (-zeros.float() * scales.float()).to(zeros.dtype).view((self.out_features, -1)).t()
                    self.W_group_mode = 4
                else: #W ~ (Wq - zeros) * scales
                    self.zeros = zeros.view((self.out_features, -1)).t()
                    self.W_group_mode = 3 
            else: #Integer
                self.zeros = int(zeros) 
                if(self.scales is not None):
                    self.W_group_mode = 3 #Symmetric with shift
                else:
                    self.W_group_mode = 1 #Shift only with integer

        assert self.W_group_mode > -1, "Invalid scales/zeros settings."

        #weight-only
        if((self.scaled_activations == False) and (self.meta_is_channelwise == True)):
            self.channel_scale_mode = 1
            self.W_group_mode = 1 if(self.zeros is not None) else 0 #only with fma_mode=False

        #activation-only
        if((self.scaled_activations == True) and (self.meta_is_channelwise == False)):
            self.channel_scale_mode = 2

        #weight + activation mode
        if((self.scaled_activations == True) and (self.meta_is_channelwise == True)):
             self.channel_scale_mode = 3
             self.W_group_mode = 1 if(self.zeros is not None) else 0 #only with fma_mode=False

        if(self.channel_scale_mode in [1, 3]):
            assert self.W_group_mode not in [3, 4], "Can't use channel_scale_mode with W_group_mode == 3 or 4."

        # if(self.input_dtype == DType.INT8):
        #     assert self.W_group_mode in [1], "Only channel-wise symmetric quantization is supported for INT8 inputs."

        #Dummy values 
        if(isinstance(self.zeros, int)): #Union[Tensor, int] not supported by custom op
            self.zeros = torch.tensor(self.zeros, dtype=torch.int32, device=self.device)
        if(self.zeros is None):
            self.zeros = torch.tensor([[]], dtype=torch.int32, device=self.device)
        if(self.scales is None):
            self.scales = torch.tensor([[]], dtype=torch.int32, device=self.device)

        #Force contiguous
        if(contiguous):
            self.data_contiguous = True
            self.W_q = self.W_q.contiguous()
        else:
            self.data_contiguous = False

        if(isinstance(self.scales, torch.Tensor)):
            self.scales = self.scales.contiguous()
        if(isinstance(self.zeros, torch.Tensor)):
            self.zeros = self.zeros.contiguous()

        #MX dtypes scaling
        if(self.input_dtype in [DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4]):
            self.scales = self.scales.to(torch.float8_e8m0fnu).view(torch.uint8)
        if(self.input_dtype in [DType.NVFP4]):
            self.scales = self.scales.to(torch.float8_e4m3fn)
        if(is_mx_dtype(self.input_dtype)):
            self.W_group_mode = 2
            self.channel_scale_mode = 0
            
            ################################
            # TMA
            K, N = self.W_q.shape
            
            if(self.input_dtype in [DType.MXFP4, DType.NVFP4]):
                K *= 2
                group_size = 2 * self.W_q.numel() // self.scales.numel()
            else:
                group_size = self.W_q.numel() // self.scales.numel()
            
            # Preshuffle weight scales to 5D TMA layout for fast loading
            # Original: [K_S, N] -> transpose to [N, K_S] -> 5D: [1, N//128, K_S//4, 2, 256]
            K_S = K // group_size
            if GEMLITE_USE_TMA and self.elements_per_sample > 1 and N % 128 == 0 and K_S % 4 == 0:
                # Currently TMA only enabled for MXFP4/NVFP4 NOT for MXFP8 because of poor performance on sm_120 (self.elements_per_sample > 1 check)
                self.scales = self.scales.T.contiguous().reshape(N // 128, 4, 32, K_S // 4, 4).permute(0, 3, 2, 1, 4).reshape(1, N // 128, K_S // 4, 2, 256).contiguous()
            else:
                # Keep 2D transposed layout for pointer-based fallback
                self.scales = self.scales.T
            ################################

        if(self.scales is not None):
            self.meta_dtype = TORCH_TO_DTYPE[self.scales.dtype]

        #Register tensors as buffers
        self.W_q = torch.nn.Parameter(self.W_q, requires_grad=False)
        self.bias = torch.nn.Parameter(self.bias, requires_grad=False) if self.bias is not None else None
        self.scales = torch.nn.Parameter(self.scales,requires_grad=False)
        self.zeros  = torch.nn.Parameter(self.zeros, requires_grad=False)

        #Register metadata
        self.metadata = torch.nn.Parameter(
            torch.tensor(self.get_meta_args(), device=self.device, dtype=torch.int32),
            requires_grad=False,
        )
        
        self.orig_shape = torch.nn.Parameter(
            torch.tensor([self.out_features, self.in_features], device=self.device, dtype=torch.int32),
            requires_grad=False,
        )
        

        self.meta_scale = torch.nn.Parameter(
            torch.tensor(float(self.meta_scale), device=self.device, dtype=torch.float32),
            requires_grad=False,
        )

        return self

    #Return the main arguments
    def get_tensor_args(self):
        meta_scale = self.meta_scale
        if not isinstance(meta_scale, torch.Tensor):
            meta_scale = torch.tensor(meta_scale, dtype=torch.float32, device=self.W_q.device)
        return [self.W_q, self.scales, self.zeros, meta_scale]

    def get_meta_args(self):
        return [int(self.scaled_activations),
                self.W_nbits,
                self.group_size,
                self.unpack_mask,
                self.elements_per_sample,
                self.input_dtype.value,
                self.output_dtype.value,
                self.acc_dtype.value,
                self.meta_dtype.value,
                self.channel_scale_mode,
                self.W_group_mode,
                int(self.data_contiguous),
                ]

    # #Main manual call
    def forward_manual(self, x: Tensor, matmul_type: str="GEMM") -> Tensor:
        return forward_functional(
            x,
            self.bias,
            self.get_tensor_args(),
            self.get_meta_args(),
            GEMLITE_MATMUL_TYPES_MAPPING[matmul_type],
        )

    # #Main auto call without exhaustive search
    def forward_auto_no_warmup(self, x: Tensor) -> Tensor:
        return forward_functional(
            x,
            self.bias,
            self.get_tensor_args(),
            self.get_meta_args(),
        )

    @staticmethod
    def cache_config(filename: str):
        global GEMLITE_TRITON_CONFIG_CACHE
        #Load existing cache if available
        try:
            with FILE_LOCK, open(filename, 'r') as json_file:
                config = json.load(json_file)
        except:
            config = {}
    
        #Can't use GEMLITE_TRITON_MAPPING for some reason kernel.cache is empty
        _GEMLITE_TRITON_MAPPING = {}
        from .triton_kernels.gemv_kernels import gemv
        _GEMLITE_TRITON_MAPPING['GEMV'] = gemv

        from .triton_kernels.gemv_revsplitK_kernels import gemv_revsplitK
        _GEMLITE_TRITON_MAPPING['GEMV_REVSPLITK'] = gemv_revsplitK

        from .triton_kernels.gemv_splitK_kernels import gemv_splitK
        _GEMLITE_TRITON_MAPPING['GEMV_SPLITK'] = gemv_splitK

        from .triton_kernels.gemm_splitK_kernels import gemm_splitK
        _GEMLITE_TRITON_MAPPING['GEMM_SPLITK'] = gemm_splitK

        from .triton_kernels.gemm_kernels import gemm
        _GEMLITE_TRITON_MAPPING['GEMM'] = gemm

        for name in _GEMLITE_TRITON_MAPPING:
            if(name not in config): 
                config[name] = {}
            
            if(name in GEMLITE_TRITON_CONFIG_CACHE):
                config[name].update(GEMLITE_TRITON_CONFIG_CACHE[name])

            for kernel in _GEMLITE_TRITON_MAPPING[name].kernel:
                config[name].update(cache_kernel_config(kernel, 6)) #5: len(prune_keys)

        #Save combined cache
        with FILE_LOCK, open(filename, "w") as json_file: 
            json.dump(config, json_file)

    @staticmethod
    def load_config(
        filename: str, print_error: bool = True, overwrite: bool = False
    ):
        global GEMLITE_TRITON_CONFIG_CACHE
        if(filename is None):
            return False
        try:
            with FILE_LOCK, open(filename, 'r') as json_file:
                config = json.load(json_file)
                if(overwrite):
                    GEMLITE_TRITON_CONFIG_CACHE = config
                else:
                    for name in config:
                        if(name not in GEMLITE_TRITON_CONFIG_CACHE): 
                            GEMLITE_TRITON_CONFIG_CACHE[name] = {}
                        GEMLITE_TRITON_CONFIG_CACHE[name].update(config[name])

        except Exception as e:
            if(print_error):
                logger.error(f"Failed to load the cache file '{filename}': {e}")
            return False
        return True 

    @staticmethod
    def reset_config():
        global GEMLITE_TRITON_CONFIG_CACHE
        GEMLITE_TRITON_CONFIG_CACHE = {}

###################################################################################################################################
###################################################################################################################################
GemLiteLinear = GemLiteLinearTriton  # Triton by default

#Setting default config
def get_default_cache_config():
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/")
    
    def get_tags(path):
        return [f.split('.')[0] for f in os.listdir(path)]

    name = torch.cuda.get_device_properties(0).name.lower().replace(' ', '_')
    tags = get_tags(root_path)
    tags.sort(key=len, reverse=True)

    selected_tag = None
    for tag in tags:
        if(tag in name):
            selected_tag = os.path.join(root_path, tag + '.json')
            break
    
    return selected_tag

selected_tag = get_default_cache_config()
if(GemLiteLinear.load_config(selected_tag)):
    logger.warning('Loaded ' + selected_tag + ' config.')
