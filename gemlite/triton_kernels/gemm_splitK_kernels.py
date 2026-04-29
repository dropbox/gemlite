# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
from ..dtypes import is_mx_dtype
from .config import AUTOTUNE, BLOCK_QUANT_SIZE
from . import config
from .utils import *

KEYS          = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'a_sizeof', 'b_sizeof', 'channel_scale_mode'] 
MATMUL_TYPE   = "GEMM_SPLITK"

def kernel_config_pruner(configs, nargs, **kwargs):
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = nargs['type_id']
    a_sizeof = nargs['a_sizeof']
    b_sizeof = nargs['b_sizeof']

    #Check cache
    load_scales_as_block = kwargs['load_scales_as_block']
    channel_scale_mode = kwargs.get('channel_scale_mode', 0)
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        signature = str(tuple([get_closest_m(m), n, k, g, e, t]))
        if(signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps  = config.pop('num_warps')
            num_ctas   = config.pop('num_ctas', 1)

            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)
            config["NUM_STAGES"] = num_stages

            config.pop('EVEN_M', None)
            config.pop('EVEN_K', None)
            config.pop('EVEN_N', None)

            # Adjust 5D TMA compatibility for cached configs
            if load_scales_as_block and n % 128 == 0 and (k // g) % 4 == 0:
                config['BLOCK_SIZE_N'] = max(config['BLOCK_SIZE_N'], 128)
                while (config['BLOCK_SIZE_K'] // g) % 4 != 0:
                    config['BLOCK_SIZE_K'] *= 2

            # INT Block-quant (mode==5): clamp block sizes <= BLOCK_QUANT_SIZE
            if channel_scale_mode == 5:
                # BLOCK_SIZE_M is unconstrained (scales_a is [M, K/BLOCK_QUANT_SIZE])
                config['BLOCK_SIZE_N'] = min(config['BLOCK_SIZE_N'], BLOCK_QUANT_SIZE)
                config['BLOCK_SIZE_K'] = min(config['BLOCK_SIZE_K'], BLOCK_QUANT_SIZE)

            yield triton.Config(config,
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=init_to_zero("c_ptr") if (config['SPLIT_K'] > 1) else None,
            )
            return

    gpu_shared_memory = get_gpu_shared_memory()
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        split_k      = config.kwargs['SPLIT_K']

        A_load_order = config.kwargs['A_load_order']
        num_stages   = config.num_stages
        num_warps    = config.num_warps

        #Autotune prune the batch_size (1..64)
        if m <= 16:   block_size_m = 16
        elif m <= 32: block_size_m = min(max(block_size_m, 16), 32) #m: [16, 32]
        elif m <= 64: block_size_m = min(max(block_size_m, 32), 64) #m: [32, 64]
        elif m > 64 : block_size_m = 64

        #Only use higher split_k values for smaller m
        if(m >= 32): split_k = min(split_k, 8)

        #Constraints
        if(load_scales_as_block):
            if(e > 1):
                block_size_k = max(block_size_k, 64) #m16n8k64
            else:
                block_size_k = max(block_size_k, 32) #m16n8k32
            # 5D TMA scale compatibility: adjust block sizes for 5D TMA descriptor
            if n % 128 == 0 and (k // g) % 4 == 0:
                block_size_n = max(block_size_n, 128)
                while (block_size_k // g) % 4 != 0:
                    block_size_k *= 2
        else:
            block_size_k = max(min(block_size_k, g), 32) #tl.dot minimum K

        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)
        split_k      = max(split_k, 1)
        
        # INT Block-quant (mode==5): block sizes must be <= BLOCK_QUANT_SIZE
        if channel_scale_mode == 5:
            # BLOCK_SIZE_M is unconstrained (scales_a is [M, K/BLOCK_QUANT_SIZE])
            block_size_n = min(block_size_n, BLOCK_QUANT_SIZE)
            block_size_k = min(block_size_k, BLOCK_QUANT_SIZE)
        
        if not IS_HIP:
            if e == 1 and num_stages == 1:
                continue

        # Reduce num_stages until config fits in shared memory
        while num_stages > 1:
            estimated_smem = estimate_shared_memory_per_block(
                block_size_m, block_size_n, block_size_k,
                a_sizeof, b_sizeof, num_stages, e, g,
                load_scales_as_block
            )
            if estimated_smem <= gpu_shared_memory:
                break
            num_stages -= 1

        key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, A_load_order, num_stages, num_warps)
        

        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
            "SPLIT_K": split_k,
            "A_load_order": A_load_order,
            "NUM_STAGES": num_stages,
        }

        if IS_HIP:
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            new_config['matrix_instr_nonkdim'] = config.kwargs.get('matrix_instr_nonkdim', 16) #MI300X
            key = key + (new_config['waves_per_eu'], new_config['matrix_instr_nonkdim'])

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config,
            num_stages=num_stages,
            num_warps=num_warps,
            pre_hook=init_to_zero("c_ptr") if split_k > 1 else None, 
        )

########################################################################################################################################################################
#Nvidia

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_max_autotune_config_nvidia():
    stages  = [1, 2, 3, 4, 5]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128, 256, 512]:
                            for split_k in [1, 2, 4, 8, 16]:
                                configs.append(
                                    triton.Config(
                                        {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, 
                                        "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A},
                                        num_warps=w, num_stages=s,
                                    )
                                )
    return configs

#Faster autotuner 
def get_fast_autotune_config_nvidia():
    configs = []
    #Small N tiles
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    #Medium N tiles (N=128 — workhorse for MX/INT types)
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    #Large N tiles
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':2,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':256, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':32,  'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    #High split_k with wide N
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'SPLIT_K':8,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    #Extra coverage
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':2,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64,  'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    #Additional M=16 configs for MX kernel coverage
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':2,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    #M=32 tiles (for M=32..64 batch sizes)
    configs.append(triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':1,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':4,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':2,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    return configs

def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=2),
            ]

########################################################################################################################################################################
#AMD - Instinct MI300X

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_max_autotune_config_amd():
    configs = []
    for A in [0]:
        for w in [4, 8]:
            for s in [1, 2]:
                for v in [0, 2, 4]:
                    for M in [16, 32, 64]:
                        for N in [32, 64, 128, 256, 512]:
                            for K in [32, 64, 128, 256, 512]:
                                for split_k in [1, 2, 4, 8, 16]:
                                    configs.append(
                                        triton.Config(
                                            {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, 
                                            "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A, 'waves_per_eu': v},
                                            num_warps=w, num_stages=s,
                                        )
                                    )
    return configs

#Faster autotuner 
def get_fast_autotune_config_amd():
    configs = [] #BLOCK_SIZE_M is automatically adapted in the config pruning.
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))

    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':4 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':8 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=1))

    return configs

def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':64, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0, 'NUM_STAGES':2}, num_warps=4, num_stages=2)]
########################################################################################################################################################################

if IS_HIP:
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia

AUTOTUNE_SETTING = AUTOTUNE.GEMM_SPLITK
if(AUTOTUNE_SETTING == 'max'):
    get_autotune_config = get_max_autotune_config
elif(AUTOTUNE_SETTING == 'fast'):
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config

@triton.autotune(
    configs=get_autotune_config(),
    key = KEYS,
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.heuristics(values={
    "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_SIZE_K"] * args["SPLIT_K"]) == 0,
})
@triton.jit
def gemm_splitK_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N: tl.constexpr, K: tl.constexpr, M_CLOSEST,
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    #################################
    type_id: tl.constexpr,
    a_sizeof: tl.constexpr,
    b_sizeof: tl.constexpr,
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    load_scales_as_block, #False | IF FALSE, RESTRICT BLOCK_SIZE_K <= 32
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
    SPLIT_K: tl.constexpr, 
    #################################
    NUM_STAGES: tl.constexpr,
    A_load_order: tl.constexpr, 
    data_contiguous: tl.constexpr,
    #################################
    EVEN_M: tl.constexpr = False,
    EVEN_K: tl.constexpr = False, 
    EVEN_N: tl.constexpr = False,
    #################################
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    meta_scale_norm_ptr = None,
    ################################# dmmy
    use_tma: tl.constexpr = True,
    use_5d_scales: tl.constexpr = False,
    block_quant_size: tl.constexpr = BLOCK_QUANT_SIZE,
    warp_specialize: tl.constexpr = False,
):
    """
    Based on https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """


    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    #Swizzle?
    if(elements_per_sample > 1):
        pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    else:
        pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) 

    #Offsets
    #############################################################################################################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < K))

    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn)
    b_mask  = ((offs_bk[:, None] < K) & (offs_bn[None, :] < N))
    q_shift = ((offs_bk % elements_per_sample) * W_nbits).to(tl.int32)[:, None] 
        
    #Meta data stuff
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr     = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_U: tl.constexpr = BLOCK_SIZE_K * SPLIT_K
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample) * SPLIT_K

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')

    # INT Block-quantization (mode==5): BxB weight scales (fp32), per-row per-B-K activation scales (fp32),
    # where B = block_quant_size (kernel arg, defaulted from BLOCK_QUANT_SIZE in config.py).
    if channel_scale_mode == 5:
        scales_a_ptrs     = scales_a_ptr + offs_am * stride_meta_a_m
        scales_b_base_ptr = scales_ptr + ((pid_n * BLOCK_SIZE_N) // block_quant_size) * stride_meta_n

    #############################################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in tl.range(num_pid_k, num_stages=NUM_STAGES, warp_specialize=warp_specialize):

        if(A_load_order == 0): #Early load            
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))

        b = load_ptr(b_ptrs, b_mask, b_evict, not (EVEN_K and EVEN_N))

        if(A_load_order == 1): #Early load
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))
        
        #Meta-data loading policy
        if(W_group_mode > 0):
            k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 

        if(W_group_mode >= 2): #[2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
            if(zero_is_scalar):
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs  + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None
        
        if(A_load_order == 2): #Mid load
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))

        # Unpack and dequantize
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

        if(A_load_order == 3): #Late load 
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))
        
        #Dot
        if channel_scale_mode == 5:
            #INT Block-quant:
            k_m = ((k * SPLIT_K + pid_k) * BLOCK_SIZE_K) // block_quant_size
            scales_a = tl.load(scales_a_ptrs + k_m * stride_meta_a_g,  mask=offs_am < M, other=0.0, eviction_policy=meta_evict_policy)
            scales_b = tl.load(scales_b_base_ptr + k_m * stride_meta_g, eviction_policy=meta_evict_policy)
            tmp = tl.dot(a, b.to(input_dtype), out_dtype=acc_dtype)
            acc += tmp * scales_a[:, None] * scales_b
        else:
            acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)
        
        #Advance
        a_ptrs += BLOCK_SIZE_K_U * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk
        
        offs_ak += BLOCK_SIZE_K * SPLIT_K
        offs_bk += BLOCK_SIZE_K_U

        if not EVEN_K:
            if EVEN_M:
                a_mask = tl.broadcast_to((offs_ak[None, :] < K), [BLOCK_SIZE_M, BLOCK_SIZE_K])
            else:
                a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < K))
            if EVEN_N:
                b_mask = tl.broadcast_to((offs_bk[:, None] < K), [BLOCK_SIZE_K, BLOCK_SIZE_N])
            else:
                b_mask = ((offs_bk[:, None] < K) & (offs_bn[None, :] < N))

    #############################################################################################################
    #Channel-wise scaling
    if channel_scale_mode == 1 or channel_scale_mode == 3:
        scales_b = load_ptr(scales_ptr + offs_bn, offs_bn < N, meta_evict_policy, not EVEN_N, other=1)

    if channel_scale_mode == 2 or channel_scale_mode == 3:
        scales_a = load_ptr(scales_a_ptr + offs_am, offs_am < M, meta_evict_policy, not EVEN_M, other=1)

    if channel_scale_mode == 1:
        acc = acc.to(meta_dtype) * scales_b[None, :]
    elif channel_scale_mode == 2:
        acc = acc.to(meta_dtype) * scales_a[:, None]
    elif channel_scale_mode == 3:
        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    #############################################################################################################
    #Output
    acc     = acc.to(output_dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if(SPLIT_K > 1):
        if EVEN_M and EVEN_N:
            tl.atomic_add(c_ptrs, acc, sem=atomic_mode)
        else:
            tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode) 
    else:
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc)
        else:
            tl.store(c_ptrs, acc, mask=mask)

@triton.autotune(
    configs=get_autotune_config(),
    key = KEYS,
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.heuristics(values={
    "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_SIZE_K"] * args["SPLIT_K"]) == 0,
})
@triton.jit
def gemm_splitK_MX_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N: tl.constexpr, K: tl.constexpr, M_CLOSEST,
    ######### Quant parms #########
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr, 
    #################################
    type_id: tl.constexpr,
    a_sizeof: tl.constexpr,
    b_sizeof: tl.constexpr,
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m: tl.constexpr, 
    stride_meta_a_g: tl.constexpr,
    stride_meta_n: tl.constexpr, 
    stride_meta_g: tl.constexpr,
    ######### Dtypes #########
    load_scales_as_block, #True | IF FALSE, RESTRICT BLOCK_SIZE_K <= 32
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr, 
    SPLIT_K: tl.constexpr, 
    NUM_STAGES: tl.constexpr,
    #################################
    A_load_order: tl.constexpr,
    data_contiguous: tl.constexpr,
    #################################
    EVEN_M: tl.constexpr = False,
    EVEN_K: tl.constexpr = False, 
    EVEN_N: tl.constexpr = False,
    #################################
    meta_evict_policy: tl.constexpr = 'evict_first',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    meta_scale_norm_ptr = None,
    #################################
    use_tma: tl.constexpr = True,
    use_5d_scales: tl.constexpr = False,
    warp_specialize: tl.constexpr = False,
):

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    a_ptr_dtype: tl.constexpr = a_ptr.dtype.element_ty
    if(a_ptr_dtype == tl.float16):
        a_dtype: tl.constexpr = "fp16"
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.bfloat16):
        a_dtype: tl.constexpr = "bf16"
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.float8e4nv):
        a_dtype: tl.constexpr = "e4m3"
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.uint8):
        a_dtype: tl.constexpr = "e2m1" #FP4
        elements_per_sample_a: tl.constexpr = 2

    if(elements_per_sample == 1): #FP8
        b_dtype: tl.constexpr = "e4m3"
    if(elements_per_sample == 2): #FP4
        b_dtype: tl.constexpr = "e2m1"

    #A
    BLOCK_SIZE_K_A_E: tl.constexpr = BLOCK_SIZE_K // elements_per_sample_a
    BLOCK_SIZE_K_A: tl.constexpr = BLOCK_SIZE_K_A_E * SPLIT_K
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = pid_k * BLOCK_SIZE_K_A_E + tl.arange(0, BLOCK_SIZE_K_A_E)
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < (K // elements_per_sample_a)))

    #B
    BLOCK_SIZE_K_B_E: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
    BLOCK_SIZE_K_B: tl.constexpr = BLOCK_SIZE_K_B_E * SPLIT_K
    offs_bk = pid_k * BLOCK_SIZE_K_B_E + tl.arange(0, BLOCK_SIZE_K_B_E)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_mask = ((offs_bk[:, None] < (K // elements_per_sample)) & (offs_bn[None, :] < N))
    
    #Scales
    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_S: tl.constexpr = BLOCK_SIZE_K // group_size
    offs_k_scales = tl.arange(0, BLOCK_SIZE_K_S)
    offs_n_b_scales = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #B scales: [BLOCK_SIZE_N, BLOCK_SIZE_K // group_size]
    if not use_5d_scales:
        scales_b_ptrs = scales_ptr + offs_n_b_scales[:, None] * stride_meta_n + offs_k_scales[None, :] * stride_meta_g

    #A scales
    if(channel_scale_mode == 4):
        scales_a_ptrs = scales_a_ptr + offs_am[:, None] * stride_meta_a_m + offs_k_scales[None, :] * stride_meta_a_g

    if use_tma:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            [M, K // elements_per_sample_a],
            [stride_am, stride_ak],
            [BLOCK_SIZE_M, BLOCK_SIZE_K_A_E]
        )

        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            [N, K // elements_per_sample],
            [stride_bn, stride_bk],
            [BLOCK_SIZE_N, BLOCK_SIZE_K_B_E]
        )

        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            [M, N],
            [stride_cm, stride_cn],
            [BLOCK_SIZE_M, BLOCK_SIZE_N]
        )

    # 5D TMA Descriptors for Scales (preshuffled layout)
    if use_5d_scales:
        rep_n: tl.constexpr = BLOCK_SIZE_N // 128
        rep_k: tl.constexpr = BLOCK_SIZE_K // group_size // 4
        stride_b4: tl.constexpr = 1
        stride_b3: tl.constexpr = 256
        stride_b2: tl.constexpr = 512
        stride_b1: tl.constexpr = 512 * (K // group_size // 4)
        stride_b0: tl.constexpr = stride_b1 * (N // 128)
        scales_b_5d_desc = tl.make_tensor_descriptor(
            scales_ptr,
            [1, N // 128, K // group_size // 4, 2, 256],
            [stride_b0, stride_b1, stride_b2, stride_b3, stride_b4],
            [1, rep_n, rep_k, 2, 256]
        )


    if group_size == 16:
        scales_a_1s = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K_S), value=1, dtype=tl.float32).to(tl.float8e4nv)
        scales_b_1s = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K_S), value=1, dtype=tl.float32).to(tl.float8e4nv)
    else:
        scales_a_1s = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K_S), value=127, dtype=tl.uint8)
        scales_b_1s = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K_S), value=127, dtype=tl.uint8)
    
    _meta_scale_norm = tl.load(meta_scale_norm_ptr, eviction_policy='evict_last') if group_size == 16 else 1.0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(num_pid_k, warp_specialize=warp_specialize):
        if use_tma:
            a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_SIZE_M, (k * SPLIT_K + pid_k) * BLOCK_SIZE_K_A_E])
            b = tl.load_tensor_descriptor(b_desc, [pid_n * BLOCK_SIZE_N, (k * SPLIT_K + pid_k) * BLOCK_SIZE_K_B_E]).T
        else:
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))
                
            b = load_ptr(b_ptrs, b_mask, b_evict, not (EVEN_K and EVEN_N))

        #k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32)
        k_m = (k * SPLIT_K + pid_k) * BLOCK_SIZE_K_S #OK for BLOCK_SIZE_K >=group_size
        if use_5d_scales:
            scale_b_raw = tl.load_tensor_descriptor(scales_b_5d_desc, [0, pid_n * rep_n, (k * SPLIT_K + pid_k) * rep_k, 0, 0])
            scales_b = scale_b_raw.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K_S)
        else:
            scale_b_mask = ((offs_k_scales[None, :] + k_m) < (K // group_size))
            scales_b = load_ptr(scales_b_ptrs + k_m * stride_meta_g, scale_b_mask, meta_evict_policy, not EVEN_K)

        if(channel_scale_mode == 4):
            scale_a_mask_m = (offs_am[:, None] < M) if not EVEN_M else True
            scale_a_mask_k = ((offs_k_scales[None, :] + k_m) < (K // group_size)) if not EVEN_K else True
            scale_a_mask = scale_a_mask_m & scale_a_mask_k
            scales_a = load_ptr(scales_a_ptrs + k_m * stride_meta_a_g, scale_a_mask, meta_evict_policy, not (EVEN_M and EVEN_K))
        else:
            scales_a = scales_a_1s

        acc = tl.dot_scaled(a, scales_a, a_dtype, b, scales_b, b_dtype, acc)

        a_ptrs += BLOCK_SIZE_K_A * stride_ak
        b_ptrs += BLOCK_SIZE_K_B * stride_bk
        
        if not use_tma:
            offs_ak += BLOCK_SIZE_K_A
            offs_bk += BLOCK_SIZE_K_B

            if not EVEN_K:
                if EVEN_M:
                    a_mask = tl.broadcast_to((offs_ak[None, :] < (K // elements_per_sample_a)), [BLOCK_SIZE_M, BLOCK_SIZE_K_A_E])
                else:
                    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < (K // elements_per_sample_a)))
                if EVEN_N:
                    b_mask = tl.broadcast_to((offs_bk[:, None] < (K // elements_per_sample)), [BLOCK_SIZE_K_B_E, BLOCK_SIZE_N])
                else:
                    b_mask = ((offs_bk[:, None] < (K // elements_per_sample)) & (offs_bn[None, :] < N))

    #NVFP4 meta-scale
    if(group_size == 16):
        acc = acc.to(tl.float32) * _meta_scale_norm

    #############################################################################################################
    #Channel-wise scaling  
    if channel_scale_mode == 2:  # activation-only
        scales_a = load_ptr(scales_a_ptr + offs_am, offs_am < M, meta_evict_policy, not EVEN_M, other=1.0)
        acc = acc * scales_a[:, None]
    
    #############################################################################################################
    #Output
    acc     = acc.to(output_dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if(SPLIT_K > 1):
        if EVEN_M and EVEN_N:
            tl.atomic_add(c_ptrs, acc, sem=atomic_mode)
        else:
            tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode)
    else:
        if use_tma:
            tl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], value=acc)
        else:
            if EVEN_M and EVEN_N:
                tl.store(c_ptrs, acc)
            else:
                tl.store(c_ptrs, acc, mask=mask)

def gemm_splitK_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                        W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                        input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                        channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id:int, 
                        meta_scale: Tensor = None,
                        ) -> Tensor: 
        
    from ..core import GEMLITE_USE_TMA
    M, K, N = x.shape[0], W_q.shape[0] * elements_per_sample, W_q.shape[1] # W
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    M_CLOSEST = get_closest_m(M)

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or gpu_supports_bfloat16_atomicadd()
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    if(scales_x is not None):
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None

    if(is_mx_dtype(input_dtype)):
        gemm_splitK_kernel = gemm_splitK_MX_kernel
        load_scales_as_block = True
        use_5d_scales = (scales.ndim == 5)
    else:
        gemm_splitK_kernel = gemm_splitK_INT_kernel
        load_scales_as_block = False
        use_5d_scales = False

    gemm_splitK_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        #############################################
        W_nbits, group_size, unpack_mask, elements_per_sample,
        type_id, x.dtype.itemsize, W_q.dtype.itemsize,
        ###############################################
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        0 if use_5d_scales else scales.stride(0), 0 if use_5d_scales else scales.stride(1),
        ################################################
        load_scales_as_block = load_scales_as_block,
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ################################################
        channel_scale_mode  = channel_scale_mode,
        W_group_mode        = W_group_mode,
        zero_is_scalar      = zeros.numel() == 1,
        data_contiguous     = data_contiguous,
        use_tma             = use_5d_scales,
        use_5d_scales       = use_5d_scales,
        meta_scale_norm_ptr = meta_scale,
        warp_specialize     = config.WARP_SPECIALIZE,
    )

    if(not native_atomic):
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output

class gemm_splitK:
    kernel = [gemm_splitK_INT_kernel, gemm_splitK_MX_kernel]
    forward = gemm_splitK_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm_splitK"]
