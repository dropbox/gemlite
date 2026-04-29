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

KEYS        = ['M_CLOSEST', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'a_sizeof', 'b_sizeof', 'channel_scale_mode'] 
MATMUL_TYPE = "GEMM"

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

            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps)
            return

    gpu_shared_memory = get_gpu_shared_memory()
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])

        A_load_order = config.kwargs['A_load_order']
        num_stages = config.num_stages
        num_warps  = config.num_warps

        #Autotune prune the batch_size
        if m <= 16:    block_size_m = 16
        elif m <= 32:  block_size_m = min(max(block_size_m, 16), 32)  #m: [16...32]
        elif m <= 64:  block_size_m = min(max(block_size_m, 32), 64)  #m: [32...64]
        elif m <= 128: block_size_m = min(max(block_size_m, 64), 128) #m: [64...128]
        elif m <= 256: block_size_m = min(max(block_size_m, 64), 256) #m: [128...256]
        elif m > 256:  block_size_m = min(max(block_size_m, 64), 256) #m > 256

        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)
        
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

        #INT Block-quant (mode==5): single fp32 scale per BxB block, so BLOCK_SIZE_{M,N,K} must be <=BLOCK_QUANT_SIZE
        if channel_scale_mode == 5:
            # BLOCK_SIZE_M is unconstrained (scales_a is [M, K/BLOCK_QUANT_SIZE])
            block_size_n = min(block_size_n, BLOCK_QUANT_SIZE)
            block_size_k = min(block_size_k, BLOCK_QUANT_SIZE)

        #Hint: skip block_size_n > block_size_k for col-major non-packed data.

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

        key = (block_size_m, block_size_n, block_size_k, group_size_m, A_load_order, num_stages, num_warps)
        

        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
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
        yield triton.Config(new_config, num_stages=num_stages, num_warps=num_warps)

########################################################################################################################################################################
#Nvidia
def get_max_autotune_config_nvidia():
    stages  = [1, 3, 4, 5]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64, 128, 256]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128, 256, 512]:
                            configs.append(
                                triton.Config(
                                    {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, "GROUP_SIZE_M": 8, "A_load_order": A},
                                    num_warps=w, num_stages=s,
                                )
                            )

    return configs

def get_fast_autotune_config_nvidia():
    configs = [] #BLOCK_SIZE_M is automatically adapted in the config pruning.
    #Small tiles (packed INT with small group_size, small N problems)
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    #Medium N tiles
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    #Large N tiles
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=5))
    #Large M×N tiles (pruner adapts M for large batch sizes)
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    # NVFP4-friendly configs: BSK=128 allows stages=3/4 within 99KB smem (NVFP4 g=16 cant fit BSK=256 stages=3)
    configs.append(triton.Config({"BLOCK_SIZE_M":128, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":8, "A_load_order":0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({"BLOCK_SIZE_M":128, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":8, "A_load_order":0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({"BLOCK_SIZE_M":128, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":8, "A_load_order":2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({"BLOCK_SIZE_M":128, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":8, "A_load_order":2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    #Extra coverage
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=5))
    #Small M tiles (for M=32..64 where more tiles improve SM utilization)
    configs.append(triton.Config({"BLOCK_SIZE_M":32, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":8, "A_load_order":0}, num_warps=4, num_stages=5))
    configs.append(triton.Config({"BLOCK_SIZE_M":32, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":256, "GROUP_SIZE_M":8, "A_load_order":0}, num_warps=4, num_stages=5))
    return configs

def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=2),
            ]

########################################################################################################################################################################
#AMD - Instinct MI300X

def get_max_autotune_config_amd():
    configs = []
    for A in [0]:
        for w in [4, 8]:
            for s in [1, 2]:
                for v in [0, 2, 4]:
                    for M in [16, 32, 64, 128, 256]:
                        for N in [32, 64, 128, 256, 512]:
                            for K in [32, 64, 128, 256, 512]:
                                configs.append(
                                    triton.Config(
                                        {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, "GROUP_SIZE_M": 8, "A_load_order": A, 'waves_per_eu': v},
                                        num_warps=w, num_stages=s,
                                    )
                                )

    return configs

def get_fast_autotune_config_amd():
    configs = [] #BLOCK_SIZE_M is automatically adapted in the config pruning.
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2)) 

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=2)) 
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    return configs

def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':64, 'GROUP_SIZE_M':8, 'A_load_order':0, 'NUM_STAGES':2}, num_warps=4, num_stages=2),]

########################################################################################################################################################################
if IS_HIP:
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia

AUTOTUNE_SETTING = AUTOTUNE.GEMM
if(AUTOTUNE_SETTING == 'max'):
    get_autotune_config = get_max_autotune_config
elif(AUTOTUNE_SETTING == 'fast'):
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config


@triton.autotune(
    configs = get_autotune_config(),
    key = KEYS, 
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.heuristics(values={
    "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def gemm_INT_kernel(
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
    load_scales_as_block, #False
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
    NUM_STAGES: tl.constexpr,
    A_load_order: tl.constexpr, 
    data_contiguous: tl.constexpr,
    #################################
    EVEN_M: tl.constexpr = False,
    EVEN_K: tl.constexpr = False, 
    EVEN_N: tl.constexpr = False,
    #################################
    meta_evict_policy: tl.constexpr = "evict_last",
    a_evict: tl.constexpr = "",
    b_evict: tl.constexpr = "evict_first",
    meta_scale_norm_ptr = None,
    #################################
    use_tma: tl.constexpr = True,
    use_5d_scales: tl.constexpr = False,
    block_quant_size: tl.constexpr = BLOCK_QUANT_SIZE,
    warp_specialize: tl.constexpr = False,
):
    """
    Based on https://github.com/fpgaminer/GPTQ-triton
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K <= group_size
    """


    pid  = tl.program_id(axis=0)

    #Swizzle?
    if(elements_per_sample > 1):
        pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    else:
        pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K) 
    
    #Offsets
    #############################################################################################################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k

    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 
    b_mask  = ((offs_bk[:, None] < K) & (offs_bn[None, :] < N))
    q_shift = ((offs_bk % elements_per_sample) * W_nbits).to(tl.int32)[:, None] 

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < K))
    
    #Meta data stuff
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr     = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample)

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')
    
    # INT Block-quantization (mode==5): BxB weight scales (fp32), per-row per-B-K activation scales (fp32), where B = BLOCK_QUANT_SIZE
    if channel_scale_mode == 5:
        K_PER_SCALE:      tl.constexpr = block_quant_size // BLOCK_SIZE_K   #k-iters per B-K block (>=1)
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
            k_m = (k * stride_mul).to(tl.int32) 

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
        
        #INT Block-quant (mode==5): load per-block activation/weight fp32 scales before the dot.
        if channel_scale_mode == 5:
            k_m = k // K_PER_SCALE
            scales_a = tl.load(scales_a_ptrs + k_m * stride_meta_a_g, mask=offs_am < M, other=0.0, eviction_policy=meta_evict_policy)
            scales_b = tl.load(scales_b_base_ptr + k_m * stride_meta_g, eviction_policy=meta_evict_policy)

        #Dot
        if channel_scale_mode == 5:
            tmp = tl.dot(a, b.to(input_dtype), out_dtype=acc_dtype)
            acc += tmp * scales_a[:, None] * scales_b
        else:
            acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)
        
        #Advance
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk
        
        offs_ak += BLOCK_SIZE_K
        offs_bk += BLOCK_SIZE_K

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
    acc = acc.to(output_dtype)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.autotune(
    configs = get_autotune_config(),
    key = KEYS, 
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.heuristics(values={
    "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def gemm_MX_kernel(
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
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_meta_a_m: tl.constexpr, stride_meta_a_g: tl.constexpr,
    stride_meta_n: tl.constexpr, stride_meta_g: tl.constexpr,
    ######### Dtypes #########
    load_scales_as_block, #True
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr,
    A_load_order: tl.constexpr,
    data_contiguous: tl.constexpr,
    #################################
    EVEN_M: tl.constexpr = False,
    EVEN_K: tl.constexpr = False, 
    EVEN_N: tl.constexpr = False,
    #################################
    meta_evict_policy: tl.constexpr = "evict_last",
    a_evict: tl.constexpr = "",
    b_evict: tl.constexpr = "",
    meta_scale_norm_ptr = None,
    #################################
    use_tma: tl.constexpr = True,
    use_5d_scales: tl.constexpr = False,
    warp_specialize: tl.constexpr = False,
):


    pid = tl.program_id(axis=0)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

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
    BLOCK_SIZE_K_A: tl.constexpr = BLOCK_SIZE_K // elements_per_sample_a
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = tl.arange(0, BLOCK_SIZE_K_A)
    if not use_tma:
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < K // elements_per_sample_a))

    #B
    BLOCK_SIZE_K_B: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
    offs_bk = tl.arange(0, BLOCK_SIZE_K_B)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if not use_tma:
        if data_contiguous:
            offs_bn_load = offs_bn
        else:
            offs_bn_load = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    else:
        offs_bn_load = offs_bn
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn_load[None, :] * stride_bn
    b_mask = ((offs_bk[:, None] < K) & (offs_bn[None, :] < N))

    #Scales
    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_S: tl.constexpr = BLOCK_SIZE_K // group_size
    offs_k_scales = tl.arange(0, BLOCK_SIZE_K_S)
    offs_n_b_scales = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if not use_5d_scales:
        scales_b_ptrs = scales_ptr + offs_n_b_scales[:, None] * stride_meta_n + offs_k_scales[None, :] * stride_meta_g #[BLOCK_SIZE_N, BLOCK_SIZE_K // group_size]

    if use_tma:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            [M, K // elements_per_sample_a],
            [stride_am, stride_ak],
            [BLOCK_SIZE_M, BLOCK_SIZE_K_A]
        )
        
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            [N, K // elements_per_sample],
            [stride_bn, stride_bk],
            [BLOCK_SIZE_N, BLOCK_SIZE_K_B]
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
        scales_b_shape1: tl.constexpr = N // 128
        scales_b_shape2: tl.constexpr = K // group_size // 4
        stride_b4: tl.constexpr = 1
        stride_b3: tl.constexpr = 256
        stride_b2: tl.constexpr = 512
        stride_b1: tl.constexpr = 512 * scales_b_shape2
        stride_b0: tl.constexpr = stride_b1 * scales_b_shape1
        scales_b_5d_desc = tl.make_tensor_descriptor(
            scales_ptr,
            [1, scales_b_shape1, scales_b_shape2, 2, 256],
            [stride_b0, stride_b1, stride_b2, stride_b3, stride_b4],
            [1, rep_n, rep_k, 2, 256]
        )
    
    #B-scales
    if(channel_scale_mode == 4):
        scales_a_ptrs = scales_a_ptr + offs_am[:, None] * stride_meta_a_m + offs_k_scales[None, :] * stride_meta_a_g
    
    # _1s dtype must match actual scale dtype: uint8 for MXFP (E8M0), float8e4nv for NVFP4 (E4M3)
    if group_size == 16:
        scales_a_1s = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K_S), value=1, dtype=tl.float32).to(tl.float8e4nv)
        #scales_b_1s = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K_S), value=1, dtype=tl.float32).to(tl.float8e4nv)
    else:
        scales_a_1s = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K_S), value=127, dtype=tl.uint8)
        #scales_b_1s = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K_S), value=127, dtype=tl.uint8)

    _meta_scale_norm = tl.load(meta_scale_norm_ptr, eviction_policy='evict_last') if group_size == 16 else 1.0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(num_pid_k, num_stages=NUM_STAGES, warp_specialize=warp_specialize):
        # Load A and B tiles        
        if use_tma:
            a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K_A])
            b = tl.load_tensor_descriptor(b_desc, [pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K_B]).T
        else:
            a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_M and EVEN_K))

            b = load_ptr(b_ptrs, b_mask, b_evict, not (EVEN_K and EVEN_N))
        ####################################################################################
        k_m = k * BLOCK_SIZE_K_S
        if use_5d_scales:
            # 5D TMA scale loads (preshuffled layout)
            scale_b_raw = tl.load_tensor_descriptor(scales_b_5d_desc, [0, pid_n * rep_n, k * rep_k, 0, 0])
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
        ####################################################################################
        
        acc = tl.dot_scaled(a, scales_a, a_dtype, b, scales_b, b_dtype, acc)

        if not use_tma:
            a_ptrs += BLOCK_SIZE_K_A * stride_ak
            b_ptrs += BLOCK_SIZE_K_B * stride_bk     
            offs_ak += BLOCK_SIZE_K
            offs_bk += BLOCK_SIZE_K

            if not EVEN_K:
                if EVEN_M:
                    a_mask = tl.broadcast_to((offs_ak[None, :] < K), [BLOCK_SIZE_M, BLOCK_SIZE_K_A])
                else:
                    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < K))
                if EVEN_N:
                    b_mask = tl.broadcast_to((offs_bk[:, None] < K), [BLOCK_SIZE_K_B, BLOCK_SIZE_N])
                else:
                    b_mask = ((offs_bk[:, None] < K) & (offs_bn[None, :] < N))

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
    acc = acc.to(output_dtype)
    if use_tma:
        tl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], value=acc)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc)
        else:
            tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    

PRINTED = False
def gemm_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int,
                channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id:int,
                meta_scale: Tensor = None,
                ) -> Tensor:
    
    
    global PRINTED
    from ..core import GEMLITE_USE_TMA
    M, K, N = x.shape[0], W_q.shape[0] * elements_per_sample, W_q.shape[1] # W
    M_CLOSEST = get_closest_m(M)
    
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    if(scales_x is not None):
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None

    if(is_mx_dtype(input_dtype)):
        gemm_kernel = gemm_MX_kernel
        load_scales_as_block = True
        use_5d_scales = (scales.ndim == 5)
    else:
        gemm_kernel = gemm_INT_kernel
        load_scales_as_block = False
        use_5d_scales = False

    gemm_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K, M_CLOSEST,
        W_nbits, group_size, unpack_mask, elements_per_sample,
        type_id, x.dtype.itemsize, W_q.dtype.itemsize,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        0 if use_5d_scales else scales.stride(0), 0 if use_5d_scales else scales.stride(1),
        load_scales_as_block = load_scales_as_block,
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        channel_scale_mode  = channel_scale_mode,
        W_group_mode        = W_group_mode,
        zero_is_scalar      = zeros.numel() == 1,
        data_contiguous     = data_contiguous,
        use_tma             = use_5d_scales,
        use_5d_scales       = use_5d_scales,
        meta_scale_norm_ptr = meta_scale,
        warp_specialize     = config.WARP_SPECIALIZE,
    )
    
    return output



# @triton.autotune(
#     configs = get_autotune_config(),
#     key = KEYS, 
#     prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
#     use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
# )
# @triton.jit
# def gemm_INT_kernel_persistent_tma(
#     a_ptr, b_ptr, c_ptr,
#     scales_ptr, zeros_ptr, scales_a_ptr,
#     M, N, K, M_CLOSEST,
#     ######### Quant parms #########
#     W_nbits: tl.constexpr, 
#     group_size: tl.constexpr, 
#     unpack_mask: tl.constexpr, 
#     elements_per_sample: tl.constexpr, 
#     #################################
#     type_id: tl.constexpr,
#     a_sizeof: tl.constexpr,
#     b_sizeof: tl.constexpr,
#     ######### Strides #########
#     stride_am: tl.constexpr, stride_ak: tl.constexpr,
#     stride_bk: tl.constexpr, stride_bn: tl.constexpr,
#     stride_cm: tl.constexpr, stride_cn: tl.constexpr,
#     stride_meta_a_m: tl.constexpr, stride_meta_a_g: tl.constexpr,
#     stride_meta_g: tl.constexpr, stride_meta_n: tl.constexpr,
#     ######### Dtypes #########
#     load_scales_as_block: tl.constexpr, #False
#     input_dtype: tl.constexpr,
#     output_dtype: tl.constexpr,
#     acc_dtype: tl.constexpr,
#     meta_dtype: tl.constexpr,
#     ######### Meta-data mode #########
#     channel_scale_mode: tl.constexpr,
#     W_group_mode: tl.constexpr,
#     zero_is_scalar: tl.constexpr,
#     ######### tuning params #########
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr,
#     #################################
#     EVEN_M: tl.constexpr = False, 
#     EVEN_K: tl.constexpr = False, 
#     EVEN_N: tl.constexpr = False,
#     #################################
#     A_load_order: tl.constexpr = 0, 
#     data_contiguous: tl.constexpr = True,
#     #################################
#     meta_evict_policy: tl.constexpr = '',
#     a_evict: tl.constexpr = '',
#     b_evict: tl.constexpr = '',
#     NUM_SMS: tl.constexpr = 8,
# ):
#     """
#     Persistent + TMA version.
#     A: (M, K) fp16/bf16
#     B_packed: (K//elements_per_sample, N) int32
#     scales/zeros: (num_groups, N) or other depending on W_group_mode
#     """

#     # ---------------------------
#     # Persistent tiling setup
#     # ---------------------------
#     start_pid = tl.program_id(0).to(tl.int32)
   
#     grid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_tiles = grid_m * grid_n
#     width = GROUP_SIZE_M * grid_n  # tiles per "group stripe"

#     a_desc = tl.make_tensor_descriptor(
#         a_ptr,
#         [M, K],
#         [stride_am, stride_ak],
#         [BLOCK_SIZE_M, BLOCK_SIZE_K]
#     )

#     # b_desc = tl.make_tensor_descriptor(
#     #     b_ptr,
#     #     [K, N],
#     #     [stride_bk, stride_bn],
#     #     [BLOCK_SIZE_K, BLOCK_SIZE_N]
#     # )
    
#     #transposed : use self.W_q = self.W_q.contiguous().t()
#     b_desc = tl.make_tensor_descriptor(
#         b_ptr,
#         [N, K],
#         [stride_bn, stride_bk],
#         [BLOCK_SIZE_N, BLOCK_SIZE_K]
#     )
        
#     # # Precompute unpack shifts (vector length = elements_per_sample)
#     # # shifts = [0, W_nbits, 2*W_nbits, ...]
#     # shifts = (tl.arange(0, elements_per_sample) * W_nbits).to(tl.int32)

#     # # Optional scalar zero
#     # if zero_is_scalar:
#     #     zero_scalar = tl.load(zeros_ptr, eviction_policy="evict_last")

#     #############################################################################################################
#     # Main loop
#     for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
#         group_id = tile_id // width
#         first_m = group_id * GROUP_SIZE_M
#         gs = tl.minimum(grid_m - first_m, GROUP_SIZE_M)
        
#         pid_m = first_m + (tile_id % gs)
#         pid_n = (tile_id % width) // gs

#         rm = pid_m * BLOCK_SIZE_M
#         rn = pid_n * BLOCK_SIZE_N

#         # Accumulator
#         acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

#         # Column indices for this tile (used for metadata + store)
#         offs_n = rn + tl.arange(0, BLOCK_SIZE_N)
#         n_mask = offs_n < N

#         # K loop
#         for k in tl.range(0, K, BLOCK_SIZE_K):
#             a = tl.load_tensor_descriptor(a_desc, [rm, k])

#             k_packed = k // elements_per_sample
#             #b = tl.load_tensor_descriptor(b_desc, [k_packed, rn])
#             b = tl.load_tensor_descriptor(b_desc, [rn, k_packed]).T #Transposed 

#             acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)

#         #############################################################################################################
#         # Channel-wise scaling
#         offs_m = rm + tl.arange(0, BLOCK_SIZE_M)
#         m_mask = offs_m < M
#         if channel_scale_mode == 1:  # weight-only
#             # expects a 1D per-N scale at scales_ptr (same as your original)
#             scales_b = tl.load(scales_ptr + offs_n, mask=n_mask, other=1.0, eviction_policy=meta_evict_policy)
#             acc = acc.to(meta_dtype) * scales_b[None, :]

#         if channel_scale_mode == 2:  # activation-only
#             scales_a = tl.load(scales_a_ptr + offs_m, mask=m_mask, other=1.0, eviction_policy=meta_evict_policy)
#             acc = acc.to(meta_dtype) * scales_a[:, None]

#         if channel_scale_mode == 3:  # weight + activation
#             scales_a = tl.load(scales_a_ptr + offs_m, mask=m_mask, other=1.0, eviction_policy=meta_evict_policy)
#             scales_b = tl.load(scales_ptr + offs_n, mask=n_mask, other=1.0, eviction_policy=meta_evict_policy)
#             acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

#         acc = acc.to(output_dtype)

#         #############################################################################################################
#         # Store
#         c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#         mask = (m_mask[:, None] & n_mask[None, :])
#         if EVEN_M and EVEN_N:
#             tl.store(c_ptrs, acc)
#         else:
#             tl.store(c_ptrs, acc, mask=mask)

# # Persistent version
# NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
# def gemm_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_5d: Tensor, scales_x: Tensor,
#                 W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
#                 input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
#                 channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id:int, 
#                 ) -> Tensor:

#     M = x.shape[0]
#     K = W_q.shape[0] * elements_per_sample
#     N = W_q.shape[1]
#     M_CLOSEST = get_closest_m(M)
#     load_scales_as_block = False
    
#     output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])

#     if scales_x is not None:
#         stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
#     else:
#         stride_meta_a_m, stride_meta_a_g = 0, 0

#     grid = (NUM_SMS,)

#     gemm_INT_kernel_persistent_tma[grid](
#         x, W_q, output, 
#         scales, zeros, scales_x,
#         M, N, K, M_CLOSEST,
#         #############################################
#         W_nbits, group_size, unpack_mask, elements_per_sample,
#         type_id, x.dtype.itemsize, W_q.dtype.itemsize,
#         ###############################################
#         x.stride(0), x.stride(1),
#         W_q.stride(0), W_q.stride(1),
#         output.stride(0), output.stride(1),
#         stride_meta_a_m, stride_meta_a_g,
#         scales.stride(0), scales.stride(1),
#         ################################################
#         load_scales_as_block = load_scales_as_block,
#         input_dtype  = DTYPE_TO_TRITON[input_dtype],
#         output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
#         acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
#         meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
#         ################################################
#         channel_scale_mode = channel_scale_mode,
#         W_group_mode       = W_group_mode,
#         zero_is_scalar     = zeros.numel() == 1,
#         data_contiguous    = data_contiguous,
#         NUM_SMS            = NUM_SMS,
#     )


#     return output

    
class gemm:
    kernel = [gemm_INT_kernel, gemm_MX_kernel]
    forward = gemm_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm"]
