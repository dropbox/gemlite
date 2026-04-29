# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
from ..dtypes import is_mx_dtype
from .config import AUTOTUNE, KERNEL_CACHE, BLOCK_QUANT_SIZE
from . import config as _gemlite_config
from .utils import *

KEYS          = ['M', 'N', 'K', 'group_size', 'elements_per_sample', 'type_id', 'channel_scale_mode']
MATMUL_TYPE   = "GEMV"

#Init MXFP workspace for dequant
fp4_mapping = []
for g_id in range(torch.cuda.device_count()):
    fp4_mapping.append(
        torch.tensor(
            [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
            dtype=torch.int8,
            device="cuda:" + str(g_id),
        )
    )

def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from ..core import GEMLITE_TRITON_CONFIG_CACHE
    
    m = nargs['M']
    n = nargs['N']
    k = nargs['K']
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = nargs['type_id']

    pre_hook = init_to_zero("c_ptr") if nargs['use_prehook'] else None

    #Check cache
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
            config['NUM_STAGES'] = num_stages

            config['EVEN_K'] = (k % config['BLOCK_SIZE_K'] == 0)
            config['EVEN_N'] = (n % config['BLOCK_SIZE_N'] == 0)

            yield triton.Config(config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)
            return

    used = set()
    for config in configs:
        block_size_m = 1 #Only 1 allowed
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
                
        #Constraints: BLOCK_SIZE_K <= group_size -> load_scales_as_block is always False for gemvs
        block_size_k = min(g, block_size_k) #Makes BLOCK_SIZE_K compatible with the group_size
        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        # INT Block-quant (channel_scale_mode==5): one tile must fit inside one BxB scale block
        if channel_scale_mode == 5:
            block_size_n = min(block_size_n, BLOCK_QUANT_SIZE)
            block_size_k = min(block_size_k, BLOCK_QUANT_SIZE)

        #tmp fix autotune getting stuck on the MI300X
        if IS_HIP:
            if block_size_n * block_size_k >= 65536:
                continue
        
        #Block size should be compatible with minimum-packing
        if(block_size_k < e):
            continue
             
        A_load_order  = config.kwargs['A_load_order']
        dot_prod_mode = config.kwargs['dot_prod_mode']
        num_stages    = config.num_stages
        num_warps     = config.num_warps

        even_k = (k % block_size_k == 0)
        even_n = (n % block_size_n == 0)

        key = (block_size_m, block_size_n, block_size_k, A_load_order, dot_prod_mode, num_stages, num_warps)

        new_config = {
            'BLOCK_SIZE_M': block_size_m,
            'BLOCK_SIZE_N': block_size_n,
            'BLOCK_SIZE_K': block_size_k,
            'A_load_order': A_load_order,
            'dot_prod_mode': dot_prod_mode,
            'NUM_STAGES': num_stages,
            'EVEN_K': even_k,
            'EVEN_N': even_n,
        }

        if IS_HIP:
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            key = key + (new_config['waves_per_eu'],)

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config, num_stages=num_stages, num_warps=num_warps, pre_hook=pre_hook)

########################################################################################################################################################################
#Nvidia

#contiguous = True
def get_max_autotune_config_nvidia():
    configs = []
    for A in [0]:
        for D in [0]: 
            for w in [1, 2, 4]:
                for s in [1, 2]:
                    for M in [1]: #ONLY 1 allowed here
                        for N in [32, 64, 128, 256, 512]:
                            for K in [8, 16, 32, 64, 128]:
                                configs.append(
                                        triton.Config(
                                            {'BLOCK_SIZE_M': M, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K, 'A_load_order': A, 'dot_prod_mode': D,}, 
                                            num_stages=s, num_warps=w, 
                                            )
                                        )

    return configs


def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':16,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=1, num_stages=1))
    
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':16,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=2, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'A_load_order':0, 'dot_prod_mode':0}, num_warps=2, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':16,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=4, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':64,  'A_load_order':0, 'dot_prod_mode':0}, num_warps=2, num_stages=1))
    
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0}, num_warps=4, num_stages=1))
    return configs


def get_default_config_nvidia():
    config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'NUM_STAGES':1}, num_warps=1, num_stages=1)
    return [config]

########################################################################################################################################################################
#AMD - Instinct MI300X

def get_max_autotune_config_amd():
    configs = []
    for A in [0]: #[0, 1] - why is 1 taking a lot of time on some devices?
        for D in [0]:
            for w in [1, 2, 4]:
                for s in [1, 2]:
                    for v in [0, 2, 4]:
                        for M in [1]: #ONLY 1 allowed here
                            for N in [32, 64, 128, 256, 512, 1024]:
                                for K in [8, 16, 32, 64, 128]:
                                    configs.append(
                                        triton.Config(
                                            {'BLOCK_SIZE_M': M, 'BLOCK_SIZE_N': N, 'BLOCK_SIZE_K': K, 
                                            'A_load_order': A, 'dot_prod_mode': D, 'waves_per_eu': v},
                                            num_stages=s, num_warps=w,
                                        )
                                    )
    return configs

def get_fast_autotune_config_amd():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':8,  'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':16, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=1, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':16, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':4}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=1, num_stages=1))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=2, num_stages=1))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':16, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=1, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=1, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':2}, num_warps=2, num_stages=1))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':8,  'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=2, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=2, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':0}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':32, 'A_load_order':0, 'dot_prod_mode':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))
    return configs


def get_default_config_amd():
    config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':16, 'A_load_order':0, 'dot_prod_mode':0, 'NUM_STAGES':1}, num_warps=1, num_stages=1)
    return [config]
########################################################################################################################################################################

if IS_HIP:
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia

AUTOTUNE_SETTING = AUTOTUNE.GEMV
if(AUTOTUNE_SETTING == 'max'):
    get_autotune_config = get_max_autotune_config
elif(AUTOTUNE_SETTING == 'fast'):
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config

@triton.autotune(
    configs=get_autotune_config(),
    key = KEYS,
    restore_value = ['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)

@triton.jit
def gemv_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    mapping_ptr,
    M, N: tl.constexpr, K: tl.constexpr, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    ######### Strides #########
    stride_am: tl.constexpr, 
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, 
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, 
    stride_cn: tl.constexpr,
    stride_meta_a_m: tl.constexpr, 
    stride_meta_a_g: tl.constexpr,
    stride_meta_g: tl.constexpr, 
    stride_meta_n: tl.constexpr,
    ######### Dtypes #########
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
    A_load_order: tl.constexpr, 
    NUM_STAGES: tl.constexpr,
    dot_prod_mode:tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0, #Improve accuracy mainly for A16W8 with post loop scaling
    #####################################
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    join_version: tl.constexpr = False,
    #################################
    load_scales_as_block: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    block_quant_size: tl.constexpr = BLOCK_QUANT_SIZE,
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros)). This is optimized for M==1
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K // elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16
    """    

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1) 
    pid_m, pid_n = pid % M, pid // M

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #Vectorized coalesced load
    ##############################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k
    ###############################

    a_ptrs  = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak 

    if(join_version):
        BLOCK_SIZE_K_E: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
        offs_bk = pid_k * BLOCK_SIZE_K_E + tl.arange(0, BLOCK_SIZE_K_E) 
        b_ptrs  = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn 
    else:
        #orig version
        b_ptrs  = b_ptr + (offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn 

    a_mask = (offs_ak[None, :] < K)
    b_mask = (offs_bk[:, None] < K) & (offs_bn[None, :] < N)
    ###################################################################
    #Load
    if(A_load_order == 0):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    b = load_ptr(b_ptrs, b_mask, b_evict, not (EVEN_K and EVEN_N))

    if(A_load_order == 1):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))
    
    if(W_group_mode > 0):
        k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)

    if(W_group_mode >= 2): #[2, 3, 4]
        scales = tl.load(scales_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy) 
    else:
        scales = None
    
    if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
        if(zero_is_scalar):
            zeros = tl.load(zeros_ptr, eviction_policy=a_evict)
        else:
            zeros = tl.load(zeros_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy) 
    else:
        zeros = None
    
    if(A_load_order == 2):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    #tl.join() version
    if(join_version):
        if(elements_per_sample == 2):
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False) 

        if(elements_per_sample == 8):
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 4, BLOCK_SIZE_N), can_reorder=False) 
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 2, BLOCK_SIZE_N), can_reorder=False) 
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False)
    ####################################################################
    # Unpack and dequantize
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

    if(A_load_order == 3):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    if(dump_b_val > 0): b = b.to(tl.float32) * dump_b_val
        
    #Dot product
    if(dot_prod_mode == 0):
        acc = tl.sum((a.reshape((BLOCK_SIZE_K, 1), can_reorder=False).to(acc_dtype)) * b.to(acc_dtype), axis=0, keep_dims=True) 
    if(dot_prod_mode == 1):
        acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b.to(input_dtype), axis=0, keep_dims=True) 

    if(dump_b_val > 0): acc /= dump_b_val

    ##################################################################
    #Channel-wise scaling
    if channel_scale_mode == 1: #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if channel_scale_mode == 2: #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        acc = acc.to(meta_dtype) * scales_a[:, None]

    if channel_scale_mode == 3: #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N,   other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if channel_scale_mode == 5: #INT block-quant: one scale_b/scale_a per tile
        k_m_bq = (pid_k * BLOCK_SIZE_K) // block_quant_size
        n_m_bq = (pid_n * BLOCK_SIZE_N) // block_quant_size
        scales_b = tl.load(scales_ptr + k_m_bq * stride_meta_g + n_m_bq * stride_meta_n, eviction_policy=meta_evict_policy)
        scales_a = tl.load(scales_a_ptr + offs_am * stride_meta_a_m + k_m_bq * stride_meta_a_g, mask=offs_am < M, other=0.0, eviction_policy=meta_evict_policy)
        acc      = acc.to(tl.float32) * scales_a[:, None] * scales_b

    ####################################################################
    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    if EVEN_N:
        tl.atomic_add(c_ptrs, acc, sem=atomic_mode)
    else:
        mask = (offs_cn[None, :] < N) # BLOCK_SIZE_M is always 1
        tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode)


@triton.autotune(
    configs=get_autotune_config(),
    key = KEYS,
    restore_value = ['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by = {'early_config_prune': lambda _c, _n, **_kw: list(kernel_config_pruner(_c, _n, **_kw))},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)

@triton.jit
def gemv_MX_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    mapping_ptr,
    M, N: tl.constexpr, K: tl.constexpr, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    ######### Strides #########
    stride_am: tl.constexpr, 
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, 
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, 
    stride_cn: tl.constexpr,
    stride_meta_a_m: tl.constexpr, 
    stride_meta_a_g: tl.constexpr,
    stride_meta_g: tl.constexpr, 
    stride_meta_n: tl.constexpr,
    ######### Dtypes #########
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
    A_load_order: tl.constexpr, 
    NUM_STAGES: tl.constexpr,
    dot_prod_mode:tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0, #Improve accuracy mainly for A16W8 with post looop scaling
    #####################################
    meta_evict_policy: tl.constexpr = 'evict_first',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    join_version: tl.constexpr = False,
    #################################
    load_scales_as_block: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros)). This is optimized for M==1
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K // elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16
    """    

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1) 
    pid_m, pid_n = pid % M, pid // M

    a_ptr_dtype: tl.constexpr = a_ptr.dtype.element_ty
    if(a_ptr_dtype == tl.float16):
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.bfloat16):
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.float8e4nv):
        elements_per_sample_a: tl.constexpr = 1
    if(a_ptr_dtype == tl.uint8):
        elements_per_sample_a: tl.constexpr = 2

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k // elements_per_sample_a
    offs_bk = offs_k // elements_per_sample
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak #[1, BLOCK_SIZE_K]
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn #[BLOCK_SIZE_K, BLOCK_SIZE_N]
    a_mask = ((offs_am[:, None] < M) & (offs_ak[None, :] < (K // elements_per_sample_a))) 
    b_mask = ((offs_bk[:, None] < (K // elements_per_sample)) & (offs_bn[None, :] < N))

    if(W_nbits == 4): #mxpf4 mapping
        mapping = tl.load(mapping_ptr + tl.arange(0, 16), eviction_policy='evict_last')[None, :].broadcast_to((BLOCK_SIZE_K, 16))

    ###################################################################
    #Load
    if(A_load_order == 0):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    b = load_ptr(b_ptrs, b_mask, b_evict, not (EVEN_K and EVEN_N))

    if(A_load_order == 1):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    #Scales: only load_scales_as_block == False is supported here
    k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)
    #B scales
    scales_b_ptrs = scales_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n
    scales_b = tl.load(scales_b_ptrs, eviction_policy=meta_evict_policy)
    if(scales_ptr.dtype.element_ty == tl.uint8):
        scales_b = (tl.exp2(scales_b.to(tl.float32) - 127) * 0.50)
    scales_b = scales_b.to(acc_dtype)

    #A scales
    if(channel_scale_mode == 4):
        scales_a_ptrs = scales_a_ptr + k_m * stride_meta_a_g + offs_am[None, :] * stride_meta_a_m
        scales_a = tl.load(scales_a_ptrs, eviction_policy=meta_evict_policy)
        if(scales_a_ptr.dtype.element_ty == tl.uint8):
            scales_a = (tl.exp2(scales_a.to(tl.float32) - 127) * 0.50)
        scales_a = scales_a.to(acc_dtype)

    if(channel_scale_mode == 2):
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy) #Scalar
        if(scales_a_ptr.dtype.element_ty == tl.uint8):
            scales_a = (tl.exp2(scales_a.to(tl.float32) - 127) * 0.50)
        scales_a = scales_a.to(acc_dtype)

    if(A_load_order == 2):
        a = load_ptr(a_ptrs, a_mask, a_evict, not (EVEN_K))

    #Unpack and dequantize A
    a = a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) #we work with transposed activations
    if(elements_per_sample_a == 2): #4-bit activations
        q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]
        a = (a >> q_shift) & 15 
        a = tl.gather(mapping, a, axis=1)
    
    a = a.to(acc_dtype)
    if(channel_scale_mode == 2):
        a = a * scales_a
    if(channel_scale_mode == 4):
        a = a * scales_a
    
    # Unpack and dequantize B
    if(elements_per_sample == 2): #4-bit weights
        q_shift = ((offs_k % 2) * 4).to(tl.int32)[:, None]
        b = (b >> q_shift) & 15 
        b = tl.gather(mapping, b, axis=1)
    
    b = b.to(acc_dtype) * scales_b

    #Dot product
    if(dot_prod_mode == 0):
        acc = tl.sum(a.to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True) 
    if(dot_prod_mode == 1):
        acc = tl.sum(a * b.to(a.dtype), axis=0, keep_dims=True) 

    ####################################################################
    #Output: tl.atomic_add only supports 1D fp16 arrays, bfp16 would crash 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    if EVEN_N:
        tl.atomic_add(c_ptrs, acc, sem=atomic_mode)
    else:
        mask = (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode)

#TODO: gemv not generating correct reuslts with mxfp dtypes use except for A16W4.
def gemv_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                         W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                         input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int,  
                                         channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id: int,
                                         meta_scale: float = 0.0,
                                         ) -> Tensor:
    

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or gpu_supports_bfloat16_atomicadd()
    kernel_output_dtype = (DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32)

    if KERNEL_CACHE.ENABLE and M == 1:
        if (M, N) not in KERNEL_CACHE.CACHE:
            KERNEL_CACHE.CACHE[(M, N)] = {
                "data": torch.empty((KERNEL_CACHE.CACHE_SIZE, M, N), device=W_q.device, dtype=kernel_output_dtype),
                "ptr": 0,
            }

        entry = KERNEL_CACHE.CACHE[(M, N)]
        # CUDA graph capture: reset ptr once per entry so batch-zero gets captured
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not entry.get("_capture_reset", False):
            entry["ptr"] = 0
            entry["_capture_reset"] = True
        elif not capturing:
            entry["_capture_reset"] = False

        if entry["ptr"] % KERNEL_CACHE.CACHE_SIZE == 0:
            entry["data"].zero_()
            entry["ptr"] = 0

        output = entry["data"][entry["ptr"] % KERNEL_CACHE.CACHE_SIZE]
        entry["ptr"] += 1
        use_prehook = False
    else:
        #output, use_prehook = torch.empty((M, N), device=W_q.device, dtype=kernel_output_dtype), True
        output, use_prehook = torch.zeros((M, N), device=W_q.device, dtype=kernel_output_dtype), False

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(K, meta['BLOCK_SIZE_K']))

    device_index = W_q.device.index

    if(scales_x is not None):
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None
        #channel_scale_mode = 0
        
    dtype = DTYPE_TO_TRITON[input_dtype]
    if(dtype in [tl.float16, tl.bfloat16, tl.float32] and _gemlite_config.GEMLITE_FAST_GEMV_ACC):
        acc_dtype = dtype
    else:
        acc_dtype = DTYPE_TO_TRITON[acc_dtype]

    if(is_mx_dtype(input_dtype)):
        gemv_kernel = gemv_MX_kernel
        scales = scales.T
        if(scales_x is not None):
            scales_x = scales_x.T
    else:
        gemv_kernel = gemv_INT_kernel

    gemv_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        fp4_mapping[device_index],
        M, N, K,
        ###########################################
        W_nbits, group_size, unpack_mask, elements_per_sample, type_id, use_prehook,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        ############################################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype    = acc_dtype,
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ############################################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
        dump_b_val         = 0.001 if(W_group_mode in [0, 1] and acc_dtype == tl.float16 and W_nbits == 8) else 0, #Warning: Only use with INT8
    )

    if(not native_atomic):
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output


class gemv:
    kernel      = [gemv_INT_kernel, gemv_MX_kernel]
    forward     = gemv_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemv"]

