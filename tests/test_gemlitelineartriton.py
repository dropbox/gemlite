#python -m unittest test_gemlitelineartriton.py

import unittest
import torch
from gemlite import reset_config, set_autotune
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE, forward_functional
from gemlite.triton_kernels.config import KERNEL
from gemlite.quant_utils import scale_activations_per_token_torch as scale_activations

def is_fp8_supported():
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(0) 
    return capability >= (8, 9)  

device        = 'cuda:0'
compute_dtype = torch.float16 #float16, bfloat16
fp8_dtype     = torch.float8_e4m3fn #float8_e4m3fn / torch.float8_e5m2 (Nvidia)
gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]
matmul_types  = ['GEMV_REVSPLITK', 'GEMV', 'GEMV_SPLITK', 'GEMM_SPLITK', 'GEMM']
reset_config()
set_autotune(False)
KERNEL.ENABLE_CACHING = False


def gen_data(in_features, out_features, W_nbits, group_size, dtype=compute_dtype):

	W_q = torch.randint(0, 2**W_nbits - 1, (out_features, in_features), device=device).to(torch.uint8)

	shape  = (out_features, in_features)
	gs     = W_q.numel() // group_size
	scales = torch.ones((gs, 1), device=device, dtype=dtype) * 0.001
	zeros  = torch.zeros((gs, 1), device=device, dtype=dtype) * ((2**W_nbits - 1)//2)
	W      = ((W_q.reshape([-1, group_size]) - zeros) * scales).to(fp8_dtype).to(dtype)

	zeros = torch.mean(W_q.reshape([-1, group_size]).float() - (W / scales).float(), axis=1, keepdim=True).to(dtype)
	W     = ((W_q.reshape([-1, group_size]).to(dtype) - zeros) * scales)
	W     = W.reshape(shape)

	return W, W_q, scales, zeros


in_features, out_features = 4096, 1024
batch_sizes               = [1, 4]
W_nbits, group_size       = 4, 128 #128 / in_features
W, W_q, scales, zeros     = gen_data(in_features, out_features, W_nbits=W_nbits, group_size=group_size)

class TestGemLiteLinearTriton(unittest.TestCase):

	def test_serialization(self):
		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=gemlite_dtype, 
						output_dtype=gemlite_dtype)


		gemlite_linear.pack(W_q, scales, zeros, None)

		torch.save(gemlite_linear.state_dict(), 'tmp.pt')

		gemlite_linear_loaded = GemLiteLinearTriton()
		gemlite_linear_loaded.load_state_dict(torch.load('tmp.pt'))

		ref_args = gemlite_linear.get_meta_args()
		loaded_args = gemlite_linear_loaded.get_meta_args()
		for i in range(len(ref_args)):
			assert ref_args[i] == loaded_args[i], "meta_args mismatch at " + str(i)

		ref_args = gemlite_linear.get_tensor_args()
		loaded_args = gemlite_linear_loaded.get_tensor_args()
		for i in range(len(ref_args)):
			assert (ref_args[i] - loaded_args[i]).float().abs().mean() == 0, "tensor_args mismatch at " + str(i)

		tol = 1e-7
		for batch_size in batch_sizes:
			x = torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.
			for matmul_type in ['GEMM']:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				
				y_ref = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				y_gem = gemlite_linear_loaded.forward_manual(x, matmul_type=matmul_type)

				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	def test_fp16xfp16(self):
		gemlite_linear = GemLiteLinearTriton(W_nbits=16, 
						group_size=None, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=gemlite_dtype, 
						output_dtype=gemlite_dtype,
						scaled_activations=False)

		gemlite_linear.pack(W, None, None, None);

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0 and gemlite_linear.channel_scale_mode == 0)
		#Use non-contiguous when data is not packed
		self.assertTrue(gemlite_linear.data_contiguous == False)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.)
			y_ref = torch.matmul(x.to(compute_dtype), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_fp16xWn_asymmetric(self):
		#FP16 x Wn / asymmetric 
		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=gemlite_dtype, 
						output_dtype=gemlite_dtype)


		gemlite_linear.pack(W_q, scales, zeros, None);

		if(group_size == in_features):
			#Weights are unpacked() then shift only if group_size == in_features (1) otherwise (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1 and gemlite_linear.channel_scale_mode == 1) or 
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 0)) 
		else:
			self.assertTrue(gemlite_linear.W_group_mode in [3, 4] and gemlite_linear.channel_scale_mode == 0)

		#Use-contiguous when data is packed
		self.assertTrue(gemlite_linear.data_contiguous == True)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.
			y_ref = torch.matmul(x.to(compute_dtype), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_symmetric_no_activation_scaling(self):
		#INT8 x Wn - symmetric / no scaling activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, #only channelwise is supported 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=False) 


		_scales = torch.randn((out_features, 1), dtype=compute_dtype, device=device) * 1e-4
		gemlite_linear.pack(W_q, scales=_scales, zeros=7, bias=None);

		#Weights are unpacked() then shifted by 7
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Since the scales are channel-wise, we perform scaling post K-sum
		self.assertTrue(gemlite_linear.channel_scale_mode == 1)

		tol = 1e-3

		for batch_size in batch_sizes:
			x = (torch.randint(-10, 10, (batch_size, in_features), device=device)).to(torch.int8)
			y_ref = torch.matmul(x.to(compute_dtype), ((W_q.to(compute_dtype) - 7) * _scales).T) 
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))


	def test_int8xWn_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales=None, zeros=7, bias=None)
		gemlite_linear.meta_dtype = DType.FP32

		#Weights are unpacked() then shifted by 7
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Activations only are scaled
		self.assertTrue(gemlite_linear.channel_scale_mode == 2)

		tol = 5e-3

		for batch_size in batch_sizes:
			x = torch.randn((batch_size, in_features), dtype=torch.float16, device=device) / 20.
			_x, _x_scaled = scale_activations(x, w_dtype=torch.int8)
			y_ref = torch.matmul(_x.to(torch.float16), (W_q.to(torch.float16) - 7).T) * _x_scaled

			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol) + ' | ' + matmul_type)

	def test_int8Wn_scaled_weights_scaled_activations(self):
		#INT8 x Wn - activation scaling only

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=in_features,  #only channel-wise supported
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=DType.INT8, 
						output_dtype=DType.FP32,
						scaled_activations=True)

		_scales = torch.randn((out_features, 1), dtype=compute_dtype, device=device) * 1e-4
		gemlite_linear.pack(W_q, scales=_scales, zeros=7, bias=None);

		#Weights are unpacked() then shifted by 7 if group_size == in_features (1), otherwise (3)
		self.assertTrue(gemlite_linear.W_group_mode == 1) 
		#Activations only are scaled if group_size != in_features (2) otherwise bot are scales merged (3)
		self.assertTrue(gemlite_linear.channel_scale_mode == 3)

		tol = 1e-3

		for batch_size in batch_sizes:
			shape = W_q.shape
			x = torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.
			_x, _x_scaled = scale_activations(x, w_dtype=torch.int8)
			y_ref = torch.matmul(_x.to(compute_dtype), ((W_q.to(compute_dtype) - 7) * _scales).T) * _x_scaled
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol) + ' | ' + matmul_type)


	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_fp8xfp8(self):
		#FP8 x FP8 - no scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=None, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=TORCH_TO_DTYPE[fp8_dtype], 
						output_dtype=gemlite_dtype,
						scaled_activations=False)


		gemlite_linear.pack(W.to(fp8_dtype), None, None, None)

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#No channel-wise scaling
		self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.).to(fp8_dtype)
			y_ref = torch.matmul(x.to(compute_dtype), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_fp8xfp8_scaled_weights_scaled_activations(self):
		#FP8 x FP8 - both activations and weights are scaled

		gemlite_linear = GemLiteLinearTriton(W_nbits=8, 
						group_size=in_features, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=TORCH_TO_DTYPE[fp8_dtype], 
						output_dtype=gemlite_dtype,
						scaled_activations=True)


		_scales = torch.randn((1, out_features), dtype=compute_dtype, device=device) * 1e-4
		gemlite_linear.pack(W.to(fp8_dtype), scales=_scales, zeros=None, bias=None);

		#No weight unpacking / dequant
		self.assertTrue(gemlite_linear.W_group_mode == 0)
		#Both activations and weights are scales
		self.assertTrue(gemlite_linear.channel_scale_mode == 3)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			shape = W.shape
			x = torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.
			_x, scales_x = scale_activations(x, w_dtype=fp8_dtype)

			y_ref = torch.matmul(_x.to(compute_dtype), W.T) * (_scales * scales_x)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_fp8xWn_scaled_activations(self):
		#FP8 x Wn - asymmetric, with activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=TORCH_TO_DTYPE[fp8_dtype], 
						output_dtype=gemlite_dtype,
						scaled_activations=True)


		gemlite_linear.pack(W_q, scales, zeros, None);

		if(group_size == in_features):
			#weight unpacking and shift if group_size == in_features else (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1) and (gemlite_linear.channel_scale_mode == 3) or
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 2))
		else:
			#activations and weights are scaled psot accumulation if group_size==in_features else (2)
			self.assertTrue(gemlite_linear.W_group_mode in [3, 4])
			self.assertTrue(gemlite_linear.channel_scale_mode == 2)


		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.).to(fp8_dtype).to(compute_dtype)
			_x, _scaled_x = scale_activations(x, w_dtype=fp8_dtype)
			y_ref = torch.matmul(_x.to(compute_dtype), W.T) * _scaled_x
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))

	@unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
	def test_fp8xWn_no_activation_scaling(self):
		#FP8 x Wn - asymmetric, no activation scaling

		gemlite_linear = GemLiteLinearTriton(W_nbits, 
						group_size=group_size, 
						in_features=in_features, 
						out_features=out_features, 
						input_dtype=TORCH_TO_DTYPE[fp8_dtype], 
						output_dtype=gemlite_dtype,
						scaled_activations=False)

		gemlite_linear.pack(W_q, scales, zeros, None)

		if(group_size == in_features):
			#Weight shift only if group_size==in_features else (3)
			self.assertTrue((gemlite_linear.W_group_mode == 1 and gemlite_linear.channel_scale_mode == 1) or
							(gemlite_linear.W_group_mode == 3 and gemlite_linear.channel_scale_mode == 0))
		else:
			#weight scaling only - post accumulator if group_size==in_features else (0) 
			self.assertTrue(gemlite_linear.W_group_mode in [3, 4])
			self.assertTrue(gemlite_linear.channel_scale_mode == 0)

		tol = 5e-3 #needs higher tolerance with fp8

		for batch_size in batch_sizes:
			x = (torch.randn((batch_size, in_features), dtype=compute_dtype, device=device) / 10.).to(fp8_dtype)
			y_ref = torch.matmul(x.to(compute_dtype), W.T)
			for matmul_type in matmul_types:
				if(batch_size>1  and 'GEMV' in matmul_type): continue
				y_gem = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
				err   = (y_ref - y_gem).abs().mean().item()
				self.assertTrue(err < tol, str(err) + ', expected < ' + str(tol))
