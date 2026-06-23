# Usage: python3 test_serialization.py [--autotune]
import sys
_autotune = '--autotune' in sys.argv
if _autotune: sys.argv.remove('--autotune')

import unittest
import torch
from gemlite import reset_config, set_autotune
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
from gemlite.triton_kernels.config import KERNEL_CACHE
from gemlite.helper import A4W4_MXFP_dynamic, A4W4_NVFP_dynamic, patch_model

def is_fp8_supported():
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(0)
    return capability >= (8, 9)

device        = 'cuda:0'
compute_dtype = torch.bfloat16
gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]
dummy_file    = "/tmp/_test_serial.pt"

reset_config()
if not _autotune: set_autotune(False)
KERNEL_CACHE.ENABLE = False

class TestFreshStateDictSchema(unittest.TestCase):
    """Fresh modules should advertise the serialized GemLite checkpoint schema."""

    def test_fresh_module_exposes_quantized_weight_keys(self):
        layer = GemLiteLinearTriton()
        state_dict = layer.state_dict()

        expected_keys = {
            "W_q",
            "bias",
            "scales",
            "zeros",
            "metadata",
            "orig_shape",
            "meta_scale",
        }
        self.assertTrue(expected_keys.issubset(state_dict.keys()))
        self.assertEqual(state_dict["W_q"].numel(), 0)
        self.assertEqual(state_dict["metadata"].dtype, torch.int32)
        self.assertEqual(state_dict["orig_shape"].dtype, torch.int32)
        self.assertEqual(state_dict["meta_scale"].dtype, torch.float32)

def _check_serialization(test_case, gemlite_linear, matmul_type='GEMM', batch_size=32, tol=1e-7):
    """Shared serialization round-trip check."""
    in_features = gemlite_linear.in_features

    torch.save(gemlite_linear.state_dict(), dummy_file)

    loaded = GemLiteLinearTriton()
    loaded.load_state_dict(torch.load(dummy_file))

    # Check meta_args match
    ref_meta = gemlite_linear.get_meta_args()
    loaded_meta = loaded.get_meta_args()
    for i in range(len(ref_meta)):
        test_case.assertEqual(ref_meta[i], loaded_meta[i], f"meta_args mismatch at {i}: {ref_meta[i]} != {loaded_meta[i]}")

    # Check tensor_args match
    ref_tensors = gemlite_linear.get_tensor_args()
    loaded_tensors = loaded.get_tensor_args()
    for i in range(len(ref_tensors)):
        if ref_tensors[i] is not None and ref_tensors[i].numel() > 0:
            diff = (ref_tensors[i].float() - loaded_tensors[i].float()).abs().mean().item()
            test_case.assertEqual(diff, 0, f"tensor_args mismatch at {i}: mean diff = {diff}")

    # Check inference matches
    x = torch.randn(batch_size, in_features, dtype=compute_dtype, device=device) / 10.
    y_ref = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
    y_loaded = loaded.forward_manual(x, matmul_type=matmul_type)
    diff = (y_ref - y_loaded).abs().mean().item()
    test_case.assertTrue(diff < tol, f"Inference mismatch: mean diff = {diff}, expected < {tol}")


class TestSerializationINT(unittest.TestCase):
    """Serialization tests for INT quantized layers."""

    def test_A16W4(self):
        in_features, out_features = 4096, 2048
        W_nbits, group_size = 4, 128

        W_q = torch.randint(0, 2**W_nbits - 1, (out_features, in_features), device=device).to(torch.uint8)
        gs = W_q.numel() // group_size
        scales = torch.ones((gs, 1), device=device, dtype=compute_dtype) * 0.001
        zeros  = torch.full((gs, 1), (2**W_nbits - 1) // 2, device=device, dtype=compute_dtype)

        gemlite_linear = GemLiteLinearTriton(W_nbits,
                        group_size=group_size,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=gemlite_dtype,
                        output_dtype=gemlite_dtype)
        gemlite_linear.pack(W_q, scales, zeros, None)

        _check_serialization(self, gemlite_linear)

    @unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
    def test_A8W8(self):
        in_features, out_features = 4096, 2048
        fp8_dtype = torch.float8_e4m3fn

        W = torch.randn((out_features, in_features), dtype=compute_dtype, device=device) / 10.
        _scales = torch.rand((1, out_features), dtype=compute_dtype, device=device) * 1e-4

        gemlite_linear = GemLiteLinearTriton(W_nbits=8,
                        group_size=in_features,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=TORCH_TO_DTYPE[fp8_dtype],
                        output_dtype=gemlite_dtype,
                        scaled_activations=True)
        gemlite_linear.pack(W.to(fp8_dtype), scales=_scales, zeros=None, bias=None)

        _check_serialization(self, gemlite_linear)


class TestSerializationMX(unittest.TestCase):
    """Serialization tests for MXFP/NVFP quantized layers."""

    def setUp(self):
        self.in_features, self.out_features = 4224, 2048
        torch.manual_seed(42)
        self.linear_layer = torch.nn.Linear(
            self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False
        )
        self.linear_layer.weight.data /= 10.
        self.linear_layer.weight.requires_grad = False

    def _quantize(self, processor_fn):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False)
        )
        model.requires_grad_(False)
        model[0].weight.data = self.linear_layer.weight.data.clone()
        processor = processor_fn(dtype=compute_dtype)
        patch_model(model, device=device, processor=processor)
        return model[0]

    def test_A4W4_MXFP(self):
        gemlite_linear = self._quantize(A4W4_MXFP_dynamic)
        _check_serialization(self, gemlite_linear, matmul_type='GEMM')
        _check_serialization(self, gemlite_linear, matmul_type='GEMM_SPLITK', batch_size=2)

    def test_A4W4_NVFP(self):
        gemlite_linear = self._quantize(A4W4_NVFP_dynamic)
        _check_serialization(self, gemlite_linear, matmul_type='GEMM')
        _check_serialization(self, gemlite_linear, matmul_type='GEMM_SPLITK', batch_size=2)



class TestReSerializer(unittest.TestCase):
    """Test that loaded models can be re-saved and produce correct results."""

    def test_resave_INT(self):
        """Save -> load -> save -> load -> compare inference."""
        in_features, out_features = 4096, 2048
        W_nbits, group_size = 4, 128
        W_q = torch.randint(0, 2**W_nbits - 1, (out_features, in_features), device=device).to(torch.uint8)
        gs = W_q.numel() // group_size
        scales = torch.ones((gs, 1), device=device, dtype=compute_dtype) * 0.001
        zeros  = torch.full((gs, 1), (2**W_nbits - 1) // 2, device=device, dtype=compute_dtype)

        orig = GemLiteLinearTriton(W_nbits,
                        group_size=group_size,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=gemlite_dtype,
                        output_dtype=gemlite_dtype)
        orig.pack(W_q, scales, zeros, None)

        # First save/load
        torch.save(orig.state_dict(), dummy_file)
        loaded1 = GemLiteLinearTriton()
        loaded1.load_state_dict(torch.load(dummy_file))

        # Verify Parameters are registered
        params = list(loaded1.parameters())
        self.assertGreater(len(params), 0, "No parameters after load_state_dict")

        # Re-save/load
        torch.save(loaded1.state_dict(), dummy_file)
        sd2 = torch.load(dummy_file)
        self.assertIn("W_q", sd2, "W_q missing from re-saved state_dict")
        self.assertIn("scales", sd2, "scales missing from re-saved state_dict")

        loaded2 = GemLiteLinearTriton()
        loaded2.load_state_dict(sd2)

        # Compare inference
        x = torch.randn(32, in_features, dtype=compute_dtype, device=device) / 10.
        y_orig = orig.forward_manual(x, matmul_type='GEMM')
        y_loaded2 = loaded2.forward_manual(x, matmul_type='GEMM')
        diff = (y_orig - y_loaded2).abs().mean().item()
        self.assertTrue(diff < 1e-7, f"Re-serialized inference mismatch: {diff}")


class TestMXScales2Dto5D(unittest.TestCase):
    """Test that 2D MX scales (old/no-TMA checkpoints) are correctly converted to 5D on load."""

    def setUp(self):
        self.in_features, self.out_features = 4224, 2048
        torch.manual_seed(42)
        self.linear_layer = torch.nn.Linear(
            self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False
        )
        self.linear_layer.weight.data /= 10.
        self.linear_layer.weight.requires_grad = False

    def test_2d_to_5d_MXFP(self):
        import gemlite.core as gc

        # Pack with TMA (produces 5D scales)
        model_tma = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False)
        )
        model_tma.requires_grad_(False)
        model_tma[0].weight.data = self.linear_layer.weight.data.clone()
        processor = A4W4_MXFP_dynamic(dtype=compute_dtype)
        old_tma = gc.GEMLITE_USE_TMA
        gc.GEMLITE_USE_TMA = True
        patch_model(model_tma, device=device, processor=processor)
        ref = model_tma[0]

        # Pack without TMA (produces 2D scales)
        model_notma = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False)
        )
        model_notma.requires_grad_(False)
        model_notma[0].weight.data = self.linear_layer.weight.data.clone()
        gc.GEMLITE_USE_TMA = False
        processor2 = A4W4_MXFP_dynamic(dtype=compute_dtype)
        patch_model(model_notma, device=device, processor=processor2)
        notma = model_notma[0]

        # Save 2D checkpoint
        torch.save(notma.state_dict(), dummy_file)

        # Load with TMA enabled (should convert 2D -> 5D correctly)
        gc.GEMLITE_USE_TMA = True
        loaded = GemLiteLinearTriton()
        loaded.load_state_dict(torch.load(dummy_file))

        # Scales should match the TMA-packed reference
        ref_scales = ref.scales.data
        loaded_scales = loaded.scales.data
        self.assertEqual(ref_scales.shape, loaded_scales.shape,
                         f"Shape mismatch: ref={ref_scales.shape} loaded={loaded_scales.shape}")
        diff = (ref_scales.float() - loaded_scales.float()).abs().mean().item()
        self.assertEqual(diff, 0.0, f"Scales differ after 2D->5D conversion: {diff}")

        # Inference should match
        x = torch.randn(32, self.in_features, dtype=compute_dtype, device=device) / 10.
        y_ref = ref.forward_manual(x, matmul_type='GEMM')
        y_loaded = loaded.forward_manual(x, matmul_type='GEMM')
        inf_diff = (y_ref - y_loaded).abs().mean().item()
        self.assertTrue(inf_diff < 1e-7, f"Inference mismatch: {inf_diff}")

        gc.GEMLITE_USE_TMA = old_tma


if __name__ == '__main__':
    unittest.main()
