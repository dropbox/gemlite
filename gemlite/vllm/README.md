# gemlite.vllm — gemlite integration for vLLM

Route vLLM's quantized forward path through gemlite's Triton kernels, or
quantize fp16/bf16 checkpoints on the fly at load time.

Supported pre-quantized formats: FP8 (dynamic block and per-tensor/channel,
weight-only), NVFP4 W4A4, MXFP4 weight-only/dynamic, INT8 weight-only and
dynamic, GPTQ / AWQ / GPTQMarlin / AWQMarlin int4 and int8, compressed-tensors
NVFP4/MXFP4/WNA16, GGUF (Q4_0 / Q4_1 / Q4_K / Q8_0 / Q2_K).

Entry points:
- `enable_gemlite(names=None)` — route pre-quantized checkpoints through gemlite.
- `set_onthefly_quant(...)` — quantize fp16/bf16 checkpoints at load time.
- `patch_vllm()` — env-driven application of both (idempotent).
- `register()` — vLLM plugin entry point (calls `patch_vllm()`).

## How the patch works

`enable_gemlite()` swaps vLLM's `get_quantization_config` registry so requested
quant types load through gemlite `LinearMethod`s. Weight materialization is
delegated to stock vLLM (so HF loading, sharding, fused qkv/gate_up all keep
working); only `process_weights_after_loading` and `apply` are replaced.

**The swap must run before vLLM resolves the quant-config class for the
model** — i.e. before `LLM(...)` or `vllm serve` finishes importing the
engine. All examples below follow that ordering.

Any checkpoint / layer that gemlite doesn't handle (K-quants other than
Q4_K / Q2_K, I-quants, MoE experts, embeddings, etc.) falls through to stock
vLLM with a warning. Nothing hard-fails.

## Requirements

- `gemlite` (this package)
- `vllm`
- `hqq` — only if you use on-the-fly `int4_weightonly`

## 1. Interactive Python / offline `LLM`

Import and enable **before** constructing `LLM`:

```python
from gemlite.vllm import enable_gemlite
enable_gemlite()                          # all schemes in SUPPORTED

from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507-FP8", dtype="bfloat16")
out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=16))
print(out[0].outputs[0].text)
```

To restrict to a subset:

```python
enable_gemlite(["A8W8_FP8_DYNAMIC", "A16W4_HQQ_INT"])
```

## 2. `vllm serve` / OpenAI-compatible server

`gemlite` registers a `vllm.general_plugins` entry point (see `setup.py`),
so plain `vllm serve` auto-discovers and installs the patch at engine
startup. Just set `VLLM_GEMLITE_ENABLE=1`:

```bash
export VLLM_GEMLITE_ENABLE=1
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --dtype bfloat16 --port 8000
```

Restrict to a subset:

```bash
export VLLM_GEMLITE_ENABLE_LIST=A8W8_FP8_DYNAMIC,A16W4_HQQ_INT
```

### Bootstrap (fallback)

If the plugin entry point isn't discovered (e.g. an editable install that
skipped `pip install -e .` after adding the entry point), pre-import
`gemlite.vllm` in a wrapper instead:

```bash
export VLLM_GEMLITE_ENABLE=1
python3 -c "
import sys, gemlite.vllm            # triggers patch_vllm() via env var
sys.argv = ['vllm', 'serve', 'Qwen/Qwen3-4B-Instruct-2507-FP8',
            '--dtype', 'bfloat16', '--port', '8000']
from vllm.entrypoints.cli.main import main
main()
"
```

## 3. Pre-quantized checkpoints

`enable_gemlite(names=None)` enables every scheme below; pass a list to
restrict.

| Scheme name          | Matches checkpoints                                  |
| -------------------- | ---------------------------------------------------- |
| `A8W8_FP8_DYNAMIC`   | FP8 dynamic (DeepSeek 128×128 block, per-tensor, per-channel) |
| `A16W8_FP8`          | FP8 weight-only per-channel, dynamic activations     |
| `A4W4_NVFP_DYNAMIC`  | NVFP4 W4A4 (ModelOpt and compressed-tensors)         |
| `A4W4_MXFP_DYNAMIC`  | MXFP4 dynamic                                        |
| `A16W4_MXFP`         | MXFP4 weight-only (compressed-tensors)               |
| `A16W8_INT8`         | INT8 weight-only                                     |
| `A8W8_INT8_DYNAMIC`  | INT8 dynamic                                         |
| `A16W4_HQQ_INT`      | GPTQ / AWQ / GPTQMarlin / AWQMarlin int4, HQQ int4, GGUF Q4_0 / Q4_1 / Q4_K, CT pack_quantized int4 |
| `A16W8_HQQ_INT`      | GPTQ / AWQ int8, GGUF Q8_0, CT pack_quantized int8   |
| `A16W2_HQQ_INT`      | GGUF Q2_K                                            |

Aliases: `A16W4_INT` → `A16W4_HQQ_INT`, `A16W8_INT` → `A16W8_HQQ_INT`.

## 4. On-the-fly quantization

Quantize an fp16/bf16 checkpoint at load time.

Programmatic:

```python
from gemlite.vllm import set_onthefly_quant
set_onthefly_quant(
    weight_bits=8, group_size=None, quant_mode="int8_weightonly",
    skip_modules=["lm_head", "vision", "visual"],
)

from vllm import LLM
llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16")
```

Via env var (uses a named preset):

```bash
export VLLM_GEMLITE_ONTHEFLY_QUANT=A16W8_INT8
export VLLM_GEMLITE_SKIP_MODULES=lm_head,visual,vision
```

`VLLM_GEMLITE_ONTHEFLY_QUANT` alone is enough — importing `gemlite.vllm`
with that env var set triggers `patch_vllm()`, which calls
`set_onthefly_quant(...)`.

### Presets

| Preset                     | weight_bits | group_size | quant_mode          | block_quant |
| -------------------------- | ----------- | ---------- | ------------------- | ----------- |
| `A16W8_INT8`               | 8           | —          | `int8_weightonly`   | —           |
| `A16W8_FP8`                | 8           | —          | `fp8_weightonly`    | —           |
| `A16W4_INT4_HQQ`           | 4           | 64         | `int4_weightonly`   | —           |
| `A8W8_INT8_DYNAMIC`        | 8           | —          | `int8_dynamic`      | false       |
| `A8W8_FP8_DYNAMIC`         | 8           | —          | `fp8_dynamic`       | false       |
| `A8W8_FP8_DYNAMIC_BLOCK`   | 8           | —          | `fp8_dynamic`       | true        |
| `MXFP8_DYNAMIC`            | 8           | 32         | `mxfp8_dynamic`     | —           |
| `MXFP4_WEIGHTONLY`         | 4           | —          | `mxfp4_weightonly`  | —           |
| `MXFP4_DYNAMIC`            | 4           | —          | `mxfp4_dynamic`     | —           |
| `A8W4_MXFP_DYNAMIC`        | 4           | —          | `mxfp8_dynamic`     | —           |
| `NVFP4_DYNAMIC`            | 4           | —          | `nvfp4_dynamic`     | —           |

`int4_weightonly` requires `pip install hqq`.

## Environment variables

| Name                          | Default   | Purpose                                              |
| ----------------------------- | --------- | ---------------------------------------------------- |
| `VLLM_GEMLITE_ENABLE`         | `0`       | Set to `"1"` to route pre-quantized checkpoints through gemlite. Required for `vllm serve` (both plugin and bootstrap paths). |
| `VLLM_GEMLITE_ENABLE_LIST`    | (unset)   | Comma-separated subset of `SUPPORTED` scheme names.  |
| `VLLM_GEMLITE_ONTHEFLY_QUANT` | (unset)   | Preset name — enables on-the-fly quantization.       |
| `VLLM_GEMLITE_SKIP_MODULES`   | `lm_head,visual,vision` | Comma-separated module names to leave unquantized (on-the-fly only). |

## Notes

- **Autotune cache** — first call on a new shape runs Triton autotune (can
  take minutes). Decisions are persisted to `/tmp/gemlite_cache.json` and
  reused on subsequent runs.
- **CUDA graphs** — keep them on (vLLM default). Gemlite kernels are
  captured correctly under `torch.compile`'s PIECEWISE mode.
- **Fallback on unsupported layers** — a warning is logged and that layer
  keeps its stock vLLM forward path. This applies per-layer, not per-model:
  a model with unsupported GGUF tensors still uses gemlite on the supported
  ones.

## Tested

Verified end-to-end on RTX PRO 6000 Blackwell (sm_120, CUDA 13, vLLM 0.19.2):
across all three activation paths (offline `LLM`, `vllm serve` bootstrap,
plain `vllm serve` via plugin):

| Model                                        | Format                     | Scheme                          |
| -------------------------------------------- | -------------------------- | ------------------------------- |
| `Firworks/Qwen3-4B-Instruct-2507-nvfp4`      | CT NVFP4 W4A4              | `A4W4_NVFP_DYNAMIC`             |
| `Qwen/Qwen3-4B-Instruct-2507-FP8`            | DeepSeek block FP8 128×128 | `A8W8_FP8_DYNAMIC`              |
| `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`   | CT pack-quantized int4     | `A16W4_HQQ_INT`                 |
| `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4`  | GPTQ int4 (→ gptq_marlin)  | `A16W4_HQQ_INT`                 |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_1`   | GGUF Q4_1                  | `A16W4_HQQ_INT`                 |

GGUF checkpoints require `--hf-config-path <hf-repo>` on `vllm serve`.

## Troubleshooting

- **`collective_rpc` with custom functions** — set
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to let vLLM pickle arbitrary
  callables to workers. Only relevant if you pass your own probe/closure,
  not for normal inference.
- **Plugin not firing under plain `vllm serve`** — verify the entry point
  is registered: `python3 -c "from importlib.metadata import entry_points;
  print([e for e in entry_points().select(group='vllm.general_plugins')])"`
  should include `gemlite`. If missing, reinstall: `pip install -e /path/to/gemlite`.
