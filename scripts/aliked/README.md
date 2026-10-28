# Optional native ALIKED N16ROT operators

This path is off by default. Build with ALIKED_NATIVE_CUDA_ENABLED=ON
alongside COLMAP's CUDA and ONNX support. The qualified scope is Linux,
CUDA 13.2, an sm86 device, and ONNX Runtime 1.31.1 with API31 headers.
Other platforms, toolchains and architectures are not qualified by these
results. The verified model-pair path requires DOWNLOAD_ENABLED=ON.

Keep the original N16ROT model. Using ONNX 1.22, write a fresh transformed output:

    python scripts/aliked/export_onnx.py --input original.onnx --output native.onnx

The exporter validates both pinned stages and their serialized checksums.
Use --check-only to validate both stages without writing a model. The accepted
original is 2,997,054 bytes, SHA-256
39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547.
The accepted final native graph is 2,995,274 bytes, SHA-256
66c1b167d88c5e5e6b3b55683f9dc2159362dc1c228d29a790ef027f057aa300.
Weights and transformed models are not included.

Select N16ROT GPU extraction on one device. Set
AlikedExtraction.n16rot_model_path to the original and
AlikedExtraction.n16rot_native_model_path to the local transformed graph.
The Python option is AlikedExtractionOptions.n16rot_native_model_path.
An empty native path preserves ordinary extraction.

Unavailable or unsupported native paths/pairs use the configured original
model. Selected model-contract/initialization failures also fall back;
CUDA query, I/O, allocation and unexpected runtime failures remain errors.
This is not a guarantee that arbitrary native graphs are accepted.

See src/colmap/feature/aliked_native/NOTICE and the accompanying license
files for source attribution. The hermetic tests are in python/tests/test_aliked_export_onnx.py and skip
explicitly when optional ONNX is absent. Run them normally and with python
-O; no weights are required. Synthetic native fixtures can be regenerated
with scripts/aliked/generate_test_graphs.py --output a-fresh-directory.



Reclaim unused GPU memory (CUDA arena) is enabled for the admitted native
graph by default, using requested-size growth and post-run shrinking. Disable it with
AlikedExtraction.n16rot_native_reclaim_arena=0, the GUI checkbox, or
AlikedExtractionOptions.n16rot_native_reclaim_arena in Python. The option is
ignored without an admitted native graph; CPU/original-model fallback keeps
the ordinary memory policy. This option makes no memory-cap guarantee.
