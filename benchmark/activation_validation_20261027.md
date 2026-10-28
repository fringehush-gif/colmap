# Source, activation and validation checkpoint — 2026-10-27

**The historical upstream COLMAP arm used our custom Ceres dependency on the
main host. It was not a fully stock stack. A separately pinned Ceres comparison
is linked below; it does not revise those historical timings.**

This separates the immutable October 25 measurement from current source integration.
It adds no speed claim. An enabled build path does not prove admission in every call.

| Checkpoint | COLMAP source | Ceres dependency | Established scope |
| --- | --- | --- | --- |
| Historical October 25 | upstream `8dd74ef6` vs custom `d4d5f2fd`, report pinned at `d297c1ed` | Both main-host arms: custom `8d41e570` plus identical private matrix/RHS and numeric-reuse overlays; active-pattern off | N=1 per build/case; seven summed native stages (eleven process calls per Façade arm), controller gaps excluded; Façade resumed. Compares COLMAP builds under that common dependency, not stock-stack/fork-stack or Ceres attribution. |
| Current composed integration | 85d329e345dc7e426ee5967fa8e6fefd3cdbeece, parentsd297c1ed/9371f7b4/db7586c2 |94a6f21e89c00a5aeadf6eff7dcab48439d419df, upstream07a95262, five unchanged custom Schur files |22 affected CPU objects; CLI/focused tests; separate CMake configure/generation.63 selected CPU + 7 RTX 3080 Ti cases, zero failure/skip. Complete C++ suite, built Python suite and current-head E2E timing remain unproved. |
| Ceres qualification | Earlier qualified 392dc75 consumer observed loading this library; later CLI records it in its link/runtime configuration | `94a6f21`; libceres.so.2.3.0, 7,528,992 bytes, XXH3_128 6397ad67edcd4712d8c3fc3d990dce50 |50 CPU + 7 CUDA vector cases. Old NO_CONVERGENCE regression fails old/passes updated. Sparse consumer: 11 images/12,936 points, preserved tracks/observations/RGB, max displacement 1.90e-12 model units. No GPU Schur or speed claim. |

Tested composed CLI: 86,492,344 bytes, XXH3_128 01434fdcf9439e2c68ecf3259c6d00b1.
These identify tested local artifacts; a rebuild can differ.

The published util/cuda.cc has a formatting-only include/whitespace difference from the compiled blob. The CMake signature correction was configured/generated after the native tests, without rebuilding production. Initial ONNX link and configure failures were repaired. Reused archives retain the recorded inherited provenance; this is not compilation of a fresh complete closure.
These are recorded local build artifacts, not a fresh full-source rebuild or proof of deployment activation.

## Ceres activation

Qualified compiler definitions below are not named upstream CMake options:

~~~text
CERES_PRIVATE_SCHUR_ACCUMULATION_CANDIDATE=1
CERES_PRIVATE_SCHUR_RHS_CANDIDATE=1
CERES_PRIVATE_SCHUR_MATRIX_REUSE_CANDIDATE=1
CERES_PRIVATE_SCHUR_ACTIVE_PATTERN_CANDIDATE=0
CERES_PRIVATE_SCHUR_ACTIVE_PATTERN_CACHE_CANDIDATE=0
~~~

Use consistent definitions in the affected library translation units and consumers.
Without enabling definitions the checkout retains ordinary paths. Admission depends on
solver structure, sparsity, blocks, workers and budget. Larger problems can deliberately
fall back. The composed build has no collected per-solve activation/fallback census.
Fixed thread-index reduction does not make chunk-to-worker scheduling deterministic.
CPU assembly checks and CUDA vector checks establish different paths.

The [separate Ceres comparison](ceres_attribution_20261027.md) now records N=4 per current library/worker setting, with COLMAP fixed, normal stopping, actual 12/24 workers and automatic selection. Its near-converged input, variable work, source pins and artifact/loader limits are explicit. It is not a fixed-work, per-optimization activation, historical-Ceres-contribution or whole-stack claim. Per-solve activation/fallback counts and broader structural scaling remain pending.

## Defaults, platforms and test selection

At `85d329e`: rotation_buffer_reuse=false; direct_source_upload=false;
reference_only_normal_materialization=true; checkpoint_hmm=true for non-HIP, false for HIP.
October 25 custom timing and the three primary HMM comparisons explicitly enabled
rotation reuse. Separate ordinary-rotation1536 evidence is only N=1,21.693 → 22.305 s
(+2.82%), exact maps. Saved memory does not demonstrate greater admitted concurrency.

Sparse sharing is automatic for eligible batches, compiled for non-HIP CUDA>=13.2,
with further device/driver/profitability/admission checks. Older-CUDA positive-test
mismatch, allocation recovery and source-generation identity are active remediation
items at this revision; they are not represented as already fixed.

Main-host evidence: Linux/WSL2, Ryzen 3900X, RTX 3080 Ti/RTX 3060, routine 12 workers.
Historical older host: Windows 10, Haswell/GTX 1080, MSVC 14.39/CUDA 12.9, eight workers.
Historical older-host results do not establish current full-suite portability. Current complete C++ and built Python suites, ONNX/LoMa fallback and HIP remain unqualified here.

Current 63 CPU selection: SIFT 21, feature utilities 12, option manager 17, CUDA compiler
helpers 3, ONNX 9, invalid GPU index 1. GPU selection: ONNX CUDA 2, CUDA helpers 4,
explicit/default stream 1. Ceres 50 CPU means Schur solver 21, corrector 7, complete
private/eliminator 22. Initial 15-case filter is replaced by 22, not added.
Seven Ceres CUDA vector tests ran on RTX 3060.

## Remaining gates

After sparse fixes, run the complete applicable C++ suite and actual built Python
bindings at a deliberate integration checkpoint, explicitly recording unavailable
dependencies and unsupported paths. Start with a short representative screen before modest balanced E2E. Retain frozen sparse/stereo
inputs for dense work and frozen clouds for meshing. Separate source membership,
build enablement, activation, output agreement and independent geometry accuracy.

Façade/Outdoor Death final-mesh quality remains open. Fitted-view SSIM identifies
disagreement, not accuracy. Original fusion/Poisson arms consumed different geometry;
their timings do not isolate those algorithms. Keep October 25 immutable, and do not
combine later observations arithmetically into its headline.

Keep wall, native CPU, host peak RSS/commit, sampled whole-board VRAM and integrated
sampled power separate. Sample maxima miss short peaks; energy is not process-attributed.
Counter running percentage is not full lifetime coverage after late attachment.

## Evidence

- [Immutable October 25 report](https://github.com/fringehush-gif/colmap/blob/d297c1ed12ee392731ec638bfbbc2ada01d53ebd/benchmark/upstream_validation_20261025.md)
- [COLMAP integration](https://github.com/fringehush-gif/colmap/commit/85d329e345dc7e426ee5967fa8e6fefd3cdbeece)
- [Ceres integration](https://github.com/fringehush-gif/ceres-solver/commit/94a6f21e89c00a5aeadf6eff7dcab48439d419df)
- [Reduction contract](https://github.com/fringehush-gif/ceres-solver/blob/94a6f21e89c00a5aeadf6eff7dcab48439d419df/internal/ceres/private_schur_rhs.h#L60-L62)

