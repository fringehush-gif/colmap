# Benchmark evidence

**The October 25 main-host comparison used the same custom Ceres dependency in
both arms. Its baseline is upstream COLMAP source, not an entirely stock software
stack. Those results do not measure Ceres's contribution or the full stock-stack
versus fork-stack improvement.**

The historical comparison is one observation per build and case. Its totals sum
seven native stages (eleven process calls per Façade arm), excluding controller
transitions; the custom Façade run
resumed. Custom rotation-buffer reuse was enabled. Preserve these conditions when
reusing the numbers. Later changes cannot be added to that headline.

- [Historical report, with neutral scene label](upstream_validation_20261025.md)
- [Earlier October 22 evidence](upstream_validation_20261022.json): Fountain
  seven-stage pipeline, two observations per build; the separate Façade section
  is dense-only from provided cameras. Both COLMAP arms used optimized Ceres P.
- [Unrounded historical observations](upstream_validation_20261025.json): per-call
  wall/CPU/host memory and sampled whole-board VRAM, with timing and missing-data
  qualifications. Linux RSS and Windows working set are distinct measurements.
- [October 27 activation and validation matrix](activation_validation_20261027.md):
  source revisions, Ceres enabling definitions, artifact identities, test selections,
  defaults and remaining runtime/quality gates.

A separate [Ceres comparison with COLMAP fixed](ceres_attribution_20261027.md)
records four measured observations per arm at 12 and 24 workers, automatic
selection, actual solver and variable stopping work on one near-converged input.
It supplies bounded Ceres-library evidence, not a whole-stack speedup or
per-optimization activation measurement.

Latest Linux checkpoint: [October 29 measurements](current_series_20261029.md)
compare three retained builds at 1536/3072 and separate fixed-eight Ceres calls.
