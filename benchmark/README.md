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

- [Pinned historical report](https://github.com/fringehush-gif/colmap/blob/d297c1ed12ee392731ec638bfbbc2ada01d53ebd/benchmark/upstream_validation_20261025.md)
- [Unrounded historical observations](upstream_validation_20261025.json): per-call
  wall/CPU/host memory and sampled whole-board VRAM, with timing and missing-data
  qualifications. Linux RSS and Windows working set are distinct measurements.
- [October 27 activation and validation matrix](activation_validation_20261027.md):
  source revisions, Ceres enabling definitions, artifact identities, test selections,
  defaults and remaining runtime/quality gates.

A compatible upstream-versus-custom Ceres comparison with COLMAP held fixed is
pending. It starts with a short screen at 12 and 24 workers and records automatic
selection, actual solver/activation and stopping behavior. No new Ceres or
whole-stack speedup is claimed here.
