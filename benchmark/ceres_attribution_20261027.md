# Ceres with COLMAP held fixed — October 27, 2026

On this retained Fountain BA input, custom Ceres used less mean elapsed and CPU
time, with about 0.4 MiB more mean peak RSS. Automatic selection used 24 workers
in both libraries. This does not extend the historical pipeline speedups.

| Workers | Arm | Wall mean / median (s) | Range (s) | CPU mean (s) | Peak RSS mean (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 | stock | 0.6925 / 0.6900 | 0.64–0.75 | 2.4275 | 395.66 |
| 12 | fork | 0.6150 / 0.5850 | 0.53–0.76 | 1.6050 | 396.06 |
| 24 | stock | 1.2125 / 1.2000 | 0.78–1.67 | 9.6975 | 399.29 |
| 24 | fork | 0.5275 / 0.5450 | 0.45–0.57 | 1.6125 | 399.73 |

Four measured calls per row, four excluded warmups; all 20 calls converged.
Normal selection chose DENSE_SCHUR. Every output preserved 12,936 point IDs,
reciprocal tracks, 118,468 observation rows (61,196 linked), and RGB exactly.
Cross-arm maximum point differences were about 6.99e-9 model units, similar
in scale to within-arm variation. This is descriptive agreement, not independent
geometry accuracy or statistical noninferiority.

**The input was already near convergence and stopping work varied.** Linear
solve counts: stock 12: 6/6/6/6; custom 12: 6/6/29/20; stock 24: 6/6/20/24;
custom 24: 6/18/6/20. These are normal-stopping BA calls, not fixed-work Schur
throughput or per-optimization activation evidence. The 12-worker wall ranges
overlap. No universal thread policy follows.

The same qualified COLMAP 85d329 build and input were held fixed. The CPU host
was Ryzen 9 3900X (12 cores/24 logical processors), Linux under Windows WSL2;
no GPU computation was measured. Ceres upstream
07a952622c2118433c783b1c3bd3fabe68420fc5 was compared with fork
94a6f21e89c00a5aeadf6eff7dcab48439d419df, with the same pinned compiler and
dependencies and sibling library placement. Actual upstream headers were used;
27 affected production consumers were rebuilt per arm. Each passed five upstream
eliminator and 21 complement-solver tests. Executing-process library paths were
verified. [Raw observations, flags and identities](ceres_attribution_20261027.json).

Qualified unaffected objects were reused identically; their historical compile
identities are not all independently proven. An initial strict export check
stopped on one extra weak standard-library helper; no stock export was missing.
The reviewed continuation allowed that exact symbol and verified current
products/inputs, without retrospectively proving the original build's final
input check. This is not a fresh whole-stack build.

GNU time has 0.01 s wall/CPU precision. Six PMU events include wrapper/descendant
overhead. Measured calls had no major page faults. No GPU or energy claim.
A four-call fast screen preceded this repeat; all 20 calls and post-run checks
finished in about 37 seconds after resource admission.

Private-path activation/fallback counts remain unmeasured. The historical
[October 22](upstream_validation_20261022.json) and
[October 25](upstream_validation_20261025.md) COLMAP comparisons used a common
optimized Ceres dependency. This result does not make them stock-stack
comparisons. These current Ceres pins also differ from the historical optimized
dependency; this does not retroactively measure its exact contribution.
No provider or default changed.
