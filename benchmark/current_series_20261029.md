# COLMAP and Ceres measurements: 29 October 2026

Fountain P11, SIFT, 12 workers, one fresh seven-stage run per build/resolution. These Linux binaries reuse recorded objects and custom Ceres; baseline/D4 and finished have different dependency versions. Remote Windows performance remains pending.

| Pixels | Build | Native wall s | CPU s | Peak stage RSS MiB | Sampled board VRAM MiB |
|---|---|---:|---:|---:|---:|
| 1536 | Baseline `8dd74ef6` | 344.044 | 578.663 | 1722.1 | 1412.1 |
| 1536 | D4 `d4d5f2fd` | 51.103 | 175.749 | 1462.4 | 2508.1 |
| 1536 | Finished `cbbed62a` | 48.744 | 170.265 | 1451.3 | 1942.1 |
| 3072 | Baseline | 802.561 | 1366.559 | 4755.1 | 2732.1 |
| 3072 | D4 | 139.269 | 471.607 | 3894.8 | 6970.1 |
| 3072 | Finished | 135.438 | 463.531 | 2847.6 | 4720.1 |

Finished wall totals were 4.6%/2.8% below D4. At 3072, its largest stage RSS was 1047 MiB lower; different stages set the maxima. N1 and unequal major faults limit interpretation.

Totals exclude controller gaps and include loader diagnostics. Three-second whole-board samples can miss peaks; energy lacks native-phase alignment. Host commit is unavailable. Fresh OMP/BLAS/MKL=12/1/1; D4 rotation/HMM=0/0, finished=0/1. Historical D4 used rotation/HMM=1/0 and OMP=1.

Every run registered 11 images and produced 44 finite maps. All three pairs match across 11 logical database tables at each resolution. Mesh maxima reach 0.723/0.204 scene units at 1536/3072; agreement does not establish accuracy.

Separately, fixed-eight Ceres whole-BA calls used one model, excluded warmups and N2 per library/worker setting.

| Workers | Stock/fork wall s | Stock/fork CPU s | Stock/fork peak RSS MiB |
|---|---:|---:|---:|
| 12 | 3.065 / 2.100 | 14.235 / 6.910 | 554.64 / 562.87 |
| 24 | 3.190 / 2.055 | 22.760 / 8.505 | 573.36 / 589.61 |

Every call completed eight solves at the iteration limit with zero major faults; ten geometry/cost checks passed. Times include I/O. Optimization admission was not counted; equal solve counts do not prove identical linear systems.

[Observations, definitions, power/counter coverage and SHA-256 identities](current_series_20261029.json).
