# Qualification results — 28 October 2026

Fountain P11, two runs per build and resolution, 12 requested workers, rotation-buffer reuse off. Both COLMAP builds used the same custom Ceres and repaired PoseLib. [Source revisions, build definitions and full observations](current_qualification_20261028.json) identify the composed builds and reused objects.

Values are means of two runs, upstream → candidate; times are seconds.

| Image limit | Seven native stages | Including controller/interstage time | CPU-s | Peak native RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| 1536 | 340.713 → 47.799 | 351.102 → 58.110 | 559.384 → 163.960 | 1655.20 → 1436.92 |
| 3072 | 807.426 → 137.784 | 818.060 → 150.400 | 1374.517 → 486.807 | 4753.37 → 2819.09 |

Mean sampled GPU peaks **increased**: 1756.82 → 1856.82 MiB at 1536 and 2826.96 → 5423.77 MiB at 3072. These are whole-board samples every three seconds and can miss peaks.

All runs registered 11 images and produced 44 finite maps; complete typed database rows matched. Outputs were not uniformly identical. One 1536 candidate mesh had p99/max disagreement of 0.199/0.803 scene units; at 3072 candidate p99 reached0.00656, maximum 0.735. These are sampled agreement distances (100k queries/direction, no ICP), not accuracy measurements.

## Separate Ceres comparison

Fixed reconstructed input, normal stopping, two measured calls per library/setting after warmup:

| Workers | Stock process wall, s | Fork process wall, s |
| --- | ---: | ---: |
|12|7.010|3.110|
|24|5.695|2.705|
|Automatic (24)|4.875|2.445|

All 14 calls converged and 13 numerical comparisons passed. Fork RSS was 8–16 MiB higher. Times include model I/O; solve counts vary. This does not attribute the pipeline speedup to Ceres. Aggregate matrix-budget fallback also passed at 12/24/automatic 24; physical OOM remains untested.

## Validation and limits

- Linux: 2288 C++ cases across 190 executables, including a separate CPU-fallback replay; 935 Python cases with explicitly qualified private dependencies.
- Older Windows, CUDA 12.9: reported 57 passing tests and 10 identical photometric fallback maps. The sparse bank was unsupported.
- Real-cache tests: exact outputs at 5/8 sources, with eviction/reload. Controller recovery under real memory pressure remains open.
- ALIKED native integration and aligned/padded MME04 were not promoted by this report.

[Activation matrix](activation_validation_20261027.md) · [Full observations and measurement limits](current_qualification_20261028.json) · Historical [October 22](upstream_validation_20261022.json) / [October 25](upstream_validation_20261025.json).
