# COLMAP: cumulative image-to-mesh comparison

Pristine `8dd74ef6ca84a0950e5b181df26c6e78e56d73d6` versus custom `d4d5f2fdf74eb79ced79755afcbe638eb4942791`, measured 2026-10-25. These are cumulative build comparisons across seven fresh native stages. This report adds measurements and validation, not an algorithm change.

## Fountain — 1536 px

| Hardware | Pristine s | Custom s | Speedup | Less time |
|---|---:|---:|---:|---:|
| Ryzen 3900X / RTX 3080 Ti | 349.854 | 49.357 | **7.09×** | **85.9%** |
| Ryzen 3900X / RTX 3060 | 388.866 | 74.533 | **5.22×** | **80.8%** |
| i7-4790K / GTX 1080 | 318.691 | 110.915 | **2.87×** | **65.2%** |

## Larger scenes — 1500 px

| Scene / hardware | Pristine s | Custom s | Speedup | Less time |
|---|---:|---:|---:|---:|
| Façade / Ryzen 3900X / RTX 3080 Ti | 2475.791 | 449.323 | **5.51×** | **81.9%** |
| Outdoor scene / Ryzen 3900X / RTX 3080 Ti | 1838.550 | 245.443 | **7.49×** | **86.7%** |

**CPU work:** Fountain falls 71.0% / 67.5% / 49.1% on Ti / 3060 / GTX 1080; Façade falls 59.8%; Outdoor scene falls 65.7%.

**GPU memory:** PatchMatch's sampled whole-board VRAM increases by roughly 1.25–2.19 GiB across cases. Rotation-buffer reuse was enabled; `--PatchMatchStereo.rotation_buffer_reuse 0` stops retaining alternate rotation buffers. The next optimization target is lower PatchMatch VRAM while retaining performance.

**Host RAM:** Outdoor scene's overall peak RSS rises 2513.8 → 2795.0 MiB (+11.2%). Façade's overall peak falls 5511 → 3685 MiB, despite higher PatchMatch RSS. Savings in another stage do not necessarily lower the pipeline's highest peak.

**Validation:** matching Fountain initialization produces 44 byte-identical maps on Ti; older-PC residuals are comparable to pristine's observed variation; Façade laser F1 differs by at most 0.0733 percentage points. Details and residual differences follow.

**One E2E observation per build and case; no confidence intervals.** Sums are native command lifetimes, excluding later diagnostics. Façade custom resumed after a registration-check stop; its sum is valid but is not an uninterrupted workflow measurement. Results should not be generalized to every scene or GPU.

The recorded larger-scene runs used a 1500 px maximum edge; Fountain used 1536 px. Each pristine/custom pair used the same resolution. Future routine comparisons will consistently use 1536 px; these completed runs are not relabelled.

Both arms used the same photos, intrinsic initialization and scene/worker settings, except for the stated custom lossless rotation setting. Each arm starts with fresh databases, features, matches and estimated SfM. Automatic SfM starting pairs can vary, as checked below. Fountain uses 11 standardized 3072×2048 RGB PNG inputs. Façade uses 76 original JPEGs, 6193–6204×4127–4132 pixels, with five intrinsic groups. Outdoor scene uses 51 camera-neutral 4752×3168 PNG exports; four prior holdouts and one overexposed image remain excluded from both builds. Larger 3060 runs were deliberately omitted because of their runtime. No reference poses seeded mapping. GPU extraction/matching were enabled in both; one GPU controller was used. Custom rotation-buffer reuse was ON after a separate exact-output guard. Pristine used its original allocation path. PHOTO and GEO PatchMatch are both included.

Main hardware: Ryzen 9 3900X, 64 GiB host RAM, Linux/WSL2, NVIDIA driver 616.56. Each main-host pair used the same qualified Ceres dependency build/environment: pristine here refers to COLMAP source, not an entirely pristine third-party stack. The common Ceres closure uses source 8d41e570 with the same local Schur matrix/RHS accumulation and numeric-buffer-reuse overlays; active-pattern options are disabled in both. The new fixed-size Ceres candidate is not included. Older hardware: i7-4790K, 32 GB DDR3, Windows 10, GTX 1080; MSVC 14.39/CUDA 12.9. Main runs used 12 CPU workers; older runs used 8. Ambient OMP/BLAS/MKL thread settings were held at 1, with explicit native worker settings.

Pristine's actual Fountain and Façade fusion logs confirm “Starting fusion with 12 threads” in default full-RAM mode (`use_cache=0`); the one-worker cache fallback was not active.

## Stage time and memory

P → C means pristine → custom. CPU is accumulated process CPU seconds. Host memory is peak RSS on Linux and peak working set on Windows, in MiB. GPU memory is sampled **whole-board** MiB, including display/other clients; short allocations can be missed. Values are rounded.

### Fountain 1536 — Ryzen 3900X / RTX 3080 Ti

| Stage | Wall s | CPU s | Host MiB | Board MiB |
|---|---:|---:|---:|---:|
| Features | 2.79 → 2.74 | 6.05 → 6.06 | 546 → 556 | 1884 → 1884 |
| Matching | 1.10 → 0.98 | 4.08 → 3.99 | 530 → 537 | 1884 → 1884 |
| SfM | 7.83 → 7.49 | 18.12 → 18.66 | 454 → 453 | 1886 → 1884 |
| Undistort | 1.35 → 1.35 | 11.30 → 11.55 | 680 → 680 | 1884 → 1884 |
| PatchMatch | 289.40 → 19.80 | 290.92 → 22.54 | 1130 → 1383 | 2523 → 3857 |
| Fusion | 2.88 → 1.00 | 19.87 → 7.52 | 831 → 632 | 1884 → 1884 |
| Poisson | 44.50 → 16.00 | 242.02 → 101.21 | 1682 → 1342 | 1884 → 2135 |

### Fountain 1536 — Ryzen 3900X / RTX 3060

| Stage | Wall s | CPU s | Host MiB | Board MiB |
|---|---:|---:|---:|---:|
| Features | 2.63 → 2.92 | 5.95 → 6.26 | 546 → 574 | 184 → 184 |
| Matching | 1.26 → 1.40 | 3.72 → 3.62 | 537 → 533 | 184 → 184 |
| SfM | 7.19 → 7.50 | 16.75 → 17.46 | 459 → 449 | 184 → 184 |
| Undistort | 1.35 → 1.45 | 11.21 → 11.96 | 683 → 685 | 184 → 184 |
| PatchMatch | 329.91 → 42.06 | 331.24 → 44.79 | 1129 → 1389 | 677 → 2011 |
| Fusion | 3.13 → 1.00 | 21.74 → 7.26 | 832 → 631 | 184 → 184 |
| Poisson | 43.39 → 18.20 | 239.93 → 113.86 | 1663 → 1450 | 184 → 289 |

### Fountain 1536 — i7-4790K / GTX 1080

| Stage | Wall s | CPU s | Host MiB | Board MiB |
|---|---:|---:|---:|---:|
| Features | 1.99 → 1.99 | 5.06 → 5.23 | 191 → 191 | 1751 → 1751 |
| Matching | 1.17 → 1.17 | 5.09 → 5.34 | 170 → 168 | 1421 → 875 |
| SfM | 11.89 → 10.94 | 38.78 → 35.80 | 118 → 114 | 438 → 438 |
| Undistort | 2.59 → 2.64 | 13.61 → 13.56 | 230 → 230 | 438 → 438 |
| PatchMatch | 258.22 → 67.61 | 262.20 → 70.27 | 788 → 1030 | 989 → 2269 |
| Fusion | 5.50 → 2.20 | 35.53 → 15.19 | 569 → 374 | 438 → 438 |
| Poisson | 37.33 → 24.37 | 179.30 → 129.08 | 1075 → 967 | 438 → 547 |

### Façade 1500 — Ryzen 3900X / RTX 3080 Ti

| Stage | Wall s | CPU s | Host MiB | Board MiB |
|---|---:|---:|---:|---:|
| Features | 34.68 → 30.71 | 87.46 → 87.32 | 733 → 747 | 3314 → 3319 |
| Matching | 26.06 → 25.70 | 302.74 → 301.21 | 645 → 643 | 3308 → 3313 |
| SfM | 84.45 → 85.96 | 213.11 → 215.99 | 744 → 750 | 1878 → 1884 |
| Undistort | 22.18 → 21.28 | 238.25 → 229.85 | 2149 → 2146 | 1881 → 1878 |
| PatchMatch | 2122.12 → 200.60 | 2130.24 → 219.92 | 3094 → 3254 | 2893 → 5102 |
| Fusion | 24.91 → 10.75 | 211.72 → 109.49 | 3255 → 2370 | 1936 → 1878 |
| Poisson | 161.39 → 74.33 | 794.72 → 435.01 | 5511 → 3685 | 2030 → 2130 |


### Outdoor scene 1500 - Ryzen 3900X / RTX 3080 Ti

| Stage | Wall s | CPU s | Host MiB | Board MiB |
|---|---:|---:|---:|---:|
| Features | 19.25 → 18.14 | 48.40 → 45.91 | 617 → 615 | 3357 → 3357 |
| Matching | 35.37 → 34.49 | 412.94 → 403.03 | 564 → 564 | 2747 → 2747 |
| SfM | 25.56 → 24.48 | 29.34 → 28.00 | 452 → 450 | 1924 → 1924 |
| Undistort | 14.14 → 13.43 | 145.96 → 139.75 | 1399 → 1398 | 1924 → 1924 |
| PatchMatch | 1698.70 → 137.59 | 1702.84 → 151.26 | 2514 → 2795 | 2912 → 5149 |
| Fusion | 15.19 → 4.21 | 131.22 → 41.65 | 1974 → 1707 | 1929 → 1924 |
| Poisson | 30.34 → 13.11 | 158.33 → 93.08 | 1171 → 905 | 1929 → 1924 |

PatchMatch uses more memory to finish sooner. On the older PC, its peak process commit also rises from 1544.7 to 2854.5 MiB. On Outdoor scene, the highest stage RSS rises from 2513.8 to 2795.0 MiB (+11.2%). There is no general RAM/VRAM-saving claim. Fusion and Poisson process different upstream geometry in the original E2E arms, so their entire observed speed differences cannot be assigned to code alone.

Rotation-buffer reuse and concurrent reference working sets both contribute to GPU memory use. In the existing Fountain 1536 / RTX 3060 ON/OFF check, OFF used 1791 MiB at 45.20 s versus ON at 2011 MiB and 43.75 s; all 44 maps were exact (N1 each). The measured D4 build automatically batches up to five compatible references and falls back under memory pressure. `PatchMatchStereo.cache_size` controls the host cache, not these GPU working sets. Disabling rotation reuse leaves resolution, source-view selection and precision unchanged.

## Same-pass counters

Instructions are billions; cache misses are millions. The main-host counters were attached to the exact native PID with default inheritance. All recorded hardware events report 100.00% running time, without reported multiplexing. Attachment has a startup gap; complete lifetime/thread/descendant coverage was not independently proved. These are reported user-mode counts, not energy or wall time.

| Case | Stage | Instructions G | Cache misses M |
|---|---|---:|---:|
| Fountain / Ti | Features | 47.69 → 47.52 | 24.86 → 25.48 |
| Fountain / Ti | Matching | 26.21 → 26.13 | 48.45 → 46.81 |
| Fountain / Ti | SfM | 98.89 → 107.26 | 452.72 → 488.35 |
| Fountain / Ti | Undistort | 46.23 → 46.23 | 33.93 → 39.06 |
| Fountain / Ti | PatchMatch | 1172.39 → 86.37 | 159.27 → 56.56 |
| Fountain / Ti | Fusion | 78.75 → 36.19 | 212.90 → 82.09 |
| Fountain / Ti | Poisson | 421.59 → 287.50 | 3626.60 → 2003.40 |
| Fountain / 3060 | Features | 48.34 → 48.07 | 26.75 → 31.39 |
| Fountain / 3060 | Matching | 27.48 → 27.52 | 41.39 → 39.84 |
| Fountain / 3060 | SfM | 99.19 → 98.89 | 447.75 → 451.19 |
| Fountain / 3060 | Undistort | 46.23 → 46.23 | 35.38 → 46.37 |
| Fountain / 3060 | PatchMatch | 1344.07 → 174.13 | 159.98 → 65.65 |
| Fountain / 3060 | Fusion | 78.78 → 36.20 | 246.73 → 78.91 |
| Fountain / 3060 | Poisson | 421.58 → 317.90 | 3627.96 → 2311.56 |
| Façade / Ti | Features | 848.91 → 840.62 | 225.27 → 236.20 |
| Façade / Ti | Matching | 1859.67 → 1860.15 | 1401.30 → 1450.51 |
| Façade / Ti | SfM | 1196.47 → 1198.16 | 5712.70 → 5754.58 |
| Façade / Ti | Undistort | 1673.41 → 1673.42 | 238.00 → 207.86 |
| Façade / Ti | PatchMatch | 8386.27 → 827.93 | 1284.15 → 466.64 |
| Façade / Ti | Fusion | 989.57 → 602.33 | 2762.63 → 1414.53 |
| Façade / Ti | Poisson | 1611.01 → 1248.62 | 14224.17 → 9637.77 |
| Outdoor scene / Ti | Features | 448.49 → 446.18 | 121.28 → 100.45 |
| Outdoor scene / Ti | Matching | 2672.50 → 2672.21 | 4714.02 → 4080.15 |
| Outdoor scene / Ti | SfM | 228.66 → 225.93 | 527.98 → 503.97 |
| Outdoor scene / Ti | Undistort | 884.33 → 884.26 | 279.04 → 238.64 |
| Outdoor scene / Ti | PatchMatch | 6729.99 → 565.00 | 899.42 → 321.56 |
| Outdoor scene / Ti | Fusion | 365.37 → 179.98 | 2037.87 → 872.48 |
| Outdoor scene / Ti | Poisson | 266.57 → 197.58 | 2360.19 → 1669.43 |

The older Windows run recorded process cycles, user/kernel CPU, working set and commit. Its cumulative process-cycle count falls from 2153.480 to 1090.785 billion (a different scope from Linux user-mode cycles). Retired-instruction and cache-miss PMU counts are unavailable there; missing values are not zero. Total process CPU seconds P → C were 592.367 → 171.529 (Fountain Ti), 630.535 → 205.223 (Fountain 3060), 539.578 → 274.469 (older Fountain), 3978.253 → 1598.779 (Façade Ti), and 2629.016 → 902.681 (Outdoor scene Ti).

Sampled PatchMatch board power peaks were 190.084 → 335.061 W, 78.776 → 166.704 W and 200.946 → 353.350 W for main Ti Fountain, 3060 Fountain and Ti Façade respectively. Outdoor scene peak board power was 207.860 → 351.790 W. Both arms reached sampled 100% GPU utilization. Observed SM clocks ranged 210–2040/210–2025 MHz on Ti and 210–1965/210–1957 MHz on 3060 (P/C). These samples include collector edges and do not establish locked clocks, identical thermal conditions or COLMAP-only energy.

## Quality and the starting-pair control

All Fountain arms register 11/11 images. Both Façade arms register the same 75/76 images; both Outdoor scene arms register all 51. The complete finite-map manifests include both builds, including the external pristine Façade reference. Depth/normal payloads were checked for finite values and correct rosters; mesh topology and attributes were inspected. Counts and finiteness do not establish geometric accuracy. Comparisons use one camera-center Sim3, no ICP; distances below use scene units, not meters.

| Original E2E observations | Fused points P → C | Mesh vertices P → C |
|---|---:|---:|
| Fountain / Ti | 432,069 → 434,946 | 1,230,217 → 1,167,614 |
| Fountain / 3060 | 432,072 → 432,068 | 1,230,220 → 1,230,210 |
| Fountain / GTX 1080 | 436,443 → 432,923 | 1,268,728 → 1,197,646 |
| Façade / Ti | 1,981,062 → 1,978,709 | 5,504,732 → 5,630,953 |
| Outdoor scene / Ti | 471,008 → 469,880 | 929,445 → 922,741 |

- **Fountain Ti originally chose different starting pairs:** pristine #9/#7, custom #6/#4. The original meshes show a material asymmetric outer-coverage difference: sampled closest-triangle p99 is 0.502581 C→P versus 0.003405 P→C. Three fitted-view silhouette IoUs span 0.878–0.952; this difference was not dismissed as generic nondeterminism.
- **A direct control reproduces that difference within pristine.** Pristine was separately rerun from the same photos with #6/#4; its freshly generated feature/match tables were checked against custom and match exactly. Comparing pristine's two starting-pair choices reproduces the large tail (p99 0.489781 in the affected direction). With both builds using #6/#4, camera rotations agree and sampled cross-build mesh p99 falls to 8.43e-8/1.19e-7; rare maxima are 0.000133/0.000466. All 44 PHOTO/GEO depth/normal maps are byte-exact, as are the logical feature/matching fields. This directly supports starting-pair variation as the source of the large observed Fountain Ti coverage difference. The matched pair also has identical silhouettes in all three fitted views, vertex-color PSNR 48.05–48.34 dB and SSIM 0.99649–0.99703. This is scene-specific evidence. The diagnostic's timing is excluded from the headline observations.
- **Fountain 3060 used #9/#7 in both builds.** Cameras agree; sampled surface p99 is approximately 4.77e-7 in both directions. Three fitted-view silhouette IoUs are 1/0.9999957/1, shared-hit vertex-color PSNR is 50.19–51.73 dB and interior SSIM is 0.9980–0.9984. Mesh counts/colors still differ, so this is not byte equality.
- **Older Fountain:** a saved pristine repeat chose custom's ordered starting pair, reducing maximum camera rotation disagreement from 0.001381° to 0.00004696°. Its fresh dense continuation uses byte-identical undistorted images, matching stereo configurations and all 22 ordered source lists. Dense residuals remain: geometric-depth p99 relative differences are 0.190–0.298% against custom, versus 0.191–0.303% against original pristine. Full-vertex mesh p99 is 0.01001/0.00767 against custom versus 0.00859/0.00672 against original pristine; each uses its target's SfM units. Barycentric RGB mean absolute differences are 4.92/4.91 versus 4.99/4.96 channel codes out of 255. This is comparable observed run-to-run variation within pristine and between builds, not a statistical equivalence test or a fixed-pose variability measurement. Original timing totals remain unchanged; no independent accuracy claim is made.
- **Façade Ti:** sampled cross-build surface p99 is 0.01222/0.01999, with larger rare tails. Official camera-aligned fused-cloud laser scores are close: at 5 cm, accuracy is 79.7461→79.8097%, completeness 60.3117→60.3399%, F1 68.6806→68.7224%. Across six tolerances, maximum absolute F1 difference is 0.0733 percentage points. Three fitted-view renders give SSIM 0.65378/0.65477/0.76446, PSNR 24.103/25.071/23.747 dB and silhouette IoU 0.88008/0.96499/0.97950; visual review found similar main structure with visible coverage/color variation. Pristine started #35/#27, custom #33/#25. Fountain's controlled result supports this interpretation but does not prove that every Façade difference comes from its pair choice. Reference poses enter postrun alignment only. Fused-cloud scores do not establish final-mesh accuracy or universal quality equivalence.

- **Outdoor scene Ti:** both builds chose the same ordered initial pair #50/#37. All 51 named feature rows and all 1,275 named raw-match multisets are exact; two photographs received swapped numeric IDs. Only their mutual verified pair differs: 576 pristine versus 578 custom inliers, with all 576 pristine inliers shared. Median/max camera rotation differences are 0.01332/0.21556 degrees; median sparse reprojection errors are 0.8494/0.8535 px. A pristine mapper repeat on its own database reproduces pristine; pristine on a backup of the custom matched database reproduces the custom sparse model to numerical floor (center max 6.55e-15, rotation 1.21e-6 degrees). The changed database input is therefore sufficient to reproduce the custom SfM outcome without custom mapper code; this does not separate ID ordering from verification variation. Sampled mesh surface p99 is 0.01424/0.01271. Three fitted-view renders give SSIM 0.68190/0.73644/0.69265, PSNR 22.987/24.151/23.757 dB and silhouette IoU 0.96792/0.98333/0.97112. Main structure looks similar, with visible coverage/colour variation. No independent ground truth or common-pose dense equivalence is claimed; no expensive dense rerun was substituted into timing totals.

Façade official fused-cloud F1 (%), with estimated-camera-center alignment and no ICP:

| Tolerance cm |1|2|5|10|20|50|
|---|---:|---:|---:|---:|---:|---:|
| Pristine |7.74326|26.5899|68.6806|82.3212|86.2356|88.9384|
| Custom |7.76336|26.6632|68.7224|82.3312|86.2369|88.8990|

Main surface summaries use 100,000 deterministic vertex queries per direction against full triangle meshes. Render metrics compare stored vertex RGB on shared fitted-view hits; they are not source-photo fidelity, texture quality or novel-view evaluation. Same-input Poisson diagnostics on both main Fountain clouds give surface p99 zero, supporting the separation between upstream reconstruction variation and mesher differences.

**AI assistance disclosure: Totally vibe coded.** AI tools assisted implementation, test tooling, analysis and reporting. The measurements and controlled comparisons were run locally and retained for review.
