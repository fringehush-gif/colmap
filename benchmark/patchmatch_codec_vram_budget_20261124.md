# PatchMatch storage codecs at a VRAM budget

At the tested VRAM budget, dual UNORM16 let us run both ETH3D scenes at a
higher image-size cap and improved laser F1 at 1, 2 and 5 cm. The higher
resolution cost more time. The comparisons vary both storage mode and image
size. The fixed-resolution Terrace runs show the observed differences at the same image cap.

Dual UNORM16 means `probability_unorm16=true` and `ncc_cost_unorm16=true`, with
FP32 normals. The three-codec mode uses probability UNORM16, NCC20 and Oct40.
Only one NCC cost encoding can be active at a time.

## Full-scene VRAM-budget results

| ETH3D scene | Arm | Image cap | Sampled peak board / above idle (MiB) | Dense wall (s) | Laser F1 at 1 cm | 2 cm | 5 cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Terrace, 23 views | FP32 | 2592 | 3778 / 2175 | 141.69 | 64.3225% | 80.6837% | 89.0896% |
| Terrace, 23 views | Dual UNORM16 | 3072 | 3557 / 2031 | 184.21 | 67.8069% | 81.6832% | 89.4004% |
| Terrace, 23 views | Three codecs | 3072 | 3483 / 1997 | 189.72 | 67.7918% | 81.6724% | 89.4217% |
| Façade, 76 views | FP32 | 2432 | 7485 / 6010 | 464.46 | 33.0085% | 59.6621% | 78.9114% |
| Façade, 76 views | Dual UNORM16 | 3072 | 6844 / 5369 | 733.09 | 40.9719% | 63.3350% | 79.7467% |

For dual UNORM16 versus the lower-resolution FP32 arm, the observed
above-idle peak was 144 MiB lower on Terrace (6.6%) and 641 MiB lower on
Façade (10.7%). Both measured gaps are larger than 5%. The Façade FP32 cap was estimated from five-view PHOTO pilots
and the observed GEO memory increase. Façade has two full arms.

Façade dual UNORM16 minus FP32 changes against the laser, in percentage
points:

| Tolerance | Completeness | Accuracy | F1 |
| ---: | ---: | ---: | ---: |
| 1 cm | +7.5818 | +2.2253 | +7.9634 |
| 2 cm | +4.0448 | +2.2035 | +3.6729 |
| 5 cm | +0.8193 | +0.8084 | +0.8353 |
| 10 cm | +0.0238 | +0.2994 | +0.1256 |
| 20 cm | -0.1175 | +0.1124 | -0.0316 |
| 50 cm | -0.0037 | +0.0506 | +0.0181 |

On Terrace the close-range F1 improvement was driven mostly by completeness.
At 1 cm dual UNORM16 improved completeness by 4.20 points and moved accuracy
by -0.08 points relative to the lower-resolution FP32 run.

## Fixed-resolution Terrace results

| Image cap | Mode | Above-idle peak (MiB) | Dense wall (s) | Laser F1 at 1 cm | 2 cm | 5 cm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2592 | FP32 | 2175 | 141.69 | 64.3225% | 80.6837% | 89.0896% |
| 2592 | NCC cost UNORM16 | 1930 | 139.24 | 64.3271% | 80.6718% | 89.1026% |
| 2592 | Three codecs | 1644 | 149.16 | 64.3535% | 80.6973% | 89.1165% |
| 3072 | FP32 | 2824 | 185.43 | 67.8131% | 81.6847% | 89.4301% |
| 3072 | NCC cost UNORM16 | 2403 | 179.19 | 67.7985% | 81.7037% | 89.4295% |
| 3072 | Dual UNORM16 | 2031 | 184.21 | 67.8069% | 81.6832% | 89.4004% |
| 3072 | Three codecs | 1997 | 189.72 | 67.7918% | 81.6724% | 89.4217% |

At 3072, dual UNORM16 saved 372 MiB above idle against NCC cost UNORM16
alone and used 34 MiB more than the three-codec mode. The observed dense
time was 5.51 s shorter than the three-codec mode. Dual UNORM16 was within
0.072 percentage points of fixed-resolution FP32 F1 at every tested
tolerance from 1 to 50 cm. These observations leave the repeatability of
small mode differences open.

## Setup and limits

The full-scene comparisons used the corrected PatchMatch CUDA archive from
[`c94b6a07`](https://github.com/fringehush-gif/colmap/commit/c94b6a074ae870e812cbf258ae8181fef47334d5)
on an RTX 3080 Ti. Separately compiled controller adapters selected the
storage flags and image caps and linked the same CUDA archive. The scenes
used the complete supplied camera models,
fixed reference and ordered source lists, up to 20 sources, radius 5, step 1,
15 samples, five iterations, checkpointed HMM and rotation-buffer reuse off.
All arms within a scene used the same 3072 preparation. Each ran PHOTO then
filtered GEO. Dense time includes map writes. Common pristine COLMAP fusion
at [`8dd74ef6`](https://github.com/colmap/colmap/commit/8dd74ef6ca84a0950e5b181df26c6e78e56d73d6)
used 12 workers with cache disabled. The same ETH3D scorer used the supplied
laser frame without fitting, 1 cm voxels and identical beam settings.

Each cell is one observation. Board VRAM is a whole-device sample at
100 ms requested cadence. Above-idle figures subtract the pre-run board
reading. Timing includes execution and map I/O on that run. Parallel fusion
can vary with identical input maps. The Façade test does not include
same-cap FP32 or a three-codec arm. These results support using compression
to buy resolution on these two scenes and this GPU. They do not assign the
cross-resolution gain to any single codec.

## Earlier measured panels

The earlier five-reference PHOTO and one-reference GEO panels used different
builds and settings. Each number compares only that panel's own arms.
Sampled whole-device memory savings cannot be added across rows. The
probability-only panel used Fountain with 10 sources and no warmup. It
compares one candidate observation with the mean of two controls at each
cap. Most other rows use the two timed observations per arm from their
respective panels.

| Storage mode versus its control | 1536 PHOTO peak saved (MiB) | 3072 PHOTO peak saved (MiB) | 3072 observed PHOTO wall change |
| --- | ---: | ---: | ---: |
| Probability UNORM16 vs FP32 | 129 | 596 | -0.2% |
| NCC cost UNORM16 vs FP32 | 300 | 1266 | -1.4% |
| NCC24 vs FP32 | 143 | 537 | +1.2% |
| NCC20 vs NCC24 | 67.5 | 297.5 | -0.2% |
| Oct40 versus FP32 normals on NCC20 | 60 | 252 | +0.8% |
| Three codecs vs FP32 | 570 | 2342 | +2.9% |

In the earlier 3072 three-codec panel, sampled peak was 8079.6 to 5737.6
MiB and dense PHOTO wall was 23.789 to 24.482 s, with two timed observations
per mode. The separate one-reference GEO score moved by -0.0176 points
at 2 cm and +0.0074 points at 5 cm in F1. Its quality setup used five matched
PHOTO source pairs with the remaining 15 sources and the reference frozen.

- [Probability UNORM16](https://github.com/fringehush-gif/colmap/blob/9b241c2cd675906379aea053fa5225b43d0566b2/benchmark/patchmatch_probability_unorm16.json)
- [NCC cost UNORM16](https://github.com/fringehush-gif/colmap/blob/9b241c2cd675906379aea053fa5225b43d0566b2/benchmark/ncc_cost_unorm16_20261029.json)
- [NCC24](https://github.com/fringehush-gif/colmap/commit/94925af30bdab2c5245df90cdb626f1f4178826b)
- [NCC20](https://github.com/fringehush-gif/colmap/commit/5705e1a59cfbdb06cd9ac3f96fae9de113e2d438)
- [Oct40](https://github.com/fringehush-gif/colmap/commit/2cef4e01998ea15b5c67fb63416abcc1487a7c2c)
- [Three-codec panel](https://github.com/fringehush-gif/colmap/blob/9b241c2cd675906379aea053fa5225b43d0566b2/benchmark/unorm16_ncc20_oct40_20261104.json)
- [1024-pixel Oct40 full-scene quality screening](https://github.com/fringehush-gif/colmap/pull/142)
