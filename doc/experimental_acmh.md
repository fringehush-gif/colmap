# Experimental weighted ACMH PatchMatch backend

This fork adds an optional checkerboard search. It transfers neighboring
planes to the target ray, evaluates 15 hypotheses with weighted source and
native view scores, and retains the winning score for the next checkerboard
color. The native backend remains the default. This experiment is inspired
by [ACMH](https://arxiv.org/abs/1904.08103); it does not implement the complete
ACMH or ACMM pipelines.

## One-reference measurements

Façade ref75 used the same ordered 20 source views, five photometric and
five geometric iterations, radius 5, 15 samples, CPU12/CUDA1 and source
depth maps. Official ETH3D 2 cm accuracy / completeness / F1:

| Size | Native step 1 | Native step 2 | Weighted experimental step 2 |
| --- | ---: | ---: | ---: |
| 1536 | 62.5130 / 6.0362 / 11.0093% | 62.3409 / 5.9966 / 10.9408% | 63.1371 / 6.2608 / 11.3920% |
| 3072 | 65.4656 / 9.2995 / 16.2856% | 65.6183 / 9.3990 / 16.4428% | 65.6797 / 10.0095 / 17.3716% |

The measured precursor's isolated N2 photometric-plus-geometric time after warmup was
6.042 / 5.179 / 6.677 seconds at 1536 and 15.370 / 12.834 / 23.741
seconds at 3072 in the table's mode order. Mean process CPU time for
native step 1 versus the experiment was 13.106 / 13.670 seconds at 1536
and 45.065 / 53.697 seconds at 3072. Peak process RSS was
940.1 / 939.8 MiB at 1536 and 2427.3 / 2427.2 MiB at 3072.
These are one-reference results; source-view solves and fusion are excluded.

## Use and checks

- Set `COLMAP_PATCH_MATCH_BACKEND=experimental-acmh-weighted` on the
  `patch_match_stereo` process.
- Use `--PatchMatchStereo.window_radius 5` and
  `--PatchMatchStereo.window_step 2`. Other settings retain their normal
  PatchMatch behavior; unsupported backend settings fail before the solve.
- GPU tests cover odd partial tiles, camera-facing normal convention,
  photometric and geometric stages, mode gates and the native pair/batch
  fallback. Native default and experimental depth and normal maps matched
  the measured precursor implementations byte for byte at both sizes.

Totally vibe coded.
