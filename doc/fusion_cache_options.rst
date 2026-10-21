.. SPDX-License-Identifier: BSD-3-Clause

Fusion memory options
=====================

Stereo Fusion reads images and FP32 depth/normal maps from a native COLMAP
or PMVS workspace. The native path uses COLMAP's OpenImageIO reader; no separate
image-loader service or Python callback is required.
They change Fusion's resident representation or loading policy; they do not
change PatchMatch's map files.

Choose a loading policy
-----------------------

* **Preload all inputs:** the default, ``use_cache=0`` and ``bounded_cache=0``.
  Inputs are loaded before Fusion. ``num_threads`` controls Fusion workers;
  ``full_ram_load_threads`` can independently limit preload workers. Its default
  of ``-1`` follows ``num_threads``. This mode needs enough RAM for the workspace.
* **Legacy LRU:** ``use_cache=1``. Inputs are loaded on demand. This path uses one
  Fusion worker. ``cache_size`` sets the cache's size in GiB.
* **Byte-bounded cache:** ``bounded_cache=1``. Pinned inputs cannot be evicted
  while a worker uses them. The budget must accommodate the live working set;
  this is not a hard process-RSS limit. ``bounded_cache_components=1`` loads
  depth, normal and RGB components separately. Neighborhood pinning and prefill
  are additional options, not prerequisites for ordinary bounded operation.

Resizing existing maps needs temporary decode/resize storage. More concurrent
workers increase the minimum admitted budget, even when the final grid is
smaller. If admission rejects a budget, reduce Fusion workers or increase the
budget; changing preload workers only controls the separate full-RAM policy.

Do not enable legacy LRU and byte-bounded caching together. Advanced combinations
are checked by ``StereoFusionOptions::Check()`` before Fusion starts. Use
``colmap stereo_fusion -h`` for the complete option list.

Resident precision in preload mode
----------------------------------

The default depth and normal representation is FP32. Octahedral normal storage
is optional and approximate; it can change fused points and later meshes.
The original FP32 files remain unchanged.

.. list-table:: Per-pixel resident payload
   :header-rows: 1

   * - Option
     - Values
     - Payload
     - Meaning
   * - ``full_ram_normal_storage``
     - ``fp32``
     - 12 bytes
     - Original FP32 normal components, including their magnitude.
   * - ``full_ram_normal_storage``
     - ``oct2x16``
     - 4 bytes
     - Octahedral direction; magnitude is discarded and decoded nonzero normals
       have unit length.

Payload sizes exclude allocator, metadata, images and Fusion working storage.
They do not predict whole-pipeline peak RAM. Resizing happens before resident
packing. Depth maps retain their original FP32 representation.

Conservative normal tiles
-------------------------

Bounded component mode can keep normal tiles rather than decoded whole normal
maps. The tile and mapped-owner budgets are separate. Mapped bytes describe
file-backed address space, not an equivalent amount of resident physical RAM.
Mapped ownership currently requires supported Linux little-endian storage.

``bounded_cache_normal_snorm16=1`` quantizes tile components. Without its separate
guard option, decisions use those approximate values. Enabling
``bounded_cache_normal_snorm16_guard=1`` screens decisions conservatively and
reads the original FP32 normal when the result is ambiguous. Supported compiler
arithmetic is required; unsupported builds reject the guard. Floating-point
rounding controls are checked at each decision.
SNORM tiles require finite normal components in the range [-1, 1]; other
values are rejected when loading the tile.

With normal tiles, ``bounded_cache_neighborhood=1`` can prefill and pin depth
maps selected for the current image's neighborhood. Fusion can then borrow
those depth pointers instead of acquiring a cache lease for each sample. RGB
components are admitted separately on demand. Admission retains per-worker
headroom; components outside the admitted subset still load on demand. Compare
this option with the same worker count and cache budgets before choosing it
for a workspace.

The original-depth diagnostic is separate: ``bounded_cache_exact_depth=1`` uses
original FP32 depth values. It requires ``bounded_cache=1``,
``bounded_cache_components=1``, ``num_threads=1``, and positive values for both
``bounded_cache_normal_mapped_bytes`` and ``bounded_cache_normal_mapped_files``.
Normal-tile size must remain zero and both SNORM options must remain off. This
mode cannot be combined with compact full-RAM normal representations.
Stored depth maps must already match the consumed depth grid; this diagnostic
does not support resizing those original depth maps during Fusion.
Every source depth sample must be finite, including invalid-depth sentinels;
non-finite samples are rejected before exact-depth reads.

Check available combinations with ``StereoFusionOptions::Check()``. Python
callers can query ``StereoFusionOptions.supports_normal_precision_guard()``
before selecting the conservative normal guard.

Measured normal-storage tradeoffs
----------------------------------

Fountain P11 uses all 11 photos and fixed geometric FP32 maps. These results
measure the complete Fusion process, including loading and PLY/VIS writing.
Map generation and meshing are excluded. Source:
``e85b31ef280101c1bf2ed34faccb411982530703``.

With default preloading, oct2x16 reduced mean peak RSS by 9.2% / 10.9%,
while elapsed time increased 10.0% / 11.4% at 1536 / 3072, respectively.
Each mode has two observations per resolution, with 12 Fusion workers.
An excluded FP32 warmup precedes forward and reverse mode order.

Peak RSS is the mean of the two native-process maxima. CPU time is user
plus system time. The default preload setting (-1) follows the configured
Fusion worker count. The resident guard requires one preloader, so its
FP32 and plain-octahedral controls also use one preloader. Bounded modes
load on demand and share the same explicit cache settings.

1536 x 1024; 12 Fusion workers

.. csv-table::
   :header: "Mode", "Preload setting", "Elapsed s", "CPU s", "Peak RSS MiB"
   :widths: 34, 16, 15, 15, 20

   "Resident FP32", "default (-1)", 2.400, 7.665, 690.760
   "Resident oct2x16", "default (-1)", 2.640, 10.025, 627.447
   "Resident FP32 / load1", "1", 4.750, 7.395, 690.521
   "Resident oct2x16 / load1", "1", 4.990, 8.880, 599.904
   "Guarded resident oct2x16", "1", 57.645, 101.245, 599.559
   "Bounded FP32 tiles", "n/a", 38.895, 276.425, 884.080
   "Bounded SNORM tiles", "n/a", 42.390, 298.835, 799.469
   "Guarded SNORM tiles", "n/a", 46.245, 330.505, 608.721

3072 x 2048; 12 Fusion workers

.. csv-table::
   :header: "Mode", "Preload setting", "Elapsed s", "CPU s", "Peak RSS MiB"
   :widths: 34, 16, 15, 15, 20

   "Resident FP32", "default (-1)", 7.355, 28.945, 1851.203
   "Resident oct2x16", "default (-1)", 8.195, 39.435, 1650.102
   "Resident FP32 / load1", "1", 15.840, 29.090, 1851.254
   "Resident oct2x16 / load1", "1", 16.785, 37.555, 1323.375
   "Guarded resident oct2x16", "1", 313.580, 564.180, 1323.346
   "Bounded FP32 tiles", "n/a", 173.885, 1259.465, 2045.297
   "Bounded SNORM tiles", "n/a", 180.045, 1294.445, 1814.066
   "Guarded SNORM tiles", "n/a", 202.955, 1482.835, 1472.059

Bounded rows use a 1 GiB decoded-admission budget, a separate 512 MiB
mapped-address budget, 32 mapped files, 256-pixel tiles and 64 tile slots.
Tile hints and cache diagnostics are enabled in all three bounded rows.
Mapped charge is not physical RSS; neither budget caps total process RSS.

All eight inexpensive CPU events were collected during those timed calls
and ran at 100%. Instructions, user cycles, cache misses, context switches
and the individual observations are retained in
``benchmark/normal_storage_tradeoffs.json``.

Output differences
~~~~~~~~~~~~~~~~~~

Thirty-two output clouds and 34 cross-mode/repeat comparisons passed the
predeclared parallel/codec gates. Deterministic symmetric nearest-neighbor
samples use 65,536 queries per direction. All sampled position and RGB
99th-percentile differences were zero in this cohort. This is approximate
output, not bitwise parity, and does not establish laser, mesh or render
accuracy. The table retains rare larger differences as well.

1536 maximum image size; 12 Fusion workers

.. csv-table::
   :header: "Mode / matched FP32", "Point count change %", "Normal P99 / max degrees", "Position max (native units)", "RGB max", "VIS mismatch %"
   :widths: 28, 14, 20, 16, 10, 12

   "Resident oct2x16 / Resident FP32", +0.000000, "0.002463 / 18.746", 0.016693, 27, 0.08392
   "Guarded resident oct2x16 / Resident FP32 / load1", -0.000461, "0.002400 / 4.709", 0.013076, 8, 0.02594
   "Bounded SNORM tiles / Bounded FP32 tiles", +0.005767, "0.001329 / 26.015", 0.016541, 44, 0.25940
   "Guarded SNORM tiles / Bounded FP32 tiles", +0.000923, "0.001220 / 18.392", 0.007386, 13, 0.00916

3072 maximum image size; 12 Fusion workers

.. csv-table::
   :header: "Mode / matched FP32", "Point count change %", "Normal P99 / max degrees", "Position max (native units)", "RGB max", "VIS mismatch %"
   :widths: 28, 14, 20, 16, 10, 12

   "Resident oct2x16 / Resident FP32", +0.001491, "0.002500 / 23.042", 0.007841, 24, 0.11444
   "Guarded resident oct2x16 / Resident FP32 / load1", +0.000477, "0.002415 / 3.786", 0.005698, 21, 0.01068
   "Bounded SNORM tiles / Bounded FP32 tiles", +0.001014, "0.001351 / 59.735", 0.011097, 25, 0.32959
   "Guarded SNORM tiles / Bounded FP32 tiles", -0.000298, "0.001227 / 2.989", 0.006567, 12, 0.00305

FP32 same-mode repeats also differ at rare rows: maximum sampled position
differences were 0.023079 / 0.005131 native units, normal differences
14.656 / 11.829 degrees, and visibility mismatch 0.04272% / 0.01221%
at 1536 / 3072. Some compressed-mode maxima exceed those repeat maxima.
The detailed data includes every repeat and comparison; passing the
P99-based gates does not promise that every output row is unchanged.

These inputs reside on a Windows volume accessed through WSL 9p/DrvFS.
The guarded resident mode used 85.135 / 484.805 system CPU seconds versus
1.005 / 3.415 seconds for plain oct2x16 with the same one-preloader setting.
This records the observed original-FP32 fallback cost on that storage setup;
native-ext4 performance has not been measured in this cohort.

All compression and guards remain opt-in. Original FP32 maps are unchanged.
