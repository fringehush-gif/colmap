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

FP32 normal tiles and exact depth
---------------------------------

Bounded component mode can keep normal tiles rather than decoded whole normal
maps. The tile and mapped-owner budgets are separate. Mapped bytes describe
file-backed address space, not an equivalent amount of resident physical RAM.
Mapped ownership currently requires supported Linux little-endian storage.

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
Normal-tile size must remain zero.
Stored depth maps must already match the consumed depth grid; this diagnostic
does not support resizing those original depth maps during Fusion.
Every source depth sample must be finite, including invalid-depth sentinels;
non-finite samples are rejected before exact-depth reads.

Check available combinations with ``StereoFusionOptions::Check()``.
