# Isolated CPU qualification harness

Use clean, pinned source checkouts and the supplied native dependency/toolchain environment. `build_selected.py` requires each checkout's HEAD to equal its full supplied commit and rejects tracked modifications. The native build must have been configured against the **same absolute baseline source path** passed to the helper. Do not substitute a dirty checkout, a different base, reduced images, a smaller feature budget or selectively retained runs.

`targets.json` pins the two independent fork drafts, their common upstream baseline, expected test counts and predeclared performance criterion. Select exactly one target per job. The requested minimum is a three-second median whole-stage saving; passing correctness and showing a positive median are insufficient if the controls undermine attribution. Keep every observation and report an inconclusive result when the evidence warrants it. This job must not publish or modify a pull request.

## Set up the baseline

Set `REPO`, `BASE_REF`, `CANDIDATE_REF`, `TASK_ROOT`, `HARNESS` and `PHOTOS` from the job's real inputs. `REPO` must already contain both commits; `TASK_ROOT` and output directories must be new. `PHOTOS` must contain P1180141.JPG through P1180148.JPG from the official South Building archive identified in `photos.json`, at their original 3072×2304 resolution. The runner verifies each JPEG's exact size and SHA-256.

`python3 "$HARNESS/fetch_photos.py" --out "$PHOTOS"` obtains only the eight compressed-data ranges, checking both compressed and decoded identities and ZIP CRCs. If the server refuses HTTP ranges, obtain the full official archive and extract those named photos; the same byte checks still apply. Preserve the photo-verification receipt.

```bash
# Use the API template's actual toolchain environment, if it supplies a script.
# Discover that path in the running environment; authoring-workspace paths do not apply.
if [[ -n "${TOOLCHAIN_ENV:-}" ]]; then source "$TOOLCHAIN_ENV"; fi

: "${REPO:?}" "${BASE_REF:?}" "${CANDIDATE_REF:?}"
: "${TASK_ROOT:?}" "${HARNESS:?}" "${PHOTOS:?}"
mkdir "$TASK_ROOT"
git -C "$REPO" worktree add --detach "$TASK_ROOT/base" "$BASE_REF"
git -C "$REPO" worktree add --detach "$TASK_ROOT/candidate" "$CANDIDATE_REF"

cmake -S "$TASK_ROOT/base" -B "$TASK_ROOT/native-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= \
  '-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG' \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG' \
  -DTESTS_ENABLED=ON -DSIMD_ENABLED=ON -DOPENMP_ENABLED=ON \
  -DCUDA_ENABLED=OFF -DHIP_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DGUI_ENABLED=OFF -DONNX_ENABLED=OFF -DIPO_ENABLED=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=OFF \
  -DASAN_ENABLED=OFF -DUBSAN_ENABLED=OFF -DTSAN_ENABLED=OFF
cmake --build "$TASK_ROOT/native-build" --parallel 2 --target \
  colmap_feature_sift_test colmap_controllers_feature_extraction_test
```

An existing baseline build may be reused only when its clean source path/ref, native configuration and dependencies match these requirements. Preserve its CMake cache, compilation database and Ninja file. Use native `-O2 -DNDEBUG`, with no fast-math, `-Ofast`, IPO/LTO or architecture/layout experiments. GPU support is disabled through CUDA, HIP and OpenGL options; `GPU_ENABLED` is derived by COLMAP.

## Build and verify both arms

```bash
python3 "$HARNESS/build_selected.py" \
  --base-source "$TASK_ROOT/base" --base-ref "$BASE_REF" \
  --candidate-source "$TASK_ROOT/candidate" --candidate-ref "$CANDIDATE_REF" \
  --native-build "$TASK_ROOT/native-build" --mode native \
  --out "$TASK_ROOT/selected-native"

python3 "$HARNESS/build_selected.py" \
  --base-source "$TASK_ROOT/base" --base-ref "$BASE_REF" \
  --candidate-source "$TASK_ROOT/candidate" --candidate-ref "$CANDIDATE_REF" \
  --native-build "$TASK_ROOT/native-build" --mode san \
  --out "$TASK_ROOT/selected-san"
```

The helper rebuilds complete `sift.cc`, VLFeat `sift.c`, `covdet.c` and `scalespace.c` owners, the maintained SIFT tests, stage driver and raw-float oracle for each arm. It removes those owners from copied support archives, verifies retained members, records source/dependency/provider identities and link maps, runs CPU tests without skips, and compares complete 2,592×128-float oracle outputs. Prepared-descriptor oracles are included when that API exists.

Sanitizer mode enables ASan/UBSan/float-cast-overflow and assertions, and derives the native Eigen allocator ABI. **Remaining support is native and uninstrumented; this is selected-owner qualification.** Leak checking is disabled in this environment. Retain failed build/test attempts and their logs; do not change production code or weaken checks to hide a harness failure.

## Run the fixed native campaign

Set `PRIMARY=covariant` or `PRIMARY=dsp10` as declared by the job. Use only the **native** manifest for timing. No compilation, sanitizer runs or competing benchmark process may overlap the campaign.

```bash
: "${PRIMARY:?Set the predeclared covariant or dsp10 primary setting}"
python3 "$HARNESS/run_campaign.py" \
  --manifest "$TASK_ROOT/selected-native/campaign-manifest.json" \
  --photos "$PHOTOS" --primary "$PRIMARY" --count 8 \
  --deadline-seconds 3600 --run-timeout 900 \
  --out "$TASK_ROOT/campaign"
```

The runner chooses an allowed CPU unless an allowed `--cpu` is supplied, pins one benchmark process, limits library threads to one and holds a shared qualification lock. Arm A is the baseline and B the candidate. Every process uses all eight originals, requested feature budget 8192 and a fresh database. The fixed **20-run** order is:

| Cohort | Setting | Order |
|---|---|---|
| Before control | Primary | AA |
| Main comparison | Primary | ABBABAAB |
| Related comparison | Other covariant/DSP setting | ABBA |
| Ordinary-path control | Ordinary SIFT | BAAB |
| After control | Primary | BB |

The timer covers controller construction, image reads, extraction, database writes and destruction. Complete ordered keypoint/descriptor verification is outside the stage timer. Keep wall, process CPU, user CPU and system CPU separately. Report all observations and paired statistics; do not discard slow runs, subtract a control from another cohort, silently resume, shrink the schedule or claim completion without `completed.json` and successful frozen-input verification.

Earlier local ordinary-path controls showed severe and inconsistent timing noise, including large changes despite unchanged substantive hot-path instructions. These controls are part of the qualification, not optional extras. Process locking and CPU pinning do **not** establish exclusive physical hardware. Submit this work as the authorized **120-minute native Agents job**, with the runner's **60-minute campaign budget** reserving time for setup, verification and delivery. A two-hour deadline is not a billing discount and does not guarantee hardware isolation. If the deadline interrupts work, preserve and report the partial result.

## Package and deliver

Write the substantive report after inspecting the evidence. Include the exact source/configuration identities, test/static-analysis results, full-output equality, all **20 individual timing rows** or explicitly missing rows, paired statistics, ordinary-path controls, limitations and a verdict on the several-seconds whole-stage requirement. Do not promote a phase diagnostic into an end-to-end saving.

```bash
python3 "$HARNESS/package_results.py" \
  --campaign "$TASK_ROOT/campaign" \
  --build "$TASK_ROOT/selected-native" --build "$TASK_ROOT/selected-san" \
  --extra "$TASK_ROOT/REPORT.md" \
  --extra "$HARNESS/README.md" \
  --output "$TASK_ROOT/qualification.zip"
```

`--build` and `--extra` are repeatable; extras must be real files. The output must be outside input evidence directories and must not exist. Packaging retains campaign JSON and every available run receipt/log; build/test/oracle metadata, maps and executables up to 64 MiB; one verified full A feature witness per setting; and one verified A raw-float oracle per build. Duplicate witnesses, SQLite databases/sidecars, objects and support archives remain in the session. Divergent or unanchored partial witnesses are retained and identified instead of being hidden or relabeled as A output.

The ZIP embeds `package-manifest.json`, including individual timing rows, source paths, omitted-file reasons and the honest complete/partial/inconsistent status. A missing campaign directory or missing/invalid completion receipt is not upgraded to success; available build evidence can still be delivered as partial. The packager rereads **every ZIP payload**, checks its SHA-256, byte count and CRC, then creates `qualification.zip.receipt.json` with the exact archive byte count/hash and payload count. It never overwrites or deletes input evidence. A failed packaging attempt has no verified receipt; use a fresh output name.

Deliver the report/summary and all timing rows in the job result as well as the evidence archive and its receipt. Include only the real provider artifact IDs/URLs returned by delivery and the real saved paths. If upload/delivery fails, say so and retain the actual paths; never invent an attachment ID or claim the user received an archive that was not delivered.
