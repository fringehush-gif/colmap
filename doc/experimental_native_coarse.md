# Experimental coarse initialization

Initialize fine photometric planes from a smaller depth/normal reconstruction,
then reconsider the original random plane in the first fine sweep. This optional
method changes the resulting geometry; it is not native-quality equivalent.

Use fresh workspaces with the same original images, model and source order:

1. Run coarse PHOTO at 384 px, five iterations, geometry off, filtering on.
   Set `COLMAP_NATIVE_COARSE_EXPORT=/path/to/bundle` to export camera and
   image/pose bindings named `INDEX.camera.bin` and `INDEX.binding.bin`.
2. Match reference image names to the logged indices. Copy or link each coarse
   `.photometric.bin` depth/normal map into the bundle as `INDEX.depth.bin`
   and `INDEX.normal.bin`.
3. Run fine PHOTO at 1536 or 3072 px, two iterations, 15 samples, radius 5, step 1,
   geometry/filtering off, with:

       COLMAP_NATIVE_COARSE_INIT=/path/to/bundle
       COLMAP_NATIVE_COARSE_REFINE_SCHEDULE=1
       COLMAP_NATIVE_COARSE_FIRST_SWEEP_RANDOM=1

4. Unset all four variables before ordinary GEO5 with geometry/filtering on.
   Include coarse generation when measuring the complete reconstruction.

The importer checks dimensions, exact lengths, calibration, image name and pose.
Invalid depth/normal samples keep native random initialization. Maps are limited
to 1024 px per side; 384 px is the tested donor size. Native initialization remains
the default. Coarse flags require photometric mode and the default backend; incompatible
settings fail before map loading.
