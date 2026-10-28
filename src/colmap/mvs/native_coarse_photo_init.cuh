// SPDX-License-Identifier: BSD-3-Clause
// Optional two-level initialization from photometric depth and normal maps.
// Include in native_coarse_v1; normal and depth remain native camera-space
// FP32.

// Bound decoded coarse map dimensions; the caller chooses the donor size.
constexpr uint32_t kMaxCoarseSide = 1024;

inline void ValidateCoarseMap(const std::filesystem::path& path,
                              uint32_t width,
                              uint32_t height,
                              uint32_t channels) {
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  THROW_CHECK(stream.is_open());
  uint32_t shape[3] = {0, 0, 0};
  int field = 0;
  size_t header_bytes = 0;
  bool has_digit = false;
  while (field < 3 && header_bytes < 128) {
    const int c = stream.get();
    THROW_CHECK_NE(c, std::char_traits<char>::eof());
    ++header_bytes;
    if (c >= '0' && c <= '9') {
      const uint32_t digit = static_cast<uint32_t>(c - '0');
      const uint32_t limit = field == 2 ? 3u : kMaxCoarseSide;
      THROW_CHECK_LE(digit, limit);
      THROW_CHECK_LE(shape[field], (limit - digit) / 10u);
      shape[field] = shape[field] * 10u + digit;
      has_digit = true;
    } else {
      THROW_CHECK(c == '&' && has_digit);
      ++field;
      has_digit = false;
    }
  }
  THROW_CHECK_EQ(field, 3);
  THROW_CHECK_EQ(shape[0], width);
  THROW_CHECK_EQ(shape[1], height);
  THROW_CHECK_EQ(shape[2], channels);
  THROW_CHECK_EQ(std::filesystem::file_size(path),
                 header_bytes + static_cast<size_t>(width) * height * channels *
                                    sizeof(float));
}

__global__ void ProlongCoarsePlanes(const RefCamera fine,
                                    const RefCamera coarse,
                                    GpuMatView<float> coarse_depth,
                                    GpuMatView<float> coarse_normal,
                                    GpuMatView<float> depth,
                                    GpuMatView<float> normal,
                                    float minimum,
                                    float maximum,
                                    unsigned int* adopted_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  bool adopted = false;
  if (c < depth.GetWidth() && r < depth.GetHeight()) {
    const float target_ray[3] = {fine.inv_K[0] * c + fine.inv_K[1],
                                 fine.inv_K[2] * r + fine.inv_K[3],
                                 1};
    // Choose nearest coarse pixel through ACTUAL intrinsic matrices. Rounded
    // image sizes and principal points need not be a common integer multiple.
    const float coarse_c = (target_ray[0] - coarse.inv_K[1]) / coarse.inv_K[0];
    const float coarse_r = (target_ray[1] - coarse.inv_K[3]) / coarse.inv_K[2];
    const int cc = min(static_cast<int>(coarse_depth.GetWidth()) - 1,
                       max(0, static_cast<int>(floorf(coarse_c + 0.5f))));
    const int rr = min(static_cast<int>(coarse_depth.GetHeight()) - 1,
                       max(0, static_cast<int>(floorf(coarse_r + 0.5f))));
    float n[3];
    coarse_normal.GetSlice(rr, cc, n);
    const float z = coarse_depth.Get(rr, cc);
    const float norm2 = DotProduct3(n, n);
    const float donor_ray[3] = {coarse.inv_K[0] * cc + coarse.inv_K[1],
                                coarse.inv_K[2] * rr + coarse.inv_K[3],
                                1};
    const float numerator = DotProduct3(n, donor_ray);
    const float denominator = DotProduct3(n, target_ray);
    const float candidate = z * numerator / denominator;
    adopted = isfinite(z) && z > 0 && isfinite(n[0]) && isfinite(n[1]) &&
              isfinite(n[2]) && isfinite(norm2) &&
              fabsf(norm2 - 1.0f) < 1e-3f && numerator < -1e-5f &&
              denominator < -1e-5f && isfinite(candidate) && candidate > 0 &&
              candidate >= minimum && candidate <= maximum;
    if (adopted) {
      depth.Set(r, c, candidate);
      normal.SetSlice(r, c, n);
    }
  }
  // No validity return before the ballot: partial image warps participate too.
  const unsigned int votes = __ballot_sync(0xffffffffu, adopted);
  if (threadIdx.x == 0) atomicAdd(adopted_count, __popc(votes));
}

inline void InitializePhotoFromCoarse(const RefCamera fine,
                                      GpuMatView<float> depth,
                                      GpuMatView<float> normal,
                                      int reference_id,
                                      const Image& reference_image,
                                      const std::vector<Image>& model_images,
                                      float minimum,
                                      float maximum) {
  const char* init = std::getenv("COLMAP_NATIVE_COARSE_INIT");
  const char* export_root = std::getenv("COLMAP_NATIVE_COARSE_EXPORT");
  const bool initialize = init && init[0] != '\0';
  const bool export_camera = export_root && export_root[0] != '\0';
  THROW_CHECK(!(initialize && export_camera))
      << "Coarse pass must not recursively initialize itself";
  const std::string stem = std::to_string(reference_id);
  // The numeric slot is not scene identity. Bind the camera sidecar to the
  // current reference image name and unscaled camera pose as well.
  const std::string image_name =
      reference_image.GetPath().filename().generic_string();
  THROW_CHECK(!image_name.empty());
  THROW_CHECK_LE(image_name.size(), 4096);
  // The binding format stores a basename, not the workspace-specific image
  // root. Reject a model where two registered images share that basename;
  // otherwise the sidecar could bind to the wrong subdirectory after a reorder.
  size_t basename_matches = 0;
  for (const Image& image : model_images) {
    basename_matches +=
        image.GetPath().filename().generic_string() == image_name ? 1 : 0;
  }
  THROW_CHECK_EQ(basename_matches, 1)
      << "Native coarse requires unique registered image basenames";
  const auto binding_path =
      std::filesystem::path(export_camera ? export_root : init) /
      (stem + ".binding.bin");
  if (export_camera) {
    const std::filesystem::path path =
        std::filesystem::path(export_root) / (stem + ".camera.bin");
    std::ofstream file(path, std::ios::out | std::ios::binary);
    THROW_CHECK(file.is_open());
    WriteBinaryLittleEndian<uint32_t>(&file,
                                      static_cast<uint32_t>(depth.GetWidth()));
    WriteBinaryLittleEndian<uint32_t>(&file,
                                      static_cast<uint32_t>(depth.GetHeight()));
    WriteBinaryLittleEndian<float>(&file, span<const float>(fine.K, 4));
    WriteBinaryLittleEndian<float>(&file, span<const float>(fine.inv_K, 4));
    file.close();
    THROW_CHECK(file.good());
    std::ofstream binding(binding_path, std::ios::out | std::ios::binary);
    THROW_CHECK(binding.is_open());
    WriteBinaryLittleEndian<uint32_t>(&binding,
                                      static_cast<uint32_t>(image_name.size()));
    binding.write(image_name.data(), image_name.size());
    WriteBinaryLittleEndian<float>(
        &binding, span<const float>(reference_image.GetR(), 9));
    WriteBinaryLittleEndian<float>(
        &binding, span<const float>(reference_image.GetT(), 3));
    binding.close();
    THROW_CHECK(binding.good());
    LOG(INFO)
        << "Native coarse camera export: reference=" << reference_id
        << " width=" << depth.GetWidth() << " height=" << depth.GetHeight()
        << " bytes=40; actual native K/inverse-K, no plane initialization";
  }
  if (!initialize) return;
  const std::filesystem::path root(init);
  THROW_CHECK_LE(std::filesystem::file_size(binding_path), 4096 + 52);
  std::ifstream binding(binding_path, std::ios::in | std::ios::binary);
  THROW_CHECK(binding.is_open());
  const uint32_t name_bytes = ReadBinaryLittleEndian<uint32_t>(&binding);
  THROW_CHECK_GT(name_bytes, 0);
  THROW_CHECK_LE(name_bytes, 4096);
  THROW_CHECK_EQ(std::filesystem::file_size(binding_path),
                 static_cast<size_t>(name_bytes) + 52);
  std::string bound_name(name_bytes, '\0');
  binding.read(bound_name.data(), name_bytes);
  THROW_CHECK(binding.good());
  THROW_CHECK_EQ(bound_name, image_name);
  for (int i = 0; i < 9; ++i) {
    const float bound = ReadBinaryLittleEndian<float>(&binding);
    THROW_CHECK_EQ(bound, reference_image.GetR()[i]);
  }
  for (int i = 0; i < 3; ++i) {
    const float bound = ReadBinaryLittleEndian<float>(&binding);
    THROW_CHECK_EQ(bound, reference_image.GetT()[i]);
  }
  THROW_CHECK(binding.good());
  THROW_CHECK_LE(static_cast<size_t>(depth.GetWidth()) * depth.GetHeight(),
                 std::numeric_limits<unsigned int>::max());
  const auto depth_path = root / (stem + ".depth.bin");
  const auto normal_path = root / (stem + ".normal.bin");
  const auto camera_path = root / (stem + ".camera.bin");
  // File sizes checked before decoded allocation. Explicitly bound the map
  // up to 1024 px; the caller still chooses the coarse resolution.
  constexpr size_t kMaxCoarsePixels =
      static_cast<size_t>(kMaxCoarseSide) * kMaxCoarseSide;
  THROW_CHECK_LE(std::filesystem::file_size(depth_path),
                 kMaxCoarsePixels * sizeof(float) + 128);
  THROW_CHECK_LE(std::filesystem::file_size(normal_path),
                 kMaxCoarsePixels * 3 * sizeof(float) + 128);
  THROW_CHECK_EQ(std::filesystem::file_size(camera_path), 40);
  std::ifstream camera_file(camera_path, std::ios::in | std::ios::binary);
  THROW_CHECK(camera_file.is_open());
  const uint32_t width = ReadBinaryLittleEndian<uint32_t>(&camera_file);
  const uint32_t height = ReadBinaryLittleEndian<uint32_t>(&camera_file);
  float K[4], inv_K[4];
  for (int i = 0; i < 4; ++i)
    K[i] = ReadBinaryLittleEndian<float>(&camera_file);
  for (int i = 0; i < 4; ++i)
    inv_K[i] = ReadBinaryLittleEndian<float>(&camera_file);
  THROW_CHECK(camera_file.good());
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_LE(width, kMaxCoarseSide);
  THROW_CHECK_LE(height, kMaxCoarseSide);
  THROW_CHECK_LE(width, depth.GetWidth());
  THROW_CHECK_LE(height, depth.GetHeight());
  THROW_CHECK(width < depth.GetWidth() || height < depth.GetHeight());
  for (int i = 0; i < 4; ++i) {
    THROW_CHECK(std::isfinite(K[i]));
    THROW_CHECK(std::isfinite(inv_K[i]));
  }
  THROW_CHECK_GT(K[0], 0);
  THROW_CHECK_GT(K[2], 0);
  THROW_CHECK_GT(inv_K[0], 0);
  THROW_CHECK_GT(inv_K[2], 0);
  // Actual coarse K must agree with the same full-image calibration scaled by
  // the actual independently rounded width/height, within FP32 rounding only.
  const float sx = static_cast<float>(width) / depth.GetWidth();
  const float sy = static_cast<float>(height) / depth.GetHeight();
  for (int i = 0; i < 4; ++i) {
    // InitTransforms stores cx,cy AFTER native pixel-center offset -0.5.
    // Image::Rescale scales the unshifted K first, then InitTransforms shifts
    // each resolution independently: stored_cx_c=(stored_cx_f+0.5)*sx-0.5.
    const float scale = i < 2 ? sx : sy;
    const float expected = (i == 1 || i == 3)
                               ? (fine.K[i] + 0.5f) * scale - 0.5f
                               : fine.K[i] * scale;
    THROW_CHECK_LE(std::abs(K[i] - expected),
                   2e-5f * std::max(1.0f, std::abs(expected)));
  }
  THROW_CHECK_LE(std::abs(K[0] * inv_K[0] - 1), 2e-5f);
  THROW_CHECK_LE(std::abs(K[2] * inv_K[2] - 1), 2e-5f);
  THROW_CHECK_LE(std::abs(K[1] * inv_K[0] + inv_K[1]), 2e-5f);
  THROW_CHECK_LE(std::abs(K[3] * inv_K[2] + inv_K[3]), 2e-5f);
  // Native Mat<float>::Read checks required length before allocation. Enforce
  // our narrower sidecar dimensions/channels and EXACT length before calling
  // it too: a malformed header cannot allocate beyond the coarse size bound.
  ValidateCoarseMap(depth_path, width, height, 1);
  ValidateCoarseMap(normal_path, width, height, 3);
  Mat<float> coarse_depth_host, coarse_normal_host;
  coarse_depth_host.Read(depth_path);
  coarse_normal_host.Read(normal_path);
  THROW_CHECK_EQ(coarse_depth_host.GetWidth(), width);
  THROW_CHECK_EQ(coarse_depth_host.GetHeight(), height);
  THROW_CHECK_EQ(coarse_depth_host.GetDepth(), 1);
  THROW_CHECK_EQ(coarse_normal_host.GetWidth(), width);
  THROW_CHECK_EQ(coarse_normal_host.GetHeight(), height);
  THROW_CHECK_EQ(coarse_normal_host.GetDepth(), 3);
  GpuMat<float> coarse_depth(width, height), coarse_normal(width, height, 3);
  coarse_depth.CopyToDevice(coarse_depth_host.GetPtr(), width * sizeof(float));
  coarse_normal.CopyToDevice(coarse_normal_host.GetPtr(),
                             width * sizeof(float));
  GpuMat<unsigned int> count(1, 1);
  CUDA_SAFE_CALL(cudaMemset(count.GetPtr(), 0, sizeof(unsigned int)));
  CudaTimer timer;
  const dim3 block(32, 8),
      grid((depth.GetWidth() + 31) / 32, (depth.GetHeight() + 7) / 8);
  ProlongCoarsePlanes<<<grid, block>>>(fine,
                                       MakeRefCamera(K, inv_K),
                                       coarse_depth.View(),
                                       coarse_normal.View(),
                                       depth,
                                       normal,
                                       minimum,
                                       maximum,
                                       count.GetPtr());
  CUDA_SYNC_AND_CHECK();
  timer.Print(" Native coarse plane prolongation");
  unsigned int adopted = 0;
  count.CopyToHost(&adopted, sizeof(adopted));
  LOG(INFO) << "Native coarse initialization: reference=" << reference_id
            << " coarse=" << width << "x" << height
            << " fine=" << depth.GetWidth() << "x" << depth.GetHeight()
            << " adopted=" << adopted << " preservedRandom="
            << depth.GetWidth() * depth.GetHeight() - adopted
            << " temporaryDeviceBytes="
            << coarse_depth.GetPitch() * height +
                   coarse_normal.GetPitch() * height * 3 + count.GetPitch()
            << "; original random initialization completed first; full fine "
               "search unchanged";
}
