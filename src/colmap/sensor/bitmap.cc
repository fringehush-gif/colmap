// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/bitmap.h"

#include "colmap/sensor/bitmap_reader.h"
#include "colmap/sensor/bitmap_rgb8.h"
#include "colmap/sensor/database.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/oiio_utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>

#include <OpenImageIO/color.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imagebufalgo_util.h>
#include <OpenImageIO/imageio.h>

namespace colmap {
namespace {

// Match OIIO 2.5.18's ordered math only in qualified compiler/library builds.
#if OIIO_VERSION == OIIO_MAKE_VERSION(2, 5, 18) && defined(__GNUC__) &&   \
    (__GNUC__ >= 9) && defined(__x86_64__) && !defined(__clang__) &&      \
    !defined(__FAST_MATH__) && !defined(__FMA__) && !defined(__FMA4__) && \
    (__FLT_EVAL_METHOD__ == 0) && (__GCC_IEC_559 >= 2) &&                 \
    !defined(__AVX512F__)
#define COLMAP_BITMAP_GRAY_TRIANGLE_ENABLED
namespace gray_triangle_resize {
namespace {
constexpr uint64_t kMaxPixels = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMaxAxisEntries = 8ULL * 1024ULL * 1024ULL;
struct Axis {
  int taps = 0;
  std::vector<int> origin;
  std::vector<float> weight;
};

float Triangle(float x) { return std::max(0.0f, 1.0f - std::abs(x)); }

bool MakeAxis(int source, int target, Axis* axis) {
  const float ratio = static_cast<float>(target) / static_cast<float>(source);
  const float width = 2.0f * std::max(1.0f, ratio);
  const float radius_value = std::ceil((width * 0.5f) / ratio);
  if (!std::isfinite(radius_value) || radius_value < 0.0f ||
      static_cast<double>(radius_value) >
          static_cast<double>((std::numeric_limits<int>::max() - 1) / 2))
    return false;
  const int radius = static_cast<int>(radius_value);
  if (static_cast<uint64_t>(target) *
          (2ULL * static_cast<uint64_t>(radius) + 1ULL) >
      kMaxAxisEntries)
    return false;
  Axis a;
  a.taps = 2 * radius + 1;
  a.origin.resize(target);
  a.weight.resize(static_cast<size_t>(target) * a.taps);
  for (int dst = 0; dst < target; ++dst) {
    const float sample = (static_cast<float>(dst) + 0.5f) *
                         (1.0f / static_cast<float>(target)) *
                         static_cast<float>(source);
    const float center_value = std::floor(sample);
    if (!std::isfinite(center_value) ||
        static_cast<double>(center_value) <
            static_cast<double>(std::numeric_limits<int>::min()) ||
        static_cast<double>(center_value) >
            static_cast<double>(std::numeric_limits<int>::max()))
      return false;
    const int center = static_cast<int>(center_value);
    const int64_t first = static_cast<int64_t>(center) - radius;
    const int64_t last = first + a.taps - 1;
    if (first < std::numeric_limits<int>::min() ||
        last > std::numeric_limits<int>::max())
      return false;
    const float fraction = sample - static_cast<float>(center);
    a.origin[dst] = center - radius;
    float sum = 0.0f;
    for (int tap = 0; tap < a.taps; ++tap) {
      const float offset =
          ratio * (static_cast<float>(tap - radius) - (fraction - 0.5f));
      const float scaled = offset * (2.0f / width);
      const float w = Triangle(scaled);
      a.weight[static_cast<size_t>(dst) * a.taps + tap] = w;
      sum += w;
    }
    if (sum != 0.0f) {
      for (int tap = 0; tap < a.taps; ++tap)
        a.weight[static_cast<size_t>(dst) * a.taps + tap] /= sum;
    }
  }
  axis->taps = a.taps;
  axis->origin.swap(a.origin);
  axis->weight.swap(a.weight);
  return true;
}

uint8_t Quantize(float value) {
  float scaled = value * 255.0f;
  scaled += scaled < 0.0f ? -0.5f : 0.5f;
  scaled = std::fmin(255.0f, std::fmax(0.0f, scaled));
  // Finite clamped nonnegative conversion truncates, matching V37's rz.
  return static_cast<uint8_t>(scaled);
}

bool Disjoint(const uint8_t* src,
              size_t source_bytes,
              const uint8_t* dst,
              size_t target_bytes) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(src);
  const uintptr_t b = reinterpret_cast<uintptr_t>(dst);
  if (source_bytes > std::numeric_limits<uintptr_t>::max() - a ||
      target_bytes > std::numeric_limits<uintptr_t>::max() - b)
    return false;
  return a + source_bytes <= b || b + target_bytes <= a;
}
}  // namespace

// Packed rows only: source byte count must equal sw*sh; destination capacity
// must be >=dw*dh. Caller owns live, disjoint input/output buffers throughout.
// Refusal/axis allocation failure returns before writes; pool exceptions
// propagate. Writes exactly dw*dh bytes on success; source and output suffix
// are untouched.
bool ResizeGrayTriangle(const uint8_t* src,
                        size_t source_bytes,
                        int sw,
                        int sh,
                        uint8_t* dst,
                        size_t target_capacity,
                        int dw,
                        int dh) {
  if (!src || !dst || sw < 1 || sh < 1 || dw < 1 || dh < 1 || dw > sw ||
      dh > sh)
    return false;
  const uint64_t source_pixels =
      static_cast<uint64_t>(sw) * static_cast<uint64_t>(sh);
  const uint64_t target_pixels =
      static_cast<uint64_t>(dw) * static_cast<uint64_t>(dh);
  if (source_pixels > kMaxPixels || target_pixels > kMaxPixels ||
      source_pixels > std::numeric_limits<size_t>::max() ||
      target_pixels > std::numeric_limits<size_t>::max() ||
      source_bytes != static_cast<size_t>(source_pixels) ||
      target_capacity < static_cast<size_t>(target_pixels) ||
      !Disjoint(src,
                static_cast<size_t>(source_pixels),
                dst,
                static_cast<size_t>(target_pixels)))
    return false;

  Axis ax, ay;
  try {
    if (!MakeAxis(sw, dw, &ax) || !MakeAxis(sh, dh, &ay)) return false;
  } catch (const std::bad_alloc&) {
    return false;
  } catch (const std::length_error&) {
    return false;
  }
  // Axis validation/allocation precedes output. The OIIO pool joins before
  // return.
  OIIO::ImageBufAlgo::parallel_image(
      OIIO::ROI(0, dw, 0, dh, 0, 1, 0, 1),
      OIIO::paropt(0),
      [&](OIIO::ROI region) {
        for (int y = region.ybegin; y < region.yend; ++y) {
          for (int x = region.xbegin; x < region.xend; ++x) {
            const float* xw =
                ax.weight.data() + static_cast<size_t>(x) * ax.taps;
            const float* yw =
                ay.weight.data() + static_cast<size_t>(y) * ay.taps;
            float sum = 0.0f;
            for (int j = 0; j < ay.taps; ++j) {
              const float vy = yw[j];
              if (vy == 0.0f) continue;
              const int sy = std::max(0, std::min(sh - 1, ay.origin[y] + j));
              for (int i = 0; i < ax.taps; ++i) {
                const float w = vy * xw[i];
                if (w == 0.0f) continue;
                const int sx = std::max(0, std::min(sw - 1, ax.origin[x] + i));
                sum += w * (static_cast<float>(
                                src[static_cast<size_t>(sy) * sw + sx]) *
                            (1.0f / 255.0f));
              }
            }
            dst[static_cast<size_t>(y) * dw + x] = Quantize(sum);
          }
        }
      });
  return true;
}
}  // namespace gray_triangle_resize

#endif

struct BitmapReaderRegistry {
  struct Entry {
    const ScopedBitmapReader* owner;
    std::weak_ptr<ScopedBitmapReader> reader;
  };
  std::mutex mutex;
  FlatHashMap<std::string, Entry> entries;
  std::atomic<size_t> count{0};
};

BitmapReaderRegistry& ReaderRegistry() {
  static BitmapReaderRegistry registry;
  return registry;
}

std::string ReaderPath(const std::filesystem::path& path) {
  return std::filesystem::absolute(path).lexically_normal().generic_string();
}

struct OIIOMetaData : public Bitmap::MetaData {
  OIIOMetaData() = default;

  OIIO::ImageSpec image_spec;

  static OIIOMetaData* Upcast(Bitmap::MetaData* meta_data) {
    return THROW_CHECK_NOTNULL(dynamic_cast<OIIOMetaData*>(meta_data));
  }

  static std::unique_ptr<MetaData> Clone(
      const std::unique_ptr<Bitmap::MetaData>& meta_data) {
    auto cloned = std::make_unique<OIIOMetaData>();
    *cloned = *Upcast(meta_data.get());
    return cloned;
  }
};

// For backwards compatibility with older OIIO versions without implicit
// conversion from std::string_view.
OIIO::string_view OIIOFromStdStringView(std::string_view value) {
  return {value.data(), value.size()};
}

// OIIO's IPTC IIM encoder reads the "IPTC:*" attributes plus the TIFF/EXIF
// synonyms from its IIM tag table (Software, Artist, Copyright, and
// ImageDescription, see the `iimtag` table in OIIO's
// src/libOpenImageIO/iptc.cpp). OIIO 2.x aborts on empty string values in
// `encode_iptc_iim_one_tag` (see colmap/colmap#4599), so detect these
// attributes here in order to omit them from written files. "Keywords" is
// included as a defensive alias for the repeatable IPTC:Keywords field. Empty
// strings carry no information, so omitting them is lossless.
bool IsIptcMetadata(const OIIO::ParamValue& attribute) {
  if (attribute.type().basetype != OIIO::TypeDesc::STRING) {
    return false;
  }

  const std::string_view name(attribute.name().c_str());
  return name.substr(0, 5) == "IPTC:" || name == "Keywords" ||
         name == "Software" || name == "Artist" || name == "Copyright" ||
         name == "ImageDescription";
}

// Check all values, as repeatable IPTC fields (e.g. Keywords) may hold
// multiple strings and OIIO encodes each of them.
bool HasEmptyIptcString(const OIIO::ParamValue& attribute) {
  if (!IsIptcMetadata(attribute)) {
    return false;
  }
  for (int i = 0; i < attribute.nvalues(); ++i) {
    if (attribute.get_string_indexed(i).empty()) {
      return true;
    }
  }
  return false;
}

// Convert a filesystem path to a UTF-8 std::string. On Windows,
// path.string() returns a locale-dependent narrow string, which mangles
// non-ASCII characters before they reach OIIO. path.u8string() always yields
// UTF-8 bytes, but its return type changes from std::string in C++17 to
// std::u8string in C++20; the reinterpret_cast keeps this portable.
std::string PathToUtf8(const std::filesystem::path& path) {
  const auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

std::vector<uint8_t> ConvertColorSpace(const uint8_t* src_data,
                                       int width,
                                       int height,
                                       int channels,
                                       const std::string_view& from,
                                       const std::string_view& to) {
  EnsureOpenImageIOInitialized();
  const OIIO::ImageSpec image_spec(
      width, height, channels, OIIO::TypeDesc::UINT8);
  const int pitch = width * channels;
  const OIIO::ImageBuf src(image_spec, const_cast<uint8_t*>(src_data));
  std::vector<uint8_t> tgt_data(height * pitch);
  OIIO::ImageBuf tgt(image_spec, tgt_data.data());
  THROW_CHECK(OIIO::ImageBufAlgo::colorconvert(
      tgt, src, OIIOFromStdStringView(from), OIIOFromStdStringView(to)));
  return tgt_data;
}

void SetImageSpecColorSpace(OIIO::ImageSpec& image_spec,
                            const OIIO::string_view& colorspace) {
  EnsureOpenImageIOInitialized();
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
  image_spec.set_colorspace(colorspace);
#else
  // Extract logic from 3.0.0 version for backwards compatibility.
  const OIIO::string_view oldspace =
      image_spec.get_string_attribute("oiio:ColorSpace");
  if (oldspace.size() && colorspace.size() && oldspace == colorspace) {
    return;
  }

  if (colorspace.empty()) {
    image_spec.erase_attribute("oiio:ColorSpace");
  } else {
    image_spec.attribute("oiio:ColorSpace", colorspace);
  }

  if (colorspace != "sRGB") {
    image_spec.erase_attribute("Exif:ColorSpace");
  }

  image_spec.erase_attribute("tiff:ColorSpace");
  image_spec.erase_attribute("tiff:PhotometricInterpretation");
  image_spec.erase_attribute("oiio:Gamma");
#endif
}

bool IsEquivalentColorSpace(const std::string_view& colorspace1,
                            const std::string_view& colorspace2) {
  EnsureOpenImageIOInitialized();
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
  return OIIO::equivalent_colorspace(colorspace1, colorspace2);
#else
  // Poor (wo)man's version of available functionality in recent OIIO versions.
  auto is_linear_srgb = [](const std::string_view& colorspace) {
    return colorspace == "linear" || colorspace == "lin_srgb" ||
           colorspace == "lin_rec709P";
  };
  if (is_linear_srgb(colorspace1) && is_linear_srgb(colorspace2)) {
    return true;
  } else {
    return colorspace1 == colorspace2;
  }
#endif
}

}  // namespace

ScopedBitmapReader::ScopedBitmapReader(
    std::vector<std::string> paths,
    Callback callback,
    FlatHashMap<std::string, std::pair<int, int>> image_sizes)
    : paths_(std::move(paths)),
      callback_(std::move(callback)),
      image_sizes_(std::move(image_sizes)) {}

std::shared_ptr<ScopedBitmapReader> ScopedBitmapReader::Create(
    const std::vector<std::filesystem::path>& paths,
    Callback callback,
    const std::vector<std::pair<int, int>>& image_sizes_wh) {
  if (paths.empty() || !callback) {
    throw std::invalid_argument(
        "Bitmap reader requires an exact roster/callback");
  }
  if (!image_sizes_wh.empty() && image_sizes_wh.size() != paths.size()) {
    throw std::invalid_argument(
        "Bitmap reader dimensions must cover its exact roster");
  }
  FlatHashMap<std::string, std::pair<int, int>> image_sizes;
  std::vector<std::string> keys;
  FlatHashSet<std::string> unique;
  keys.reserve(paths.size());
  for (const auto& path : paths) {
    if (!path.is_absolute()) {
      throw std::invalid_argument("Bitmap reader paths must be absolute");
    }
    auto key = ReaderPath(path);
    if (!unique.insert(key).second) {
      throw std::invalid_argument("Bitmap reader roster has duplicate paths");
    }
    keys.push_back(std::move(key));
    if (!image_sizes_wh.empty()) {
      const auto size = image_sizes_wh.at(keys.size() - 1);
      if (size.first <= 0 || size.second <= 0 ||
          size.first > std::numeric_limits<int>::max() / 3 ||
          size.second > std::numeric_limits<int>::max() / (size.first * 3)) {
        throw std::invalid_argument(
            "Bitmap reader RGB8 dimensions are invalid");
      }
      image_sizes.emplace(keys.back(), size);
    }
  }
  auto reader = std::shared_ptr<ScopedBitmapReader>(new ScopedBitmapReader(
      std::move(keys), std::move(callback), std::move(image_sizes)));
  auto& registry = ReaderRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto& key : reader->paths_) {
    const auto found = registry.entries.find(key);
    if (found != registry.entries.end() && !found->second.reader.expired()) {
      // Unlock before the unregistered reader's destructor calls Close.
      throw std::invalid_argument("Bitmap reader path already has an owner");
    }
  }
  for (const auto& key : reader->paths_) {
    registry.entries[key] = {reader.get(), reader};
  }
  registry.count.store(registry.entries.size(), std::memory_order_release);
  return reader;
}

std::shared_ptr<ScopedBitmapReader> ScopedBitmapReader::Find(
    const std::filesystem::path& path) {
  auto& registry = ReaderRegistry();
  // Stock Bitmap readers incur no path normalization/mutex when unused.
  if (!registry.count.load(std::memory_order_acquire)) return nullptr;
  const auto key = ReaderPath(path);
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.entries.find(key);
  return found == registry.entries.end() ? nullptr
                                         : found->second.reader.lock();
}

ScopedBitmapReader::~ScopedBitmapReader() { Close(); }

void ScopedBitmapReader::Close() {
  auto& registry = ReaderRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto& key : paths_) {
    const auto found = registry.entries.find(key);
    if (found != registry.entries.end() && found->second.owner == this) {
      registry.entries.erase(key);
    }
  }
  registry.count.store(registry.entries.size(), std::memory_order_release);
}

bool ScopedBitmapReader::Read(const std::filesystem::path& path,
                              bool as_rgb,
                              bool linearize_colorspace,
                              Bitmap* target) const {
  const auto key = ReaderPath(path);
  const bool read = callback_(
      std::filesystem::path(key), as_rgb, linearize_colorspace, target);
  const auto expected = image_sizes_.find(key);
  if (read && expected != image_sizes_.end() &&
      (target->Width() != expected->second.first ||
       target->Height() != expected->second.second)) {
    throw std::invalid_argument(
        "Bitmap payload grid differs from its declared resident dimensions");
  }
  return read;
}

std::optional<std::pair<int, int>> ScopedBitmapReader::ImageSize(
    const std::filesystem::path& path) const {
  const auto found = image_sizes_.find(ReaderPath(path));
  if (found == image_sizes_.end()) return std::nullopt;
  return found->second;
}

Bitmap::Bitmap()
    : width_(0), height_(0), channels_(0), linear_colorspace_(false) {
  EnsureOpenImageIOInitialized();
}

Bitmap::Bitmap(const int width,
               const int height,
               const bool as_rgb,
               const bool linear_colorspace) {
  EnsureOpenImageIOInitialized();
  width_ = width;
  height_ = height;
  channels_ = as_rgb ? 3 : 1;
  linear_colorspace_ = linear_colorspace;
  data_.resize(width_ * height_ * channels_);
  auto meta_data = std::make_unique<OIIOMetaData>();
  meta_data->image_spec =
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8);
  SetImageSpecColorSpace(meta_data->image_spec,
                         linear_colorspace ? "linear" : "sRGB");
  meta_data_ = std::move(meta_data);
}

Bitmap::Bitmap(const Bitmap& other) {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = other.data_;
  if (!other.IsEmpty()) {
    meta_data_ = OIIOMetaData::Clone(other.meta_data_);
  }
}

Bitmap::Bitmap(Bitmap&& other) noexcept {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = std::move(other.data_);
  meta_data_ = std::move(other.meta_data_);
  other.width_ = 0;
  other.height_ = 0;
  other.channels_ = 0;
}

Bitmap& Bitmap::operator=(const Bitmap& other) {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = other.data_;
  if (other.IsEmpty()) {
    meta_data_ = nullptr;
  } else {
    meta_data_ = OIIOMetaData::Clone(other.meta_data_);
  }
  return *this;
}

Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) {
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    linear_colorspace_ = other.linear_colorspace_;
    data_ = std::move(other.data_);
    meta_data_ = std::move(other.meta_data_);
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
  }
  return *this;
}

void Bitmap::Fill(const BitmapColor<uint8_t>& color) {
  if (IsGrey()) {
    std::fill(data_.begin(), data_.end(), color.r);
  } else {
    THROW_CHECK_EQ(data_.size() % 3, 0);
    size_t i = 0;
    while (i < data_.size()) {
      data_[i++] = color.r;
      data_[i++] = color.g;
      data_[i++] = color.b;
    }
  }
}

std::optional<int> Bitmap::ExifOrientation() const {
  int orientation = 0;
  if (GetMetaData("Orientation", "int", &orientation)) {
    return orientation;
  }
  return std::nullopt;
}

std::optional<std::string> Bitmap::ExifCameraModel() const {
  // Read camera make and model
  const std::optional<std::string> make_str = GetMetaData("Make");
  if (!make_str.has_value()) {
    return std::nullopt;
  }
  const std::optional<std::string> model_str = GetMetaData("Model");
  if (!model_str.has_value()) {
    return std::nullopt;
  }
  float focal_length = 0;
  if (!GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length) &&
      !GetMetaData("Exif:FocalLength", "float", &focal_length)) {
    return std::nullopt;
  }
  std::string camera_model = StringPrintf("%s-%s-%.6f-%dx%d",
                                          make_str->c_str(),
                                          model_str->c_str(),
                                          static_cast<double>(focal_length),
                                          width_,
                                          height_);
  return camera_model;
}

std::optional<double> Bitmap::ExifFocalLength() const {
  const double max_size = std::max(width_, height_);

  float focal_length_35mm = 0;
  if (GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length_35mm)) {
    if (focal_length_35mm > 0) {
      // Based on https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
      // According to CIPA guidelines, 35 mm equivalent focal length is to be
      // calculated like this:
      // "focal length in 35 mm camera" =
      //   (Diagonal distance of image area in the 35 mm camera (43.27 mm) /
      //    Diagonal distance of image area on the image sensor of the DSC)
      //    * focal length of the lens of the DSC.
      const double diagonal = std::sqrt(width_ * width_ + height_ * height_);
      return focal_length_35mm / 43.27 * diagonal;
    }
  }

  float focal_length_mm = 0.f;
  if (GetMetaData("Exif:FocalLength", "float", &focal_length_mm)) {
    float focal_x_res = 0.f;
    int focal_x_res_unit = 0;
    if (GetMetaData("Exif:FocalPlaneXResolution", "float", &focal_x_res) &&
        GetMetaData(
            "Exif:FocalPlaneResolutionUnit", "int", &focal_x_res_unit)) {
      if (focal_length_mm > 0 && focal_x_res_unit > 1 &&
          focal_x_res_unit <= 5) {
        double pixels_per_mm = 0;
        switch (focal_x_res_unit) {
          case 2:  // inches
            pixels_per_mm = focal_x_res / 25.4;
            break;
          case 3:  // cm
            pixels_per_mm = focal_x_res / 10.0;
            break;
          case 4:  // mm
            pixels_per_mm = focal_x_res * 1.0;
            break;
          case 5:  // um
            pixels_per_mm = focal_x_res * 1000.0;
            break;
          default:
            LOG(FATAL) << "Unexpected FocalPlaneXResolution value";
        }
        return focal_length_mm * pixels_per_mm;
      }
    }

    // Lookup sensor width in database.
    const std::optional<std::string> make_str = GetMetaData("Make");
    const std::optional<std::string> model_str = GetMetaData("Model");
    if (make_str.has_value() && model_str.has_value()) {
      CameraDatabase database;
      double sensor_width_mm;
      if (database.QuerySensorWidth(*make_str, *model_str, &sensor_width_mm)) {
        return focal_length_mm / sensor_width_mm * max_size;
      }
    }
  }

  return std::nullopt;
}

std::optional<double> Bitmap::ExifLatitude() const {
  const std::optional<std::string> latitude_ref =
      GetMetaData("GPS:LatitudeRef");
  double sign = 1.0;
  if (latitude_ref.has_value()) {
    if (*latitude_ref == "N" || *latitude_ref == "n") {
      sign = 1.0;
    } else if (*latitude_ref == "S" || *latitude_ref == "s") {
      sign = -1.0;
    }
  }
  float deg_min_sec[3] = {0.0};
  if (GetMetaData("GPS:Latitude", "point", &deg_min_sec)) {
    double latitude =
        deg_min_sec[0] + deg_min_sec[1] / 60.0 + deg_min_sec[2] / 3600.0;
    if (latitude > 0 && sign < 0) {
      latitude *= sign;
    }
    return latitude;
  }
  return std::nullopt;
}

std::optional<double> Bitmap::ExifLongitude() const {
  const std::optional<std::string> longitude_ref =
      GetMetaData("GPS:LongitudeRef");
  double sign = 1.0;
  if (longitude_ref.has_value()) {
    if (*longitude_ref == "W" || *longitude_ref == "w") {
      sign = -1.0;
    } else if (*longitude_ref == "E" || *longitude_ref == "e") {
      sign = 1.0;
    }
  }
  float deg_min_sec[3] = {0.0};
  if (GetMetaData("GPS:Longitude", "point", &deg_min_sec)) {
    double longitude =
        deg_min_sec[0] + deg_min_sec[1] / 60.0 + deg_min_sec[2] / 3600.0;
    if (longitude > 0 && sign < 0) {
      longitude *= sign;
    }
    return longitude;
  }
  return std::nullopt;
}

std::optional<double> Bitmap::ExifAltitude() const {
  const std::optional<std::string> altitude_ref =
      GetMetaData("GPS:AltitudeRef");
  double sign = 1.0;
  if (altitude_ref.has_value()) {
    if (*altitude_ref == "0") {
      sign = 1.0;
    } else if (*altitude_ref == "1") {
      sign = -1.0;
    }
  }
  float altitude_float = 0.f;
  if (GetMetaData("GPS:Altitude", "float", &altitude_float)) {
    double altitude = altitude_float;
    if (altitude > 0 && sign < 0) {
      altitude *= sign;
    }
    return altitude;
  }
  return std::nullopt;
}

bool Bitmap::Read(const std::filesystem::path& path,
                  const bool as_rgb,
                  const bool linearize_colorspace) {
  if (auto reader = ScopedBitmapReader::Find(path)) {
    return reader->Read(path, as_rgb, linearize_colorspace, this);
  }
  return ReadFromFile(path, as_rgb, linearize_colorspace);
}

std::optional<bool> Bitmap::ReadIfUnmanaged(const std::filesystem::path& path,
                                            const bool as_rgb,
                                            const bool linearize_colorspace) {
  if (ScopedBitmapReader::Find(path)) {
    return std::nullopt;
  }
  return ReadFromFile(path, as_rgb, linearize_colorspace);
}

std::optional<bool> Bitmap::ReadBilinearSamplesIfUnmanaged(
    const std::filesystem::path& path,
    const std::vector<std::array<double, 2>>& coordinates,
    std::vector<std::optional<BitmapColor<uint8_t>>>* samples) {
  THROW_CHECK_NOTNULL(samples);
  if (ScopedBitmapReader::Find(path)) {
    return std::nullopt;
  }
  for (const auto& xy : coordinates) {
    if (!(xy[0] >= std::numeric_limits<int>::min() &&
          xy[0] < std::numeric_limits<int>::max() &&
          xy[1] >= std::numeric_limits<int>::min() &&
          xy[1] < std::numeric_limits<int>::max())) {
      return std::nullopt;
    }
  }

  // Provider choice is fixed above. An unsupported format must not perform
  // another managed-provider lookup when taking the ordinary file fallback.
  const auto read_full_image = [&]() -> bool {
    Bitmap bitmap;
    if (!bitmap.ReadFromFile(
            path, /*as_rgb=*/true, /*linearize_colorspace=*/false)) {
      return false;
    }
    std::vector<std::optional<BitmapColor<uint8_t>>> result;
    result.reserve(coordinates.size());
    for (const auto& xy : coordinates) {
      const auto color = bitmap.InterpolateBilinear(xy[0], xy[1]);
      result.emplace_back(
          color ? std::optional<BitmapColor<uint8_t>>(color->Cast<uint8_t>())
                : std::nullopt);
    }
    *samples = std::move(result);
    return true;
  };
  if (!ExistsFile(path)) {
    return false;
  }
  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  config["oiio:UnassociatedAlpha"] = 1;
  auto input = OIIO::ImageInput::open(PathToUtf8(path), &config);
  if (!input) {
    const std::string error = OIIO::geterror();
    VLOG(3) << "Failed to read bitmap: " << error;
    return false;
  }
  const auto& spec = input->spec();
  constexpr int kBatchRows = 65;
  const std::string format = input->format_name();
  if ((format != "jpeg" && format != "png") ||
      spec.format != OIIO::TypeDesc::UINT8 || !spec.channelformats.empty() ||
      spec.nchannels != 3 || spec.width <= 0 || spec.height <= 0 ||
      spec.width > std::numeric_limits<int>::max() / (3 * kBatchRows) ||
      spec.depth != 1 || spec.x != 0 || spec.y != 0 || spec.z != 0 ||
      spec.tile_width != 0 || spec.tile_height != 0 || spec.deep) {
    input->close();
    input.reset();
    return read_full_image();
  }
  struct Request {
    int row;
    size_t index;
  };
  std::vector<Request> requests;
  requests.reserve(coordinates.size());
  std::vector<std::optional<BitmapColor<uint8_t>>> result(coordinates.size());
  for (size_t index = 0; index < coordinates.size(); ++index) {
    const int x = static_cast<int>(std::floor(coordinates[index][0]));
    const int y = static_cast<int>(std::floor(coordinates[index][1]));
    // Check global bounds: spare rows in the final batch are not image data.
    if (x >= 0 && x < spec.width - 1 && y >= 0 && y < spec.height - 1) {
      requests.push_back({y, index});
    }
  }
  std::sort(requests.begin(),
            requests.end(),
            [](const Request& a, const Request& b) { return a.row < b.row; });
  Bitmap rows(spec.width, std::min(kBatchRows, spec.height), /*as_rgb=*/true);
  auto& data = rows.RowMajorData();
  const size_t pitch = static_cast<size_t>(spec.width) * 3;
  size_t next_request = 0;
  int begin = 0;
  bool carry = false;
  while (begin < spec.height) {
    const int end = begin + std::min(kBatchRows, spec.height - begin);
    const int read_begin = begin + (carry ? 1 : 0);
    if (read_begin < end &&
        !input->read_scanlines(0,
                               0,
                               read_begin,
                               end,
                               0,
                               0,
                               3,
                               OIIO::TypeDesc::UINT8,
                               data.data() + (carry ? pitch : 0))) {
      const std::string error = input->geterror();
      input->close();
      VLOG(3) << "Failed to read bitmap: " << error;
      return false;
    }
    while (next_request < requests.size() &&
           requests[next_request].row < end - 1) {
      const size_t index = requests[next_request++].index;
      const auto& xy = coordinates[index];
      // Subtracting the integer batch start preserves the original fractional
      // y for these nonnegative bounded coordinates. Reuse unchanged pixel
      // math.
      const auto color = rows.InterpolateBilinear(xy[0], xy[1] - begin);
      result[index] = color->Cast<uint8_t>();
    }
    // Consume the tail even when no requested samples remain, preserving read
    // errors that the full-image reader can encounter after the last sample.
    if (end == spec.height) {
      break;
    }
    std::copy_n(data.data() + static_cast<size_t>(end - begin - 1) * pitch,
                pitch,
                data.data());
    begin = end - 1;
    carry = true;
  }
  input->close();
  *samples = std::move(result);
  return true;
}

bool Bitmap::ReadFromFile(const std::filesystem::path& path,
                          const bool as_rgb,
                          const bool linearize_colorspace) {
  if (!ExistsFile(path)) {
    VLOG(3) << "Failed to read bitmap, because file does not exist";
    return false;
  }

  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  // Preserve color values when reading formats with unassociated alpha. Alpha
  // is discarded below, so automatic premultiplication would darken pixels.
  config["oiio:UnassociatedAlpha"] = 1;

  const auto input = OIIO::ImageInput::open(PathToUtf8(path), &config);
  if (!input) {
    // Always retrieve the error to clear OIIO's pending error state.
    const std::string error = OIIO::geterror();
    VLOG(3) << "Failed to read bitmap: " << error;
    return false;
  }

  const OIIO::ImageSpec& image_spec = input->spec();
  const int width = image_spec.width;
  const int height = image_spec.height;
  const int file_channels = image_spec.nchannels;
  int channels;
  // Handle images with alpha channel by dropping alpha:
  // - 4 channels (RGBA) -> 3 channels (RGB)
  // - 2 channels (grayscale + alpha) -> 1 channel (grayscale)
  if (file_channels == 4) {
    channels = 3;
  } else if (file_channels == 2) {
    channels = 1;
  } else if (file_channels == 1 || file_channels == 3) {
    channels = file_channels;
  } else {
    VLOG(3) << "Unsupported number of channels: " << file_channels;
    return false;
  }

  // The ordinary RGB8-to-gray route otherwise holds a full RGB image while
  // CloneAsGrey allocates its output. Keep the same decoder and gray
  // expression, but convert bounded RGB rows directly into the final gray pixel
  // storage.
  constexpr int kGrayBatchRows = 64;
  const std::string format = input->format_name();
  if (!as_rgb && !linearize_colorspace && format == "jpeg" &&
      image_spec.format == OIIO::TypeDesc::UINT8 &&
      image_spec.channelformats.empty() && file_channels == 3 && width > 0 &&
      height > 0 &&
      width <= std::numeric_limits<int>::max() / (3 * kGrayBatchRows) &&
      height <= std::numeric_limits<int>::max() / width &&
      image_spec.depth == 1 && image_spec.x == 0 && image_spec.y == 0 &&
      image_spec.z == 0 && image_spec.tile_width == 0 &&
      image_spec.tile_height == 0 && !image_spec.deep) {
    const size_t row_width = static_cast<size_t>(width);
    std::vector<uint8_t> gray(row_width * static_cast<size_t>(height));
    std::vector<uint8_t> rgb_rows(
        row_width * 3 * static_cast<size_t>(std::min(kGrayBatchRows, height)));
    for (int begin = 0; begin < height;) {
      const int end = begin + std::min(kGrayBatchRows, height - begin);
      if (!input->read_scanlines(0,
                                 0,
                                 begin,
                                 end,
                                 0,
                                 0,
                                 3,
                                 OIIO::TypeDesc::UINT8,
                                 rgb_rows.data())) {
        const std::string error = input->geterror();
        input->close();
        VLOG(3) << "Failed to read bitmap: " << error;
        return false;
      }
      ConvertPackedRGB8ToGrey(
          rgb_rows.data(),
          row_width * static_cast<size_t>(end - begin),
          gray.data() + row_width * static_cast<size_t>(begin));
      begin = end;
    }
    input->close();
    // Match ReadFromFile followed by CloneAsGrey: retain the complete decoder
    // ImageSpec (including EXIF), changing only its channel count. Preserve the
    // bitmap's existing linear_colorspace_ field, also for reused objects.
    auto meta_data = std::make_unique<OIIOMetaData>();
    meta_data->image_spec = image_spec;
    meta_data->image_spec.nchannels = 1;
    meta_data_ = std::move(meta_data);
    data_ = std::move(gray);
    width_ = width;
    height_ = height;
    channels_ = 1;
    return true;
  }

  width_ = width;
  height_ = height;
  channels_ = channels;
  data_.resize(width_ * height_ * channels_);
  // Note that OIIO supports reading a subset of channels.
  if (!input->read_image(
          0, 0, 0, channels_, OIIO::TypeDesc::UINT8, data_.data())) {
    const std::string error = input->geterror();
    input->close();
    VLOG(3) << "Failed to read bitmap: " << error;
    return false;
  }
  input->close();

  auto meta_data = std::make_unique<OIIOMetaData>();
  meta_data->image_spec = image_spec;
  meta_data->image_spec.nchannels = channels_;
  meta_data_ = std::move(meta_data);

  if (linearize_colorspace) {
    const std::string colorspace = image_spec["oiio:ColorSpace"];
    if (IsEquivalentColorSpace(colorspace, "linear")) {
      data_ = ConvertColorSpace(
          data_.data(), width_, height_, channels_, colorspace, "linear");
    }
  }

  if (as_rgb && channels_ != 3) {
    *this = CloneAsRGB();
  } else if (!as_rgb && channels_ != 1) {
    *this = CloneAsGrey();
  }

  return true;
}

bool Bitmap::Write(const std::filesystem::path& path,
                   const bool delinearize_colorspace) const {
  const std::string utf8_path = PathToUtf8(path);
  const auto output = OIIO::ImageOutput::create(utf8_path);
  if (!output) {
    std::cerr << "Could not create an ImageOutput for " << path
              << ", error = " << OIIO::geterror() << "\n";
    return false;
  }

  // Create a copy of the metadata to avoid modifying the original.
  OIIOMetaData meta_data = *OIIOMetaData::Upcast(meta_data_.get());

  const uint8_t* output_data_ptr = data_.data();
  std::vector<uint8_t> maybe_linearized_output_data;
  if (delinearize_colorspace && linear_colorspace_) {
    std::optional<std::string> colorspace = GetMetaData("oiio:ColorSpace");
    if (!colorspace.has_value()) {
      // Assume sRGB color space if not specified.
      colorspace = "sRGB";
      SetImageSpecColorSpace(meta_data.image_spec,
                             OIIOFromStdStringView(*colorspace));
    }

    maybe_linearized_output_data = ConvertColorSpace(
        data_.data(), width_, height_, channels_, "linear", *colorspace);
    output_data_ptr = maybe_linearized_output_data.data();
  }

  // OIIO writers that encode IPTC metadata (e.g. JPEG, or TIFF with
  // tiff:write_iptc) abort on empty fields in OIIO 2.x. Filter the copied spec
  // for all formats, as empty IPTC strings carry no information anyway, while
  // preserving populated metadata.
  auto& attributes = meta_data.image_spec.extra_attribs;
  attributes.erase(
      std::remove_if(attributes.begin(), attributes.end(), HasEmptyIptcString),
      attributes.end());

  if (HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg")) {
    if (!GetMetaData("Compression").has_value()) {
      // Save JPEG in superb quality by default to reduce compression artifacts.
      meta_data.image_spec["Compression"] = "jpeg:100";
    }
  }

  if (!output->open(utf8_path, meta_data.image_spec)) {
    VLOG(3) << "Could not open " << path << ", error = " << output->geterror()
            << "\n";
    return false;
  }

  if (!output->write_image(OIIO::TypeDesc::UINT8, output_data_ptr)) {
    VLOG(3) << "Could not write pixels to " << path
            << ", error = " << output->geterror() << "\n";
    return false;
  }

  if (!output->close()) {
    VLOG(3) << "Error closing " << path << ", error = " << output->geterror()
            << "\n";
    return false;
  }

  return true;
}

void Bitmap::SetImageData(int width, int height, std::vector<uint8_t>&& data) {
  width_ = width;
  height_ = height;
  data_ = std::move(data);
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  meta_data->image_spec.width = width;
  meta_data->image_spec.height = height;
}

void Bitmap::Rescale(const int new_width,
                     const int new_height,
                     RescaleFilter filter) {
  // A valid image already on the requested grid needs no OIIO filtering or
  // replacement allocation. Preserve SetImageData's metadata-axis repair;
  // leave malformed and empty inputs and unknown filters on the old path.
  if (new_width == width_ && new_height == height_ && width_ > 0 &&
      height_ > 0 && channels_ > 0 && meta_data_ != nullptr &&
      (filter == RescaleFilter::kBilinear || filter == RescaleFilter::kBox) &&
      data_.size() == static_cast<size_t>(width_) * height_ * channels_) {
    auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
    meta_data->image_spec.width = width_;
    meta_data->image_spec.height = height_;
    return;
  }

#if defined(COLMAP_BITMAP_GRAY_TRIANGLE_ENABLED)
  if (filter == RescaleFilter::kBilinear && channels_ == 1 && width_ > 0 &&
      height_ > 0 && new_width > 0 && new_height > 0 && new_width <= width_ &&
      new_height <= height_ && meta_data_ != nullptr &&
      static_cast<uint64_t>(width_) * height_ <=
          gray_triangle_resize::kMaxPixels &&
      data_.size() == static_cast<size_t>(width_) * height_) {
    const size_t bytes = static_cast<size_t>(new_width) * new_height;
    std::vector<uint8_t> gray_data;
    try {
      gray_data.resize(bytes);
    } catch (const std::bad_alloc&) {
      // Retain the unmodified source for the ordinary OIIO path.
    }
    if (gray_data.size() == bytes &&
        gray_triangle_resize::ResizeGrayTriangle(data_.data(),
                                                 data_.size(),
                                                 width_,
                                                 height_,
                                                 gray_data.data(),
                                                 gray_data.size(),
                                                 new_width,
                                                 new_height)) {
      SetImageData(new_width, new_height, std::move(gray_data));
      return;
    }
  }
#endif

  const OIIO::ImageBuf buf(
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8),
      data_.data());
  std::vector<uint8_t> new_data(new_width * new_height * channels_);
  OIIO::ImageBuf new_buf(
      OIIO::ImageSpec(new_width, new_height, channels_, OIIO::TypeDesc::UINT8),
      new_data.data());
  switch (filter) {
    case RescaleFilter::kBilinear:
      // Unlike resample(), resize() applies antialiasing when downsampling.
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
      THROW_CHECK(OIIO::ImageBufAlgo::resize(
          new_buf, buf, {{"filtername", "triangle"}}));
#else
      THROW_CHECK(OIIO::ImageBufAlgo::resize(
          new_buf, buf, /*filtername=*/"triangle", /*filterwidth=*/0.0f));
#endif
      break;
    case RescaleFilter::kBox:
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
      THROW_CHECK(
          OIIO::ImageBufAlgo::resize(new_buf, buf, {{"filtername", "box"}}));
#else
      THROW_CHECK(OIIO::ImageBufAlgo::resize(
          new_buf, buf, /*filtername=*/"box", /*filterwidth=*/0.0f));
#endif
      break;
  }

  SetImageData(new_width, new_height, std::move(new_data));
}

double Bitmap::Thumbnail(const int max_image_size, RescaleFilter filter) {
  THROW_CHECK_GT(max_image_size, 0);
  if (width_ <= max_image_size && height_ <= max_image_size) {
    return 1.0;
  }
  // Fit the down-sampled version exactly into the max dimensions.
  const double scale =
      static_cast<double>(max_image_size) / std::max(width_, height_);
  Rescale(static_cast<int>(std::round(width_ * scale)),
          static_cast<int>(std::round(height_ * scale)),
          filter);
  return scale;
}

void Bitmap::Rot90(int k) {
  if (IsEmpty()) {
    return;
  }
  k = k % 4;
  if (k < 0) {
    k += 4;
  }
  if (k == 0) {
    return;
  }

  const OIIO::ImageBuf buf(
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8),
      data_.data());

  const bool swap_dims = (k == 1 || k == 3);
  const int new_width = swap_dims ? height_ : width_;
  const int new_height = swap_dims ? width_ : height_;

  std::vector<uint8_t> new_data(new_width * new_height * channels_);
  OIIO::ImageBuf new_buf(
      OIIO::ImageSpec(new_width, new_height, channels_, OIIO::TypeDesc::UINT8),
      new_data.data());

  // OIIO rotates clockwise, we rotte counter-clockwise.
  if (k == 1) {  // 90 CCW = 270 CW
    THROW_CHECK(OIIO::ImageBufAlgo::rotate270(new_buf, buf));
  } else if (k == 2) {  // 180 CCW = 180 CW
    THROW_CHECK(OIIO::ImageBufAlgo::rotate180(new_buf, buf));
  } else if (k == 3) {  // 270 CCW = 90 CW
    THROW_CHECK(OIIO::ImageBufAlgo::rotate90(new_buf, buf));
  }

  SetImageData(new_width, new_height, std::move(new_data));
}

void Bitmap::Crop(const int left,
                  const int top,
                  const int width,
                  const int height) {
  THROW_CHECK_GE(left, 0);
  THROW_CHECK_GE(top, 0);
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_LE(left + width, width_);
  THROW_CHECK_LE(top + height, height_);
  if (left == 0 && top == 0 && width == width_ && height == height_) {
    return;
  }

  std::vector<uint8_t> cropped(width * height * channels_);
  const int src_pitch = Pitch();
  const int dst_pitch = width * channels_;
  for (int y = 0; y < height; ++y) {
    std::memcpy(cropped.data() + y * dst_pitch,
                data_.data() + (top + y) * src_pitch + left * channels_,
                dst_pitch);
  }

  SetImageData(width, height, std::move(cropped));
}

Bitmap Bitmap::Clone() const { return *this; }

void ConvertPackedRGB8ToGrey(const uint8_t* rgb,
                             size_t num_pixels,
                             uint8_t* gray) {
  for (size_t i = 0; i < num_pixels; ++i) {
    // Same expression/order as the original CloneAsGrey loop. Non-negative
    // weighted sum plus 0.5 permits truncation and compiler vectorization.
    // NOLINTNEXTLINE(bugprone-incorrect-roundings)
    gray[i] =
        static_cast<uint8_t>(.2126f * rgb[3 * i + 0] + .7152f * rgb[3 * i + 1] +
                             .0722f * rgb[3 * i + 2] + .5f);
  }
}

Bitmap Bitmap::CloneAsGrey() const {
  if (IsGrey()) {
    return Clone();
  } else {
    Bitmap cloned;
    cloned.width_ = width_;
    cloned.height_ = height_;
    cloned.channels_ = 1;
    cloned.linear_colorspace_ = linear_colorspace_;
    cloned.data_.resize(width_ * height_);
    ConvertPackedRGB8ToGrey(
        data_.data(), cloned.data_.size(), cloned.data_.data());
    cloned.meta_data_ = OIIOMetaData::Clone(meta_data_);
    auto* cloned_meta_data = OIIOMetaData::Upcast(cloned.meta_data_.get());
    cloned_meta_data->image_spec.nchannels = 1;
    return cloned;
  }
}

Bitmap Bitmap::CloneAsRGB() const {
  if (IsRGB()) {
    return Clone();
  } else {
    THROW_CHECK_EQ(channels_, 1);
    Bitmap cloned;
    cloned.width_ = width_;
    cloned.height_ = height_;
    cloned.channels_ = 3;
    cloned.linear_colorspace_ = linear_colorspace_;
    cloned.data_.resize(width_ * height_ * 3);
    for (size_t i = 0; i < data_.size(); ++i) {
      cloned.data_[3 * i + 0] = data_[i];
      cloned.data_[3 * i + 1] = data_[i];
      cloned.data_[3 * i + 2] = data_[i];
    }
    cloned.meta_data_ = OIIOMetaData::Clone(meta_data_);
    auto* cloned_meta_data = OIIOMetaData::Upcast(cloned.meta_data_.get());
    cloned_meta_data->image_spec.nchannels = 3;
    return cloned;
  }
}

void Bitmap::SetJpegQuality(int quality) {
  THROW_CHECK_GT(quality, 0);
  THROW_CHECK_LE(quality, 100);
  SetMetaData("Compression", "jpeg:" + std::to_string(quality));
}

void Bitmap::SetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         const void* value) {
  THROW_CHECK_NE(type, "string");
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(OIIOFromStdStringView(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  meta_data->image_spec.attribute(
      OIIOFromStdStringView(name), type_desc, value);
}

void Bitmap::SetMetaData(const std::string_view& name,
                         const std::string_view& value) {
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  meta_data->image_spec.attribute(OIIOFromStdStringView(name),
                                  OIIOFromStdStringView(value));
}

bool Bitmap::GetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         void* value) const {
  THROW_CHECK_NE(type, "string");
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(OIIOFromStdStringView(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  return meta_data->image_spec.getattribute(
      OIIOFromStdStringView(name), type_desc, value);
}

std::optional<std::string> Bitmap::GetMetaData(
    const std::string_view& name) const {
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::ustring ustring_value;
  if (meta_data->image_spec.getattribute(
          OIIOFromStdStringView(name), OIIO::TypeString, &ustring_value)) {
    return std::string(ustring_value.data(), ustring_value.size());
  }
  return std::nullopt;
}

void Bitmap::CloneMetadata(Bitmap* target) const {
  THROW_CHECK_NOTNULL(target);
  target->meta_data_ = OIIOMetaData::Clone(meta_data_);
  auto* target_meta_data = OIIOMetaData::Upcast(target->meta_data_.get());
  target_meta_data->image_spec.width = target->Width();
  target_meta_data->image_spec.height = target->Height();
}

std::ostream& operator<<(std::ostream& stream, const Bitmap& bitmap) {
  stream << "Bitmap(width=" << bitmap.Width() << ", height=" << bitmap.Height()
         << ", channels=" << bitmap.Channels() << ")";
  return stream;
}

float JetColormap::Red(const float gray) { return Base(gray - 0.25f); }

float JetColormap::Green(const float gray) { return Base(gray); }

float JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

BitmapColor<uint8_t> JetColormap::ToBitmapColor(const float gray) {
  return BitmapColor<float>(
             255 * Red(gray), 255 * Green(gray), 255 * Blue(gray))
      .Cast<uint8_t>();
}

float JetColormap::Base(const float val) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (val <= 0.125f) {
    return 0.0f;
  } else if (val <= 0.375f) {
    return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
  } else if (val <= 0.625f) {
    return 1.0f;
  } else if (val <= 0.87f) {
    return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
  } else {
    return 0.0f;
  }
}

float JetColormap::Interpolate(const float val,
                               const float y0,
                               const float x0,
                               const float y1,
                               const float x1) {
  return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

}  // namespace colmap
