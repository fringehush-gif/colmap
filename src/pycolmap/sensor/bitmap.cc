// SPDX-License-Identifier: BSD-3-Clause

#include "pycolmap/sensor/bitmap.h"

#include "colmap/sensor/bitmap_reader.h"
#include "colmap/sensor/bitmap_rgb8.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

#ifdef _MSC_VER  // If compiling with MSVC
#include <stddef.h>
typedef ptrdiff_t ssize_t;
#endif

namespace {

std::atomic<uint64_t> next_bitmap_scope_identity{1};

struct PythonBitmapReaderState {
  explicit PythonBitmapReaderState(py::function loader)
      : loader(std::move(loader)) {}
  py::function loader;
  std::weak_ptr<ScopedBitmapReader> reader;
  const uint64_t identity =
      next_bitmap_scope_identity.fetch_add(1, std::memory_order_relaxed);
  std::atomic<size_t> reads{0};
  std::atomic<size_t> rgb_copy_bytes{0};
  std::atomic<size_t> gray_bytes{0};
};

class PythonBitmapReadScope {
 public:
  enum class PayloadMode { kRgb8, kGray8 };

  PythonBitmapReadScope(const std::vector<std::filesystem::path>& paths,
                        py::function loader,
                        bool normalized_srgb,
                        bool camera_metadata_external,
                        const std::vector<std::pair<int, int>>& image_sizes_wh)
      : PythonBitmapReadScope(paths,
                              std::move(loader),
                              normalized_srgb,
                              camera_metadata_external,
                              image_sizes_wh,
                              PayloadMode::kRgb8) {}

  static std::unique_ptr<PythonBitmapReadScope> GrayOnly(
      const std::vector<std::filesystem::path>& paths,
      py::function loader,
      bool normalized_srgb,
      bool camera_metadata_external,
      const std::vector<std::pair<int, int>>& image_sizes_wh) {
    return std::make_unique<PythonBitmapReadScope>(paths,
                                                   std::move(loader),
                                                   normalized_srgb,
                                                   camera_metadata_external,
                                                   image_sizes_wh,
                                                   PayloadMode::kGray8);
  }

  PythonBitmapReadScope(const std::vector<std::filesystem::path>& paths,
                        py::function loader,
                        bool normalized_srgb,
                        bool camera_metadata_external,
                        const std::vector<std::pair<int, int>>& image_sizes_wh,
                        PayloadMode mode) {
    if (!normalized_srgb || !camera_metadata_external) {
      throw std::invalid_argument(
          mode == PayloadMode::kGray8
              ? "Bitmap gray8 ingress requires normalized sRGB and external "
                "camera metadata"
              : "Bitmap RGB8 ingress requires normalized sRGB and external "
                "camera metadata");
    }
    if (mode == PayloadMode::kGray8 && image_sizes_wh.size() != paths.size()) {
      throw std::invalid_argument(
          "Gray-only Bitmap scope requires image_sizes_wh for its exact "
          "roster");
    }
    // A final in-flight native reader may release this owner on its worker
    // thread. Python callback destruction must still occur with the GIL held.
    state_ = std::shared_ptr<PythonBitmapReaderState>(
        new PythonBitmapReaderState(std::move(loader)),
        [](PythonBitmapReaderState* state) {
          py::gil_scoped_acquire acquire;
          delete state;
        });
    // Reader retains callback state for admitted in-flight reads. The reverse
    // state->reader edge is weak, so this does not create an ownership cycle.
    const auto state = state_;
    reader_ = ScopedBitmapReader::Create(
        paths,
        [state, mode](const std::filesystem::path& path,
                      bool as_rgb,
                      bool linearize,
                      Bitmap* target) {
          const bool gray_only = mode == PayloadMode::kGray8;
          if (gray_only && as_rgb) {
            throw std::invalid_argument(
                "Gray-only Bitmap scope refuses RGB reads");
          }
          if (linearize) {
            throw std::invalid_argument(
                "Bitmap RGB8 ingress does not silently quantize linear-light "
                "output");
          }
          py::gil_scoped_acquire acquire;
          py::object pixels;
          try {
            pixels = state->loader(path);
          } catch (const py::error_already_set& error) {
            throw std::runtime_error(std::string("Bitmap RGB8 loader: ") +
                                     error.what());
          }
          if (!PyObject_CheckBuffer(pixels.ptr())) {
            throw std::invalid_argument(
                "Bitmap loader must return a uint8 HWC buffer");
          }
          const py::buffer buffer = py::reinterpret_borrow<py::buffer>(pixels);
          const auto info = buffer.request();
          if (gray_only) {
            if (info.ndim != 2 || info.itemsize != 1 || info.format != "B" ||
                info.shape[0] <= 0 || info.shape[1] <= 0 ||
                info.shape[1] > std::numeric_limits<int>::max() ||
                info.shape[0] >
                    std::numeric_limits<int>::max() / info.shape[1] ||
                info.strides[1] != 1 || info.strides[0] != info.shape[1]) {
              throw std::invalid_argument(
                  "Gray-only Bitmap loader requires positive, packed 2D uint8");
            }
          } else if (info.ndim != 3 || info.itemsize != 1 ||
                     info.format != "B" || info.shape[0] <= 0 ||
                     info.shape[1] <= 0 || info.shape[2] != 3 ||
                     info.shape[1] > std::numeric_limits<int>::max() / 3 ||
                     info.shape[0] > std::numeric_limits<int>::max() /
                                         (info.shape[1] * 3) ||
                     info.strides[2] != 1 || info.strides[1] != 3 ||
                     info.strides[0] != info.shape[1] * 3) {
            throw std::invalid_argument(
                "Bitmap loader requires positive, packed, unsigned uint8 HWC "
                "RGB");
          }
          const auto reader = state->reader.lock();
          if (!reader)
            throw std::runtime_error("Bitmap RGB8 reader scope is closed");
          const auto expected = reader->ImageSize(path);
          if (expected && (info.shape[1] != expected->first ||
                           info.shape[0] != expected->second)) {
            throw std::invalid_argument(
                "Bitmap payload grid differs from declared resident "
                "dimensions");
          }
          Bitmap owned(static_cast<int>(info.shape[1]),
                       static_cast<int>(info.shape[0]),
                       as_rgb,
                       false);
          const size_t source_bytes = static_cast<size_t>(info.shape[0]) *
                                      info.shape[1] * (gray_only ? 1 : 3);
          const size_t bytes = owned.NumBytes();
          {
            // Buffer export/pixels/state retain the source lease allocation
            // while memcpy and the native grayscale conversion release the GIL.
            py::gil_scoped_release release;
            if (gray_only || as_rgb) {
              std::memcpy(owned.RowMajorData().data(), info.ptr, bytes);
            } else {
              ConvertPackedRGB8ToGrey(static_cast<const uint8_t*>(info.ptr),
                                      bytes,
                                      owned.RowMajorData().data());
            }
            *target = std::move(owned);
          }
          state->reads.fetch_add(1, std::memory_order_relaxed);
          if (as_rgb)
            state->rgb_copy_bytes.fetch_add(source_bytes,
                                            std::memory_order_relaxed);
          else
            state->gray_bytes.fetch_add(bytes, std::memory_order_relaxed);
          return true;
        },
        image_sizes_wh);
    state_->reader = reader_;
  }

  ~PythonBitmapReadScope() { Close(); }
  void Close() {
    if (reader_) {
      reader_->Close();
      reader_.reset();
    }
  }
  bool Closed() const { return !reader_; }
  uint64_t IdentityToken() const { return state_->identity; }
  std::optional<std::pair<int, int>> ImageSize(
      const std::filesystem::path& path) const {
    if (!reader_) throw std::runtime_error("Bitmap reader scope is closed");
    if (ScopedBitmapReader::Find(path) != reader_) {
      throw std::invalid_argument(
          "Bitmap dimensions requested outside this scope");
    }
    return reader_->ImageSize(path);
  }
  py::dict Stats() const {
    py::dict result;
    result["bitmapReads"] = state_->reads.load(std::memory_order_relaxed);
    result["ownedRgbCopyBytes"] =
        state_->rgb_copy_bytes.load(std::memory_order_relaxed);
    result["grayOutputBytes"] =
        state_->gray_bytes.load(std::memory_order_relaxed);
    result["memfdPublicationBytes"] = 0;
    return result;
  }

 private:
  std::shared_ptr<PythonBitmapReaderState> state_;
  std::shared_ptr<ScopedBitmapReader> reader_;
};

}  // namespace

py::array_t<uint8_t> ArrayFromBitmap(const Bitmap& self) {
  std::vector<ssize_t> shape = {static_cast<ssize_t>(self.Height()),
                                static_cast<ssize_t>(self.Width())};
  const auto channels = static_cast<ssize_t>(self.Channels());
  if (channels != 1) {
    if (channels != 3) {
      throw std::runtime_error(
          "Can only convert grayscale or 3-channel RGB image to "
          "array");
    }
    shape.push_back(channels);
  }
  py::array_t<uint8_t> output(shape);
  py::buffer_info output_into = output.request();
  uint8_t* output_row_ptr = reinterpret_cast<uint8_t*>(output.request().ptr);
  std::memcpy(output_row_ptr, self.RowMajorData().data(), self.NumBytes());
  return output;
}

Bitmap BitmapFromArray(py::array_t<uint8_t, py::array::c_style> array,
                       bool linear_colorspace) {
  int channels = 1;
  if (array.ndim() == 3) {
    channels = array.shape(2);
  } else if (array.ndim() != 2) {
    throw std::runtime_error("Input array must have 2 or 3 dimensions!");
  }

  const int width = array.shape(1);
  const int height = array.shape(0);
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Input array must have positive width and height!");
  }

  if (channels != 1 && channels != 3) {
    throw std::runtime_error("Input array must have 1 or 3 channels!");
  }

  const bool as_rgb = channels != 1;
  Bitmap output(width,
                height,
                /*as_rgb=*/as_rgb,
                /*linear_colorspace=*/linear_colorspace);

  const uint8_t* input_row_ptr = static_cast<uint8_t*>(array.request().ptr);
  std::memcpy(output.RowMajorData().data(), input_row_ptr, output.NumBytes());
  return output;
}

void BindBitmap(pybind11::module& m) {
  py::classh<PythonBitmapReadScope>(m, "BitmapReadScope")
      .def(
          py::init<const std::vector<std::filesystem::path>&,
                   py::function,
                   bool,
                   bool,
                   const std::vector<std::pair<int, int>>&>(),
          "paths"_a,
          "loader"_a,
          "normalized_srgb"_a,
          "camera_metadata_external"_a,
          "image_sizes_wh"_a = std::vector<std::pair<int, int>>{},
          "Scoped exact-path RGB8 ingress for Bitmap::Read. The loader returns "
          "immutable packed HWC uint8 from the shared cache. RGB copies into "
          "owned storage; grayscale converts directly into owned gray. No disk "
          "fallback on a managed-path error. Keep open until readers join; "
          "not a CLI-process callback or external Bitmap-storage borrow.")
      .def_static(
          "gray_only",
          &PythonBitmapReadScope::GrayOnly,
          "paths"_a,
          "loader"_a,
          "normalized_srgb"_a,
          "camera_metadata_external"_a,
          "image_sizes_wh"_a,
          "Explicit gray-only exact-path scope. Loader returns positive packed "
          "2D uint8 that stays immutable through the copy. RGB reads fail "
          "without invoking it; native Bitmap owns a grayscale copy. "
          "Keep open through worker join.")
      .def("close", &PythonBitmapReadScope::Close)
      .def("stats", &PythonBitmapReadScope::Stats)
      .def("image_size",
           &PythonBitmapReadScope::ImageSize,
           "path"_a,
           "Declared width/height without invoking the pixel loader.")
      .def_property_readonly("closed", &PythonBitmapReadScope::Closed)
      .def_property_readonly("identity_token",
                             &PythonBitmapReadScope::IdentityToken)
      .def(
          "__enter__",
          [](PythonBitmapReadScope& self) -> PythonBitmapReadScope& {
            if (self.Closed())
              throw std::runtime_error("Bitmap reader scope is closed");
            return self;
          },
          py::return_value_policy::reference_internal)
      .def("__exit__",
           [](PythonBitmapReadScope& self, py::object, py::object, py::object) {
             self.Close();
             return false;
           });
  using BitmapRescaleFilter = Bitmap::RescaleFilter;
  py::enum_<BitmapRescaleFilter> PyRescaleFilter(m, "BitmapRescaleFilter");
  PyRescaleFilter.value("BILINEAR", BitmapRescaleFilter::kBilinear)
      .value("BOX", BitmapRescaleFilter::kBox);
  AddStringToEnumConstructor(PyRescaleFilter);

  py::classh<Bitmap>(m, "Bitmap")
      .def(py::init<>())
      .def(py::init<>(
               [](int width, int height, bool as_rgb, bool linear_colorspace) {
                 return Bitmap(width,
                               height,
                               /*as_rgb=*/as_rgb,
                               /*linear_colorspace=*/linear_colorspace);
               }),
           "width"_a,
           "height"_a,
           "as_rgb"_a,
           "linear_colorspace"_a = false)
      .def("to_array", &ArrayFromBitmap)
      .def_static("from_array",
                  &BitmapFromArray,
                  "array"_a,
                  "linear_colorspace"_a = false,
                  "Create bitmap as a copy of array. Returns RGB bitmap, "
                  "if array has shape (H, W, 3), or grayscale bitmap, if "
                  "array has shape (H, W[, 1]).")
      .def("write",
           &Bitmap::Write,
           "path"_a,
           "delinearize_colorspace"_a = true,
           "Write bitmap to file at given path. Defaults to converting to "
           "sRGB colorspace for file storage.")
      .def_static(
          "read",
          [](const std::filesystem::path& path,
             bool as_rgb,
             bool linearize_colorspace) -> py::typing::Optional<Bitmap> {
            Bitmap bitmap;
            if (!bitmap.Read(path,
                             /*as_rgb=*/as_rgb,
                             /*linearize_colorspace=*/linearize_colorspace)) {
              return py::none();
            }
            return py::cast(std::move(bitmap));
          },
          "path"_a,
          "as_rgb"_a,
          "linearize_colorspace"_a = false,
          "Read bitmap at given path and convert to grey- or colorscale. "
          "Defaults to keeping the original colorspace (potentially "
          "non-linear) for image processing.")
      .def("rescale",
           &Bitmap::Rescale,
           "new_width"_a,
           "new_height"_a,
           "filter"_a = BitmapRescaleFilter::kBilinear,
           "Rescale image to the new dimensions.")
      .def("thumbnail",
           &Bitmap::Thumbnail,
           "max_image_size"_a,
           "filter"_a = BitmapRescaleFilter::kBilinear,
           "Downscale the image so neither dimension exceeds max_image_size, "
           "preserving the aspect ratio. Images already within the bound are "
           "left unchanged. Returns the applied scale factor (1 if no "
           "rescaling was necessary).")
      .def("rot90",
           &Bitmap::Rot90,
           "Rotate image by k * 90 degrees counter-clockwise.")
      .def_property_readonly("width", &Bitmap::Width, "Width of the image.")
      .def_property_readonly("height", &Bitmap::Height, "Height of the image.")
      .def_property_readonly(
          "channels", &Bitmap::Channels, "Number of channels of the image.")
      .def_property_readonly(
          "is_rgb", &Bitmap::IsRGB, "Whether the image is colorscale.")
      .def_property_readonly(
          "is_grey", &Bitmap::IsGrey, "Whether the image is greyscale.")
      .def_property_readonly(
          "is_empty", &Bitmap::IsEmpty, "Whether the image is empty.")
      .def_property_readonly(
          "bits_per_pixel",
          &Bitmap::BitsPerPixel,
          "Number of bits per pixel (8 for grey, 24 for RGB).")
      .def_property_readonly(
          "pitch", &Bitmap::Pitch, "Scan line size in bytes (stride).")
      .def("clone", &Bitmap::Clone, "Clone the image to a new bitmap.")
      .def("clone_as_grey",
           &Bitmap::CloneAsGrey,
           "Clone the image as grayscale.")
      .def("clone_as_rgb", &Bitmap::CloneAsRGB, "Clone the image as RGB.")
      .def(
          "set_jpeg_quality",
          &Bitmap::SetJpegQuality,
          "quality"_a,
          "Set compression quality when writing to JPEG in the range [1, 100]. "
          "Lower values reduce quality and file size. By default, bitmaps are "
          "written in superb (100) quality, if not otherwise specified.")
      .def("exif_orientation",
           &Bitmap::ExifOrientation,
           "Extract EXIF orientation. Returns None if not available.")
      .def("exif_camera_model",
           &Bitmap::ExifCameraModel,
           "Extract EXIF camera model. Returns None if not available.")
      .def("exif_focal_length",
           &Bitmap::ExifFocalLength,
           "Extract EXIF focal length. Returns None if not available.")
      .def("exif_latitude",
           &Bitmap::ExifLatitude,
           "Extract EXIF latitude. Returns None if not available.")
      .def("exif_longitude",
           &Bitmap::ExifLongitude,
           "Extract EXIF longitude. Returns None if not available.")
      .def("exif_altitude",
           &Bitmap::ExifAltitude,
           "Extract EXIF altitude. Returns None if not available.")
      .def("__repr__", &CreateRepresentation<Bitmap>);
}
