// SPDX-License-Identifier: BSD-3-Clause

#include "pycolmap/sensor/bitmap.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <filesystem>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

#ifdef _MSC_VER  // If compiling with MSVC
#include <stddef.h>
typedef ptrdiff_t ssize_t;
#endif

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

Bitmap BitmapFromArray(py::object input, bool linear_colorspace) {
  // Retain NumPy's buffer view without requiring uint8 or contiguous storage.
  // Validate that view before normalization can copy its pixels. Other array-
  // like objects retain the existing typed conversion before shape inspection.
  py::array array;
  if (PyObject_CheckBuffer(input.ptr())) {
    array = py::array::ensure(input);
  } else {
    array = py::array_t<uint8_t, py::array::c_style>::ensure(input);
  }
  if (!array) {
    throw py::type_error("Input must be convertible to a uint8 array!");
  }
  py::ssize_t channels = 1;
  if (array.ndim() == 3) {
    channels = array.shape(2);
  } else if (array.ndim() != 2) {
    throw std::runtime_error("Input array must have 2 or 3 dimensions!");
  }

  const py::ssize_t width = array.shape(1);
  const py::ssize_t height = array.shape(0);
  if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() ||
      height > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "Input array dimensions must be positive and fit in int!");
  }

  if (channels != 1 && channels != 3) {
    throw std::runtime_error("Input array must have 1 or 3 channels!");
  }

  const int bitmap_width = static_cast<int>(width);
  const int bitmap_height = static_cast<int>(height);
  if (!Bitmap::IsValidSize(
          bitmap_width, bitmap_height, static_cast<int>(channels))) {
    throw std::runtime_error("Input array exceeds bitmap storage capacity!");
  }

  // Validate the owner's storage domain before input normalization, then
  // preserve the legacy dtype conversion before allocating the output raster.
  auto contiguous = py::array_t<uint8_t, py::array::c_style>::ensure(array);
  if (!contiguous) {
    throw py::type_error("Input must be convertible to a uint8 array!");
  }
  Bitmap output(bitmap_width,
                bitmap_height,
                /*as_rgb=*/channels != 1,
                /*linear_colorspace=*/linear_colorspace);

  const uint8_t* input_row_ptr = contiguous.data();
  std::memcpy(output.RowMajorData().data(), input_row_ptr, output.NumBytes());
  return output;
}

void BindBitmap(pybind11::module& m) {
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
            return py::cast(bitmap);
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
