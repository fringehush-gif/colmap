// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/bitmap.h"

#include <pybind11/numpy.h>

namespace py = pybind11;

colmap::Bitmap BitmapFromArray(py::object array,
                               bool linear_colorspace = false);
