// SPDX-License-Identifier: BSD-3-Clause
// clang-format off
#include "PreProcessor.h"
#include "PlyFile.h"
// clang-format on
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace PoissonRecon {
namespace {
class TemporaryFile {
 public:
  TemporaryFile() {
    static std::atomic<unsigned int> sequence{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
      directory_ = std::filesystem::temp_directory_path() /
                   ("poisson-ply-" + std::to_string(stamp) + "-" +
                    std::to_string(sequence++));
      if (std::filesystem::create_directory(directory_)) return;
    }
    throw std::runtime_error("Could not create a unique test directory");
  }
  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  std::string Name() const { return (directory_ / "scalar.ply").string(); }

 private:
  std::filesystem::path directory_;
};

TEST(PlyFloatingProperties, RoundTripsFloatingScalarAndListDestinations) {
  TemporaryFile temporary;
  struct Element {
    double scalar;
    int count;
    double* list;
  };
  for (int type : {PLY_FLOAT, PLY_FLOAT_32, PLY_DOUBLE, PLY_FLOAT_64}) {
    for (int format : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
      SCOPED_TRACE(std::to_string(type) + "/" + std::to_string(format));
      std::vector<double> values{-2.0, 0.1, 1e20};
      if (type == PLY_DOUBLE || type == PLY_FLOAT_64) {
        // These observed world positions collapse with six ASCII digits.
        values.push_back(-0.8250000342726711);
        values.push_back(-0.8250000178813934);
      }
      Element input{values[0], static_cast<int>(values.size()), values.data()};
      PlyProperty properties[] = {
          PlyProperty("x", type, PLY_DOUBLE, offsetof(Element, scalar)),
          PlyProperty("values",
                      type,
                      PLY_DOUBLE,
                      offsetof(Element, list),
                      PLY_LIST,
                      PLY_UCHAR,
                      PLY_INT,
                      offsetof(Element, count))};
      float version;
      std::vector<std::string> names{"vertex"};
      {
        std::unique_ptr<PlyFile> file(
            PlyFile::Write(temporary.Name(), names, format, version));
        ASSERT_NE(file, nullptr);
        file->describe_element("vertex", 1, 2, properties);
        file->header_complete();
        file->put_element_setup("vertex");
        file->put_element(&input);
      }
      Element output{};
      {
        std::unique_ptr<PlyFile> file(
            PlyFile::Read(temporary.Name(), names, format, version));
        ASSERT_NE(file, nullptr);
        file->get_element_setup("vertex", 2, properties);
        file->get_element(&output);
      }
      std::unique_ptr<double, decltype(&std::free)> allocation(output.list,
                                                               &std::free);
      ASSERT_EQ(output.count, values.size());
      EXPECT_EQ(output.scalar, -2.0);
      for (int i = 0; i < output.count; ++i) {
        const double expected =
            type == PLY_FLOAT || type == PLY_FLOAT_32
                ? static_cast<double>(static_cast<float>(values[i]))
                : values[i];
        EXPECT_EQ(output.list[i], expected);
      }
    }
  }
}

TEST(PlyFloatingProperties, HonorsAsciiSourceWidthAndIgnoresUnusedFloats) {
  TemporaryFile temporary;
  {
    std::ofstream file(temporary.Name());
    file << "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty double y\nproperty float32 z\n"
            "property double unused\nproperty list uchar double unused_list\n"
            "end_header\n0.1 0.2 0.3 -1e20 2 -2 1e20\n";
  }
  struct Position {
    double x, y, z;
  } position{};
  PlyProperty properties[] = {
      PlyProperty("x", PLY_FLOAT, PLY_DOUBLE, offsetof(Position, x)),
      PlyProperty("y", PLY_DOUBLE, PLY_DOUBLE, offsetof(Position, y)),
      PlyProperty("z", PLY_FLOAT_32, PLY_DOUBLE, offsetof(Position, z))};
  int format;
  float version;
  std::vector<std::string> names;
  std::unique_ptr<PlyFile> file(
      PlyFile::Read(temporary.Name(), names, format, version));
  ASSERT_NE(file, nullptr);
  file->get_element_setup("vertex", 3, properties);
  file->get_element(&position);
  EXPECT_EQ(position.x, static_cast<double>(0.1f));
  EXPECT_EQ(position.y, 0.2);
  EXPECT_EQ(position.z, static_cast<double>(0.3f));
}

TEST(PlyFloatingProperties, PreservesExistingIntegerRequestedConversion) {
  TemporaryFile temporary;
  for (const std::string type : {"float", "float32", "double", "float64"}) {
    {
      std::ofstream file(temporary.Name());
      file << "ply\nformat ascii 1.0\nelement vertex 1\nproperty " << type
           << " x\nend_header\n16777217\n";
    }
    // Preserve the prior integer-requested interpretation of this ASCII token.
    int result = 0;
    PlyProperty property("x", PLY_FLOAT, PLY_INT, 0);
    int format;
    float version;
    std::vector<std::string> names;
    std::unique_ptr<PlyFile> file(
        PlyFile::Read(temporary.Name(), names, format, version));
    ASSERT_NE(file, nullptr);
    file->get_element_setup("vertex", 1, &property);
    file->get_element(&result);
    EXPECT_EQ(result, 16777217);
  }
}
}  // namespace
}  // namespace PoissonRecon
