// SPDX-License-Identifier: BSD-3-Clause

// clang-format off
#include "PreProcessor.h"
#include "PlyFile.h"
// clang-format on

#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace PoissonRecon {
namespace {

constexpr struct {
  const char* name;
  const char* contents;
} kRejectedHeaders[] = {
    {"empty", ""},
    {"bad_magic", "not_ply\n"},
    {"format_arity", "ply\nformat ascii\n"},
    {"unknown_encoding", "ply\nformat unknown 1.0\n"},
    {"missing_format", "ply\nend_header\n"},
    {"missing_end_header",
     "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"},
};

void WriteFixture(const std::filesystem::path& path,
                  std::string_view contents) {
  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  stream << contents;
  stream.close();
  ASSERT_FALSE(stream.fail());
}

TEST(PlyRead, RejectedHeadersPreserveOutputs) {
  const auto path = colmap::CreateTestDir() / "header.ply";
  for (const auto& fixture : kRejectedHeaders) {
    SCOPED_TRACE(fixture.name);
    WriteFixture(path, fixture.contents);
    std::vector<std::string> names{"sentinel"};
    int file_type = -1;
    float version = -1.0f;
    std::unique_ptr<PlyFile> ply(
        PlyFile::Read(path.string(), names, file_type, version));
    EXPECT_EQ(ply, nullptr);
    EXPECT_EQ(names, std::vector<std::string>{"sentinel"});
    EXPECT_EQ(file_type, -1);
    EXPECT_EQ(version, -1.0f);

    WriteFixture(path, "ply\nformat ascii 1.0\nend_header\n");
    ply.reset(PlyFile::Read(path.string(), names, file_type, version));
    ASSERT_NE(ply, nullptr);
    EXPECT_TRUE(names.empty());
    EXPECT_EQ(file_type, PLY_ASCII);
    EXPECT_EQ(version, 1.0f);
  }
}

TEST(PlyRead, ReadsSupportedEncodings) {
  const auto path = colmap::CreateTestDir() / "point.ply";
  const struct {
    const char* format;
    int file_type;
    std::string payload;
  } fixtures[] = {
      {"ascii", PLY_ASCII, "1.5\n"},
      {"binary_little_endian", PLY_BINARY_LE, std::string("\0\0\xc0\x3f", 4)},
      {"binary_big_endian", PLY_BINARY_BE, std::string("\x3f\xc0\0\0", 4)},
  };
  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.format);
    WriteFixture(path,
                 "ply\nformat " + std::string(fixture.format) +
                     " 1.0\nelement vertex 1\nproperty float x\nend_header\n" +
                     fixture.payload);
    std::vector<std::string> names;
    int file_type = -1;
    float version = -1.0f;
    std::unique_ptr<PlyFile> ply(
        PlyFile::Read(path.string(), names, file_type, version));
    ASSERT_NE(ply, nullptr);
    EXPECT_EQ(names, std::vector<std::string>{"vertex"});
    EXPECT_EQ(file_type, fixture.file_type);
    EXPECT_EQ(version, 1.0f);
    size_t num_vertices = 0;
    const auto properties =
        ply->get_element_description("vertex", num_vertices);
    EXPECT_EQ(num_vertices, 1);
    ASSERT_EQ(properties.size(), 1);
    EXPECT_EQ(properties[0].external_type, PLY_FLOAT);
    const PlyProperty property("x", PLY_FLOAT, PLY_FLOAT, 0);
    ASSERT_EQ(ply->get_property("vertex", &property), 1);
    float x = 0.0f;
    ply->get_element(&x);
    EXPECT_FLOAT_EQ(x, 1.5f);
  }
}

TEST(PlyRead, PreservesHeaderOrderingAndUnknownKeywords) {
  const auto path = colmap::CreateTestDir() / "ordered_header.ply";
  WriteFixture(path,
               "ply\ncomment before format\nunknown_extension ignored\n"
               "element vertex 0\nproperty float x\n"
               "format ascii 1.0\nend_header\n");
  std::vector<std::string> names;
  int file_type = -1;
  float version = -1.0f;
  std::unique_ptr<PlyFile> ply(
      PlyFile::Read(path.string(), names, file_type, version));
  ASSERT_NE(ply, nullptr);
  EXPECT_EQ(names, std::vector<std::string>{"vertex"});
  EXPECT_EQ(file_type, PLY_ASCII);
  EXPECT_EQ(version, 1.0f);
  size_t num_vertices = 1;
  const auto properties = ply->get_element_description("vertex", num_vertices);
  EXPECT_EQ(num_vertices, 0);
  ASSERT_EQ(properties.size(), 1);
  EXPECT_EQ(properties[0].external_type, PLY_FLOAT);
}

#ifdef __linux__
size_t CountOpenDescriptors(const std::filesystem::path& path) {
  size_t count = 0;
  for (const auto& entry :
       std::filesystem::directory_iterator("/proc/self/fd")) {
    std::error_code error;
    const auto target = std::filesystem::read_symlink(entry.path(), error);
    if (!error && target == path) {
      ++count;
    }
  }
  return count;
}

TEST(PlyRead, RejectedHeadersCloseFile) {
  if (!std::filesystem::is_directory("/proc/self/fd")) {
    GTEST_SKIP() << "Descriptor inspection requires /proc/self/fd";
  }
  const auto path = colmap::CreateTestDir() / "lifetime.ply";
  WriteFixture(path, "ply\nformat ascii 1.0\nend_header\n");
  const auto canonical_path = std::filesystem::canonical(path);
  std::vector<std::string> names;
  int file_type = -1;
  float version = -1.0f;
  std::unique_ptr<PlyFile> ply(
      PlyFile::Read(path.string(), names, file_type, version));
  ASSERT_NE(ply, nullptr);
  EXPECT_EQ(CountOpenDescriptors(canonical_path), 1);
  ply.reset();
  EXPECT_EQ(CountOpenDescriptors(canonical_path), 0);

  for (const auto& fixture : kRejectedHeaders) {
    SCOPED_TRACE(fixture.name);
    WriteFixture(path, fixture.contents);
    ply.reset(PlyFile::Read(path.string(), names, file_type, version));
    ASSERT_EQ(ply, nullptr);
    EXPECT_EQ(CountOpenDescriptors(canonical_path), 0);
  }
}
#endif

}  // namespace
}  // namespace PoissonRecon
