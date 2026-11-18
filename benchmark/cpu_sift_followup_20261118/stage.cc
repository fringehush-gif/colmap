// SPDX-License-Identifier: BSD-3-Clause
// CPU qualification driver, outside production changes.
// Derived from the previously qualified exact-output extraction driver.
#include "colmap/controllers/feature_extraction.h"
#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
#include "colmap/sensor/bitmap.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace colmap;

struct Product {
  std::string name;
  int width;
  int height;
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

void Check(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

double CpuSeconds() {
  timespec value{};
  Check(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) == 0, "clock_gettime");
  return value.tv_sec + value.tv_nsec * 1e-9;
}

void U32(std::ostream& out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.put(static_cast<char>(value >> (8 * i)));
}

void U64(std::ostream& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>(value >> (8 * i)));
}

void Float(std::ostream& out, float value) {
  uint32_t bits;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  U32(out, bits);
}

uint64_t WriteWitness(const std::filesystem::path& path,
                      const std::vector<Product>& products) {
  std::ofstream out(path, std::ios::binary);
  Check(static_cast<bool>(out), "open witness");
  out.write("SIFTOUT1", 8);
  U64(out, products.size());
  uint64_t count = 0;
  for (const auto& product : products) {
    const auto& keypoints = product.keypoints;
    const auto& descriptors = product.descriptors;
    Check(!keypoints.empty(), "empty extracted keypoints");
    Check(descriptors.type == FeatureExtractorType::SIFT, "wrong descriptor type");
    Check(descriptors.data.rows() == static_cast<Eigen::Index>(keypoints.size()),
          "descriptor count mismatch");
    Check(descriptors.data.cols() == 128, "descriptor dimension mismatch");
    U64(out, product.name.size());
    out.write(product.name.data(), product.name.size());
    U32(out, product.width);
    U32(out, product.height);
    U64(out, keypoints.size());
    for (const auto& k : keypoints) {
      Float(out, k.x);
      Float(out, k.y);
      Float(out, k.a11);
      Float(out, k.a12);
      Float(out, k.a21);
      Float(out, k.a22);
    }
    U32(out, static_cast<uint32_t>(descriptors.type));
    U64(out, descriptors.data.rows());
    U64(out, descriptors.data.cols());
    out.write(reinterpret_cast<const char*>(descriptors.data.data()),
              descriptors.data.size());
    count += keypoints.size();
  }
  out.flush();
  Check(static_cast<bool>(out), "write witness");
  return count;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(argc == 7,
          "usage: stage extract|controller PHOTO_DIR COUNT ordinary|covariant|dsp10 WITNESS NEW_DB");
    const std::string stage = argv[1];
    const std::filesystem::path photos = argv[2];
    const size_t count = std::stoul(argv[3]);
    const std::string setting = argv[4];
    const std::filesystem::path witness = argv[5];
    const std::filesystem::path database_path = argv[6];
    Check(stage == "extract" || stage == "controller", "unknown stage");
    Check(setting == "ordinary" || setting == "covariant" || setting == "dsp10",
          "unknown setting");
    Check(count > 0 && count <= 8, "photo count outside original eight-photo fixture");
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i) {
      names.push_back("P1180" + std::to_string(141 + i) + ".JPG");
      Check(std::filesystem::is_regular_file(photos / names.back()), "missing photo");
    }
    Check(!std::filesystem::exists(database_path), "database must be fresh");

    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = false;
    options.num_threads = 1;
    options.sift->max_num_features = 8192;
    options.sift->domain_size_pooling = setting == "dsp10";
    options.sift->dsp_num_scales = 10;
    options.sift->force_covariant_extractor = setting == "covariant";
    Check(options.Check(), "invalid extraction options");
    std::vector<Product> products;
    products.reserve(count);
    std::vector<double> extract_seconds;
    extract_seconds.reserve(count);
    rusage usage_start{};
    Check(getrusage(RUSAGE_SELF, &usage_start) == 0, "getrusage start");
    const double cpu_start = CpuSeconds();
    const auto start = Clock::now();
    if (stage == "extract") {
      auto extractor = CreateSiftFeatureExtractor(options);
      Check(extractor != nullptr, "create extractor");
      for (const auto& name : names) {
        Bitmap bitmap;
        Check(bitmap.Read(photos / name, false), "read photo");
        Check(bitmap.Width() <= options.EffMaxImageSize() &&
                  bitmap.Height() <= options.EffMaxImageSize(),
              "fixture unexpectedly needs resizing");
        // Fresh products prevent the independent upstream covariant reuse bug
        // from contaminating this qualification.
        Product product{name, bitmap.Width(), bitmap.Height(), {}, {}};
        const auto extract_start = Clock::now();
        Check(extractor->Extract(bitmap, &product.keypoints, &product.descriptors),
              "extract failed");
        extract_seconds.push_back(
            std::chrono::duration<double>(Clock::now() - extract_start).count());
        products.push_back(std::move(product));
      }
    } else {
      ImageReaderOptions reader;
      reader.image_path = photos;
      reader.image_names = names;
      auto controller = CreateFeatureExtractorController(database_path, reader, options);
      Check(controller != nullptr, "create controller");
      controller->Start();
      controller->Wait();
      Check(controller->IsFinished(), "controller did not finish");
    }
    const auto stop = Clock::now();
    const double cpu_stop = CpuSeconds();
    rusage usage{};
    Check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage");
    if (stage == "controller") {
      auto database = Database::Open(database_path);
      Check(database->NumImages() == count, "controller did not produce every image");
      auto images = database->ReadAllImages();
      std::sort(images.begin(), images.end(), [](const auto& a, const auto& b) {
        return a.Name() < b.Name();
      });
      for (size_t i = 0; i < images.size(); ++i) {
        const auto& image = images[i];
        Check(image.Name() == names[i], "unexpected image name");
        const auto camera = database->ReadCamera(image.CameraId());
        products.push_back(Product{image.Name(),
                                   static_cast<int>(camera.width),
                                   static_cast<int>(camera.height),
                                   database->ReadKeypoints(image.ImageId()),
                                   database->ReadDescriptors(image.ImageId())});
      }
      database->Close();
    }
    const uint64_t features = WriteWitness(witness, products);
    std::cerr << std::setprecision(17)
              << "METRICS {\"stage\":\"" << stage << "\",\"setting\":\"" << setting
              << "\",\"photos\":" << count << ",\"features\":" << features
              << ",\"wall_seconds\":" << std::chrono::duration<double>(stop - start).count()
              << ",\"cpu_seconds\":" << cpu_stop - cpu_start
              << ",\"user_seconds\":"
              << (usage.ru_utime.tv_sec - usage_start.ru_utime.tv_sec) +
                     (usage.ru_utime.tv_usec - usage_start.ru_utime.tv_usec) * 1e-6
              << ",\"system_seconds\":"
              << (usage.ru_stime.tv_sec - usage_start.ru_stime.tv_sec) +
                     (usage.ru_stime.tv_usec - usage_start.ru_stime.tv_usec) * 1e-6
              << ",\"peak_rss_kib\":" << usage.ru_maxrss
              << ",\"per_photo_features\":[";
    for (size_t i = 0; i < products.size(); ++i) {
      if (i) std::cerr << ',';
      std::cerr << products[i].keypoints.size();
    }
    std::cerr << "],\"per_photo_extract_seconds\":[";
    for (size_t i = 0; i < extract_seconds.size(); ++i) {
      if (i) std::cerr << ',';
      std::cerr << extract_seconds[i];
    }
    std::cerr << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
