// SPDX-License-Identifier: BSD-3-Clause

#include "thirdparty/PoissonRecon/Geometry.h"
#include "thirdparty/PoissonRecon/RegularTree.h"

#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

template <unsigned int Dim, unsigned int LeftRadius, unsigned int RightRadius>
void CheckNeighborWindows() {
  struct NodeData {};
  using Node = PoissonRecon::RegularTreeNode<Dim, NodeData, int>;
  using LeftPack = ParameterPack::IsotropicUIntPack<Dim, LeftRadius>;
  using RightPack = ParameterPack::IsotropicUIntPack<Dim, RightRadius>;
  using Key = typename Node::template ConstNeighborKey<LeftPack, RightPack>;
  constexpr int kWidth = LeftRadius + RightRadius + 1;
  constexpr int kNumSlots = Window::IsotropicSize<Dim, kWidth>();

  struct Entry {
    Node* node;
    int depth;
    std::array<int, Dim> offset;
  };

  PoissonRecon::Allocator<Node> allocator;
  allocator.set(1024);
  Node root;
  std::vector<Entry> entries{{&root, 0, {}}};
  for (size_t index = 0; index < entries.size(); ++index) {
    const Entry entry = entries[index];
    // Retain complete broods while leaving holes at several depths.
    if (entry.depth == 3 || (entry.depth > 0 && index % 5 == 2)) {
      continue;
    }
    ASSERT_TRUE(entry.node->template initChildren<false>(&allocator));
    for (unsigned int corner = 0; corner < (1u << Dim); ++corner) {
      Entry child{entry.node->children + corner, entry.depth + 1, {}};
      for (unsigned int axis = 0; axis < Dim; ++axis) {
        child.offset[axis] =
            2 * entry.offset[axis] + static_cast<int>((corner >> axis) & 1u);
      }
      entries.push_back(child);
    }
  }

  Key key;
  key.set(3);
  int num_null_slots = 0;
  int num_nonnull_slots = 0;
  const auto check_window = [&](const auto& window, const Entry& query) {
    for (int slot = 0; slot < kNumSlots; ++slot) {
      std::array<int, Dim> offset = query.offset;
      int index = slot;
      for (int axis = Dim - 1; axis >= 0; --axis) {
        offset[axis] += index % kWidth - static_cast<int>(LeftRadius);
        index /= kWidth;
      }
      // The oracle searches independently recorded coordinates, not a parent
      // window or the implementation's brood-selection arithmetic.
      const auto match =
          std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
            return entry.depth == query.depth && entry.offset == offset;
          });
      const Node* expected = match == entries.end() ? nullptr : match->node;
      EXPECT_EQ(window.neighbors.data[slot], expected)
          << "depth=" << query.depth << " slot=" << slot;
      if (expected == nullptr) {
        ++num_null_slots;
      } else {
        ++num_nonnull_slots;
      }
    }
  };

  // Forward, reverse and interleaved orders exercise cache invalidation across
  // branches and depths. There are an odd number of entries in every tree.
  for (int pass = 0; pass < 3; ++pass) {
    for (size_t index = 0; index < entries.size(); ++index) {
      const size_t query_index = pass == 0   ? index
                                 : pass == 1 ? entries.size() - 1 - index
                                             : (2 * index) % entries.size();
      const Entry& query = entries[query_index];
      check_window(key.getNeighbors(query.node), query);
      check_window(key.getNeighbors(query.node), query);

      typename Key::NeighborType external;
      std::fill_n(external.neighbors.data, kNumSlots, &root);
      key.getNeighbors(LeftPack(), RightPack(), query.node, external);
      check_window(external, query);
      key.getNeighbors(LeftPack(), RightPack(), nullptr, external);
      for (const Node* node : external.neighbors.data) {
        EXPECT_EQ(node, nullptr);
      }
    }
  }
  EXPECT_GT(num_null_slots, 0);
  EXPECT_GT(num_nonnull_slots, 0);
  root.cleanChildren(false);
}

TEST(RegularTreeNeighbors, OneDimensional) { CheckNeighborWindows<1, 1, 1>(); }
TEST(RegularTreeNeighbors, TwoDimensional) { CheckNeighborWindows<2, 1, 1>(); }
TEST(RegularTreeNeighbors, ThreeDimensional) {
  CheckNeighborWindows<3, 1, 1>();
}
TEST(RegularTreeNeighbors, WiderRadius) { CheckNeighborWindows<3, 2, 2>(); }
TEST(RegularTreeNeighbors, AsymmetricRadius) {
  CheckNeighborWindows<3, 0, 2>();
}

}  // namespace
}  // namespace colmap
