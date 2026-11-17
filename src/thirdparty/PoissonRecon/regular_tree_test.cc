// SPDX-License-Identifier: BSD-3-Clause

#include "thirdparty/PoissonRecon/Geometry.h"
#include "thirdparty/PoissonRecon/RegularTree.h"

#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct ChildInitializationData {
  int num_calls = 0;
  int index = -1;
};

template <unsigned int Dim, bool ThreadSafe>
void CheckChildInitialization(const bool use_allocator) {
  using Node = PoissonRecon::RegularTreeNode<Dim, ChildInitializationData, int>;
  constexpr int kNumChildren = 1 << Dim;
  Node root;
  int root_offset[Dim];
  for (unsigned int axis = 0; axis < Dim; ++axis) {
    root_offset[axis] = axis + 1;
  }
  Node::ResetDepthAndOffset(&root, 2, root_offset);
  PoissonRecon::Allocator<Node> allocator;
  allocator.set(2 * kNumChildren);
  auto* node_allocator = use_allocator ? &allocator : nullptr;

  int num_callbacks = 0;
  bool callback_state_matches = true;
  auto initialize = [&](Node& child) {
    const int child_index = num_callbacks % kNumChildren;
    child.nodeData.index = num_callbacks++;
    ++child.nodeData.num_calls;
    callback_state_matches &=
        child.parent == &root && child.children == nullptr;
    // The ordinary initializer assigns geometry after invoking the callback.
    if constexpr (ThreadSafe) {
      const auto position = child.depthAndOffset();
      callback_state_matches &= position.depth == 3;
      for (unsigned int axis = 0; axis < Dim; ++axis) {
        callback_state_matches &=
            position.offset[axis] ==
            (2 * root_offset[axis] | ((child_index >> axis) & 1));
      }
    }
  };

  ASSERT_TRUE(
      root.template initChildren<ThreadSafe>(node_allocator, initialize));
  std::unique_ptr<Node[]> owned_children(use_allocator ? nullptr
                                                       : root.children);
  ASSERT_NE(root.children, nullptr);
  ::testing::Test::RecordProperty("initial_callbacks", num_callbacks);
  EXPECT_EQ(num_callbacks, kNumChildren);
  EXPECT_TRUE(callback_state_matches);
  for (int child_index = 0; child_index < kNumChildren; ++child_index) {
    const Node& child = root.children[child_index];
    const auto position = child.depthAndOffset();
    EXPECT_EQ(child.parent, &root);
    EXPECT_EQ(child.children, nullptr);
    EXPECT_EQ(child.nodeData.num_calls, 1);
    EXPECT_EQ(child.nodeData.index, child_index);
    EXPECT_EQ(position.depth, 3);
    for (unsigned int axis = 0; axis < Dim; ++axis) {
      EXPECT_EQ(position.offset[axis],
                2 * root_offset[axis] | ((child_index >> axis) & 1));
    }
  }

  if constexpr (ThreadSafe) {
    // A losing publication still initializes its own children exactly once.
    Node* const published = root.children;
    EXPECT_FALSE(root.template initChildren<true>(node_allocator, initialize));
    ::testing::Test::RecordProperty("total_callbacks", num_callbacks);
    EXPECT_EQ(num_callbacks, 2 * kNumChildren);
    EXPECT_TRUE(callback_state_matches);
    EXPECT_EQ(root.children, published);
    for (int child_index = 0; child_index < kNumChildren; ++child_index) {
      EXPECT_EQ(root.children[child_index].nodeData.num_calls, 1);
      EXPECT_EQ(root.children[child_index].nodeData.index, child_index);
    }
  }
  root.cleanChildren(false);
}

class RegularTreeTest : public ::testing::TestWithParam<bool> {};

TEST_P(RegularTreeTest, ThreadSafeInitialization2D) {
  CheckChildInitialization<2, true>(GetParam());
}

TEST_P(RegularTreeTest, ThreadSafeInitialization3D) {
  CheckChildInitialization<3, true>(GetParam());
}

TEST_P(RegularTreeTest, OrdinaryInitialization2D) {
  CheckChildInitialization<2, false>(GetParam());
}

TEST_P(RegularTreeTest, OrdinaryInitialization3D) {
  CheckChildInitialization<3, false>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Allocation, RegularTreeTest, ::testing::Bool());

}  // namespace
}  // namespace colmap
