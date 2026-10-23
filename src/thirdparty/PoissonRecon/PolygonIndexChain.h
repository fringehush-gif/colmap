// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace PoissonRecon {

// A component partition owns each polygon index exactly once. Store its next
// polygon index in one shared arena instead of allocating one list node/face.
// Arena and node addresses remain fixed throughout graph merging.
template <typename Index>
class PolygonIndexChain {
 public:
  static constexpr Index Missing = std::numeric_limits<Index>::max();
  static std::vector<Index> MakeArena(std::size_t count) {
    if (count > static_cast<std::size_t>(Missing))
      throw std::length_error("Polygon index chain exceeds index range");
    return std::vector<Index>(count, Missing);
  }
  PolygonIndexChain() = default;
  PolygonIndexChain(const PolygonIndexChain&) = delete;
  PolygonIndexChain& operator=(const PolygonIndexChain&) = delete;

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Index;
    using difference_type = std::ptrdiff_t;
    using pointer = const Index*;
    using reference = Index;
    const_iterator() = default;
    const_iterator(const std::vector<Index>* links, Index index)
        : links_(links), index_(index) {}
    Index operator*() const {
      assert(index_ != Missing);
      return index_;
    }
    const_iterator& operator++() {
      assert(index_ != Missing);
      index_ = (*links_)[static_cast<std::size_t>(index_)];
      return *this;
    }
    const_iterator operator++(int) {
      auto previous = *this;
      ++*this;
      return previous;
    }
    bool operator==(const const_iterator& other) const {
      return links_ == other.links_ && index_ == other.index_;
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    const std::vector<Index>* links_ = nullptr;
    Index index_ = Missing;
  };

  void bind(std::vector<Index>& links) {
    assert(!links_ && empty());
    links_ = &links;
  }
  bool empty() const { return count_ == 0; }
  std::size_t size() const { return count_; }
  const_iterator begin() const { return {links_, head_}; }
  const_iterator end() const { return {links_, Missing}; }
  void clear() {
    head_ = tail_ = Missing;
    count_ = 0;
  }

  // Called once with disjoint component IDs from SetConnectedComponents.
  template <typename Iterator>
  void append(Iterator first, Iterator last) {
    assert(links_);
    for (; first != last; ++first) {
      const Index index = *first;
      assert(index >= 0 && static_cast<std::size_t>(index) < links_->size());
      (*links_)[static_cast<std::size_t>(index)] = Missing;
      if (empty())
        head_ = index;
      else
        (*links_)[static_cast<std::size_t>(tail_)] = index;
      tail_ = index;
      ++count_;
    }
  }
  // Same ordered append as whole-list splice, with no face copies or scans.
  void append(PolygonIndexChain& other) {
    assert(this != &other && links_ == other.links_);
    if (other.empty()) return;
    if (empty())
      head_ = other.head_;
    else
      (*links_)[static_cast<std::size_t>(tail_)] = other.head_;
    tail_ = other.tail_;
    count_ += other.count_;
    other.clear();
  }

 private:
  std::vector<Index>* links_ = nullptr;
  Index head_ = Missing, tail_ = Missing;
  std::size_t count_ = 0;
};

}  // namespace PoissonRecon
