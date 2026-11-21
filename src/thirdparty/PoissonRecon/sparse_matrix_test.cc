// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, COLMAP contributors.

#include <array>
#include <cstddef>
#include <limits>
#include <utility>

#include "MultiThreading.h"
#include "SparseMatrix.h"
#include <gtest/gtest.h>

namespace {

using Matrix = PoissonRecon::SparseMatrix<float, int, 27>;

void Fill(Matrix& matrix) {
  for (size_t row = 0; row < matrix.rows(); ++row) {
    matrix.setRowSize(row, 1);
    matrix[row][0] =
        PoissonRecon::MatrixEntry<float, int>(0, static_cast<float>(row + 1));
  }
}

void ExpectFilled(const Matrix& matrix, const size_t rows) {
  ASSERT_EQ(matrix.rows(), rows);
  for (size_t row = 0; row < rows; ++row) {
    ASSERT_EQ(matrix.rowSize(row), 1);
    EXPECT_EQ(matrix[row][0].Value, static_cast<float>(row + 1));
  }
}

TEST(SparseMatrixMoveTest, AssignSmallerMatrixAndGrowDestination) {
  Matrix destination(64);
  Matrix source(4);
  Fill(destination);
  Fill(source);
  destination = std::move(source);
  ExpectFilled(destination, 4);
  destination.resize(32);
  Fill(destination);
  ExpectFilled(destination, 32);
}

TEST(SparseMatrixMoveTest, ConstructByMoveAndReuseSource) {
  Matrix source(16);
  Fill(source);
  Matrix destination(std::move(source));
  ExpectFilled(destination, 16);
  // Exercise the moved-from container's reuse contract.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  source.resize(8);
  Fill(source);
  ExpectFilled(source, 8);
}

TEST(SparseMatrixMoveTest, SwapDifferentCapacitiesAndReuseBoth) {
  Matrix first(8);
  Matrix second(4);
  Fill(first);
  Fill(second);
  Matrix::Swap(first, second);
  ExpectFilled(first, 4);
  ExpectFilled(second, 8);
  first.resize(6);
  second.resize(12);
  Fill(first);
  Fill(second);
  ExpectFilled(first, 6);
  ExpectFilled(second, 12);
}

TEST(SparseMatrixMoveTest, AssignLargerMatrixAndReuseSource) {
  Matrix destination(4);
  Matrix source(64);
  Fill(destination);
  Fill(source);
  destination = std::move(source);
  ExpectFilled(destination, 64);
  // Exercise the moved-from container's reuse contract.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  source.resize(32);
  Fill(source);
  ExpectFilled(source, 32);
}

TEST(SparseMatrixMoveTest, MoveIntoEmptyMatrixAndReuseSource) {
  Matrix destination;
  Matrix source(32);
  Fill(source);
  destination = std::move(source);
  ExpectFilled(destination, 32);
  // Exercise the moved-from container's reuse contract.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  source.resize(12);
  Fill(source);
  ExpectFilled(source, 12);
}

TEST(SparseMatrixMoveTest, SelfMovePreservesMatrix) {
  Matrix matrix(8);
  Fill(matrix);
  Matrix& alias = matrix;
  matrix = std::move(alias);
  ExpectFilled(matrix, 8);
  matrix.resize(7);
  Fill(matrix);
  ExpectFilled(matrix, 7);
}

constexpr std::array<size_t, 4> kCopyRowSizes{2, 0, 1, 3};
using ConvertedMatrix = PoissonRecon::SparseMatrix<double, long long, 27>;

template <typename MatrixType>
void FillCopyFixture(MatrixType& matrix) {
  ASSERT_EQ(matrix.rows(), kCopyRowSizes.size());
  for (size_t row = 0; row < matrix.rows(); ++row) {
    matrix.setRowSize(row, kCopyRowSizes[row]);
    for (size_t entry = 0; entry < matrix.rowSize(row); ++entry) {
      matrix[row][entry].N = static_cast<int>((row + entry) % matrix.rows());
      matrix[row][entry].Value = static_cast<float>(10 * row + entry) + 0.25f;
    }
  }
}

template <typename MatrixType>
void ExpectCopyFixture(const MatrixType& matrix, const double scale = 1.0) {
  ASSERT_EQ(matrix.rows(), kCopyRowSizes.size());
  for (size_t row = 0; row < matrix.rows(); ++row) {
    ASSERT_EQ(matrix.rowSize(row), kCopyRowSizes[row]);
    for (size_t entry = 0; entry < matrix.rowSize(row); ++entry) {
      EXPECT_EQ(matrix[row][entry].N, (row + entry) % matrix.rows());
      EXPECT_EQ(matrix[row][entry].Value, scale * (10 * row + entry + 0.25));
    }
  }
}

template <typename MatrixType>
void FillCopyDestination(MatrixType& matrix) {
  for (size_t row = 0; row < matrix.rows(); ++row) {
    matrix.setRowSize(row, 27);
    for (size_t entry = 0; entry < matrix.rowSize(row); ++entry) {
      matrix[row][entry].N = -1;
      matrix[row][entry].Value = -99;
    }
  }
}

TEST(SparseMatrixCopyTest, ConstructorsPreserveRowsAndOwnStorage) {
  Matrix source(4);
  FillCopyFixture(source);
  const Matrix copy(source);
  const ConvertedMatrix converted(source);
  source[0][0].N = 3;
  source[0][0].Value = -7;
  ExpectCopyFixture(copy);
  ExpectCopyFixture(converted);
}

TEST(SparseMatrixCopyTest, AssignmentPreservesRowsAcrossCapacities) {
  Matrix source(4);
  FillCopyFixture(source);
  for (const size_t rows : std::array<size_t, 3>{2, 4, 8}) {
    Matrix copy(rows);
    ConvertedMatrix converted(rows);
    FillCopyDestination(copy);
    FillCopyDestination(converted);
    EXPECT_EQ(&(copy = source), &copy);
    EXPECT_EQ(&(converted = source), &converted);
    ExpectCopyFixture(copy);
    ExpectCopyFixture(converted);
    copy[0][0].Value = -7;
    converted[0][0].Value = -8;
    EXPECT_EQ(source[0][0].Value, 0.25f);
  }
}

TEST(SparseMatrixCopyTest, SelfAssignmentPreservesRows) {
  Matrix matrix(4);
  FillCopyFixture(matrix);
  const Matrix& alias = matrix;
  matrix = alias;
  ExpectCopyFixture(matrix);
  EXPECT_EQ(&(matrix.template operator= <float, int>(alias)), &matrix);
  ExpectCopyFixture(matrix);
}

TEST(SparseMatrixCopyTest, EmptyCopiesAndCopyAfterMove) {
  Matrix source(4);
  FillCopyFixture(source);
  const Matrix moved(std::move(source));
  Matrix copy(moved);
  ConvertedMatrix converted(moved);
  ExpectCopyFixture(copy);
  ExpectCopyFixture(converted);
  // Exercise the moved-from container's reuse contract.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  source.resize(4);
  FillCopyFixture(source);
  ExpectCopyFixture(source);
  const Matrix empty;
  // Exercise the copy constructor for an empty container.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const Matrix empty_copy(empty);
  const ConvertedMatrix empty_converted(empty);
  EXPECT_EQ(empty_copy.rows(), 0);
  EXPECT_EQ(empty_converted.rows(), 0);
  copy = empty;
  converted = empty;
  EXPECT_EQ(copy.rows(), 0);
  EXPECT_EQ(converted.rows(), 0);
}

using UnboundedMatrix = PoissonRecon::SparseMatrix<float, int>;
using ConvertedUnboundedMatrix = PoissonRecon::SparseMatrix<double, long long>;

TEST(SparseMatrixCopyTest, UnboundedSelfCopiesPreserveRows) {
  UnboundedMatrix matrix(4);
  const UnboundedMatrix& alias = matrix;
  FillCopyFixture(matrix);
  matrix = alias;
  ExpectCopyFixture(matrix);
  FillCopyFixture(matrix);
  EXPECT_EQ(&matrix.copy(alias), &matrix);
  ExpectCopyFixture(matrix);
  FillCopyFixture(matrix);
  EXPECT_EQ(&(matrix.template operator= <float, int>(alias)), &matrix);
  ExpectCopyFixture(matrix);
}

TEST(SparseMatrixCopyTest, UnboundedNonAliasingCopiesOwnStorage) {
  UnboundedMatrix source(4);
  FillCopyFixture(source);
  const UnboundedMatrix copy(source);
  const ConvertedUnboundedMatrix converted(source);
  UnboundedMatrix assigned(2);
  ConvertedUnboundedMatrix converted_assigned(8);
  ConvertedUnboundedMatrix explicitly_copied;
  assigned = source;
  converted_assigned = source;
  explicitly_copied.copy(source);
  source[0][0].Value = -7;
  ExpectCopyFixture(copy);
  ExpectCopyFixture(converted);
  ExpectCopyFixture(assigned);
  ExpectCopyFixture(converted_assigned);
  ExpectCopyFixture(explicitly_copied);
}

TEST(SparseMatrixScalarTest, ScalesOnlyActiveBoundedEntries) {
  Matrix matrix(kCopyRowSizes.size());
  // Initialize full rows before hiding entries, so the regression also runs
  // safely against the original implementation that scales reserved storage.
  FillCopyDestination(matrix);
  FillCopyFixture(matrix);
  EXPECT_EQ(&(matrix *= 2.0f), &matrix);
  ExpectCopyFixture(matrix, 2.0);
  for (size_t row = 0; row < matrix.rows(); ++row) {
    matrix.setRowSize(row, 27);
    for (size_t entry = kCopyRowSizes[row]; entry < matrix.rowSize(row);
         ++entry) {
      EXPECT_EQ(matrix[row][entry].N, -1);
      EXPECT_EQ(matrix[row][entry].Value, -99);
    }
  }
}

template <typename MatrixType>
class SparseMatrixDivisionTest : public testing::Test {};
using DivisionMatrices = testing::Types<Matrix, UnboundedMatrix>;
TYPED_TEST_SUITE(SparseMatrixDivisionTest, DivisionMatrices);

TYPED_TEST(SparseMatrixDivisionTest, DivisionPreservesRowsAndReturnsSelf) {
  TypeParam matrix(kCopyRowSizes.size());
  FillCopyFixture(matrix);
  EXPECT_EQ(&(matrix /= 2.0f), &matrix);
  ExpectCopyFixture(matrix, 0.5);
}

void ExpectTransposedCopyFixture(const UnboundedMatrix& matrix,
                                 const size_t rows,
                                 const float scale = 1.0f) {
  ASSERT_EQ(matrix.rows(), rows);
  const std::array<size_t, 4> row_sizes{2, 2, 1, 1};
  const std::array<std::array<int, 2>, 4> indices{
      {{{0, 3}}, {{0, 3}}, {{2, 0}}, {{3, 0}}}};
  const std::array<std::array<float, 2>, 4> values{
      {{{0.25f, 31.25f}}, {{1.25f, 32.25f}}, {{20.25f, 0}}, {{30.25f, 0}}}};
  for (size_t row = 0; row < rows; ++row) {
    ASSERT_EQ(matrix.rowSize(row), row < row_sizes.size() ? row_sizes[row] : 0);
    for (size_t entry = 0; entry < matrix.rowSize(row); ++entry) {
      EXPECT_EQ(matrix[row][entry].N, indices[row][entry]);
      EXPECT_EQ(matrix[row][entry].Value, scale * values[row][entry]);
    }
  }
}

TEST(SparseMatrixTransposeTest, PreservesValuesDimensionsAndCallback) {
  UnboundedMatrix source(4);
  Matrix bounded(4);
  FillCopyFixture(source);
  FillCopyFixture(bounded);
  for (const bool negate : {false, true}) {
    const auto function =
        negate ? +[](const float& value) { return -value; } : nullptr;
    const float scale = negate ? -1.0f : 1.0f;
    ExpectTransposedCopyFixture(source.transpose(function), 4, scale);
    ExpectTransposedCopyFixture(
        UnboundedMatrix::Transpose(source, function), 4, scale);
    for (const size_t rows : {4, 6}) {
      ExpectTransposedCopyFixture(
          source.transpose(rows, function), rows, scale);
      ExpectTransposedCopyFixture(
          UnboundedMatrix::Transpose(source, rows, function), rows, scale);
      ExpectTransposedCopyFixture(
          UnboundedMatrix::Transpose(bounded, rows, function), rows, scale);
    }
  }
  ExpectCopyFixture(source);
  ExpectCopyFixture(bounded);
  PoissonRecon::SparseMatrix<float, bool> boolean_index(1);
  boolean_index.setRowSize(0, 1);
  boolean_index[0][0] = {true, 7.0f};
  const auto boolean_result =
      UnboundedMatrix::Transpose(boolean_index, size_t{2});
  ASSERT_EQ(boolean_result.rows(), 2);
  EXPECT_EQ(boolean_result.rowSize(0), 0);
  ASSERT_EQ(boolean_result.rowSize(1), 1);
  EXPECT_EQ(boolean_result[1][0].N, 0);
  EXPECT_EQ(boolean_result[1][0].Value, 7.0f);
  const UnboundedMatrix empty;
  EXPECT_EQ(empty.transpose().rows(), 0);
  EXPECT_EQ(UnboundedMatrix::Transpose(empty).rows(), 0);
  for (const size_t rows : {0, 3}) {
    const auto member = empty.transpose(rows);
    const auto generic = UnboundedMatrix::Transpose(empty, rows);
    ASSERT_EQ(member.rows(), rows);
    ASSERT_EQ(generic.rows(), rows);
    for (size_t row = 0; row < rows; ++row) {
      EXPECT_EQ(member.rowSize(row), 0);
      EXPECT_EQ(generic.rowSize(row), 0);
    }
  }
}

TEST(SparseMatrixTransposeTest, RejectsInvalidExplicitDimensions) {
  struct Case {
    size_t input_rows;
    size_t output_rows;
    std::array<int, 2> columns;
  };
  const std::array<Case, 6> cases{{{4, 2, {3, 0}},
                                   {2, 4, {8, 3}},
                                   {1, 3, {3, 0}},
                                   {1, 0, {0, 0}},
                                   {1, 4, {std::numeric_limits<int>::max(), 0}},
                                   {1, 4, {-1, 0}}}};
  for (const auto& test : cases) {
    SCOPED_TRACE(testing::Message()
                 << test.input_rows << " rows, " << test.output_rows
                 << " output rows, column " << test.columns[0]);
    UnboundedMatrix source(test.input_rows);
    source.setRowSize(0, 2);
    for (size_t entry = 0; entry < test.columns.size(); ++entry) {
      source[0][entry] = {test.columns[entry], 1.0f};
    }
    EXPECT_THROW(source.transpose(test.output_rows), PoissonRecon::Exception);
    EXPECT_THROW(UnboundedMatrix::Transpose(source, test.output_rows),
                 PoissonRecon::Exception);
  }
  using UnsignedMatrix = PoissonRecon::SparseMatrix<float, unsigned long long>;
  UnsignedMatrix source(1);
  source.setRowSize(0, 1);
  source[0][0] = {std::numeric_limits<unsigned long long>::max(), 1.0f};
  EXPECT_THROW(source.transpose(size_t{4}), PoissonRecon::Exception);
  // The input iterator's index type can differ from the result's index type.
  EXPECT_THROW(UnboundedMatrix::Transpose(source, size_t{4}),
               PoissonRecon::Exception);
}

}  // namespace
