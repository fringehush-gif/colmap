// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, COLMAP contributors.

#include <array>
#include <cstddef>
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
void ExpectCopyFixture(const MatrixType& matrix) {
  ASSERT_EQ(matrix.rows(), kCopyRowSizes.size());
  for (size_t row = 0; row < matrix.rows(); ++row) {
    ASSERT_EQ(matrix.rowSize(row), kCopyRowSizes[row]);
    for (size_t entry = 0; entry < matrix.rowSize(row); ++entry) {
      EXPECT_EQ(matrix[row][entry].N, (row + entry) % matrix.rows());
      EXPECT_EQ(matrix[row][entry].Value, 10 * row + entry + 0.25);
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
  source.resize(4);
  FillCopyFixture(source);
  ExpectCopyFixture(source);
  const Matrix empty;
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

}  // namespace
