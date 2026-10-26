// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, COLMAP contributors.

#include "MultiThreading.h"
#include "SparseMatrix.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

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

}  // namespace
