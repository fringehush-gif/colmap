// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstring>

#include "MultiThreading.h"
#include "MyExceptions.h"
// PPolynomial depends on declarations from PreProcessor.
// clang-format off
#include "PreProcessor.h"
#include "PPolynomial.h"
// clang-format on
#include "SparseMatrix.h"
#include "Window.h"
#include <gtest/gtest.h>

namespace {

template <size_t MaxRowSize>
void ExpectDivisionInPlace() {
  PoissonRecon::SparseMatrix<float, int, MaxRowSize> matrix(1);
  matrix.setRowSize(0, 1);
  matrix[0][0] = PoissonRecon::MatrixEntry<float, int>(0, 4.0f);
  EXPECT_EQ(&(matrix /= 2.0f), &matrix);
  EXPECT_FLOAT_EQ(matrix[0][0].Value, 2.0f);
}

TEST(PoissonTemplateReturnTest, SparseMatrixDivisionUpdatesBothStorageForms) {
  ExpectDivisionInPlace<0>();
  ExpectDivisionInPlace<27>();
}

TEST(PoissonTemplateReturnTest, PolynomialScalarUpdatesReturnSelf) {
  PoissonRecon::PPolynomial<1> polynomial;
  polynomial.set(1);
  EXPECT_EQ(&(polynomial += 1.0), &polynomial);
  EXPECT_DOUBLE_EQ(polynomial(1.0), 1.0);
  EXPECT_EQ(&(polynomial -= 0.5), &polynomial);
  EXPECT_DOUBLE_EQ(polynomial(1.0), 0.5);
}

TEST(PoissonTemplateReturnTest, ConstDynamicWindowReturnsSlice) {
  Window::DynamicWindow<int, 2, 2> window;
  window[1][0] = 7;
  const auto& const_window = window;
  EXPECT_EQ(const_window[1][0], 7);
}

}  // namespace
