#pragma once
#include <cstddef>

#include "matrix.hpp"

namespace algebra
{
    template <typename T>
    Matrix2d<T> eye(std::size_t n);
    template <typename T>
    Matrix2d<T> eye(std::size_t n, const T &value);
    template <typename T>
    Matrix2d<T> zeros(std::size_t n);
    template <typename T>
    Matrix2d<T> zeros(std::size_t n, std::size_t m);
    template <typename T, class RNG>
    Matrix2d<T> randn(std::size_t n, std::size_t m, RNG &gen);
    template <typename T>
    void swapRows(Matrix2d<T> &matrix, std::size_t r1, std::size_t r2);
} // namespace algebra