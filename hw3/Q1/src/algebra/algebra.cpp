#include <algebra/algebra.hpp>

#include <valarray>
#include <random>

namespace algebra
{
    template <typename T>
    Matrix2d<T> eye(std::size_t n)
    {
        return eye<T>(n, 1);
    }

    template <typename T>
    Matrix2d<T> eye(std::size_t n, T value)
    {
        Matrix2d<T> result(n, n, 0);
        for (std::size_t i = 0; i < n; i++)
        {
            result(i, i) = value;
        }
        return result;
    }

    template <typename T>
    Matrix2d<T> zeros(std::size_t n)
    {
        return Matrix2d<T>(n, n, 0);
    }

    template <typename T>
    Matrix2d<T> zeros(std::size_t n, std::size_t m)
    {
        return Matrix2d<T>(n, m, 0);
    }

    template <typename T, class RNG>
    Matrix2d<T> randn(std::size_t n, std::size_t m, RNG &gen)
    {
        algebra::Matrix2d<T> result(n, m);
        auto dist = std::normal_distribution<double>(0, 1);
        for (auto &item : result)
        {
            item = dist(gen);
        }
        return result;
    }

    template <typename T, class RNG>
    Matrix2d<T> randu(std::size_t n, std::size_t m, RNG &gen, T a, T b)
    {
        algebra::Matrix2d<T> result(n, m);
        auto dist = std::uniform_real_distribution<double>(a, b);
        for (auto &item : result)
        {
            item = dist(gen);
        }
        return result;
    }

    template <typename T>
    void swapRows(Matrix2d<T> &matrix, std::size_t r1, std::size_t r2)
    {
        std::valarray<T> row1 = matrix.row(r1);
        std::valarray<T> row2 = matrix.row(r2);
        matrix.row(r1) = row2;
        matrix.row(r2) = row1;
    }

#define DEFINE_TEMPLATE_FUNCTIONS(_Type, _RNG)                                      \
    template Matrix2d<_Type> eye(std::size_t);                                \
    template Matrix2d<_Type> eye(std::size_t, _Type);                 \
    template Matrix2d<_Type> zeros(std::size_t);                              \
    template Matrix2d<_Type> zeros(std::size_t, std::size_t);                 \
    template Matrix2d<_Type> randn(std::size_t, std::size_t, _RNG &); \
    template Matrix2d<_Type> randu(std::size_t, std::size_t, _RNG &, _Type, _Type); \
    template void swapRows(Matrix2d<_Type> &matrix, std::size_t r1, std::size_t r2);

    DEFINE_TEMPLATE_FUNCTIONS(double, std::mt19937)
    DEFINE_TEMPLATE_FUNCTIONS(long double, std::mt19937)
    DEFINE_TEMPLATE_FUNCTIONS(unsigned char, std::mt19937)
#undef DEFINE_BINARY_OPERATOR

} // namespace algebra
