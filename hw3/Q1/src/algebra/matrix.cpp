#include <algebra/matrix.hpp>
#include <algebra/algebra.hpp>

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace algebra
{
    const std::string NOT_DIVISIBLE_SIZE_MSG = "The size of array is not divisible by rows.";
    const std::string INVALID_SIZE_MSG = "The value rows * cols is not equal to the size of array.";
    const std::string ZERO_SIZE_MSG = "Invalid rows and cols value.";
    const std::string NOT_SAME_SIZE_MSG = "The size of two matrices are not equal.";
    const std::string NOT_MULTIPLIABLE_MSG = "The size of cols of the first matrix is not equal to the size of rows of the second matrix.";
    const std::string NOT_SQUARE_MSG = "The matrix is not a square matrix.";
    const std::string NOT_ONLY_ONE_VALUE_MSG = "The matrix contains more than one value.";

    // define constructors
    template <typename T>
    Matrix2d<T>::Matrix2d() : _rows(0), _cols(0), _data()
    {
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, const std::valarray<T> &data) : _rows(rows), _cols(data.size() / rows), _data(data)
    {
        if ((_rows * _cols) != _data.size())
        {
            throw std::runtime_error(NOT_DIVISIBLE_SIZE_MSG);
        }
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, std::size_t cols, const std::valarray<T> &data) : _rows(rows), _cols(cols), _data(data)
    {
        if ((_rows * _cols) != _data.size())
        {
            throw std::runtime_error(INVALID_SIZE_MSG);
        }
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, const std::vector<T> &data) : _rows(rows), _cols(data.size() / rows), _data(data.data(), data.size())
    {
        if ((_rows * _cols) != _data.size())
        {
            throw std::runtime_error(NOT_DIVISIBLE_SIZE_MSG);
        }
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, std::size_t cols, const std::vector<T> &data) : _rows(rows), _cols(cols), _data(data.data(), data.size())
    {
        if ((_rows * _cols) != _data.size())
        {
            throw std::runtime_error(INVALID_SIZE_MSG);
        }
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, std::size_t cols) : _rows(rows), _cols(cols), _data(std::valarray<T>(rows * cols))
    {
        if (rows * cols == 0)
        {
            throw std::runtime_error(ZERO_SIZE_MSG);
        }
    }

    template <typename T>
    Matrix2d<T>::Matrix2d(std::size_t rows, std::size_t cols, const T &value) : _rows(rows), _cols(cols), _data(std::valarray<T>(value, rows * cols))
    {
        if (rows * cols == 0)
        {
            throw std::runtime_error(ZERO_SIZE_MSG);
        }
    }

    // define public functions
    template <typename T>
    std::valarray<T> Matrix2d<T>::row(std::size_t r) const
    {
        if (r >= this->_rows)
        {
            throw std::out_of_range("");
        }

        return this->_data[std::slice(r * this->_cols, this->_cols, 1)];
    }

    template <typename T>
    std::slice_array<T> Matrix2d<T>::row(std::size_t r)
    {
        if (r >= this->_rows)
        {
            throw std::out_of_range("");
        }

        return this->_data[std::slice(r * this->_cols, this->_cols, 1)];
    }

    template <typename T>
    std::valarray<T> Matrix2d<T>::col(std::size_t c) const
    {
        if (c >= this->_cols)
        {
            throw std::out_of_range("");
        }

        return this->_data[std::slice(c, this->_rows, this->_cols)];
    }

    template <typename T>
    std::slice_array<T> Matrix2d<T>::col(std::size_t c)
    {
        if (c >= this->_cols)
        {
            throw std::out_of_range("");
        }

        return this->_data[std::slice(c, this->_rows, this->_cols)];
    }

    template <typename T>
    std::valarray<T> Matrix2d<T>::array() const
    {
        return this->_data;
    }

    template <typename T>
    T Matrix2d<T>::item() const
    {
        return this->_data[0];
    }

    template <typename T>
    Matrix2d<T> Matrix2d<T>::transpose() const
    {
        Matrix2d<T> result(this->_cols, this->_rows);
        for (std::size_t i = 0; i < result._rows; i++)
        {
            result.row(i) = this->col(i);
        }
        return result;
    }

    template <typename T>
    Matrix2d<T> Matrix2d<T>::inverse() const
    {
        if (this->_rows != this->_cols)
        {
            throw std::runtime_error(NOT_SQUARE_MSG);
        }

        std::size_t n = this->_rows;
        Matrix2d<T> b = eye<T>(n);
        Matrix2d<T> l = zeros<T>(n);
        Matrix2d<T> u = Matrix2d<T>(*this);

        // i = current pivot
        for (std::size_t i = 0; i < n; i++)
        {
            // find pivot
            std::size_t maxI = i;
            T maxPivot = u(maxI, i);
            for (std::size_t j = i + 1; j < n; j++)
            {
                T &&absPivot = std::abs((*this)(j, i));
                if (maxPivot < absPivot)
                {
                    maxPivot = absPivot;
                    maxI = j;
                }
            }

            // interchange rows
            swapRows(l, i, maxI);
            swapRows(u, i, maxI);
            swapRows(b, i, maxI);

            // TODO: Support matrix slicing
            std::valarray<T> upTriangleRow = u._data[std::slice(i * n + i, n - i, 1)];
            for (std::size_t j = i + 1; j < n; j++)
            {
                l(j, i) = u(j, i) * (1 / u(i, i));
                u._data[std::slice(j * n + i, n - i, 1)] -= l(j, i) * upTriangleRow;
            }
        }

        // LUx = b
        // Ly = b
        for (std::size_t i = 0; i < n; i++)
        {
            std::valarray<T> bRow = b.row(i);
            for (std::size_t j = i + 1; j < n; j++)
            {
                b.row(j) -= l(j, i) * bRow;
            }
        }

        // Ux = y
        for (std::size_t i = n; i > 0; i--)
        {
            auto realI = i - 1;
            b.row(realI) *= std::valarray<T>(1 / u(realI, realI), n);
            std::valarray<T> bRow = b.row(realI);
            for (std::size_t j = i - 1; j > 0; j--)
            {
                auto realJ = j - 1;
                b.row(realJ) -= u(realJ, realI) * bRow;
            }
        }
        return b;
    }

    template <typename T>
    T Matrix2d<T>::sum() const
    {
        return this->_data.sum();
    }

    template <typename T>
    T Matrix2d<T>::mean() const
    {
        return this->_data.sum() / this->_data.size();
    }

    template <typename T>
    T Matrix2d<T>::min() const
    {
        return this->_data.min();
    }

    template <typename T>
    T Matrix2d<T>::max() const
    {
        return this->_data.max();
    }

    template <typename T>
    Matrix2d<T> Matrix2d<T>::pow(double power) const
    {
        return Matrix2d<T>(this->_rows, this->_cols, std::pow(this->_data, power));
    }

    // define operators
    template <typename T>
    const T &Matrix2d<T>::operator()(std::size_t r, std::size_t c) const
    {
        if (r >= this->_rows || c >= this->_cols)
        {
            throw std::out_of_range("");
        }

        return this->_data[r * this->_cols + c];
    }

    template <typename T>
    T &Matrix2d<T>::operator()(std::size_t r, std::size_t c)
    {
        if (r >= this->_rows || c >= this->_cols)
        {
            throw std::out_of_range("");
        }

        return this->_data[r * this->_cols + c];
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator+=(const Matrix2d<T> &rhs)
    {
        if (this->_rows != rhs._rows || this->_cols != rhs._cols)
        {
            throw std::runtime_error(NOT_SAME_SIZE_MSG);
        }

        this->_data += rhs._data;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator+=(const T &rhs)
    {
        this->_data += rhs;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator-=(const Matrix2d<T> &rhs)
    {
        if (this->_rows != rhs._rows || this->_cols != rhs._cols)
        {
            throw std::runtime_error(NOT_SAME_SIZE_MSG);
        }

        this->_data -= rhs._data;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator-=(const T &rhs)
    {
        this->_data -= rhs;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator*=(const Matrix2d<T> &rhs)
    {
        if (this->_cols != rhs._rows)
        {
            throw std::runtime_error(NOT_MULTIPLIABLE_MSG);
        }

        std::valarray<T> result(this->_rows * rhs._cols);
        for (std::size_t i = 0; i < rhs._cols; i++)
        {
            for (std::size_t j = 0; j < this->_rows; j++)
            {
                std::valarray<T> &&tmp = std::valarray<T>(this->row(j)) * rhs.col(i);
                result[j * rhs._cols + i] = tmp.sum();
            }
        }
        this->_data = result;
        this->_cols = rhs._cols;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator*=(const T &rhs)
    {
        this->_data *= rhs;
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator/=(const Matrix2d<T> &rhs)
    {
        if (this->_cols != rhs._rows)
        {
            throw std::runtime_error(NOT_MULTIPLIABLE_MSG);
        }

        (*this) *= Matrix2d<T>(rhs._rows, (1 / rhs._data));
        return *this;
    }

    template <typename T>
    Matrix2d<T> &Matrix2d<T>::operator/=(const T &rhs)
    {
        this->_data *= (1 / rhs);
        return *this;
    }

    template <typename T>
    bool Matrix2d<T>::operator==(const Matrix2d<T> &rhs) const
    {
        return (this->_data == rhs._data).min() != 0;
    }

    template <typename T>
    bool Matrix2d<T>::operator!=(const Matrix2d<T> &rhs) const
    {
        return !(*this == rhs);
    }

    template class Matrix2d<double>;
    template class Matrix2d<long double>;

    template <>
    Matrix2d<unsigned char> Matrix2d<unsigned char>::pow(double power) const = delete;
    template class Matrix2d<unsigned char>;
}