#include <algebra/matrix.hpp>
#include <algebra/algebra.h>

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>

namespace algebra
{
    const std::string NOT_DIVISIBLE_SIZE_MSG = "The size of array is not divisible by rows.";
    const std::string INVALID_SIZE_MSG = "The value rows * cols is not equal to the size of array.";
    const std::string ZERO_SIZE_MSG = "Invalid rows and cols value.";
    const std::string NOT_SAME_SIZE_MSG = "The size of two matrices are not equal.";
    const std::string NOT_MULTIPLIABLE_MSG = "The size of cols of the first matrix is not equal to the size of rows of the second matrix.";
    const std::string NOT_SQUARE_MSG = "The matrix is not a square matrix.";
    const std::string NOT_ONLY_ONE_VALUE_MSG = "The matrix contains more than one value.";
    const std::string NOT_INVERTIBLE_MSG = "The matrix is not invertible.";

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
            throw std::runtime_error(NOT_DIVISIBLE_SIZE_MSG);
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
    Matrix2d<T> Matrix2d<T>::row(std::size_t r) const
    {
        if (r >= this->_rows)
        {
            throw std::out_of_range("");
        }

        return Matrix2d<T>{this->_cols, 1, this->_data[std::slice(r * this->_cols, this->_cols, 1)]};
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
    Matrix2d<T> Matrix2d<T>::col(std::size_t c) const
    {
        if (c >= this->_cols)
        {
            throw std::out_of_range("");
        }

        return Matrix2d<T>{this->_rows, 1, this->_data[std::slice(c, this->_rows, this->_cols)]};
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
            result.row(i) = this->_data[std::slice(i, this->_rows, this->_cols)];
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

        using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        Eigen::Map<MatrixXd> ref(std::vector<T>(std::begin(this->_data), std::end(this->_data)).data(), this->_rows, this->_cols);
        const Eigen::FullPivLU<MatrixXd> &lu = ref.fullPivLu();

        if (!lu.isInvertible())
        {
            throw std::runtime_error(NOT_INVERTIBLE_MSG);
        }

        MatrixXd inversed_m = lu.inverse();
        return Matrix2d<T>(lu.rows(), lu.cols(), std::valarray<T>(inversed_m.data(), lu.rows() * lu.cols()));
    }

    template <typename T>
    Matrix2d<T> Matrix2d<T>::mm(const Matrix2d<T> &rhs) const
    {
        if (this->_cols != rhs._rows)
        {
            throw std::runtime_error(NOT_MULTIPLIABLE_MSG);
        }

        Matrix2d<T> result(this->_rows, rhs._cols);
        for (std::size_t i = 0; i < rhs._cols; i++)
        {
            for (std::size_t j = 0; j < this->_rows; j++)
            {
                result(j, i) = (this->row(j) * rhs.col(i)).sum();
            }
        }
        return result;
    }

    template <typename T>
    T Matrix2d<T>::sum() const
    {
        return this->_data.sum();
    }

    template <typename T>
    T Matrix2d<T>::mean() const
    {
        auto size = static_cast<double>(this->_data.size());
        return this->_data.sum() / size;
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
    Matrix2d<T> Matrix2d<T>::abs() const
    {
        return Matrix2d<T>(this->_rows, this->_cols, std::abs(this->_data));
    }

    template <typename T>
    Matrix2d<T> Matrix2d<T>::exp() const
    {
        return Matrix2d<T>(this->_rows, this->_cols, std::exp(this->_data));
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
        if (this->_rows != rhs._rows || this->_cols != rhs._cols)
        {
            throw std::runtime_error(NOT_SAME_SIZE_MSG);
        }

        this->_data *= rhs._data;
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
        if (this->_rows != rhs._rows || this->_cols != rhs._cols)
        {
            throw std::runtime_error(NOT_SAME_SIZE_MSG);
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
}