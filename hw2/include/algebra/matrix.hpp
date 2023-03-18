#pragma once
#include <vector>
#include <valarray>
#include <iterator>
#include <utility>
#include <cstddef>

namespace algebra
{
    template <typename T>
    class Matrix2d
    {
    public:
        using value_type = T;

        Matrix2d();
        // Matrix2d(std::initializer_list<value_type>);
        // creates based on the rows and data size
        explicit Matrix2d(std::size_t rows, const std::valarray<T> &data);
        // creates based on the rows and cols directly
        explicit Matrix2d(std::size_t rows, std::size_t cols, const std::valarray<T> &data);
        // creates based on the rows and data size
        explicit Matrix2d(std::size_t rows, const std::vector<T> &data);
        // creates based on the rows and cols directly
        explicit Matrix2d(std::size_t rows, std::size_t cols, const std::vector<T> &data);
        // creates an empty n x m matrix2d
        explicit Matrix2d(std::size_t rows, std::size_t cols);
        // creates an n x m matrix initialized with value
        explicit Matrix2d(std::size_t rows, std::size_t cols, const value_type &value);

        // get the number of rows in the matrix2d
        inline constexpr std::size_t rows() const
        {
            return this->_rows;
        }

        // get the number of columns in the matrix2d
        inline constexpr std::size_t cols() const
        {
            return this->_cols;
        }

        // get the total size of the matrix2d
        inline constexpr std::size_t numel() const
        {
            return this->_data.size();
        }

        //  retrieve the copied data from row r of the matrix2d
        //  due to slice_array does not support const reference of valarray
        //  therefore, we generate a new valarray with the copied values of original valarray
        std::valarray<value_type> row(std::size_t r) const;
        // retrieve refernce to the data from row r of the matrix2d
        std::slice_array<value_type> row(std::size_t r);
        // retrieve the copied data from col c of the matrix2d
        std::valarray<value_type> col(std::size_t c) const;
        // retrieve refernce to the data from col c of the matrix2d
        std::slice_array<value_type> col(std::size_t c);
        // create a new array from data
        std::valarray<value_type> array() const;

        // TODO: Support the view of matrix

        // genetate a new matrix2d that is the transposition of this one
        Matrix2d<value_type> transpose() const;
        Matrix2d<value_type> inverse() const;
        value_type sum() const;
        value_type mean() const;
        Matrix2d<value_type> pow(double power) const;

        // define iterators
        inline auto begin()
        {
            return std::begin(this->_data);
        }
        inline auto begin() const
        {
            return std::cbegin(this->_data);
        }
        inline auto end()
        {
            return std::end(this->_data);
        }
        inline auto end() const
        {
            return std::cend(this->_data);
        }
        inline auto cbegin() const
        {
            return std::cbegin(this->_data);
        }
        inline auto cend() const
        {
            return std::cend(this->_data);
        }

        // TODO: Support slicing matrix?
        const value_type &operator()(std::size_t r, std::size_t c) const;
        value_type &operator()(std::size_t r, std::size_t c);

        Matrix2d<value_type> &operator+=(const Matrix2d<value_type> &rhs);
        Matrix2d<value_type> &operator+=(const value_type &rhs);

        Matrix2d<value_type> &operator-=(const Matrix2d<value_type> &rhs);
        Matrix2d<value_type> &operator-=(const value_type &rhs);

        Matrix2d<value_type> &operator*=(const Matrix2d<value_type> &rhs);
        Matrix2d<value_type> &operator*=(const value_type &rhs);

        Matrix2d<value_type> &operator/=(const Matrix2d<value_type> &rhs);
        Matrix2d<value_type> &operator/=(const value_type &rhs);

        bool operator==(const Matrix2d<value_type> &rhs) const;
        bool operator!=(const Matrix2d<value_type> &rhs) const;

    private:
        std::size_t _rows, _cols;
        std::valarray<value_type> _data;
    };

    // TODO: figure out why dependent type can work for primitive types
    // References:
    // https://en.cppreference.com/w/cpp/language/sfinae
    // https://www.cppstories.com/2016/02/notes-on-c-sfinae/

    // ##: https://ithelp.ithome.com.tw/articles/10207697
#define DEFINE_BINARY_OPERATOR(_Op)                                                                      \
    template <typename T>                                                                                \
    inline Matrix2d<T> operator _Op(const Matrix2d<T> &lhs, const Matrix2d<T> &rhs)                      \
    {                                                                                                    \
        Matrix2d<T> result(lhs);                                                                         \
        result _Op## = rhs;                                                                              \
        return result;                                                                                   \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    inline Matrix2d<T> operator _Op(const typename Matrix2d<T>::value_type &lhs, const Matrix2d<T> &rhs) \
    {                                                                                                    \
        Matrix2d<T> result(rhs);                                                                         \
        result _Op## = lhs;                                                                              \
        return result;                                                                                   \
    }                                                                                                    \
                                                                                                         \
    template <typename T>                                                                                \
    inline Matrix2d<T> operator _Op(const Matrix2d<T> &lhs, const typename Matrix2d<T>::value_type &rhs) \
    {                                                                                                    \
        Matrix2d<T> result(lhs);                                                                         \
        result _Op## = rhs;                                                                              \
        return result;                                                                                   \
    }

    DEFINE_BINARY_OPERATOR(+)
    DEFINE_BINARY_OPERATOR(-)
    DEFINE_BINARY_OPERATOR(*)
    DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR
} // namespace algebra