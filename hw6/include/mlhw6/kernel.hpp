#pragma once
#include <cmath>

#include <Eigen/Core>
#include <omp.h>

namespace mlhw6
{
    template <typename DerivedA, typename DerivedB>
    double rbf(const Eigen::MatrixBase<DerivedA> &x1, const Eigen::MatrixBase<DerivedB> &x2, double gamma)
    {
        Eigen::VectorX<typename DerivedA::Scalar> diff = x1 - x2;
        return std::exp(-gamma * diff.dot(diff));
    }

    template <typename DerivedA, typename DerivedB, typename Out = Eigen::Matrix<double, DerivedA::RowsAtCompileTime, DerivedB::RowsAtCompileTime>>
    Out rbf(const Eigen::MatrixBase<DerivedA> &x1, const Eigen::MatrixBase<DerivedB> &x2, double gamma)
    {
        Out result(x1.rows(), x2.rows());
#pragma omp parallel for collapse(2)
        for (Eigen::Index i = 0; i < x1.rows(); i++)
        {
            for (Eigen::Index j = 0; j < x2.rows(); j++)
            {
                result(i, j) = rbf(x1.row(i).transpose(), x2.row(j).transpose(), gamma);
            }
        }
        return result;
    }
}
