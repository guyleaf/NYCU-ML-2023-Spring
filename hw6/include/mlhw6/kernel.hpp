#pragma once
#include <cmath>

#include <Eigen/Core>
#include <omp.h>

namespace mlhw6
{
    template <typename DerivedA, typename DerivedB, typename Out = Eigen::Matrix<double, DerivedA::RowsAtCompileTime, DerivedB::RowsAtCompileTime>>
    Out rbf(const Eigen::MatrixBase<DerivedA> &x1, const Eigen::MatrixBase<DerivedB> &x2, double gamma)
    {
        Out result(x1.rows(), x2.rows());
#pragma omp parallel for collapse(2)
        for (Eigen::Index i = 0; i < x1.rows(); i++)
        {
            for (Eigen::Index j = 0; j < x2.rows(); j++)
            {
                Eigen::RowVectorXd &&diff = x1.row(i) - x2.row(j);
                result(i, j) = std::exp(-gamma * diff.dot(diff));
            }
        }
        return result;
    }
}
