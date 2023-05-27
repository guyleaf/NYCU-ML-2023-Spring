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
#pragma omp parallel for
        for (Eigen::Index i = 0; i < x1.rows(); i++)
        {
            result.row(i) = (-gamma * (x2.rowwise() - x1.row(i)).rowwise().squaredNorm().transpose()).array().exp();
        }
        return result;
    }
}
