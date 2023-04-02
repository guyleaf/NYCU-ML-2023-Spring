#pragma once
#include <utility>

#include <algebra/matrix.hpp>

namespace generator
{
    template <typename RNG>
    double generate_from_normal_distribution(double m, double var, RNG& rng);

    template <typename RNG>
    std::pair<double, double> generate_point_from_ploynomial_basis(const algebra::Matrix2d<double> &weights, double var, RNG& rng);
}