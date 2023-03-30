#pragma once
#include <random>
#include <cmath>
#include <utility>

#include <algebra/matrix.hpp>

namespace generator
{
    template <typename RNG>
    double generate_from_normal_distribution(double m, double var, RNG& rng)
    {
        std::uniform_real_distribution uniform_dist(std::nextafter(0.0, 1.0));

        double value = 0;
        for (int i = 0; i < 12; i++)
        {
            value += uniform_dist(rng);
        }

        return m + std::sqrt(var) * (value - 6);
    }

    template <typename RNG>
    std::pair<double, double> generate_point_from_ploynomial_basis(algebra::Matrix2d<double> weights, double var, RNG& rng)
    {
        // [a, b)
        std::uniform_real_distribution uniform_dist(-1.0 + 1e-15, 1.0);
        const double e = generate_from_normal_distribution(0, var, rng);

        double x = uniform_dist(rng);
        std::size_t n = weights.rows();
        algebra::Matrix2d<double> designMatrix(n, 1, 1.0);
        for (std::size_t i = 1; i < n; i++)
        {
            designMatrix(i, 0) *= x;
            x *= x;
        }

        auto y = (weights.transpose() * designMatrix + e).item();
        return std::make_pair(x, y);
    }
}