#include <generator/generator.h>

#include <random>
#include <cmath>
#include <utility>

#include <algebra/algebra.h>
#include <algebra/matrix.hpp>

namespace generator
{
    template <typename RNG>
    double generate_from_normal_distribution(double m, double var, RNG& rng)
    {
        auto samples = algebra::randu(12, 1, rng, 0.0, 1.0);
        return m + std::sqrt(var) * (samples.sum() - 6);
    }

    template <typename RNG>
    std::pair<double, double> generate_point_from_ploynomial_basis(const algebra::Matrix2d<double> &weights, double var, RNG& rng)
    {
        // [a, b)
        std::uniform_real_distribution uniform_dist(-1.0, 1.0);
        const double e = generate_from_normal_distribution(0, var, rng);

        double x = uniform_dist(rng);
        std::size_t n = weights.rows();
        algebra::Matrix2d designMatrix(n, 1, 1.0);
        for (std::size_t i = 1; i < n; i++)
        {
            designMatrix(i, 0) = std::pow(x, i);
        }

        auto y = (weights.transpose().mm(designMatrix)).item() + e;
        return std::make_pair(x, y);
    }

#define DEFINE_GENERATORS(_RNG)                                         \
    template double generate_from_normal_distribution(double, double, _RNG&);    \
    template std::pair<double, double> generate_point_from_ploynomial_basis(const algebra::Matrix2d<double> &, double, _RNG&);

    DEFINE_GENERATORS(std::mt19937)
    DEFINE_GENERATORS(std::mt19937_64)
#undef DEFINE_GENERATORS
}