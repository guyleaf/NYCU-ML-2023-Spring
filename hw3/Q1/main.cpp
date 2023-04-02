#include <random>
#include <string>
#include <iostream>
#include <vector>

#include <algebra/matrix.hpp>
#include <generator/generator.h>

int main(int argc, char *argv[])
{
    // mode = 0 mean variance
    // mode = 1 n variance weights
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <mode> #mode0# #mode1#" << std::endl
            << "mode0: <mean> <variance>" << std::endl
            << "mode1: <n> <variance> <weights>" << std::endl;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    switch (int mode = std::stoi(argv[1]))
    {
        case 0:
        {
            double mean = std::stod(argv[2]);
            double variance = std::stod(argv[3]);

            std::cout << "Result: " << generator::generate_from_normal_distribution(mean, variance, gen) << std::endl;
            break;
        }
        case 1:
        {
            int n = std::stoi(argv[2]);
            if (n < 1 || (argc - 4) != n)
            {
                std::cerr << "Usage: " << argv[0] << " <n> <weights>" << std::endl;
                return 1;
            }

            double variance = std::stod(argv[3]);

            algebra::Matrix2d<double> weights(n, 1);
            for (int i = 0; i < n; i++)
            {
                weights(i, 0) = std::stod(argv[4 + i]);
            }

            auto [x, y] = generator::generate_point_from_ploynomial_basis(weights, variance, gen);
            std::cout << "Result: x = " << x << ", y = " << y << std::endl;
            break;
        }
        default:
        {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }
    }

    return 0;
}