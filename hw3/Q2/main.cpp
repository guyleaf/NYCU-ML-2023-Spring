#include <random>
#include <string>
#include <iostream>
#include <vector>
#include <limits>

#include <algebra/matrix.hpp>
#include <generator/generator.h>

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-4;

class UnivariateGaussianGenerator
{
public:
    UnivariateGaussianGenerator(double mean, double variance) : mean(mean), variance(variance)
    {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    double generate()
    {
        return generator::generate_from_normal_distribution(this->mean, this->variance, this->rng);
    }

    const double mean;
    const double variance;

private:
    std::mt19937 rng;
};

void estimateGenerator(UnivariateGaussianGenerator &generator)
{
    std::cout.precision(std::numeric_limits<double>::digits10);
    int n = 0;
    double mean = 0;
    double variance = 0;

    double dMean;
    double dVariance;
    do
    {
        dMean = mean;
        dVariance = variance;

        auto sample = generator.generate();

        n++;
        mean += (sample - mean) / n;
        variance += ((sample - dMean) * (sample - mean) - variance) / n;

        std::cout << "Add data point: " << sample << std::endl
                  << "Mean = " << mean << "\tVariance = " << variance << std::endl;

        dMean = std::abs(dMean - mean);
        dVariance = std::abs(dVariance - variance);
    } while (dMean > STOP_APPROXIMATION_THRESHOLD || dVariance > STOP_APPROXIMATION_THRESHOLD);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <mean> <variance>" << std::endl;
    }

    double mean = std::stod(argv[1]);
    double variance = std::stod(argv[2]);

    std::cout << "Data point source function: N("
              << mean << ", " << variance << ")" << std::endl << std::endl;

    UnivariateGaussianGenerator generator(mean, variance);
    estimateGenerator(generator);

    return 0;
}