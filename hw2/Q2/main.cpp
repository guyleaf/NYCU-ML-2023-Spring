#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>

using ulonglong = unsigned long long;

ulonglong factorial(ulonglong n)
{
    if (n < 2)
    {
        return 1;
    }

    return n * factorial(n - 1);
}

ulonglong combination(ulonglong n, ulonglong k)
{
    ulonglong dominator = factorial(n - k);

    ulonglong result = 1;
    while (n > k)
    {
        result *= n;
        n--;
    }
    result /= dominator;

    return result;
}

double calculateLikelihood(ulonglong n, ulonglong m)
{
    // binomial likelihood
    double p = static_cast<double>(m) / n;
    ulonglong c = combination(n, m);
    return c * std::pow(p, m) * std::pow(1 - p, n - m);
}

ulonglong countSuccesses(std::string line, char value)
{
    return std::count(line.begin(), line.end(), value);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "The arguments should include a, b initial parameters." << std::endl;
        return 1;
    }

    ulonglong a, b;

    try
    {
        a = std::stoull(argv[1]);
        b = std::stoull(argv[2]);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    int count = 1;
    std::string line;
    auto originalPrecision = std::cout.precision();
    while (std::getline(std::cin, line, '\n'))
    {
        std::cout << "case" << count << ":\t" << line << std::endl;

        ulonglong successes = countSuccesses(line, '1');
        ulonglong total = line.length();

        std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout << "Likelihood:\t" << calculateLikelihood(total, successes) << std::endl;
        std::cout.precision(originalPrecision);

        std::cout << "Beta prior:\ta = " << a << " b = " << b << std::endl;

        a += successes;
        b += total - successes;

        std::cout << "Beta posterior:\ta = " << a << " b = " << b
                  << std::endl
                  << std::endl;
        count++;
    }
    return 0;
}