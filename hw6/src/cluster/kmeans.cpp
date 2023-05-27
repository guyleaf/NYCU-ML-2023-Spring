#include <mlhw6/cluster/kmeans.h>
#include <random>
#include <numeric>
#include <algorithm>

#include <omp.h>

namespace mlhw6
{
    Eigen::VectorXi kMeansPlusPlusInitialization(const Eigen::Ref<const Eigen::MatrixXd> &x, int numberOfClusters, int seed, bool precomputed)
    {
        auto rng = std::mt19937_64(seed);
        std::vector<int> candidates(numberOfClusters);

        // 1. Choose one center uniformly at random among the data points.
        // closed interval [0, rows - 1]
        auto sampler = std::uniform_int_distribution(0, static_cast<int>(x.rows()) - 1);
        candidates.push_back(sampler(rng));

        auto probabilityDistribution = std::uniform_real_distribution();
        while (numberOfClusters > 0)
        {
            Eigen::Map<Eigen::VectorXi> eigenCandidates = Eigen::VectorXi::Map(candidates.data(), candidates.size());

            // 2. For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
            Eigen::MatrixXd distances = x;
            if (!precomputed)
            {
#pragma omp parallel for collapse(2)
                for (Eigen::Index i = 0; i < x.rows(); i++)
                {
                    for (Eigen::Index j = 0; j < x.rows(); j++)
                    {
                        Eigen::RowVectorXd &&diff = x.row(i) - x.row(j);
                        distances(i, j) = diff.dot(diff);
                    }
                }
            }

            // 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen.
            Eigen::VectorXd weights = distances(Eigen::all, eigenCandidates).rowwise().minCoeff();
            weights /= weights.sum();

            auto probability = probabilityDistribution(rng);
            for (int i = 0; i < weights.rows(); i++)
            {
                auto weight = weights[i];
                if (probability < weight)
                {
                    candidates.push_back(i);
                    break;
                }
                probability -= weight;
            }

            numberOfClusters--;
        }

        return Eigen::VectorXi::Map(candidates.data(), candidates.size());
    }

    Eigen::VectorXi randomInitialization(const Eigen::Ref<const Eigen::MatrixXd> &x, int numberOfClusters, int seed)
    {
        auto rng = std::mt19937_64(seed);
        std::vector<int> sequence(x.rows());
        std::iota(sequence.begin(), sequence.end(), 0);
        std::shuffle(sequence.begin(), sequence.end(), rng);

        Eigen::Map<Eigen::VectorXi> tmp = Eigen::VectorXi::Map(sequence.data(), sequence.size());
        return tmp.topRows(numberOfClusters);
    }

#pragma region BaseKMeans

    BaseKMeans::BaseKMeans(int numberOfClusters, int maximumEpochs, double minimumTolerance, int seed, KMeansInitMethods init) : numberOfClusters(numberOfClusters), maximumEpochs(maximumEpochs), minimumTolerance(minimumTolerance), seed(seed)
    {
        if (!numberOfClusters > 0)
        {
            throw std::runtime_error("The number of clusters should be larger than 0.");
        }
    }

    Eigen::VectorXd BaseKMeans::fitAndPredict(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        this->fit(x);
        return this->predict(x);
    }

    std::vector<Eigen::VectorXd> BaseKMeans::getFittingHistory() const
    {
        return this->fittingHistory;
    }

    Eigen::VectorXi BaseKMeans::initializeCenters(const Eigen::Ref<const Eigen::MatrixXd> &x, KMeansInitMethods init, int seed, bool precomputed) const
    {
        if (precomputed)
        {
            if (x.rows() != x.cols())
            {
                throw std::runtime_error("The precomputed x should be a squared matrix.");
            }
        }

        switch (init)
        {
        case KMeansInitMethods::Random:
            return randomInitialization(x, this->numberOfClusters, seed);
        case KMeansInitMethods::Kmeansplusplus:
            return kMeansPlusPlusInitialization(x, this->numberOfClusters, seed, precomputed);
        default:
            throw std::runtime_error("The initialization method is not supported.");
        };
    }

#pragma endregion
#pragma region KMeans

    KMeans::KMeans(int numberOfClusters, int maximumEpochs, double minimumTolerance, int seed, KMeansInitMethods init) : BaseKMeans(numberOfClusters, maximumEpochs, minimumTolerance, seed, init)
    {
    }

    void KMeans::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        auto centers = this->initializeCenters(x, this->init, this->seed);
    }

    Eigen::VectorXd KMeans::predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
    }

#pragma endregion
#pragma region KernelKMeans

    KernelKMeans::KernelKMeans(int numberOfClusters, int maximumEpochs, double minimumTolerance, int seed, KMeansInitMethods init) : BaseKMeans(numberOfClusters, maximumEpochs, minimumTolerance, seed, init)
    {
    }

    void KernelKMeans::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        // x is similarity matrix (gram matrix)
        auto centers = this->initializeCenters(x, this->init, this->seed, true);
    }

    Eigen::VectorXd KernelKMeans::predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
    }

#pragma endregion
}