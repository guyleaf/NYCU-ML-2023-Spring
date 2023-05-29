#include <mlhw6/cluster/kmeans.h>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <omp.h>

namespace mlhw6
{
    Eigen::VectorXi kMeansPlusPlusInitialization(const Eigen::Ref<const Eigen::MatrixXd> &x, int numberOfClusters, int seed, bool precomputed)
    {
        auto rng = std::mt19937_64(seed);
        std::vector<int> candidates;

        // 1. Choose one center uniformly at random among the data points.
        // closed interval [0, rows - 1]
        auto sampler = std::uniform_int_distribution(0, static_cast<int>(x.rows()) - 1);
        candidates.push_back(sampler(rng));
        numberOfClusters--;

        // 2. For each data point x not chosen yet, compute D(x)^2, the distance between x and the nearest center that has already been chosen.
        Eigen::MatrixXd distances(x.rows(), x.rows());
        if (!precomputed)
        {
#pragma omp parallel for
            for (Eigen::Index i = 0; i < x.rows(); i++)
            {
                distances.row(i) = (x.rowwise() - x.row(i)).rowwise().squaredNorm().transpose();
            }
        }
        else
        {
            distances = x;
        }

        auto probabilityDistribution = std::uniform_real_distribution();
        while (numberOfClusters > 0)
        {
            Eigen::Map<Eigen::VectorXi> eigenCandidates = Eigen::VectorXi::Map(candidates.data(), candidates.size());

            // 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
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

    BaseKMeans::BaseKMeans(int numberOfClusters, KMeansInitMethods init, int maximumEpochs, int seed) : numberOfClusters(numberOfClusters), maximumEpochs(maximumEpochs), seed(seed), init(init)
    {
        if (!numberOfClusters > 0)
        {
            throw std::runtime_error("The number of clusters should be larger than 0.");
        }
    }

    Eigen::VectorXi BaseKMeans::fitAndPredict(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        this->fit(x);
        return this->fittingHistory.back();
    }

    const std::vector<Eigen::VectorXi> &BaseKMeans::getFittingHistory() const
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
            std::cout << "Use random initialization." << std::endl;
            return randomInitialization(x, this->numberOfClusters, seed);
        case KMeansInitMethods::Kmeansplusplus:
            std::cout << "Use K-Means++ initialization." << std::endl;
            return kMeansPlusPlusInitialization(x, this->numberOfClusters, seed, precomputed);
        default:
            throw std::runtime_error("The initialization method is not supported.");
        };
    }

#pragma endregion
#pragma region KMeans

    void KMeans::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        this->fittingHistory = std::vector<Eigen::VectorXi>{Eigen::VectorXi::Constant(x.rows(), -1)};

        auto centers = this->initializeCenters(x, this->init, this->seed);
        this->fittingHistory.back()(centers).setLinSpaced(0, this->numberOfClusters - 1);
        this->centers = x(centers, Eigen::all);

        int epoch = 0;
        bool sameLabels = false;
        do
        {
            std::cout << "Epoch: " << epoch << std::endl;

            // E step
            this->fittingHistory.push_back(this->assignLabels(x));

            // M step
#pragma omp parallel for
            for (int k = 0; k < this->numberOfClusters; k++)
            {
                Eigen::VectorXd selector = this->fittingHistory.back().cwiseEqual(k).cast<double>();
                // calculate center
                this->centers.row(k) = (selector.asDiagonal() * x).colwise().sum() / selector.sum();
            }

            sameLabels = (this->fittingHistory[epoch].array() == this->fittingHistory[epoch + 1].array()).all();
            epoch++;
        } while (!sameLabels && epoch < this->maximumEpochs);
    }

    Eigen::VectorXi KMeans::predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
        return this->assignLabels(x);
    }

    Eigen::VectorXi KMeans::assignLabels(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
        std::vector<int> labels(x.rows());
#pragma omp parallel for
        for (Eigen::Index i = 0; i < x.rows(); i++)
        {
            // find nearest neighbor
            (this->centers.rowwise() - x.row(i)).rowwise().squaredNorm().minCoeff(&labels[i]);
        }
        return Eigen::VectorXi::Map(labels.data(), labels.size());
    }

#pragma endregion
#pragma region KernelKMeans

    void KernelKMeans::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        this->fittingHistory = std::vector<Eigen::VectorXi>{Eigen::VectorXi::Constant(x.rows(), -1)};

        // x is similarity matrix (gram matrix)
        auto centers = this->initializeCenters(1 - x.array(), this->init, this->seed, true);
        this->fittingHistory.back()(centers).setLinSpaced(0, this->numberOfClusters - 1);

        int epoch = 0;
        bool sameLabels = false;
        do
        {
            std::cout << "Epoch: " << epoch << std::endl;

            this->fittingHistory.push_back(this->assignLabels(x));

            sameLabels = (this->fittingHistory[epoch].array() == this->fittingHistory[epoch + 1].array()).all();
            epoch++;
        } while (!sameLabels && epoch < this->maximumEpochs);

        std::cout << "Finished at " << epoch << " epoch" << std::endl;
    }

    Eigen::VectorXi KernelKMeans::predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
        return this->assignLabels(x);
    }

    Eigen::VectorXi KernelKMeans::assignLabels(const Eigen::Ref<const Eigen::MatrixXd> &x) const
    {
        Eigen::MatrixXd distance(x.rows(), this->numberOfClusters);
#pragma omp parallel for
        for (int k = 0; k < this->numberOfClusters; k++)
        {
            Eigen::VectorXd selector = this->fittingHistory.back().cwiseEqual(k).cast<double>();
            auto numberOfXInKCluster = selector.sum();
            Eigen::MatrixXd xToKCluster = x * selector.asDiagonal();

            Eigen::VectorXd secondTerm = 2 * xToKCluster.rowwise().sum() / numberOfXInKCluster;
            auto thirdTerm = (selector.asDiagonal() * xToKCluster).sum() / std::pow(numberOfXInKCluster, 2);
            distance(Eigen::all, k) = (x.diagonal() - secondTerm).array() + thirdTerm;
        }

        std::vector<int> labels(x.rows());
#pragma omp parallel for
        for (Eigen::Index i = 0; i < x.rows(); i++)
        {
            // find nearest neighbor
            distance.row(i).minCoeff(&labels[i]);
        }
        return Eigen::VectorXi::Map(labels.data(), labels.size());
    }

#pragma endregion
}