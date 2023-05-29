#include <mlhw6/cluster/spectral.h>
#include <iostream>
#include <algorithm>
#include <complex>
#include <utility>
#include <string>
#include <thread>
#include <Eigen/Eigenvalues>
#include <boost/sort/sort.hpp>

#include <omp.h>

namespace mlhw6
{
#pragma region BaseSpectralClustering

    BaseSpectralClustering::BaseSpectralClustering(int numberOfClusters, KMeansInitMethods init, int maximumEpochs, int seed) : kMeans(numberOfClusters, init, maximumEpochs, seed), numberOfClusters(numberOfClusters)
    {
        this->numberOfThreads = std::thread::hardware_concurrency();
        auto tmp = std::getenv("OMP_NUM_THREADS");
        if (tmp != nullptr)
        {
            this->numberOfThreads = std::stoul(tmp);
        }

        std::cout << "SpectralClustering: Using " << this->numberOfThreads << " threads." << std::endl;
    }

    Eigen::VectorXi BaseSpectralClustering::fitAndPredict(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        this->fit(x);
        return this->kMeans.getFittingHistory().back();
    }

    const std::vector<Eigen::VectorXi> &BaseSpectralClustering::getFittingHistory() const
    {
        return this->kMeans.getFittingHistory();
    }

    const Eigen::MatrixXd &BaseSpectralClustering::getEigenMatrix() const
    {
        return this->eigenMatrix;
    }

#pragma endregion
#pragma region SpectralClustering

    void SpectralClustering::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> degreeMatrix = x.rowwise().sum().asDiagonal();

        // L = D - W
        Eigen::MatrixXd laplacianMatrix = -x;
        laplacianMatrix.diagonal() += degreeMatrix.diagonal();

        // solve eigen decomposition
        Eigen::EigenSolver<Eigen::MatrixXd> solver(laplacianMatrix, true);
        const Eigen::VectorXcd &eigenValues = solver.eigenvalues();
        const Eigen::MatrixXcd &eigenVectors = solver.eigenvectors();

        std::cout << eigenValues.topRows(5) << std::endl;
        std::cout << eigenVectors.leftCols(5) << std::endl;

        // sort eigenvalues
        std::vector<std::pair<double, Eigen::Index>> eigenPairs(eigenValues.rows());
#pragma omp parallel for
        for (Eigen::Index i = 0; i < eigenValues.rows(); i++)
        {
            eigenPairs[i] = std::make_pair(eigenValues[i].real(), i);
        }
        boost::sort::parallel_stable_sort(eigenPairs.begin(), eigenPairs.end(), this->numberOfThreads);
        // std::partial_sort(eigenPairs.begin(), eigenPairs.begin() + this->numberOfClusters, eigenPairs.end());

        // pick k eigenvectors
        this->eigenMatrix = Eigen::MatrixXd(eigenVectors.rows(), this->numberOfClusters);
#pragma omp parallel for
        for (int i = 0; i < this->numberOfClusters; i++)
        {
            this->eigenMatrix.col(i) = eigenVectors.col(eigenPairs[i].second).real();
        }

        // perform k-means clustering on eigen space
        this->kMeans.fit(this->eigenMatrix);
    }

#pragma endregion
#pragma region NormalizedSpectralClustering

    void NormalizedSpectralClustering::fit(const Eigen::Ref<const Eigen::MatrixXd> &x)
    {
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> degreeMatrix = x.rowwise().sum().asDiagonal();

        // L = D - W
        Eigen::MatrixXd laplacianMatrix = -x;
        laplacianMatrix.diagonal() += degreeMatrix.diagonal();

        // solve generalized eigen decomposition
        Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver(laplacianMatrix, degreeMatrix, true);
        const Eigen::VectorXcd &eigenValues = solver.eigenvalues();
        const Eigen::MatrixXcd &eigenVectors = solver.eigenvectors();

        std::cout << eigenValues.topRows(5) << std::endl;
        std::cout << eigenVectors.leftCols(5) << std::endl;

        // sort eigenvalues
        std::vector<std::pair<double, Eigen::Index>> eigenPairs(eigenValues.rows());
#pragma omp parallel for
        for (Eigen::Index i = 0; i < eigenValues.rows(); i++)
        {
            eigenPairs[i] = std::make_pair(eigenValues[i].real(), i);
        }
        boost::sort::parallel_stable_sort(eigenPairs.begin(), eigenPairs.end(), this->numberOfThreads);
        // std::partial_sort(eigenPairs.begin(), eigenPairs.begin() + this->numberOfClusters, eigenPairs.end());

        // pick k eigenvectors
        this->eigenMatrix = Eigen::MatrixXd(eigenVectors.rows(), this->numberOfClusters);
#pragma omp parallel for
        for (int i = 0; i < this->numberOfClusters; i++)
        {
            this->eigenMatrix.col(i) = eigenVectors.col(eigenPairs[i].second).real();
        }

        // perform k-means clustering on eigen space
        this->kMeans.fit(this->eigenMatrix);
    }

#pragma endregion
}