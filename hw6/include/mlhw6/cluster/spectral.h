#pragma once
#include <vector>
#include <Eigen/Core>

#include <mlhw6/cluster/kmeans.h>

namespace mlhw6
{
    class BaseSpectralClustering
    {
    public:
        BaseSpectralClustering(int numberOfClusters, KMeansInitMethods init = KMeansInitMethods::Kmeansplusplus, int maximumEpochs = 200, int seed = 1234);

        virtual void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) = 0;
        virtual Eigen::VectorXi fitAndPredict(const Eigen::Ref<const Eigen::MatrixXd> &x);

        const std::vector<Eigen::VectorXi> &getFittingHistory() const;
        const Eigen::MatrixXd &getEigenMatrix() const;

    protected:
        KMeans kMeans;
        Eigen::MatrixXd eigenMatrix;

        int numberOfClusters;
        unsigned int numberOfThreads;
    };

    class SpectralClustering : public virtual BaseSpectralClustering
    {
    public:
        using BaseSpectralClustering::BaseSpectralClustering;

        void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) override;
    };

    class NormalizedSpectralClustering : public virtual BaseSpectralClustering
    {
    public:
        using BaseSpectralClustering::BaseSpectralClustering;

        void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) override;
    };
}