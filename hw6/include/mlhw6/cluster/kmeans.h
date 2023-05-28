#pragma once
#include <vector>
#include <Eigen/Dense>

namespace mlhw6
{
    enum KMeansInitMethods
    {
        Random,
        Kmeansplusplus,
    };

    class BaseKMeans
    {
    public:
        BaseKMeans(int numberOfClusters, int maximumEpochs, int seed, KMeansInitMethods init);

        virtual void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) = 0;
        virtual Eigen::VectorXi predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const = 0;
        virtual Eigen::VectorXi fitAndPredict(const Eigen::Ref<const Eigen::MatrixXd> &x);

        const std::vector<Eigen::VectorXi>& getFittingHistory() const;

    protected:
        Eigen::VectorXi initializeCenters(const Eigen::Ref<const Eigen::MatrixXd> &x, KMeansInitMethods init, int seed, bool precomputed = false) const;

        int numberOfClusters;
        int maximumEpochs;
        int seed;
        KMeansInitMethods init;

        std::vector<Eigen::VectorXi> fittingHistory;
    };

    class KMeans : public virtual BaseKMeans
    {
    public:
        KMeans(int numberOfClusters, int maximumEpochs = 200, int seed = 1234, KMeansInitMethods init = KMeansInitMethods::Kmeansplusplus);
        
        void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) override;
        Eigen::VectorXi predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const override;
    private:
        Eigen::VectorXi assignLabels(const Eigen::Ref<const Eigen::MatrixXd> &x) const;
        Eigen::MatrixXd centers;
    };

    class KernelKMeans : public virtual BaseKMeans
    {
    public:
        KernelKMeans(int numberOfClusters, int maximumEpochs = 200, int seed = 1234, KMeansInitMethods init = KMeansInitMethods::Kmeansplusplus);

        void fit(const Eigen::Ref<const Eigen::MatrixXd> &x) override;
        Eigen::VectorXi predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const override;
    private:
        Eigen::VectorXi assignLabels(const Eigen::Ref<const Eigen::MatrixXd> &x) const;
    };
}