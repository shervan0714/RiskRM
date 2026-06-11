#pragma once

#include <Eigen/Dense>
#include <string>
#include <random>
#include <vector>

/**
 * @class HighPerformanceRSW
 * @brief Enterprise-grade, parallelized implementation of Covariate-Shift Generalization 
 * via Random Sample Weighting using Eigen SIMD operations and OpenMP.
 */
class HighPerformanceRSW {
public:
    std::string task;
    int num_envs;
    int epochs;
    int warmup_epochs;
    int seed;
    double lambda_reg;
    double alpha;
    double lr_theta;
    double lr_mu;

    // Model parameters
    Eigen::VectorXd theta; // Core prediction weights
    Eigen::VectorXd mu;    // Continuous pre-activation mask parameters
    Eigen::VectorXd mask;  // Derived soft mask: sigmoid(mu)

    /**
     * @brief Constructor initializing hyperparameter parameters for the training engine.
     */
    HighPerformanceRSW(std::string task = "regression", int num_envs = 10, 
                       double lambda_reg = 1000.0, double alpha = 10.0, 
                       double lr_theta = 0.02, double lr_mu = 0.05, 
                       int epochs = 2000, int warmup_epochs = 500, int seed = 42);

    /**
     * @brief Fits the invariant predictor using multi-threaded environment scaling.
     * @param X Input feature matrix of shape (N x D)
     * @param y Target matrix of shape (N x 1)
     */
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    /**
     * @brief Predicts out-of-distribution outcomes using the masked stable features.
     * @param X Input feature matrix of shape (N x D)
     * @return Eigen::VectorXd Evaluated predictions
     */
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

private:
    /**
     * @brief Core internal loss evaluation routine designed to be side-effect free.
     */
    double compute_loss_internal(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                 const Eigen::VectorXd& target_theta, const Eigen::VectorXd& target_mu,
                                 const std::vector<Eigen::VectorXd>& all_env_weights, double current_penalty_weight) const;
};