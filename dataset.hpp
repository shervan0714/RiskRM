// File: dataset.hpp

#ifndef DATASET_HPP
#define DATASET_HPP

#include <Eigen/Dense>
#include <tuple>
#include <string>

/**
 * @class Dataset
 * @brief Handles data ingestion and synthetic generation for quantitative backtesting.
 */
class Dataset {
public:
    /**
     * @brief Generates a synthetic dataset to test covariate-shift robustness.
     * @param num_samples Number of rows to generate.
     * @param spurious_strength How tightly the fake feature (V) is correlated to the true feature (S).
     * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Returns the X feature matrix and Y target vector.
     */
    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd> generate_synthetic_data(
        int num_samples, double spurious_strength);

    /**
     * @brief Loads real-world market data from a CSV file.
     * @param filepath Path to the CSV file (e.g., "market_data.csv").
     * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Returns the X feature matrix and Y target vector.
     */
    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd> load_csv(const std::string& filepath);
};

#endif // DATASET_HPP