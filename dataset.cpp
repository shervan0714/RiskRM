// File: dataset.cpp

#include "dataset.hpp"
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Dataset::generate_synthetic_data(
    int num_samples, double spurious_strength) {

    Eigen::MatrixXd X(num_samples, 2);
    Eigen::VectorXd Y(num_samples);

    std::mt19937 gen(42); 
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        double S = dist(gen);
        // V is highly correlated with S
        double V = S + 0.1 * dist(gen);

        X(i, 0) = S;
        X(i, 1) = V;

        // The secret sauce for RSW: Non-linear misspecification
        double misspecification = 0.5 * S * S * S;
        double noise = 0.3 * dist(gen);

        Y(i) = (2.0 * S) + misspecification + noise; 
    }

    return {X, Y};
}

// --- Enterprise CSV Reader ---
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Dataset::load_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "CRITICAL ERROR: Could not open " << filepath << "\n";
        exit(1);
    }

    std::string line;
    std::vector<std::vector<double>> data;

    // Skip the header row (Order_Imbalance, Trade_Volume, Future_Return)
    std::getline(file, line);

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val)); // Convert string to double
        }
        
        // Prevent crashing on empty lines at the end of the file
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    int rows = data.size();
    int cols = data[0].size() - 1; // The last column is the Target (Y)

    Eigen::MatrixXd X(rows, cols);
    Eigen::VectorXd Y(rows);

    // Transfer from standard vectors into Eigen Matrices for high-speed math
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X(i, j) = data[i][j];
        }
        Y(i) = data[i].back();
    }

    return {X, Y};
}