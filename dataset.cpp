// File: dataset.cpp

#include "dataset.hpp"
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

using namespace std;

tuple<Eigen::MatrixXd, Eigen::VectorXd> Dataset::generate_synthetic_data(
    int num_samples, double spurious_strength) {

    Eigen::MatrixXd X(num_samples, 2);
    Eigen::VectorXd Y(num_samples);

    mt19937 gen(42); 
    normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        double S = dist(gen);
        
        // V is trap variable. it follows S mostly, so normal models will easily get confused
        double V = S + 0.1 * dist(gen);

        X(i, 0) = S;
        X(i, 1) = V;

        // main trick for RSW: adding non linear cubic curve
        // linear models cant fit this properly so they will distribute the error
        double misspecification = 0.5 * S * S * S;
        double noise = 0.3 * dist(gen);

        Y(i) = (2.0 * S) + misspecification + noise; 
    }

    return {X, Y};
}

// --- CSV Loader ---
tuple<Eigen::MatrixXd, Eigen::VectorXd> Dataset::load_csv(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: File not found at " << filepath << ". did you run python script first?\n";
        exit(1);
    }

    string line;
    vector<vector<double>> data;

    // skip header line otherwise stod will give error for strings
    getline(file, line);

    // read file line by line
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        vector<double> row;
        
        while (getline(ss, val, ',')) {
            row.push_back(stod(val)); 
        }
        
        // handle empty lines at end so code doesnt crash
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    int rows = data.size();
    int cols = data[0].size() - 1; // last col is Y

    Eigen::MatrixXd X(rows, cols);
    Eigen::VectorXd Y(rows);

    // shifting from normal vectors to Eigen matrix because Eigen is very fast for math operations
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X(i, j) = data[i][j];
        }
        Y(i) = data[i].back();
    }

    return {X, Y};
}
