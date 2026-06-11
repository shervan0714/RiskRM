// File: main.cpp
#include<fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "rsw_model.hpp"
#include "dataset.hpp"

int main() {
    const int NUM_SAMPLES = 5000;
    
    std::cout << "==================================================\n";
    std::cout << "  RSW Covariate-Shift Benchmarking Engine\n";
    std::cout << "==================================================\n\n";

    std::cout << "[1/3] Generating synthetic market data...\n";
    auto [X, Y] = Dataset::generate_synthetic_data(NUM_SAMPLES, 4.0);

    // Using the exact hyperparameters from the research paper
    // (lambda = 1000.0, lr_theta = 0.02, epochs = 2000)
    HighPerformanceRSW model("regression", 10, 1000.0, 10.0, 0.02, 0.05, 2000, 500, 42);

    std::cout << "[2/3] Initiating Multi-Threaded Training Phase...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    model.fit(X, Y);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

    Eigen::VectorXd final_weights = model.theta.cwiseProduct(model.mask);
    
    std::cout << "\n==================================================\n";
    std::cout << "  Final Benchmark Results\n";
    std::cout << "==================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Execution Time : " << (duration_ms.count() / 1000.0) << " seconds\n\n";
    
    std::cout << "--- Learned Coefficients ---\n";
    std::cout << "True Target Eq : Y = 2.0000 * S + 0.0000 * V\n";
    std::cout << "RSW Model      : Y = " << final_weights(0) << " * S + " << final_weights(1) << " * V\n\n";

    if (std::abs(final_weights(1)) < 0.1) {
        std::cout << "[SUCCESS] Variant feature isolated and eliminated.\n";
    } else {
        std::cout << "[WARNING] Model failed to penalize variant feature.\n";
    }
    std::cout << "==================================================\n";
    // --- NEW: Export Results for Python ---
    std::cout << "[3/3] Exporting data to Python interface...\n";
    std::ofstream out_file("rsw_results.csv");
    out_file << "Feature,True_Weight,Learned_Weight\n";
    out_file << "Signal_S,2.0000," << final_weights(0) << "\n";
    out_file << "Trap_V,0.0000," << final_weights(1) << "\n";
    out_file.close();
    std::cout << "[SUCCESS] Data saved to rsw_results.csv\n";

    return 0;
}