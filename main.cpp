// File: main.cpp
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "rsw_model.hpp"
#include "dataset.hpp"

using namespace std;

int main() {
    const int NUM_SAMPLES = 5000;
    
    cout << "--------------------------------------------------\n";
    cout << " RSW Covariate-Shift Benchmark\n";
    cout << "--------------------------------------------------\n\n";

    cout << "Step 1: Spinning up synthetic market data (" << NUM_SAMPLES << " samples)...\n";
    auto [X, Y] = Dataset::generate_synthetic_data(NUM_SAMPLES, 4.0);

    // Sticking to the exact hyperparams from the AAAI paper so we know the test is legit
    // (lambda = 1000.0, lr_theta = 0.02, epochs = 2000)
    HighPerformanceRSW model("regression", 10, 1000.0, 10.0, 0.02, 0.05, 2000, 500, 42);

    cout << "Step 2: Kicking off multi-threaded training...\n";
    auto start_time = chrono::high_resolution_clock::now();

    model.fit(X, Y);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_ms = end_time - start_time;

    // Grab the final effective weights (theta * mask)
    Eigen::VectorXd final_weights = model.theta.cwiseProduct(model.mask);
    
    cout << "\n--------------------------------------------------\n";
    cout << " Benchmark Results\n";
    cout << "--------------------------------------------------\n";
    cout << fixed << setprecision(4);
    cout << "Time taken : " << (duration_ms.count() / 1000.0) << " seconds\n\n";
    
    cout << "--- The Weights ---\n";
    cout << "What it should be : Y = 2.0000 * S + 0.0000 * V\n";
    cout << "What we got       : Y = " << final_weights(0) << " * S + " << final_weights(1) << " * V\n\n";

    if (abs(final_weights(1)) < 0.1) {
        cout << "The model successfully ignored the trap variable.\n";
    } else {
        cout << "The model took the bait and didn't penalize the noise.\n";
    }
    cout << "--------------------------------------------------\n";
    
    // Push the results out so our Python scripts can pick them up for plotting
    cout << "Step 3: Dumping results to CSV for Python...\n";
    ofstream out_file("rsw_results.csv");
    out_file << "Feature,True_Weight,Learned_Weight\n";
    out_file << "Signal_S,2.0000," << final_weights(0) << "\n";
    out_file << "Trap_V,0.0000," << final_weights(1) << "\n";
    out_file.close();
    
    cout << "All done. Saved to rsw_results.csv\n";

    return 0;
}
