
---

# RiskRM: High-Performance C++ Engine for Covariate Shift Robustness

**Author:** E. Sai Shervan

**Domain:** Quantitative Research / High-Frequency Machine Learning Infrastructure

## 🚀 Overview: The Financial Covariate Shift Problem

In quantitative finance, standard machine learning models are highly susceptible to **spurious correlations**. In a quiet, normal market regime, a model might observe that Trade Volume is perfectly correlated with Future Returns. An Ordinary Least Squares (OLS) or Ridge regression model will aggressively assign a high weight to this volume.

However, this correlation is a trap. The true causal driver is hidden Order Imbalance, and the trade volume is simply a downstream byproduct. When a market regime shifts (e.g., a sudden Flash Crash), the correlation between trade volume and returns breaks violently, and standard models suffer catastrophic drawdowns.

This repository implements the core concepts from the AAAI 2023 paper, **"Covariate-Shift Generalization via Random Sample Weighting"**, but rebuilds them entirely in modern C++. It mathematically isolates invariant, causal features across multiple distinct market environments, crushing spurious traps to zero and ensuring the trading strategy survives out-of-distribution (OOD) regime shifts.

---

## 🔬 The Research: What is Random Sample Weighting (RSW)?

The underlying research paper proves that by generating heterogeneous environments via Random Sample Weighting (RSW), we can expose which features are stable (causal) and which are unstable (spurious).

By applying variance penalization across these generated environments, the optimizer is forced to drop variables that behave inconsistently when the data distribution shifts. This repository takes that theoretical framework and engineers it into a high-speed, production-ready backend.

---

## 🛠️ Evolution & Engineering Upgrades

This engine underwent severe refactoring to bridge the gap between academic theory and high-performance trading infrastructure. Here is exactly how the repository was improved from its initial prototype:

### 1. From Random Noise to Covariate-Determining Environments

* **Previous State:** The model generated experimental environments using a uniform random distribution, which introduced unnecessary variance and was not strictly true to the underlying math.
* **Production State:** Implemented strict covariate-determining environments as outlined in the literature. By generating random projection vectors and mapping them through a Sigmoid function, the environmental weights are mathematically bound to the actual market data. This forces the optimizer to explicitly evaluate "High Volume" vs. "Low Volume" regimes, exposing spurious variables instantly.

### 2. From Finite Differences to Exact Analytical Gradients

* **Previous State:** The optimizer relied on numerical approximations (Finite Differences), requiring costly computational complexity as it recalculated the loss twice for every parameter using a tiny nudge.
* **Production State:** Completely eliminated numerical guessing by deriving and implementing the exact closed-form matrix calculus using the **Hessian-Vector Product Trick**. The invariance penalty is now calculated in a single vectorized forward pass.

### 3. From Static Binaries to PyBind11 Microservices

* **Previous State:** A standalone C++ executable that required manual CSV reading, writing, and compilation for every test.
* **Production State:** A dynamic Python-bound library utilizing PyBind11. It acts as a backend microservice (`import rsw_engine`), allowing researchers to pass Pandas DataFrames and NumPy arrays directly into the C++ memory space with zero overhead.

---

## ⚡ System Architecture

Unlike standard Python/PyTorch implementations designed for slow academic research, this engine is built for low-latency infrastructure:

* **Core Engine:** Written from scratch in **C++17** for strict memory control.
* **Linear Algebra Backend:** Leverages **Eigen3** for cache-friendly, vectorized matrix calculus.
* **Multithreading:** Utilizes **OpenMP** to concurrently evaluate environmental gradients across CPU cores.

---

## 📐 Mathematical Derivations

To bypass the computational overhead of standard Automatic Differentiation (Autograd) memory graphs, the exact gradients are derived and calculated manually in C++.

**Base Empirical Risk Gradient:** The base loss is the Mean Squared Error over the training set. Its exact analytical gradient with respect to the effective weights is:
$\nabla_{\theta_{eff}} L_{base} = \frac{2}{N} X^T (X\theta_{eff} - Y)$

**Invariance Penalty Gradient (Hessian-Vector Trick):** To calculate the variance of the gradients across environments without constructing an expensive dense Hessian matrix, we use a vectorized analytical approach. Let $g_e$ be the environment-specific gradient and $\bar{g}$ be the mean gradient across all environments:
$\nabla_{\theta_{eff}} R_{inv} = \frac{2}{E-1} \sum_{e=1}^{E} \left( \frac{2}{N} X^T \Big( W_e \odot \big(X (g_e - \bar{g})\big) \Big) \right)$

By combining these and applying the Chain Rule via the sparse Sigmoid mask derivative, the parameters are updated in a highly optimized multithreaded pass.

---

## 📊 Empirical Out-of-Distribution (OOD) Benchmarks

The engine was benchmarked against standard linear models. All models were trained **strictly on Normal Market data** to prevent data leakage, and then tested against an unseen Flash Crash regime.

| Model | True Signal | Trap Weight | Normal RMSE | Crash RMSE (OOD) |
| --- | --- | --- | --- | --- |
| **OLS** | 4.1785 | -0.1813 | **1.3202** | 11.2440 |
| **Ridge** | 4.0771 | -0.0808 | 1.3202 | 11.3599 |
| **Lasso** | 3.8968 | 0.0000 | 1.3241 | 11.6042 |
| **RiskRM (C++)** | **4.9346** | **0.0074** | 1.6218 | **10.3023** |

### Benchmark Analysis

* **The Normal Market:** Standard models (OLS/Ridge) overfit to the Normal regime, slightly edging out RiskRM in the quiet market.
* **The Regime Shift (Crash):** When the Crash hits, the standard models' reliance on the Trap variable causes their error rates to explode.
* **Vs. Lasso:** While Lasso shrinks the trap to zero, it is blind to causality and accidentally over-shrinks the True Signal. RiskRM successfully balances causal invariance, maintaining a strong grip on the True Signal while successfully ignoring the trap, resulting in the best OOD survival rate.

**Execution Speed:** ~1.21 seconds (Analytical Gradients, 2000 Epochs).

---

## 📂 Repository Structure

This codebase is cleanly separated into the high-performance C++ backend, the Python interface, and the evaluation scripts. Here is a guide to navigating the repository:

### Core C++ Engine (The Backend)

* **`rsw_model.cpp` & `rsw_model.hpp`:** The heart of the RiskRM engine. Contains the core mathematics, including the covariate-determining environment generation (via Sigmoid projections) and the exact analytical gradients using Eigen matrix calculus.
* **`bindings.cpp`:** The PyBind11 translation layer. This file explicitly maps the C++ classes and functions (like `.fit()` and `.predict()`) so they can be called natively within Python.
* **`dataset.cpp` & `dataset.hpp`:** C++ side data structures and memory management utilities for handling market matrices before they hit the optimizer.
* **`main.cpp`:** The legacy standalone C++ executable used for pure C++ benchmarking before the PyBind11 Python microservice architecture was implemented.

### Python Evaluation & Visualization (The Frontend)

* **`generate_market_data.py`:** Generates the synthetic causal `market_data.csv`. It injects the specific nonlinear signal ($2.5S + 0.5S^3$) and the spurious trap variable ($V$) to create the Normal and Flash Crash market regimes.
* **`test_wrapper.py`:** The primary evaluation pipeline. It loads the data, strictly trains the baseline models (OLS, Ridge, Lasso) and the C++ RiskRM engine on the Normal regime, and calculates Out-of-Distribution (OOD) RMSE on the Crash regime.
* **`plot_results.py` & `plot_variance.py`:** Visualization scripts used to generate the benchmark comparison graphs proving the elimination of the spurious correlation.

### Build & Configuration

* **`CMakeLists.txt`:** The cross-platform build configuration file. It dictates how the C++ compiler links Eigen, OpenMP, and PyBind11 to generate the final `.pyd` or `.so` binary.
* **`requirements.txt`:** The Python dependencies (`scikit-learn`, `pandas`, `numpy`, `matplotlib`) required to run the evaluation wrappers.
* **`.gitignore`:** Ensures that massive build artifacts (`/build`), virtual environments, and external C++ libraries (`/eigen`) are kept out of the production repository.

## 🚀 Quickstart & Build Instructions

This project uses CMake for cross-platform compilation.

**1. Clone and Configure**

```bash
git clone https://github.com/shervan0714/RiskRM.git
cd RiskRM
mkdir build && cd build
cmake ..

```

**2. Compile in Release Mode (Maximum Speed)**

```bash
cmake --build . --config Release

```

**3. Run the Python Pipeline**
Ensure the generated `.pyd` (Windows) or `.so` (Linux/Mac) file is located in your working directory alongside the Python scripts.

```bash
python generate_market_data.py  # Generates the synthetic causal market CSV
python test_wrapper.py          # Executes the baseline tests and C++ engine

```
