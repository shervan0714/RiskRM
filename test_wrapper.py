# File: test_wrapper.py
import rsw_engine
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

print("Loading market data...")
try:
    df = pd.read_csv("market_data.csv")
except FileNotFoundError:
    print("Error: Could not find market_data.csv.")
    exit()

X = df[['Order_Imbalance', 'Trade_Volume']].to_numpy()
Y = df['Future_Return'].to_numpy()

# The first 4000 rows are the Normal Regime, the last 1000 are the Crash Regime
X_normal, Y_normal = X[:4000], Y[:4000]
X_crash, Y_crash   = X[4000:], Y[4000:]

models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

results = []

# --- Train and Evaluate Baselines (STRICTLY ON NORMAL DATA) ---
for name, model in models.items():
    model.fit(X_normal, Y_normal) # <-- FIXED: No peeking at the crash data
    weights = model.coef_
    
    # Predict on both regimes
    preds_normal = model.predict(X_normal)
    preds_crash = model.predict(X_crash)
    
    rmse_normal = np.sqrt(mean_squared_error(Y_normal, preds_normal))
    rmse_crash = np.sqrt(mean_squared_error(Y_crash, preds_crash))
    
    results.append((name, weights[0], weights[1], rmse_normal, rmse_crash))

# --- Train and Evaluate C++ RiskRM (STRICTLY ON NORMAL DATA) ---
print("Training C++ RiskRM Engine...")
rsw = rsw_engine.HighPerformanceRSW("regression", 10, 1000.0, 10.0, 0.02, 0.05, 2000, 500, 42)
start_time = time.time()
rsw.fit(X_normal, Y_normal) # <-- FIXED: No peeking at the crash data
exec_time = time.time() - start_time

rsw_weights = rsw.theta * rsw.mask
preds_normal_rsw = rsw.predict(X_normal)
preds_crash_rsw = rsw.predict(X_crash)

rmse_normal_rsw = np.sqrt(mean_squared_error(Y_normal, preds_normal_rsw))
rmse_crash_rsw = np.sqrt(mean_squared_error(Y_crash, preds_crash_rsw))

results.append(("RiskRM (C++)", rsw_weights[0], rsw_weights[1], rmse_normal_rsw, rmse_crash_rsw))

# --- Print Research-Grade Evaluation Table ---
print("\n" + "="*85)
print(f"{'Model':<15} | {'True Signal (S)':<15} | {'Trap (V)':<12} | {'Normal RMSE':<15} | {'Crash RMSE':<15}")
print("-" * 85)
for res in results:
    print(f"{res[0]:<15} | {res[1]:<15.4f} | {res[2]:<12.4f} | {res[3]:<15.4f} | {res[4]:<15.4f}")
print("=" * 85)
print(f"\n[INFO] RiskRM Analytical Gradient Execution Time: {exec_time:.4f} seconds")