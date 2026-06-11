# File: generate_market_data.py
import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------
# The Quant Scenario: 
# Target (Y) = Future Price Return
# True Signal (S) = Order Book Imbalance (Drives price)
# Trap Signal (V) = Trade Volume 
# ---------------------------------------------------------

# REGIME 1: Normal Market (4,000 samples)
# Volume is highly correlated with Order Imbalance. 
# A naive ML model will think Volume causes Price Returns.
S_normal = np.random.normal(0, 1, 4000)
V_normal = S_normal + np.random.normal(0, 0.1, 4000)
Y_normal = 2.5 * S_normal + 0.5 * (S_normal ** 3) + np.random.normal(0, 0.5, 4000)

# REGIME 2: Flash Crash / High Volatility (1,000 samples)
# The correlation breaks. Volume spikes randomly and is no longer tied to Imbalance.
# A naive ML model will blow up here. RiskRM (RSW) will survive.
S_crash = np.random.normal(0, 2, 1000)
V_crash = np.random.normal(0, 5, 1000) 
Y_crash = 2.5 * S_crash + 0.5 * (S_crash ** 3) + np.random.normal(0, 0.5, 1000)

# Combine and save
S = np.concatenate([S_normal, S_crash])
V = np.concatenate([V_normal, V_crash])
Y = np.concatenate([Y_normal, Y_crash])

df = pd.DataFrame({'Order_Imbalance': S, 'Trade_Volume': V, 'Future_Return': Y})
df.to_csv('market_data.csv', index=False)
print("[SUCCESS] market_data.csv generated! The Covariate Shift trap is set.")