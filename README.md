# LDI with Dynamic Risk Budgeting Simulator

A simple, interactive tool to explore Liability-Driven Investing (LDI) strategies with dynamic risk budgeting, built with Streamlit. Developed with the knowledge and skills gained from the Coursera course *Introduction to Portfolio Construction and Analysis with Python* by EDHEC Business School.

**(https://ldi-dynamic-risk-budgeting-simulator.streamlit.app)**  

---

## Features
This app demonstrates key financial concepts through hands-on simulation:
- **Liability-Driven Investing (LDI)**: Balances a Performance Seeking Portfolio (PSP, risky assets) and Goal Hedging Portfolio (GHP, safe assets) to meet liability goals.
- **Dynamic Risk Budgeting**: 
  - **Floor Allocation**: Protects wealth above a minimum level (e.g., 75%) using a Constant Proportion Portfolio Insurance (CPPI)-style approach.
  - **Max Drawdown Allocation**: Limits losses (e.g., 25%) by adjusting exposure based on peak wealth.
- **Monte Carlo Simulation**: Generates thousands of scenarios using Geometric Brownian Motion (GBM) for PSP and Cox-Ingersoll-Ross (CIR) for GHP to model uncertainty.
- **Real Data Integration**: Backtested SPHD (risky) and T-bills (safe) and benchmarks against ^GSPC (market) for practical insights.
- **Interactive Analysis**: Adjust parameters (e.g., floor, max drawdown, time horizon) and visualize wealth trajectories, terminal value distributions, and risk metrics (e.g., Sharpe ratio, VaR).

---

## Dependencies
- Python 3.8+
- Required libraries (install via `pip`):
  - `streamlit==1.43.2`
  - `pandas==2.2.3`
  - `numpy==2.2.3`
  - `matplotlib==3.10.1`
  - `plotly==6.0.0`
  - `seaborn==0.13.2`
  - `yfinance==0.2.54`

---
