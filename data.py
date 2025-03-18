import yfinance as yf
import pandas as pd
import os
import pandas_datareader as pdr

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

start_date = "1995-01-01"
end_date = "2025-01-01"

# S&P 500 low volatility ETF data (using SPY as proxy for time before inception)
spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo")
sphd = yf.download("SPHD", start=start_date, end=end_date, interval="1mo")
# 3-Month T-Bill data
t_bill = pdr.get_data_fred("TB3MS", start=start_date, end=end_date)
# S&P 500 data
sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1mo")

# S&P 500 low volatility ETF monthly returns
spy_returns = spy['Close'].pct_change().dropna()
sphd_returns = sphd['Close'].pct_change().dropna()
spy_returns.loc[sphd_returns.index] = sphd_returns
sphd_returns = spy_returns
sphd_returns = pd.DataFrame(sphd_returns.unstack().droplevel(0), columns=['monthly_return'])
sphd_returns.index.name = 'date'
# 3-Month T-Bill monthly returns
t_bill_returns = ((1 + t_bill['TB3MS'] / 100) ** (1/12) - 1).dropna()
t_bill_returns = pd.DataFrame(t_bill_returns).rename(columns={'TB3MS': 'monthly_return'})
t_bill_returns.index.name = 'date'
# S&P 500 monthly returns
sp500_returns = sp500['Close'].pct_change().dropna()
sp500_returns = pd.DataFrame(sp500_returns.unstack().droplevel(0), columns=['monthly_return'])
sp500_returns.index.name = 'date'

# Align dates (intersection of indices)
aligned_dates = sphd_returns.index.intersection(t_bill_returns.index)
aligned_dates = aligned_dates.intersection(sp500_returns.index)
sphd_returns = sphd_returns.loc[aligned_dates]
t_bill_returns = t_bill_returns.loc[aligned_dates]
sp500_returns = sp500_returns.loc[aligned_dates]

# Save to CSV
sphd_returns.to_csv("data/sphd_returns.csv")
t_bill_returns.to_csv("data/us_cash_returns.csv")
sp500_returns.to_csv("data/sp500_returns.csv")

print("S&P 500 low volatility ETF returns saved to data/sphd_returns.csv")
print("T-bill (cash proxy) returns saved to data/us_cash_returns.csv")
print("S&P 500 returns saved to data/sp500_returns.csv")