# utils.py
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import streamlit as st

@st.cache_data
def load_returns(file_path_risky="data/sphd_returns.csv", file_path_safe="data/us_cash_returns.csv", file_path_market="data/sp500_returns.csv"):
    """Load real returns for risky (SPHD Etf), safe (T-bill) assets and the market (S&P 500)."""
    risky = pd.read_csv(file_path_risky, index_col=0, parse_dates=True)
    safe = pd.read_csv(file_path_safe, index_col=0, parse_dates=True)
    market = pd.read_csv(file_path_market, index_col=0, parse_dates=True)
    return pd.concat([risky, safe, market], axis=1, keys=["risky", "safe", "market"])

@st.cache_data
def simulate_gbm(n_years=10, n_scenarios=100, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Simulate asset price paths using Geometric Brownian Motion (GBM).
    
    Parameters:
    - n_years: Number of years to simulate
    - n_scenarios: Number of simulation paths
    - mu: Annual drift (expected return)
    - sigma: Annual volatility
    - steps_per_year: Number of time steps per year
    - s_0: Initial price
    - prices: If True, return prices; if False, return returns
    
    Returns:
    - A DataFrame of simulated paths (rows=time steps, columns=scenarios)
    """
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1  # Initial step
    if prices:
        return s_0 * pd.DataFrame(rets_plus_1).cumprod()
    else:
        return pd.DataFrame(rets_plus_1 - 1)
    
@st.cache_data
def simulate_cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = np.log1p(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=np.expm1(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bt_mix(returns_1, returns_2, allocator, **kwargs):
    """
    Simulates portfolio allocation between two return sets.
    Parameters:
        returns_1: T x N DataFrame of returns (T = time steps, N = scenarios)
        returns_2: T x N DataFrame of returns, matching returns_1 shape
        allocator: Function that takes two return sets and parameters, returning
                  a T x 1 DataFrame of weights for returns_1 (remainder to returns_2)
    Returns:
        T x N DataFrame of blended portfolio returns from N scenarios
    Raises:
        ValueError: If returns_1 and returns_2 shapes differ, or allocator
                   returns weights with incorrect shape
    """
    if returns_1.shape != returns_2.shape:
        raise ValueError("returns_1 and returns_2 must have identical shapes")
    weights = allocator(returns_1, returns_2, **kwargs)
    if weights.shape != returns_1.shape:
        raise ValueError("Allocator weights shape does not match returns shape")
    # print("weights index:", weights.index)
    # print("returns_1 index:", returns_1.index)
    # print("returns_2 index:", returns_2.index)
    return weights * returns_1 + (1 - weights) * returns_2

def fixedmix_allocation(returns_1, returns_2, weights_1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = weights_1, index=returns_1.index, columns=returns_1.columns)

@st.cache_data
def floor_allocation(psp_returns, ghp_returns, floor, zc_prices, multiplier=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_returns.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_returns.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    weights = pd.DataFrame(index=psp_returns.index, columns=psp_returns.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_weights = (multiplier*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_weights
        psp_alloc = account_value*psp_weights
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_returns.iloc[step]) + ghp_alloc*(1+ghp_returns.iloc[step])
        weights.iloc[step] = psp_weights
    return weights

@st.cache_data
def drawdown_allocation(psp_returns, ghp_returns, max_drawdown=0.25, multiplier=5, momentum_window=3):
    """
    Dynamically allocate between PSP (performance seeking portfolio) and GHP (goal hedging portfolio) assets using a drawdown-constrained CPPI strategy.
    
    Parameters:
    - psp_returns: DataFrame of PSP returns (T x N, T=time steps, N=scenarios)
    - ghp_returns: DataFrame of GHP returns (same shape)
    - max_drawdown: Maximum allowed drawdown (e.g., 0.2 for 20%)
    - multiplier: Multiplier for the cushion (just like CPPI)
    
    Returns:
    - DataFrame of weights allocated to PSP (same shape as inputs)
    """
    if psp_returns.shape != ghp_returns.shape:
        raise ValueError("PSP and GHP returns must have the same shape")
    
    n_steps, n_scenarios = psp_returns.shape
    account_value = np.ones(n_scenarios)  # Start with normalized value of 1
    peak_value = np.ones(n_scenarios)     # Track peak value for drawdown
    weights = pd.DataFrame(index=psp_returns.index, columns=psp_returns.columns, dtype=float)
    for step in range(n_steps):
        # Compute the floor based on the maximum drawdown from the peak
        floor_value = (1 - max_drawdown) * peak_value
        # Cushion is the proportion of the account value above the floor
        cushion = (account_value - floor_value) / account_value
        # Allocate to PSP asset: m * cushion, clipped between 0 and 1
        psp_weight = (multiplier * cushion).clip(0, 1)
        # # debug
        # if psp_weight.sum() == 0:
        #     print('psp weight:', psp_weight)
        # Momentum-based override for re-entry
        if step >= momentum_window - 1:
            momentum = psp_returns.rolling(momentum_window).mean().iloc[step]
            re_entry_condition = (cushion < 0) & (momentum > 0)
            # # debug
            # print('cushion:', cushion < 0)
            # print('momentum', momentum > 0)
            # print('re_entry_condition:', re_entry_condition[0])
            # quit()
            # if re_entry_condition.shape[0] < 2 and re_entry_condition.iloc[0]:
            #     print('cushion:', cushion)
            #     print('momentum:', momentum)
            #     print('psp_returns:', psp_returns.iloc[step-2:step+1])
            #     print('ghp_returns:', ghp_returns.iloc[step-2:step+1])
            # Gradual re-entry
            psp_weight = np.where(re_entry_condition, np.minimum(0.5, multiplier * 0.1), psp_weight)

        # Allocate the rest to GHP
        ghp_weight = 1 - psp_weight
        # Update account value based on the allocation
        account_value = (account_value * psp_weight * (1 + psp_returns.iloc[step]) + \
                         account_value * ghp_weight * (1 + ghp_returns.iloc[step]))
        # Update the peak
        peak_value = np.maximum(peak_value, account_value)
        weights.iloc[step] = psp_weight
    
    return weights

def terminal_values(returns):
    """
    Compute the terminal value of a portfolio given its returns.
    
    Parameters:
    - returns: DataFrame of returns (T x N)
    
    Returns:
    - Series of terminal values for each scenario
    """
    return (1 + returns).prod()

def terminal_stats(returns, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (returns+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def drawdown(returns: pd.Series, starting_wealth=1000):
    """Calculate wealth index, peaks, and drawdowns."""
    wealth = (1 + returns).cumprod() * starting_wealth
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks) / peaks
    return pd.DataFrame({"Wealth": wealth, "Peaks": peaks, "Drawdown": drawdowns})

def annualized_return(returns, periods_per_year):
        return (1 + returns).prod() ** (periods_per_year / returns.shape[0]) - 1

def annualized_volatility(returns, periods_per_year):
    return returns.std() * (periods_per_year ** 0.5)

def sharpe_ratio(returns, riskfree_rate, periods_per_year):
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    # print(returns, rf_per_period)
    excess_ret = returns - rf_per_period
    ann_ex_ret = annualized_return(excess_ret, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    sharpe = ann_ex_ret / ann_vol
    # print("ann ex ret", ann_ex_ret)
    return sharpe

def skewness(returns):
    demeaned = returns - returns.mean()
    sigma = returns.std(ddof=0)
    skew = (demeaned ** 3).mean() / (sigma ** 3)
    return skew

def kurtosis(returns):
    demeaned = returns - returns.mean()
    sigma = returns.std(ddof=0)
    kurt = (demeaned ** 4).mean() / (sigma ** 4)
    return kurt

def cf_var(returns, level=5):
    """
    Calculate the Cornish-Fisher Value at Risk (VaR) of a set of returns.
    """
    z = norm.ppf(level)
    skew = skewness(returns)
    kurt = kurtosis(returns)
    z = (z +
        (z**2 - 1)*skew/6 +
        (z**3 -3*z)*(kurt-3)/24 -
        (2*z**3 - 5*z)*(skew**2)/36
        )
    cf_var = -(returns.mean() + z * returns.std(ddof=0))
    return cf_var

def cvar_historic(returns, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(returns, pd.Series):
        is_beyond = returns <= -cf_var(returns, level=level)
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def summary_statistics(returns, riskfree_rate=0.03, periods_per_year=12, level=0.05):
    """
    Compute summary statistics for a set of returns.
    
    Parameters:
    - returns: DataFrame of returns (T x N)
    - riskfree_rate: Annual risk-free rate
    - periods_per_year: Number of periods per year
    
    Returns:
    - DataFrame of statistics (e.g., annualized return, volatility, Sharpe ratio)
    """
    ann_ret = returns.aggregate(annualized_return, periods_per_year=periods_per_year)
    ann_vol = returns.aggregate(annualized_volatility, periods_per_year=periods_per_year)
    sharpe = returns.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    max_dd = returns.aggregate(lambda r: drawdown(r)['Drawdown'].min())
    skew = returns.aggregate(skewness)
    kurt = returns.aggregate(kurtosis)
    var = returns.aggregate(cf_var, level=level)
    cvar = returns.aggregate(cvar_historic, level=level)
    
    return pd.DataFrame({
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR": var,
        "Historic Conditional VaR": cvar
    })