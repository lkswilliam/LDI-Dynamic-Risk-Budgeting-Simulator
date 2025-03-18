import streamlit as st
import pandas as pd
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

# --- Page configuration ---
st.set_page_config(
    page_title="LDI with Dynamic Risk Budgeting Simulator",
    layout="wide",
    initial_sidebar_state="expanded")

# --- Sidebar Info ---
with st.sidebar:
    st.title("LDI with Dynamic Risk Budgeting Simulator")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/william-lks/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Lam Kwong Sing, William`</a>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
data_source = st.sidebar.radio(
    "Select Data Source",
    options=["Real Data (SPHD & T-bill)", "Simulated Data (GBM & CIR)"],
    index=0  # Default to first option: Real Data
)

start = st.sidebar.number_input("Initial Wealth ($)", 1, None, 1000)
n_years = st.sidebar.slider("Years", 1, 30, 20)

if data_source == "Real Data (SPHD & T-bill)":
    pass

if data_source == "Simulated Data (GBM & CIR)":
    n_scenarios = st.sidebar.slider("Number of Scenarios", 1, 5000, 1000, step=1)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mu_market = st.number_input("Market Expected Return", 0.0, 0.15, 0.07, step=0.01)
        sigma_market = st.number_input("Market Volatility", 0.05, 0.3, 0.15, step=0.01)
        mu_risky = st.number_input("PSP Expected Return", 0.0, 0.15, 0.09, step=0.01)
        sigma_risky = st.number_input("PSP Volatility", 0.05, 0.3, 0.20, step=0.01)
    with col2:
        r_0 = st.number_input("Initial Interest Rate", 0.0, 0.1, 0.03, step=0.005)
        a_safe = st.number_input("Interest Rate Mean Reversion Rate", 0.01, 0.2, 0.05, step=0.01)
        b_safe = st.number_input("Long-Term Interest Rate Mean", 0.005, 0.15, 0.03, step=0.005)
        sigma_safe = st.number_input("Interest Rate Volatility", 0.01, 0.2, 0.05, step=0.01)

st.sidebar.markdown("---")

# --- Load Data ---
if data_source == "Real Data (SPHD & T-bill)":
    returns = ut.load_returns()
    real_market = returns['market'][-n_years*12:]
    real_risky = returns['risky'][-n_years*12:]
    real_safe = returns['safe'][-n_years*12:]

if data_source == "Simulated Data (GBM & CIR)":
    gbm_market = ut.simulate_gbm(n_years=n_years, n_scenarios=n_scenarios, mu=mu_market, sigma=sigma_market).pct_change().dropna()
    gbm_risky = ut.simulate_gbm(n_years=n_years, n_scenarios=n_scenarios, mu=mu_risky, sigma=sigma_risky).pct_change().dropna()
    cir_rates, cir_zc_prices = ut.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, a=a_safe, b=b_safe, sigma=sigma_safe, r_0=r_0)
    cir_safe = cir_zc_prices.pct_change().dropna()

# --- Main Content ---
st.header("LDI with Dynamic Risk Budgeting Simulator")
if data_source == "Real Data (SPHD & T-bill)":
    st.write('Using SPHD and T-bill as proxies for Performance Seeking Portfolio (PSP) and Goal Hedging Portfolio (GHP).')
if data_source == "Simulated Data (GBM & CIR)":
    st.write('Using simulated data: Geometric Brownian Motion (GBM) for the Performance Seeking Portfolio (PSP, risky assets) and Cox-Ingersoll-Ross (CIR) for the Goal Hedging Portfolio (GHP, zero-coupon bonds).')

# Allocation Strategy
options = []
if data_source == "Real Data (SPHD & T-bill)":
    options = ["Max Drawdown Allocation"]
if data_source == "Simulated Data (GBM & CIR)":
    options = ["Max Drawdown Allocation", "Floor Allocation"]

allocation_strategy = st.pills("Allocation Strategy", options, default=["Max Drawdown Allocation"], selection_mode="multi")

if not allocation_strategy:
    st.error("Please select at least one allocation strategy.")

# Tabs with Dynamic Content
tab1, tab2, tab3 = st.tabs(["Simulation Results", "Data Overview", "statistics"])

with tab1:
    if data_source == "Real Data (SPHD & T-bill)":
        col1, col2 = st.columns([1, 5])  
        # Place sliders in the first column
        with col1:
            if "Max Drawdown Allocation" in allocation_strategy:
                max_drawdown = st.slider("Max Drawdown", 0.05, 0.50, 0.25, step=0.05)
                multiplier = st.slider("MaxDD Multiplier", 1.0, 10.0, 3.0, step=0.5)
                momentum_window = st.slider("Momentum Window", 1, 12, 3, step=1)
            # Place the graph in the second column
        with col2:
            if "Max Drawdown Allocation" in allocation_strategy:
                returns_maxdd = ut.bt_mix(real_risky, real_safe, 
                                        allocator=ut.drawdown_allocation, 
                                        max_drawdown=max_drawdown, multiplier=multiplier, 
                                        momentum_window=momentum_window)
                dd_market = ut.drawdown(real_market['monthly_return'], starting_wealth=start)
                dd_maxdd = ut.drawdown(returns_maxdd['monthly_return'], starting_wealth=start)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dd_market.index, y=dd_market["Wealth"], mode="lines", line=dict(color="goldenrod"), name="Market Wealth"))
                fig.add_trace(go.Scatter(x=dd_market.index, y=dd_market["Peaks"], mode="lines", line=dict(color="goldenrod", dash="dot"), name="Market Peaks"))
                fig.add_trace(go.Scatter(x=dd_maxdd.index, y=dd_maxdd["Wealth"], mode="lines", line=dict(color="cornflowerblue"), name="MaxDD Wealth"))
                fig.add_trace(go.Scatter(x=dd_maxdd.index, y=dd_maxdd["Peaks"], mode="lines", line=dict(color="cornflowerblue", dash="dot"), name="MaxDD Peaks"))
                fig.update_layout(title="Max Drawdown Allocation Strategy benchmarked against Market (^GSPC)", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
    if data_source == "Simulated Data (GBM & CIR)":
        col1, col2 = st.columns([1, 5])  
        with col1:
            if "Floor Allocation" in allocation_strategy:
                floor = st.slider("Floor Level", 0.5, 0.95, 0.75, step=0.05)
                multiplier = st.slider("Floor Multiplier", 1.0, 10.0, 3.0, step=0.5, key="floor")
            if "Max Drawdown Allocation" in allocation_strategy:
                max_drawdown = st.slider("Max Drawdown", 0.05, 0.50, 0.25, step=0.05)
                multiplier = st.slider("MaxDD Multiplier", 1.0, 10.0, 3.0, step=0.5, key="maxdd")
                momentum_window = st.slider("Momentum Window", 1, 12, 3, step=1)
        with col2:
            plt.figure(figsize=(10,6))
            if "Floor Allocation" in allocation_strategy:
                returns_floor = ut.bt_mix(gbm_risky, cir_safe, 
                                            allocator=ut.floor_allocation, floor=floor, 
                                            zc_prices=cir_zc_prices[1:], 
                                            multiplier=multiplier)
                tv_floor = start * ut.terminal_values(returns_floor)
                sns.histplot(tv_floor, kde=True, stat="density", color="red", label=f"{floor*100}% floor", bins=100)
                plt.axvline(tv_floor.mean(), ls="--", color="red")
            if "Max Drawdown Allocation" in allocation_strategy:
                tmp = ut.load_returns()['safe'][-n_years*12:]
                tmp = pd.concat([tmp] * n_scenarios, axis=1, ignore_index=True).reset_index(drop=True)
                tmp2 = gbm_risky[1:] if n_years == 30 else gbm_risky
                sim_returns_maxdd = ut.bt_mix(tmp2, tmp, 
                                        allocator=ut.drawdown_allocation, 
                                        max_drawdown=max_drawdown, multiplier=multiplier, 
                                        momentum_window=momentum_window)
                tv_maxdd = start * ut.terminal_values(sim_returns_maxdd)
                sns.histplot(tv_maxdd, kde=True, stat="density", color="blue", label=f"{max_drawdown*100}% max drawdown", bins=100)
                plt.axvline(tv_maxdd.mean(), ls="--", color="blue")
            plt.legend()
            plt.title("Terminal Values")
            st.pyplot(plt)  
            
        if st.button("Show Wealth Trajectories"):
            if "Floor Allocation" in allocation_strategy:
                n = min(returns_floor.shape[0], 50)
                fig = go.Figure()
                for column in (start * (1 + returns_floor).cumprod()).iloc[:, :n].columns:
                    fig.add_trace(go.Scatter(x=returns_floor.index, y=(start * (1 + returns_floor).cumprod())[column], name=None))
                fig.update_layout(title=f"Floor Allocation Simulation ({floor*100}%)", showlegend=False)
                st.plotly_chart(fig)
            if "Max Drawdown Allocation" in allocation_strategy:
                n = min(sim_returns_maxdd.shape[0], 50)
                fig = go.Figure()
                for column in (start * (1 + sim_returns_maxdd).cumprod()).iloc[:, :n].columns:
                    fig.add_trace(go.Scatter(x=sim_returns_maxdd.index, y=(start * (1 + sim_returns_maxdd).cumprod())[column], name=None))
                fig.update_layout(title=f"Max Drawdown Allocation Simulation ({max_drawdown*100}%)", showlegend=False)
                st.plotly_chart(fig)

with tab2:
    if data_source == 'Real Data (SPHD & T-bill)': # Real Data
        st.write("Showing real S&P 500 and T-bill data.")
        chart_data = pd.DataFrame({
            'Market (S&P 500)': (start*(1+real_market['monthly_return']).cumprod()),
            'Risky (SPHD)': (start*(1+real_risky['monthly_return']).cumprod()),
            'Safe (T-bill)': (start*(1+real_safe['monthly_return']).cumprod())
        })
        st.line_chart(chart_data)

    elif data_source == 'Simulated Data (GBM & CIR)':  # Simulated Data
        n = min(gbm_market.shape[0], 50)
        st.write(f"Showing simulated data ({n} samples) from Geometric Brownian Motion (GBM) and Cox-Ingersoll-Ross (CIR) models.")
        fig = go.Figure()
        for column in (start * (1 + gbm_market).cumprod()).iloc[:, :n].columns:
            fig.add_trace(go.Scatter(x=gbm_market.index, y=(start * (1 + gbm_market).cumprod())[column], name=None))
        fig.update_layout(title="Market (GBM)", showlegend=False)
        st.plotly_chart(fig)

        fig = go.Figure()
        for column in (start * (1 + gbm_risky).cumprod()).iloc[:, :n].columns:
            fig.add_trace(go.Scatter(x=gbm_risky.index, y=(start * (1 + gbm_risky).cumprod())[column], name=None))
        fig.update_layout(title="Risky (GBM)", showlegend=False)
        st.plotly_chart(fig)

        fig = go.Figure()
        for column in (start * (1 + cir_safe).cumprod()).iloc[:, :n].columns:
            fig.add_trace(go.Scatter(x=cir_safe.index, y=(start * (1 + cir_safe).cumprod())[column], name=None))
        fig.update_layout(title="Safe (CIR)", showlegend=False)
        st.plotly_chart(fig)
    else:
        st.write("No data to show...sth went wrong.")

with tab3:
    real_safe = ut.load_returns()['safe'][-n_years*12:]
    risk_free_rate = ((1 + real_safe).prod() ** (12 / len(real_safe)) - 1).iloc[0]
    level = 1 - st.number_input("Level of Confidence", 0.8, 0.999, 0.95, step=0.005)
    if data_source == 'Real Data (SPHD & T-bill)': # Real Data
        if "Max Drawdown Allocation" in allocation_strategy:
            tmp = pd.concat([returns_maxdd, real_market], axis=1)
            tmp.columns = ['MaxDD Portfolio', 'Market']
            st.write("Summary Statistics on Returns")
            st.write(ut.summary_statistics(tmp, riskfree_rate=risk_free_rate, level=level))
    elif data_source == 'Simulated Data (GBM & CIR)':  # Simulated Data
        combined_summary = []
        if "Max Drawdown Allocation" in allocation_strategy:
            sim_maxdd_sum = ut.summary_statistics(sim_returns_maxdd, riskfree_rate=risk_free_rate, level=level).mean(axis=0)
            sim_maxdd_sum = pd.DataFrame(sim_maxdd_sum).T
            sim_maxdd_sum.index = ["Max Drawdown Allocation"]
            combined_summary.append(sim_maxdd_sum)
        if "Floor Allocation" in allocation_strategy:
            sim_floor_sum = ut.summary_statistics(returns_floor, riskfree_rate=risk_free_rate, level=level).mean(axis=0)
            sim_floor_sum = pd.DataFrame(sim_floor_sum).T
            sim_floor_sum.index = ["Floor Allocation"]
            combined_summary.append(sim_floor_sum)
            sim_floor_ter = ut.terminal_stats(returns_floor, floor=floor, name='Floor Allocation')

        final_summary = pd.concat(combined_summary) if combined_summary else None
        if final_summary is not None:
            st.write('Averaged Summary Statistics for Simulated Returns')
            st.write(final_summary)
        else:
            st.error("No allocation strategy selected.")
        if "Floor Allocation" in allocation_strategy:
            st.write('Summary Statistics of Terminal Values for Floor Allocation (calculated per $1 invested, multiply by your initial wealth for absolute values)')
            st.write(sim_floor_ter.T)
    else:
        st.write("No data to show...sth went wrong.")