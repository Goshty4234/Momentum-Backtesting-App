import streamlit as st
import datetime
from datetime import timedelta, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import json

# Initialize page-specific session state for Strategy Comparison page
if 'strategy_comparison_page_initialized' not in st.session_state:
    st.session_state.strategy_comparison_page_initialized = True
    # Initialize global ticker configuration (shared across all portfolios)
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
    ]

# Initialize portfolio configs only if they don't exist
if 'strategy_comparison_portfolio_configs' not in st.session_state:
    st.session_state.strategy_comparison_portfolio_configs = [
        # 1) Equal weight portfolio (no momentum) - baseline strategy
        {
            'name': 'Equal Weight (Baseline)',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'use_momentum': False,
            'use_relative_momentum': False,
            'equal_if_all_negative': False,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
        },
        # 2) Momentum-based portfolio using SPY, QQQ, GLD, TLT
        {
            'name': 'Momentum Strategy',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'use_momentum': True,
            'use_relative_momentum': True,
            'equal_if_all_negative': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {'lookback': 365, 'exclude': 30, 'weight': 0.5},
                {'lookback': 180, 'exclude': 30, 'weight': 0.3},
                {'lookback': 120, 'exclude': 30, 'weight': 0.2},
            ],
            'calc_beta': True,
            'calc_volatility': True,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
        },
        # 3) Conservative portfolio (no momentum, no beta/volatility)
        {
            'name': 'Conservative (No Momentum)',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
            'added_amount': 10000,
            'added_frequency': 'Annually',
            'rebalancing_frequency': 'Annually',
            'start_date_user': None,
            'end_date_user': None,
            'start_with': 'all',
            'use_momentum': False,
            'use_relative_momentum': False,
            'equal_if_all_negative': False,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [],
            'calc_beta': False,
            'calc_volatility': False,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
        },
    ]
    st.session_state.strategy_comparison_active_portfolio_index = 0
    st.session_state.strategy_comparison_rerun_flag = False
    # Portfolio selection will be initialized when the selector is first rendered

# Initialize other session state variables if they don't exist
if 'strategy_comparison_active_portfolio_index' not in st.session_state:
    st.session_state.strategy_comparison_active_portfolio_index = 0
if 'strategy_comparison_rerun_flag' not in st.session_state:
    st.session_state.strategy_comparison_rerun_flag = False

st.set_page_config(layout="wide", page_title="Strategy Comparison", page_icon="📈")
st.markdown("""
<style>
    /* Global Styles for the App */
    .st-emotion-cache-1f87s81 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1v0bb62 button {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-1v0bb62 button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        transform: translateY(-2px);
    }
    /* Fix for the scrollable dataframe - forces it to be non-scrollable */
    div.st-emotion-cache-1ftv8z > div {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Make the 'View Details' button more obvious */
    button[aria-label="View Details"] {
        background-color: #0ea5e9 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 4px 8px rgba(14,165,233,0.16) !important;
    }
    button[aria-label="View Details"]:hover {
        background-color: #0891b2 !important;
    }
</style>
<a id="top"></a>
<button id="back-to-top" onclick="window.scrollTo(0, 0);">⬆️</button>
<style>
    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        opacity: 0.7;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: opacity 0.3s;
    }
    #back-to-top:hover {
        opacity: 1;
    }
</style>
<script>
    window.onscroll = function() {
        var button = document.getElementById("back-to-top");
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    };
</script>
""", unsafe_allow_html=True)



# ...existing code...

# Place rerun logic after first portfolio input widget
active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] if 'strategy_comparison_portfolio_configs' in st.session_state and 'strategy_comparison_active_portfolio_index' in st.session_state else None
if active_portfolio:
    ## Removed duplicate Portfolio Name input field
    if st.session_state.get('strategy_comparison_rerun_flag', False):
        st.session_state.strategy_comparison_rerun_flag = False
        st.rerun()
import numpy as np
import pandas as pd
def calculate_mwrr(values, cash_flows, dates):
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr * 100
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
    # Exact logic from app.py for MWRR calculation
    try:
        from scipy.optimize import brentq
        values = pd.Series(values).dropna()
        flows = pd.Series(cash_flows).reindex(values.index, fill_value=0.0)
        if len(values) < 2:
            return np.nan
        dates = pd.to_datetime(values.index)
        start_date = dates[0]
        time_periods = np.array([(d - start_date).days / 365.25 for d in dates])
        initial_investment = -values.iloc[0]
        significant_flows = flows[flows != 0]
        cash_flow_dates = [start_date]
        cash_flow_amounts = [initial_investment]
        cash_flow_times = [0.0]
        for date, flow in significant_flows.items():
            if date != dates[0] and date != dates[-1]:
                cash_flow_dates.append(pd.to_datetime(date))
                cash_flow_amounts.append(flow)
                cash_flow_times.append((pd.to_datetime(date) - start_date).days / 365.25)
        cash_flow_dates.append(dates[-1])
        cash_flow_amounts.append(values.iloc[-1])
        cash_flow_times.append((dates[-1] - start_date).days / 365.25)
        cash_flow_amounts = np.array(cash_flow_amounts)
        cash_flow_times = np.array(cash_flow_times)
        def npv(rate):
            return np.sum(cash_flow_amounts / (1 + rate) ** cash_flow_times)
        try:
            irr = brentq(npv, -0.999, 10)
            return irr
        except (ValueError, RuntimeError):
            return np.nan
    except Exception:
        return np.nan
# Backtest_Engine.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import contextlib
import json
from datetime import datetime, timedelta
from warnings import warn
from scipy.optimize import newton, brentq, root_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Custom CSS for a better layout, a distinct primary button, and the fixed 'Back to Top' button
st.markdown("""
<style>
    /* Global Styles for the App */
    .st-emotion-cache-1f87s81 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1v0bb62 button {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-1v0bb62 button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        transform: translateY(-2px);
    }
    /* Fix for the scrollable dataframe - forces it to be non-scrollable */
    div.st-emotion-cache-1ftv8z > div {
        overflow: visible !important;
        max-height: none !important;
    }
</style>
<a id="top"></a>
<button id="back-to-top" onclick="window.scrollTo(0, 0);">⬆️</button>
<style>
    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        opacity: 0.7;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 24px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: opacity 0.3s;
    }
    #back-to-top:hover {
        opacity: 1;
    }
</style>
<script>
    window.onscroll = function() {
        var button = document.getElementById("back-to-top");
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    };
</script>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Strategy Comparison")

st.title("Strategy Comparison")
st.markdown("Use the forms below to configure and run backtests for multiple portfolios.")

# Portfolio name is handled in the main UI below

# -----------------------
# Default JSON configs (for initialization)
# -----------------------
default_configs = [
    # 1) Benchmark only (SPY) - yearly rebalancing and yearly additions
    {
        'name': 'Benchmark Only (SPY)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 1.0, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
    'added_amount': 10000,
    'added_frequency': 'Annually',
    'rebalancing_frequency': 'Annually',
    'start_date_user': None,
    'end_date_user': None,
    'start_with': 'all',
    'use_momentum': False,
        'use_relative_momentum': False,
        'equal_if_all_negative': False,
        'momentum_windows': [],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 2) Momentum-based portfolio using SPY, QQQ, GLD, TLT
    {
        'name': 'Momentum-Based Portfolio',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'year',
    'rebalancing_frequency': 'year',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'all',
        'use_momentum': True,
        'use_relative_momentum': True,
        'equal_if_all_negative': True,
        'momentum_strategy': 'Classic',
        'negative_momentum_strategy': 'Cash',
        'momentum_windows': [
            {'lookback': 365, 'exclude': 30, 'weight': 0.5},
            {'lookback': 180, 'exclude': 30, 'weight': 0.3},
            {'lookback': 120, 'exclude': 30, 'weight': 0.2},
        ],
        'calc_beta': True,
        'calc_volatility': True,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
    # 3) Equal weight (No Momentum) using the same tickers
    {
        'name': 'Equal Weight Portfolio (No Momentum)',
        'stocks': [
            {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
            {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
        ],
        'benchmark_ticker': '^GSPC',
        'initial_value': 10000,
        'added_amount': 10000,
        'added_frequency': 'year',
    'rebalancing_frequency': 'year',
        'start_date_user': None,
        'end_date_user': None,
        'start_with': 'all',
        'use_momentum': False,
        'use_relative_momentum': False,
        'equal_if_all_negative': False,
        'momentum_windows': [],
        'calc_beta': False,
        'calc_volatility': False,
        'beta_window_days': 365,
        'exclude_days_beta': 30,
        'vol_window_days': 365,
        'exclude_days_vol': 30,
    },
]

# -----------------------
# Helper functions
# -----------------------
def get_trading_days(start_date, end_date):
    return pd.bdate_range(start=start_date, end=end_date)

def get_dates_by_freq(freq, start, end, market_days):
    market_days = sorted(market_days)
    
    # Ensure market_days are timezone-naive for consistent comparison
    market_days_naive = [d.tz_localize(None) if d.tz is not None else d for d in market_days]
    
    if freq == "market_day":
        return set(market_days)
    elif freq == "calendar_day":
        return set(pd.date_range(start=start, end=end, freq='D'))
    elif freq == "Weekly":
        base = pd.date_range(start=start, end=end, freq='W-MON')
    elif freq == "Biweekly":
        base = pd.date_range(start=start, end=end, freq='2W-MON')
    elif freq == "Monthly":
        base = pd.date_range(start=start, end=end, freq='MS')
    elif freq == "Quarterly":
        base = pd.date_range(start=start, end=end, freq='3MS')
    elif freq == "Semiannually":
        # First day of Jan and Jul each year
        semi = []
        for y in range(start.year, end.year + 1):
            for m in [1, 7]:
                semi.append(pd.Timestamp(year=y, month=m, day=1))
        base = pd.DatetimeIndex(semi)
    elif freq == "Annually" or freq == "year":
        base = pd.date_range(start=start, end=end, freq='YS')
    elif freq == "Never" or freq == "none" or freq is None:
        return set()
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    dates = []
    for d in base:
        # Ensure d is timezone-naive for comparison
        d_naive = d.tz_localize(None) if d.tz is not None else d
        idx = np.searchsorted(market_days_naive, d_naive, side='right')
        if idx > 0 and market_days_naive[idx-1] >= d_naive:
            dates.append(market_days[idx-1])  # Use original market_days for return
        elif idx < len(market_days_naive):
            dates.append(market_days[idx])  # Use original market_days for return
    return set(dates)

def calculate_cagr(values, dates):
    if len(values) < 2:
        return np.nan
    start_val = values[0]
    end_val = values[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or start_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / np.where(peak == 0, 1, peak)
    return np.nanmin(drawdowns), drawdowns

def calculate_volatility(returns):
    # Annualized volatility
    return np.std(returns) * np.sqrt(252) if len(returns) > 1 else np.nan

def calculate_beta(returns, benchmark_returns):
    # Use exact logic from app.py
    portfolio_returns = pd.Series(returns)
    benchmark_returns = pd.Series(benchmark_returns)
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return np.nan
    pr = portfolio_returns.reindex(common_idx).dropna()
    br = benchmark_returns.reindex(common_idx).dropna()
    # Re-align after dropping NAs
    common_idx = pr.index.intersection(br.index)
    if len(common_idx) < 2 or br.loc[common_idx].var() == 0:
        return np.nan
    cov = pr.loc[common_idx].cov(br.loc[common_idx])
    var = br.loc[common_idx].var()
    return cov / var

# FIXED: Correct Sortino Ratio calculation
def calculate_sortino(returns, risk_free_rate=0):
    # Annualized Sortino ratio
    target_return = risk_free_rate / 252  # Daily target
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return np.nan
    downside_std = np.std(downside_returns) * np.sqrt(252)
    if downside_std == 0:
        return np.nan
    expected_return = returns.mean() * 252
    return (expected_return - risk_free_rate) / downside_std

# FIXED: Correct Ulcer Index calculation
def calculate_ulcer_index(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    peak[peak == 0] = 1 # Avoid division by zero
    drawdown_sq = ((values - peak) / peak)**2
    return np.sqrt(np.mean(drawdown_sq)) if len(drawdown_sq) > 0 else np.nan

# FIXED: Correct UPI calculation
def calculate_upi(cagr, ulcer_index, risk_free_rate=0):
    if pd.isna(cagr) or pd.isna(ulcer_index) or ulcer_index == 0:
        return np.nan
    return (cagr - risk_free_rate) / ulcer_index

# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data):
    stocks_list = config['stocks']
    tickers = [s['ticker'] for s in stocks_list if s['ticker']]
    # Filter tickers to those present in reindexed_data to avoid KeyErrors for invalid tickers
    available_tickers = [t for t in tickers if t in reindexed_data]
    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
        print(f"[WARN] The following tickers were not found in price data and will be ignored: {sorted(list(missing))}")
    tickers = available_tickers
    # Recompute allocations and include_dividends to only include valid tickers
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s['ticker']
            allocation = s.get('allocation', 0)
            include_div = s.get('include_dividends', False)
            
            if ticker in allocations:
                # If ticker already exists, add the allocation
                allocations[ticker] += allocation
                # For include_dividends, use True if any instance has it True
                include_dividends[ticker] = include_dividends[ticker] or include_div
            else:
                # First occurrence of this ticker
                allocations[ticker] = allocation
                include_dividends[ticker] = include_div
    
    # Update tickers to only include unique tickers after deduplication
    tickers = list(allocations.keys())
    benchmark_ticker = config['benchmark_ticker']
    initial_value = config.get('initial_value', 0)
    added_amount = config.get('added_amount', 0)
    added_frequency = config.get('added_frequency', 'none')
    rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    use_relative_momentum = config.get('use_relative_momentum', False)
    equal_if_all_negative = config.get('equal_if_all_negative', True)
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    dates_added = get_dates_by_freq(added_frequency, sim_index[0], sim_index[-1], sim_index)
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows):
        cumulative_returns, valid_assets = {}, []
        filtered_windows = [w for w in momentum_windows if w["weight"] > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w["weight"] for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w["weight"] / total_weight for w in filtered_windows]
        start_dates_config = {t: reindexed_data[t].first_valid_index() for t in tickers if t in reindexed_data}
        for t in current_assets:
            is_valid, asset_returns = True, 0.0
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window["lookback"], window["exclude"]
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                if start_dates_config.get(t, pd.Timestamp.max) > start_mom:
                    is_valid = False; break
                df_t = current_data[t]
                price_start_index = df_t.index.asof(start_mom)
                price_end_index = df_t.index.asof(end_mom)
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False; break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False; break
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        # Mirror approach used in allocations/app.py: compute weights from raw momentum
        # (Classic or Relative) and then optionally post-filter by inverse volatility
        # and inverse absolute beta (multiplicative), then renormalize. This avoids
        # dividing by beta directly which flips signs when beta is negative.
        if not valid_assets:
            return {}, {}
        # Keep only non-nan momentum values
        rets = {t: returns.get(t, np.nan) for t in valid_assets}
        rets = {t: rets[t] for t in rets if not pd.isna(rets[t])}
        if not rets:
            return {}, {}

        metrics = {t: {} for t in rets.keys()}

        # compute beta and volatility metrics when requested
        beta_vals = {}
        vol_vals = {}
        df_bench = current_data.get(benchmark_ticker)
        if calc_beta:
            start_beta = date - pd.Timedelta(days=beta_window_days)
            end_beta = date - pd.Timedelta(days=exclude_days_beta)
        if calc_volatility:
            start_vol = date - pd.Timedelta(days=vol_window_days)
            end_vol = date - pd.Timedelta(days=exclude_days_vol)

        for t in list(rets.keys()):
            df_t = current_data.get(t)
            if calc_beta and df_bench is not None and isinstance(df_t, pd.DataFrame):
                mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                returns_t_beta = df_t.loc[mask_beta, 'Price_change']
                mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                returns_bench_beta = df_bench.loc[mask_bench_beta, 'Price_change']
                if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                    beta_vals[t] = np.nan
                else:
                    variance = np.var(returns_bench_beta)
                    beta_vals[t] = (np.cov(returns_t_beta, returns_bench_beta)[0,1] / variance) if variance > 0 else np.nan
                metrics[t]['Beta'] = beta_vals[t]
            if calc_volatility and isinstance(df_t, pd.DataFrame):
                mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                returns_t_vol = df_t.loc[mask_vol, 'Price_change']
                if len(returns_t_vol) < 2:
                    vol_vals[t] = np.nan
                else:
                    vol_vals[t] = returns_t_vol.std() * np.sqrt(252)
                metrics[t]['Volatility'] = vol_vals[t]

        # attach raw momentum
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Build initial weights from raw momentum (Classic or Relative)
        weights = {}
        rets_keys = list(rets.keys())
        all_negative = all(rets[t] <= 0 for t in rets_keys)
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0.0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1.0 / len(rets_keys) for t in rets_keys}
            else:  # Relative momentum
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets[t] for t in rets_keys)
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: rets[t] for t in rets_keys if rets[t] > 0}
                if positive_scores:
                    ssum = sum(positive_scores.values())
                    weights = {t: (positive_scores.get(t, 0.0) / ssum) for t in rets_keys}
                else:
                    weights = {t: 0.0 for t in rets_keys}

        # Post-filtering: multiply weights by inverse vol and inverse |beta| when requested
        if (calc_volatility or calc_beta) and weights:
            filter_scores = {}
            for t in weights:
                score = 1.0
                if calc_volatility:
                    v = metrics.get(t, {}).get('Volatility', np.nan)
                    if not pd.isna(v) and v > 0:
                        score *= 1.0 / v
                if calc_beta:
                    b = metrics.get(t, {}).get('Beta', np.nan)
                    if not pd.isna(b) and b != 0:
                        score *= 1.0 / abs(b)
                filter_scores[t] = score

            filtered = {t: weights.get(t, 0.0) * filter_scores.get(t, 1.0) for t in weights}
            ssum = sum(filtered.values())
            # If filtering removes all weight (sum==0), fall back to unfiltered weights
            if ssum > 0:
                weights = {t: filtered[t] / ssum for t in filtered}

        # Attach calculated weights to metrics and return
        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0.0)

        # Debug print when beta/vol are used
        if calc_beta or calc_volatility:
            try:
                for t in rets_keys:
                    print(f"[MOM DEBUG] Date: {date} | Ticker: {t} | Momentum: {metrics[t].get('Momentum')} | Beta: {metrics[t].get('Beta')} | Vol: {metrics[t].get('Volatility')} | Weight: {metrics[t].get('Calculated_Weight')}")
            except Exception:
                pass

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        current_allocations = {t: allocations.get(t,0) for t in tickers}
    else:
        returns, valid_assets = calculate_momentum(sim_index[0], set(tickers), momentum_windows)
        current_allocations, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=sim_index[0],
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        historical_metrics[sim_index[0]] = metrics_on_rebal
    
    sum_alloc = sum(current_allocations.get(t,0) for t in tickers)
    if sum_alloc > 0:
        for t in tickers:
            values[t][0] = initial_value * current_allocations.get(t,0) / sum_alloc
        unallocated_cash[0] = 0
    else:
        unallocated_cash[0] = initial_value
    
    historical_allocations[sim_index[0]] = {t: values[t][0] / initial_value if initial_value > 0 else 0 for t in tickers}
    historical_allocations[sim_index[0]]['CASH'] = unallocated_cash[0] / initial_value if initial_value > 0 else 0
    
    for i in range(len(sim_index)):
        date = sim_index[i]
        if i == 0: continue
        
        date_prev = sim_index[i-1]
        total_unreinvested_dividends = 0
        total_portfolio_prev = sum(values[t][-1] for t in tickers) + unreinvested_cash[-1]
        daily_growth_factor = 1
        if total_portfolio_prev > 0:
            total_portfolio_current_before_changes = 0
            for t in tickers:
                df = reindexed_data[t]
                price_prev = df.loc[date_prev, "Close"]
                val_prev = values[t][-1]
                nb_shares = val_prev / price_prev if price_prev > 0 else 0
                # --- Dividend fix: find the correct trading day for dividend ---
                div = 0.0
                if "Dividends" in df.columns:
                    # If dividend is not on a trading day, roll forward to next available trading day
                    if date in df.index:
                        div = df.loc[date, "Dividends"]
                    else:
                        # Find next trading day in index after 'date'
                        future_dates = df.index[df.index > date]
                        if len(future_dates) > 0:
                            div = df.loc[future_dates[0], "Dividends"]
                var = df.loc[date, "Price_change"] if date in df.index else 0.0
                if include_dividends.get(t, False):
                    rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                    val_new = val_prev * (1 + rate_of_return)
                else:
                    val_new = val_prev * (1 + var)
                    # If dividends are not included, do NOT add to unreinvested cash or anywhere else
                total_portfolio_current_before_changes += val_new
            total_portfolio_current_before_changes += unreinvested_cash[-1] + total_unreinvested_dividends
            daily_growth_factor = total_portfolio_current_before_changes / total_portfolio_prev
        for t in tickers:
            df = reindexed_data[t]
            price_prev = df.loc[date_prev, "Close"]
            val_prev = values[t][-1]
            # --- Dividend fix: find the correct trading day for dividend ---
            div = 0.0
            if "Dividends" in df.columns:
                if date in df.index:
                    div = df.loc[date, "Dividends"]
                else:
                    future_dates = df.index[df.index > date]
                    if len(future_dates) > 0:
                        div = df.loc[future_dates[0], "Dividends"]
            var = df.loc[date, "Price_change"] if date in df.index else 0.0
            if include_dividends.get(t, False):
                rate_of_return = var + (div / price_prev if price_prev > 0 else 0)
                val_new = val_prev * (1 + rate_of_return)
            else:
                val_new = val_prev * (1 + var)
            values[t].append(val_new)
        unallocated_cash.append(unallocated_cash[-1])
        if date in dates_added:
            unallocated_cash[-1] += added_amount
        unreinvested_cash.append(unreinvested_cash[-1] + total_unreinvested_dividends)
        portfolio_no_additions.append(portfolio_no_additions[-1] * daily_growth_factor)
        
        current_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
        if date in dates_rebal and set(tickers):
            if use_momentum:
                returns, valid_assets = calculate_momentum(date, set(tickers), momentum_windows)
                if valid_assets:
                    weights, metrics_on_rebal = calculate_momentum_weights(
                        returns, valid_assets, date=date,
                        momentum_strategy=config.get('momentum_strategy', 'Classic'),
                        negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
                    )
                    historical_metrics[date] = metrics_on_rebal
                    if all(w == 0 for w in weights.values()):
                        # All cash: move total to unallocated_cash, set asset values to zero
                        for t in tickers:
                            values[t][-1] = 0
                        unallocated_cash[-1] = current_total
                        unreinvested_cash[-1] = 0
                    else:
                        for t in tickers:
                            values[t][-1] = current_total * weights.get(t, 0)
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
            else:
                sum_alloc = sum(allocations.get(t,0) for t in tickers)
                if sum_alloc > 0:
                    for t in tickers:
                        weight = allocations.get(t,0)/sum_alloc
                        values[t][-1] = current_total * weight
                    unreinvested_cash[-1] = 0
                    unallocated_cash[-1] = 0
            
            # Store allocations at rebalancing date
            current_total_after_rebal = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
            if current_total_after_rebal > 0:
                allocs = {t: values[t][-1] / current_total_after_rebal for t in tickers}
                allocs['CASH'] = unallocated_cash[-1] / current_total_after_rebal if current_total_after_rebal > 0 else 0
                historical_allocations[date] = allocs
            else:
                allocs = {t: 0 for t in tickers}
                allocs['CASH'] = 0
                historical_allocations[date] = allocs

    # Store last allocation
    last_date = sim_index[-1]
    last_total = sum(values[t][-1] for t in tickers) + unallocated_cash[-1] + unreinvested_cash[-1]
    if last_total > 0:
        historical_allocations[last_date] = {t: values[t][-1] / last_total for t in tickers}
        historical_allocations[last_date]['CASH'] = unallocated_cash[-1] / last_total if last_total > 0 else 0
    else:
        historical_allocations[last_date] = {t: 0 for t in tickers}
        historical_allocations[last_date]['CASH'] = 0
    
    # Store last metrics: always add a last-rebalance snapshot so the UI has a metrics row
    # If momentum is used, compute metrics; otherwise build metrics from the last allocation snapshot
    if use_momentum:
        returns, valid_assets = calculate_momentum(last_date, set(tickers), momentum_windows)
        weights, metrics_on_rebal = calculate_momentum_weights(
            returns, valid_assets, date=last_date,
            momentum_strategy=config.get('momentum_strategy', 'Classic'),
            negative_momentum_strategy=config.get('negative_momentum_strategy', 'Cash')
        )
        # Add CASH line to metrics
        cash_weight = 1.0 if all(w == 0 for w in weights.values()) else 0.0
        metrics_on_rebal['CASH'] = {'Calculated_Weight': cash_weight}
        historical_metrics[last_date] = metrics_on_rebal
    else:
        # Build a metrics snapshot from the last allocation so there's always a 'last rebalance' metrics entry
        if last_date in historical_allocations:
            alloc_snapshot = historical_allocations.get(last_date, {})
            metrics_on_rebal = {}
            for ticker_sym, alloc_val in alloc_snapshot.items():
                metrics_on_rebal[ticker_sym] = {'Calculated_Weight': alloc_val}
            # Ensure CASH entry exists
            if 'CASH' not in metrics_on_rebal:
                metrics_on_rebal['CASH'] = {'Calculated_Weight': alloc_snapshot.get('CASH', 0)}
            # Only set if not already present
            if last_date not in historical_metrics:
                historical_metrics[last_date] = metrics_on_rebal

    results = pd.DataFrame(index=sim_index)
    for t in tickers:
        results[f"Value_{t}"] = values[t]
    results["Unallocated_cash"] = unallocated_cash
    results["Unreinvested_cash"] = unreinvested_cash
    results["Total_assets"] = results[[f"Value_{t}" for t in tickers]].sum(axis=1)
    results["Total_with_dividends_plus_cash"] = results["Total_assets"] + results["Unallocated_cash"] + results["Unreinvested_cash"]
    results['Portfolio_Value_No_Additions'] = portfolio_no_additions

    return results["Total_with_dividends_plus_cash"], results['Portfolio_Value_No_Additions'], historical_allocations, historical_metrics


# -----------------------
# PAGE-SCOPED SESSION STATE INITIALIZATION - STRATEGY COMPARISON PAGE
# -----------------------
# Ensure complete independence from other pages by using page-specific session keys
if 'strategy_comparison_page_initialized' not in st.session_state:
    st.session_state.strategy_comparison_page_initialized = True
    # Clear any shared session state that might interfere with other pages
    keys_to_clear = [
        # Main app keys
        'main_portfolio_configs', 'main_active_portfolio_index', 'main_rerun_flag',
        'main_all_results', 'main_all_allocations', 'main_all_metrics',
        'main_drawdowns', 'main_stats_df', 'main_years_data', 'main_portfolio_map',
        'main_backtest_ran', 'main_raw_data', 'main_running', 'main_run_requested',
        'main_pending_backtest_params', 'main_tickers', 'main_allocs', 'main_divs',
        'main_use_momentum', 'main_mom_windows', 'main_use_beta', 'main_use_vol',
        'main_initial_value_input_decimals', 'main_initial_value_input_int',
        'main_added_amount_input_decimals', 'main_added_amount_input_int',
        'main_start_date', 'main_end_date', 'main_use_custom_dates',
        'main_momentum_strategy', 'main_negative_momentum_strategy',
        'main_beta_window_days', 'main_beta_exclude_days', 'main_vol_window_days', 'main_vol_exclude_days',
        # Allocations page keys
        'alloc_portfolio_configs', 'alloc_active_portfolio_index', 'alloc_rerun_flag',
        'alloc_all_results', 'alloc_all_allocations', 'alloc_all_metrics',
        'alloc_paste_json_text', 'allocations_page_initialized',
        # Any other potential shared keys
        'strategy_comparison_all_results', 'strategy_comparison_all_allocations', 'strategy_comparison_all_metrics',
        'all_drawdowns', 'stats_df_display', 'all_years', 'portfolio_key_map',
        'strategy_comparison_ran', 'raw_data'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Main App Logic
# -----------------------

if 'strategy_comparison_portfolio_configs' not in st.session_state:
    st.session_state.strategy_comparison_portfolio_configs = default_configs
if 'strategy_comparison_active_portfolio_index' not in st.session_state:
    st.session_state.strategy_comparison_active_portfolio_index = 0
if 'strategy_comparison_paste_json_text' not in st.session_state:
    st.session_state.strategy_comparison_paste_json_text = ""
if 'strategy_comparison_rerun_flag' not in st.session_state:
    st.session_state.strategy_comparison_rerun_flag = False
if 'strategy_comparison_global_tickers' not in st.session_state:
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
    ]

# Sync global tickers to all portfolios on page load
def sync_global_tickers_to_all_portfolios():
    """Sync global tickers to all portfolios"""
    for portfolio in st.session_state.strategy_comparison_portfolio_configs:
        portfolio['stocks'] = st.session_state.strategy_comparison_global_tickers.copy()

# Initial sync
sync_global_tickers_to_all_portfolios()

# -----------------------
# Timer function for next rebalance date
# -----------------------
def calculate_next_rebalance_date(rebalancing_frequency, last_rebalance_date):
    """
    Calculate the next rebalance date based on rebalancing frequency and last rebalance date.
    Excludes today and yesterday as mentioned in the requirements.
    """
    if not last_rebalance_date or rebalancing_frequency == 'none':
        return None, None, None
    
    # Convert to datetime if it's a pandas Timestamp
    if hasattr(last_rebalance_date, 'to_pydatetime'):
        last_rebalance_date = last_rebalance_date.to_pydatetime()
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # If last rebalance was today or yesterday, use the day before yesterday as base
    if last_rebalance_date.date() >= yesterday:
        base_date = yesterday - timedelta(days=1)
    else:
        base_date = last_rebalance_date.date()
    
    # Calculate next rebalance date based on frequency
    if rebalancing_frequency == 'market_day':
        # Next market day (simplified - just next day for now)
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'calendar_day':
        next_date = base_date + timedelta(days=1)
    elif rebalancing_frequency == 'week':
        next_date = base_date + timedelta(weeks=1)
    elif rebalancing_frequency == '2weeks':
        next_date = base_date + timedelta(weeks=2)
    elif rebalancing_frequency == 'month':
        # Add one month
        if base_date.month == 12:
            next_date = base_date.replace(year=base_date.year + 1, month=1)
        else:
            next_date = base_date.replace(month=base_date.month + 1)
    elif rebalancing_frequency == '3months':
        # Add three months
        new_month = base_date.month + 3
        new_year = base_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        next_date = base_date.replace(year=new_year, month=new_month)
    elif rebalancing_frequency == '6months':
        # Add six months
        new_month = base_date.month + 6
        new_year = base_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        next_date = base_date.replace(year=new_year, month=new_month)
    elif rebalancing_frequency == 'year':
        next_date = base_date.replace(year=base_date.year + 1)
    else:
        return None, None, None
    
    # Calculate time until next rebalance
    now = datetime.now()
    # Ensure both datetimes are offset-naive for comparison and subtraction
    if hasattr(next_date, 'tzinfo') and next_date.tzinfo is not None:
        next_date = next_date.replace(tzinfo=None)
    next_rebalance_datetime = datetime.combine(next_date, time(9, 30))  # Assume 9:30 AM market open
    if hasattr(next_rebalance_datetime, 'tzinfo') and next_rebalance_datetime.tzinfo is not None:
        next_rebalance_datetime = next_rebalance_datetime.replace(tzinfo=None)
    if hasattr(now, 'tzinfo') and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    if next_rebalance_datetime <= now:
        # If next rebalance is in the past, calculate the next one
        if rebalancing_frequency == 'market_day' or rebalancing_frequency == 'calendar_day':
            next_rebalance_datetime = now + timedelta(days=1)
            next_date = next_rebalance_datetime.date()
        else:
            # For other frequencies, recalculate from the next date
            return calculate_next_rebalance_date(rebalancing_frequency, next_rebalance_datetime)
    time_until = next_rebalance_datetime - now
    
    return next_date, time_until, next_rebalance_datetime

def format_time_until(time_until):
    """Format the time until next rebalance in a human-readable format."""
    if not time_until:
        return "Unknown"
    
    total_seconds = int(time_until.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days} days, {hours} hours, {minutes} minutes"
    elif hours > 0:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"

def add_portfolio_callback():
    new_portfolio = default_configs[1].copy()
    new_portfolio['name'] = f"New Portfolio {len(st.session_state.strategy_comparison_portfolio_configs) + 1}"
    st.session_state.strategy_comparison_portfolio_configs.append(new_portfolio)
    st.session_state.strategy_comparison_active_portfolio_index = len(st.session_state.strategy_comparison_portfolio_configs) - 1
    st.session_state.strategy_comparison_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
        st.session_state.strategy_comparison_portfolio_configs.pop(st.session_state.strategy_comparison_active_portfolio_index)
        st.session_state.strategy_comparison_active_portfolio_index = max(0, st.session_state.strategy_comparison_active_portfolio_index - 1)
        st.session_state.strategy_comparison_rerun_flag = True

def add_stock_callback():
    st.session_state.strategy_comparison_global_tickers.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()
    # Don't trigger immediate re-run for better performance
    # st.session_state.strategy_comparison_rerun_flag = True

def remove_stock_callback(index):
    if len(st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks']) > 1:
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'].pop(index)
        # Don't trigger immediate re-run for better performance
        # st.session_state.strategy_comparison_rerun_flag = True

def normalize_stock_allocations_callback():
    if 'strategy_comparison_global_tickers' not in st.session_state:
        return
    stocks = st.session_state.strategy_comparison_global_tickers
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.strategy_comparison_global_tickers = stocks
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()
    st.session_state.strategy_comparison_rerun_flag = True

def equal_stock_allocation_callback():
    if 'strategy_comparison_global_tickers' not in st.session_state:
        return
    stocks = st.session_state.strategy_comparison_global_tickers
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"strategy_comparison_global_alloc_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.strategy_comparison_global_tickers = stocks
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()
    st.session_state.strategy_comparison_rerun_flag = True
    
def reset_portfolio_callback():
    current_name = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    # Clear any saved momentum settings when resetting
    if 'saved_momentum_settings' in default_cfg_found:
        del default_cfg_found['saved_momentum_settings']
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = default_cfg_found
    st.session_state.strategy_comparison_rerun_flag = True

def reset_stock_selection_callback():
    # Reset global tickers to default
    st.session_state.strategy_comparison_global_tickers = [
        {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
        {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
    ]
    # Sync to all portfolios
    sync_global_tickers_to_all_portfolios()
    st.session_state.strategy_comparison_rerun_flag = True

def reset_momentum_windows_callback():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    # Don't trigger immediate re-run for better performance
    # st.session_state.strategy_comparison_rerun_flag = True

def update_stock_allocation(index):
    try:
        key = f"strategy_comparison_alloc_input_{st.session_state.strategy_comparison_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return

def update_stock_ticker(index):
    try:
        key = f"strategy_comparison_ticker_{st.session_state.strategy_comparison_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            # key not yet initialized (race condition). Skip update; the widget's key will be present on next rerender.
            return
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'][index]['ticker'] = val
    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return

def update_stock_dividends(index):
    try:
        key = f"strategy_comparison_div_{st.session_state.strategy_comparison_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Global ticker management functions
def update_global_stock_ticker(index):
    try:
        key = f"strategy_comparison_global_ticker_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.strategy_comparison_global_tickers[index]['ticker'] = val
        # Sync to all portfolios
        sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def update_global_stock_allocation(index):
    try:
        key = f"strategy_comparison_global_alloc_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.strategy_comparison_global_tickers[index]['allocation'] = float(val) / 100.0
        # Sync to all portfolios
        sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def update_global_stock_dividends(index):
    try:
        key = f"strategy_comparison_global_div_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.strategy_comparison_global_tickers[index]['include_dividends'] = bool(val)
        # Sync to all portfolios
        sync_global_tickers_to_all_portfolios()
    except Exception:
        return

def remove_global_stock_callback(index):
    if len(st.session_state.strategy_comparison_global_tickers) > 1:
        st.session_state.strategy_comparison_global_tickers.pop(index)
        # Sync to all portfolios
        sync_global_tickers_to_all_portfolios()

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation
    idx = st.session_state.strategy_comparison_active_portfolio_index
    st.session_state.strategy_comparison_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.strategy_comparison_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.strategy_comparison_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['strategy_comparison_active_calc_beta'] = True
    # Update UI widget values to reflect reset
    st.session_state['strategy_comparison_active_beta_window'] = 365
    st.session_state['strategy_comparison_active_beta_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.strategy_comparison_active_portfolio_index
    st.session_state.strategy_comparison_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.strategy_comparison_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.strategy_comparison_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['strategy_comparison_active_calc_vol'] = True
    # Update UI widget values to reflect reset
    st.session_state['strategy_comparison_active_vol_window'] = 365
    st.session_state['strategy_comparison_active_vol_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def add_momentum_window_callback():
    # Append a new momentum window with modest defaults
    idx = st.session_state.strategy_comparison_active_portfolio_index
    cfg = st.session_state.strategy_comparison_portfolio_configs[idx]
    if 'momentum_windows' not in cfg:
        cfg['momentum_windows'] = []
    # default new window
    cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
    # Don't trigger immediate re-run for better performance
    # st.session_state.strategy_comparison_rerun_flag = True
    st.session_state.strategy_comparison_portfolio_configs[idx] = cfg
    # Don't trigger immediate re-run for better performance
    # st.session_state.strategy_comparison_rerun_flag = True

def remove_momentum_window_callback():
    idx = st.session_state.strategy_comparison_active_portfolio_index
    cfg = st.session_state.strategy_comparison_portfolio_configs[idx]
    if 'momentum_windows' in cfg and cfg['momentum_windows']:
        cfg['momentum_windows'].pop()
        st.session_state.strategy_comparison_portfolio_configs[idx] = cfg
        # Don't trigger immediate re-run for better performance
        # st.session_state.strategy_comparison_rerun_flag = True

def normalize_momentum_weights_callback():
    if 'strategy_comparison_portfolio_configs' not in st.session_state or 'strategy_comparison_active_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio['momentum_windows']):
            w['weight'] /= total_weight
            weight_key = f"strategy_comparison_weight_input_active_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'] = active_portfolio['momentum_windows']
    st.session_state.strategy_comparison_rerun_flag = True

def paste_json_callback():
    try:
        json_data = json.loads(st.session_state.strategy_comparison_paste_json_text)
        
        # Debug: Show what we received
        st.info(f"Received JSON keys: {list(json_data.keys())}")
        if 'tickers' in json_data:
            st.info(f"Tickers in JSON: {json_data['tickers']}")
        if 'stocks' in json_data:
            st.info(f"Stocks in JSON: {json_data['stocks']}")
        if 'momentum_windows' in json_data:
            st.info(f"Momentum windows in JSON: {json_data['momentum_windows']}")
        if 'use_momentum' in json_data:
            st.info(f"Use momentum in JSON: {json_data['use_momentum']}")
        
        # Handle momentum strategy value mapping from other pages
        momentum_strategy = json_data.get('momentum_strategy', 'Classic')
        if momentum_strategy == 'Classic momentum':
            momentum_strategy = 'Classic'
        elif momentum_strategy not in ['Classic', 'Relative']:
            momentum_strategy = 'Classic'  # Default fallback
        
        # Handle negative momentum strategy value mapping from other pages
        negative_momentum_strategy = json_data.get('negative_momentum_strategy', 'Cash')
        if negative_momentum_strategy == 'Go to cash':
            negative_momentum_strategy = 'Cash'
        elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
            negative_momentum_strategy = 'Cash'  # Default fallback
        
        # Handle stocks field - convert from legacy format if needed
        stocks = json_data.get('stocks', [])
        if not stocks and 'tickers' in json_data:
            # Convert legacy format (tickers, allocs, divs) to stocks format
            tickers = json_data.get('tickers', [])
            allocs = json_data.get('allocs', [])
            divs = json_data.get('divs', [])
            stocks = []
            
            # Ensure we have valid arrays
            if tickers and isinstance(tickers, list):
                for i in range(len(tickers)):
                    if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                        # Convert allocation from percentage (0-100) to decimal (0.0-1.0) format
                        allocation = 0.0
                        if i < len(allocs) and allocs[i] is not None:
                            alloc_value = float(allocs[i])
                            if alloc_value > 1.0:
                                # Already in percentage format, convert to decimal
                                allocation = alloc_value / 100.0
                            else:
                                # Already in decimal format, use as is
                                allocation = alloc_value
                        
                        stock = {
                            'ticker': tickers[i].strip(),
                            'allocation': allocation,
                            'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                        }
                        stocks.append(stock)
            
            # Debug output
            st.info(f"Converted {len(stocks)} stocks from legacy format: {[s['ticker'] for s in stocks]}")
        
        # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
        momentum_windows = json_data.get('momentum_windows', [])
        for window in momentum_windows:
            if 'weight' in window:
                weight = window['weight']
                # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                if isinstance(weight, (int, float)) and weight > 1.0:
                    # Cap at 100% and convert to decimal
                    weight = min(weight, 100.0) / 100.0
                elif isinstance(weight, (int, float)) and weight <= 1.0:
                    # Already in decimal format, ensure it's valid
                    weight = max(0.0, min(weight, 1.0))
                else:
                    # Invalid weight, set to default
                    weight = 0.1
                window['weight'] = weight
        
                        # Map frequency values from app.py format to Strategy Comparison format
        def map_frequency(freq):
            if freq is None:
                return 'Never'
            freq_map = {
                'Never': 'Never',
                'Weekly': 'Weekly',
                'Biweekly': 'Biweekly',
                'Monthly': 'Monthly',
                'Quarterly': 'Quarterly',
                'Semiannually': 'Semiannually',
                'Annually': 'Annually',
                # Legacy format mapping
                'none': 'Never',
                'week': 'Weekly',
                '2weeks': 'Biweekly',
                'month': 'Monthly',
                '3months': 'Quarterly',
                '6months': 'Semiannually',
                'year': 'Annually'
            }
            return freq_map.get(freq, 'Monthly')
        
                        # Strategy Comparison page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        strategy_comparison_config = {
            'name': json_data.get('name', 'New Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 1000),
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Monthly')),
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': json_data.get('start_date_user'),
            'end_date_user': json_data.get('end_date_user'),
                            'start_with': json_data.get('start_with', 'all'),
            'use_momentum': json_data.get('use_momentum', True),
            'use_relative_momentum': json_data.get('use_relative_momentum', False),
            'equal_if_all_negative': json_data.get('equal_if_all_negative', False),
            'momentum_strategy': momentum_strategy,
            'negative_momentum_strategy': negative_momentum_strategy,
            'momentum_windows': momentum_windows,
            'calc_beta': json_data.get('calc_beta', True),
            'calc_volatility': json_data.get('calc_volatility', True),
            'beta_window_days': json_data.get('beta_window_days', 365),
            'exclude_days_beta': json_data.get('exclude_days_beta', 30),
            'vol_window_days': json_data.get('vol_window_days', 365),
            'exclude_days_vol': json_data.get('exclude_days_vol', 30),
        }
        
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = strategy_comparison_config
        st.success("Portfolio configuration updated from JSON (Strategy Comparison page).")
        st.info(f"Final stocks list: {[s['ticker'] for s in strategy_comparison_config['stocks']]}")
        st.info(f"Final momentum windows: {strategy_comparison_config['momentum_windows']}")
        st.info(f"Final use_momentum: {strategy_comparison_config['use_momentum']}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.strategy_comparison_rerun_flag = True

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('strategy_comparison_paste_all_json_text', '')
    if not txt:
        st.warning('No JSON provided')
        return
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            # Process each portfolio configuration for Strategy Comparison page
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy not in ['Classic', 'Relative']:
                    momentum_strategy = 'Classic'  # Default fallback
                
                # Handle negative momentum strategy value mapping from other pages
                negative_momentum_strategy = cfg.get('negative_momentum_strategy', 'Cash')
                if negative_momentum_strategy == 'Go to cash':
                    negative_momentum_strategy = 'Cash'
                elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
                    negative_momentum_strategy = 'Cash'  # Default fallback
                
                # Handle stocks field - convert from legacy format if needed
                stocks = cfg.get('stocks', [])
                if not stocks and 'tickers' in cfg:
                    # Convert legacy format (tickers, allocs, divs) to stocks format
                    tickers = cfg.get('tickers', [])
                    allocs = cfg.get('allocs', [])
                    divs = cfg.get('divs', [])
                    stocks = []
                    
                    # Ensure we have valid arrays
                    if tickers and isinstance(tickers, list):
                        for i in range(len(tickers)):
                            if tickers[i] and tickers[i].strip():  # Check for non-empty ticker
                                stock = {
                                    'ticker': tickers[i].strip(),
                                    'allocation': float(allocs[i]) if i < len(allocs) and allocs[i] is not None else 0.0,
                                    'include_dividends': bool(divs[i]) if i < len(divs) and divs[i] is not None else True
                                }
                                stocks.append(stock)
                
                # Sanitize momentum window weights to prevent StreamlitValueAboveMaxError
                momentum_windows = cfg.get('momentum_windows', [])
                for window in momentum_windows:
                    if 'weight' in window:
                        weight = window['weight']
                        # If weight is a percentage (e.g., 50 for 50%), convert to decimal
                        if isinstance(weight, (int, float)) and weight > 1.0:
                            # Cap at 100% and convert to decimal
                            weight = min(weight, 100.0) / 100.0
                        elif isinstance(weight, (int, float)) and weight <= 1.0:
                            # Already in decimal format, ensure it's valid
                            weight = max(0.0, min(weight, 1.0))
                        else:
                            # Invalid weight, set to default
                            weight = 0.1
                        window['weight'] = weight
                
                # Debug: Show what we received for this portfolio
                if 'momentum_windows' in cfg:
                    st.info(f"Momentum windows for {cfg.get('name', 'Unknown')}: {cfg['momentum_windows']}")
                if 'use_momentum' in cfg:
                    st.info(f"Use momentum for {cfg.get('name', 'Unknown')}: {cfg['use_momentum']}")
                
                # Map frequency values from app.py format to Strategy Comparison format
                def map_frequency(freq):
                    if freq is None:
                        return 'Never'
                    freq_map = {
                        'Never': 'Never',
                        'Weekly': 'Weekly',
                        'Biweekly': 'Biweekly',
                        'Monthly': 'Monthly',
                        'Quarterly': 'Quarterly',
                        'Semiannually': 'Semiannually',
                        'Annually': 'Annually',
                        # Legacy format mapping
                        'none': 'Never',
                        'week': 'Weekly',
                        '2weeks': 'Biweekly',
                        'month': 'Monthly',
                        '3months': 'Quarterly',
                        '6months': 'Semiannually',
                        'year': 'Annually'
                    }
                    return freq_map.get(freq, 'Monthly')
                
                # Strategy Comparison page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                strategy_comparison_config = {
                    'name': cfg.get('name', 'New Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                    'added_amount': cfg.get('added_amount', 1000),
                    'added_frequency': map_frequency(cfg.get('added_frequency', 'Monthly')),
                    'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                    'start_date_user': cfg.get('start_date_user'),
                    'end_date_user': cfg.get('end_date_user'),
                    'start_with': cfg.get('start_with', 'all'),
                    'use_momentum': cfg.get('use_momentum', True),
                    'use_relative_momentum': cfg.get('use_relative_momentum', False),
                    'equal_if_all_negative': cfg.get('equal_if_all_negative', False),
                    'momentum_strategy': momentum_strategy,
                    'negative_momentum_strategy': negative_momentum_strategy,
                    'momentum_windows': momentum_windows,
                    'calc_beta': cfg.get('calc_beta', True),
                    'calc_volatility': cfg.get('calc_volatility', True),
                    'beta_window_days': cfg.get('beta_window_days', 365),
                    'exclude_days_beta': cfg.get('exclude_days_beta', 30),
                    'vol_window_days': cfg.get('vol_window_days', 365),
                    'exclude_days_vol': cfg.get('exclude_days_vol', 30),
                }
                processed_configs.append(strategy_comparison_config)
            
            st.session_state.strategy_comparison_portfolio_configs = processed_configs
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.strategy_comparison_active_portfolio_index = 0
                st.session_state.strategy_comparison_portfolio_selector = processed_configs[0].get('name', '')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['strategy_comparison_active_name'] = processed_configs[0].get('name', '')
                st.session_state['strategy_comparison_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['strategy_comparison_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['strategy_comparison_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'none')
                st.session_state['strategy_comparison_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['strategy_comparison_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['strategy_comparison_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
            else:
                st.session_state.strategy_comparison_active_portfolio_index = None
                st.session_state.strategy_comparison_portfolio_selector = ''
            st.session_state.strategy_comparison_portfolio_key_map = {}
            st.session_state.strategy_comparison_ran = False
            st.success('All portfolio configurations updated from JSON (Strategy Comparison page).')
            # Debug: Show final momentum windows for first portfolio
            if processed_configs:
                st.info(f"Final momentum windows for first portfolio: {processed_configs[0]['momentum_windows']}")
                st.info(f"Final use_momentum for first portfolio: {processed_configs[0]['use_momentum']}")
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.strategy_comparison_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')

def update_active_portfolio_index():
    # Use safe accessors to avoid AttributeError when keys are not yet set
    selected_name = st.session_state.get('strategy_comparison_portfolio_selector', None)
    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    if selected_name and selected_name in portfolio_names:
        st.session_state.strategy_comparison_active_portfolio_index = portfolio_names.index(selected_name)
    else:
        # default to first portfolio if selector is missing or value not found
        st.session_state.strategy_comparison_active_portfolio_index = 0 if portfolio_names else None
    st.session_state.strategy_comparison_rerun_flag = True

def update_name():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['name'] = st.session_state.strategy_comparison_active_name

def update_initial():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['initial_value'] = st.session_state.strategy_comparison_active_initial

def update_added_amount():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_amount'] = st.session_state.strategy_comparison_active_added_amount

def update_add_freq():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_frequency'] = st.session_state.strategy_comparison_active_add_freq

def update_rebal_freq():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['rebalancing_frequency'] = st.session_state.strategy_comparison_active_rebal_freq

def update_benchmark():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['benchmark_ticker'] = st.session_state.strategy_comparison_active_benchmark

def update_use_momentum():
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['use_momentum']
    new_val = st.session_state.strategy_comparison_active_use_momentum
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling momentum - restore saved settings or use defaults
            if 'saved_momentum_settings' in portfolio:
                # Restore previously saved momentum settings
                saved_settings = portfolio['saved_momentum_settings']
                portfolio['momentum_windows'] = saved_settings.get('momentum_windows', [
                    {"lookback": 365, "exclude": 30, "weight": 0.5},
                    {"lookback": 180, "exclude": 30, "weight": 0.3},
                    {"lookback": 120, "exclude": 30, "weight": 0.2},
                ])
                portfolio['momentum_strategy'] = saved_settings.get('momentum_strategy', 'Classic')
                portfolio['negative_momentum_strategy'] = saved_settings.get('negative_momentum_strategy', 'Cash')
                portfolio['use_relative_momentum'] = saved_settings.get('use_relative_momentum', False)
                portfolio['equal_if_all_negative'] = saved_settings.get('equal_if_all_negative', False)
                portfolio['calc_beta'] = saved_settings.get('calc_beta', True)
                portfolio['calc_volatility'] = saved_settings.get('calc_volatility', True)
                portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_momentum_strategy'] = portfolio['momentum_strategy']
                st.session_state['strategy_comparison_active_negative_momentum_strategy'] = portfolio['negative_momentum_strategy']
                st.session_state['strategy_comparison_active_rel_mom'] = portfolio['use_relative_momentum']
                st.session_state['strategy_comparison_active_equal_neg'] = portfolio['equal_if_all_negative']
                st.session_state['strategy_comparison_active_calc_beta'] = portfolio['calc_beta']
                st.session_state['strategy_comparison_active_calc_vol'] = portfolio['calc_volatility']
                st.session_state['strategy_comparison_active_beta_window'] = portfolio['beta_window_days']
                st.session_state['strategy_comparison_active_beta_exclude'] = portfolio['exclude_days_beta']
                st.session_state['strategy_comparison_active_vol_window'] = portfolio['vol_window_days']
                st.session_state['strategy_comparison_active_vol_exclude'] = portfolio['exclude_days_vol']
            else:
                # No saved settings, use defaults
                portfolio['momentum_windows'] = [
                    {"lookback": 365, "exclude": 30, "weight": 0.5},
                    {"lookback": 180, "exclude": 30, "weight": 0.3},
                    {"lookback": 120, "exclude": 30, "weight": 0.2},
                ]
        else:
            # Disabling momentum - save current settings before clearing
            saved_settings = {
                'momentum_windows': portfolio.get('momentum_windows', []),
                'momentum_strategy': portfolio.get('momentum_strategy', 'Classic'),
                'negative_momentum_strategy': portfolio.get('negative_momentum_strategy', 'Cash'),
                'use_relative_momentum': portfolio.get('use_relative_momentum', False),
                'equal_if_all_negative': portfolio.get('equal_if_all_negative', False),
                'calc_beta': portfolio.get('calc_beta', True),
                'calc_volatility': portfolio.get('calc_volatility', True),
                'beta_window_days': portfolio.get('beta_window_days', 365),
                'exclude_days_beta': portfolio.get('exclude_days_beta', 30),
                'vol_window_days': portfolio.get('vol_window_days', 365),
                'exclude_days_vol': portfolio.get('exclude_days_vol', 30),
            }
            portfolio['saved_momentum_settings'] = saved_settings
            portfolio['momentum_windows'] = []
        
        portfolio['use_momentum'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_rel_mom():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['use_relative_momentum'] = st.session_state.strategy_comparison_active_rel_mom

def update_equal_neg():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['equal_if_all_negative'] = st.session_state.strategy_comparison_active_equal_neg

def update_calc_beta():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_beta'] = st.session_state.strategy_comparison_active_calc_beta

def update_beta_window():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['beta_window_days'] = st.session_state.strategy_comparison_active_beta_window

def update_beta_exclude():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['exclude_days_beta'] = st.session_state.strategy_comparison_active_beta_exclude

def update_calc_vol():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_volatility'] = st.session_state.strategy_comparison_active_calc_vol

def update_vol_window():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['vol_window_days'] = st.session_state.strategy_comparison_active_vol_window

def update_vol_exclude():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['exclude_days_vol'] = st.session_state.strategy_comparison_active_vol_exclude

# Sidebar for portfolio selection
st.sidebar.title("Manage Portfolios")
portfolio_names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs]
selected_portfolio_name = st.sidebar.selectbox(
    "Select Portfolio",
    options=portfolio_names,
    index=st.session_state.strategy_comparison_active_portfolio_index,
    key="strategy_comparison_portfolio_selector",
    on_change=update_active_portfolio_index
)

active_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]

if st.sidebar.button("Add New Portfolio", on_click=add_portfolio_callback):
    pass
if len(st.session_state.strategy_comparison_portfolio_configs) > 1:
    if st.sidebar.button("Remove Selected Portfolio", on_click=remove_portfolio_callback):
        pass
if st.sidebar.button("Reset Selected Portfolio", on_click=reset_portfolio_callback):
    pass

# Global Ticker Management Section (moved to sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("Global Ticker Management")
st.sidebar.markdown("*All portfolios use the same tickers*")

# Stock management buttons
col_stock_buttons = st.sidebar.columns([1, 1])
with col_stock_buttons[0]:
    if st.sidebar.button("Normalize Stocks %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_stock_buttons[1]:
    if st.sidebar.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass

col_stock_buttons2 = st.sidebar.columns([1, 1])
with col_stock_buttons2[0]:
    if st.sidebar.button("Reset Stocks", on_click=reset_stock_selection_callback, use_container_width=True):
        pass
with col_stock_buttons2[1]:
    if st.sidebar.button("Add Stock", on_click=add_stock_callback, use_container_width=True):
        pass

# Calculate live total stock allocation for global tickers
valid_stocks = [s for s in st.session_state.strategy_comparison_global_tickers if s['ticker']]
total_stock_allocation = sum(s['allocation'] for s in valid_stocks)

# Always show allocation status (not hidden by momentum)
if abs(total_stock_allocation - 1.0) > 0.001:
    st.sidebar.warning(f"Total allocation: {total_stock_allocation*100:.1f}%")
else:
    st.sidebar.success(f"Total allocation: {total_stock_allocation*100:.1f}%")

# Stock inputs in sidebar (using global tickers) - Layout similar to app.py
for i in range(len(st.session_state.strategy_comparison_global_tickers)):
    stock = st.session_state.strategy_comparison_global_tickers[i]
    
    # Use columns to display ticker, allocation, dividends, and remove button on same line
    col1, col2, col3, col4 = st.sidebar.columns([1, 1, 1, 0.2])
    
    with col1:
        # Ticker input
        ticker_key = f"strategy_comparison_global_ticker_{i}"
        if ticker_key not in st.session_state:
            st.session_state[ticker_key] = stock['ticker']
        ticker_val = st.text_input(f"Ticker {i+1}", key=ticker_key, on_change=update_global_stock_ticker, args=(i,))
    
    with col2:
        # Allocation input (always visible)
        alloc_key = f"strategy_comparison_global_alloc_{i}"
        if alloc_key not in st.session_state:
            st.session_state[alloc_key] = int(stock['allocation'] * 100)
        alloc_val = st.number_input(f"Alloc % {i+1}", min_value=0, step=1, format="%d", key=alloc_key, on_change=update_global_stock_allocation, args=(i,))
    
    with col3:
        # Dividends checkbox
        div_key = f"strategy_comparison_global_div_{i}"
        if div_key not in st.session_state:
            st.session_state[div_key] = stock['include_dividends']
        div_val = st.checkbox("Dividends", key=div_key, on_change=update_global_stock_dividends, args=(i,))
    
    with col4:
        # Remove button
        if st.button("x", key=f"strategy_comparison_global_rem_{i}", on_click=remove_global_stock_callback, args=(i,), help="Remove this ticker"):
            pass

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Run Backtest button
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    # Pre-validation check for all portfolios
    configs_to_run = st.session_state.strategy_comparison_portfolio_configs
    valid_configs = True
    validation_errors = []
    
    for cfg in configs_to_run:
        if cfg['use_momentum']:
            total_momentum_weight = sum(w['weight'] for w in cfg['momentum_windows'])
            if abs(total_momentum_weight - 1.0) > (_TOTAL_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' has momentum enabled but the total momentum weight is {total_momentum_weight*100:.2f}% (must be 100%)")
                valid_configs = False
        else:
            valid_stocks_for_cfg = [s for s in cfg['stocks'] if s['ticker']]
            total_stock_allocation = sum(s['allocation'] for s in valid_stocks_for_cfg)
            if abs(total_stock_allocation - 1.0) > (_ALLOC_TOL / 100.0):
                validation_errors.append(f"Portfolio '{cfg['name']}' is not using momentum, but the total stock allocation is {total_stock_allocation*100:.2f}% (must be 100%)")
                valid_configs = False
                
    if not valid_configs:
        for error in validation_errors:
            st.error(error)
        # Don't set the run flag, but continue showing the UI
        pass
    else:
        st.session_state.strategy_comparison_run_backtest = True

# Start with option
st.sidebar.markdown("---")
st.sidebar.subheader("Data Options")
if "strategy_comparison_start_with" not in st.session_state:
    st.session_state.strategy_comparison_start_with = "all"
st.sidebar.radio(
    "How to handle assets with different start dates?",
    ["all", "oldest"],
    index=0 if st.session_state.strategy_comparison_start_with == "all" else 1,
    format_func=lambda x: "Start when ALL assets are available" if x == "all" else "Start with OLDEST asset",
    help="""
    **All:** Starts the backtest when all selected assets are available.
    **Oldest:** Starts at the oldest date of any asset and adds assets as they become available.
    """,
    key="strategy_comparison_start_with_radio",
    on_change=lambda: setattr(st.session_state, 'strategy_comparison_start_with', st.session_state.strategy_comparison_start_with_radio)
)

# JSON section for all portfolios
st.sidebar.markdown("---")
with st.sidebar.expander('All Portfolios JSON (Export / Import)', expanded=False):
    all_json = json.dumps(st.session_state.get('strategy_comparison_portfolio_configs', []), indent=2)
    st.code(all_json, language='json')
    import streamlit.components.v1 as components
    copy_html_all = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(all_json)});' style='margin-bottom:10px;'>Copy All Configs to Clipboard</button>
    """
    components.html(copy_html_all, height=40)
    st.text_area('Paste JSON Here to Replace All Portfolios', key='strategy_comparison_paste_all_json_text', height=240)
    st.button('Update All Portfolios from JSON', on_click=paste_all_json_callback)

st.header(f"Editing Portfolio: {active_portfolio['name']}")
# Ensure session-state key exists before creating widgets to avoid duplicate-default warnings
if "strategy_comparison_active_name" not in st.session_state:
    st.session_state["strategy_comparison_active_name"] = active_portfolio['name']
active_portfolio['name'] = st.text_input("Portfolio Name", key="strategy_comparison_active_name", on_change=update_name)

col_left, col_right = st.columns([1, 1])
with col_left:
    if "strategy_comparison_active_initial" not in st.session_state:
        st.session_state["strategy_comparison_active_initial"] = int(active_portfolio['initial_value'])
    st.number_input("Initial Value ($)", min_value=0, step=1000, format="%d", key="strategy_comparison_active_initial", on_change=update_initial, help="Starting cash", )
with col_right:
    if "strategy_comparison_active_added_amount" not in st.session_state:
        st.session_state["strategy_comparison_active_added_amount"] = int(active_portfolio['added_amount'])
    st.number_input("Added Amount ($)", min_value=0, step=1000, format="%d", key="strategy_comparison_active_added_amount", on_change=update_added_amount, help="Amount added at each Added Frequency")

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "strategy_comparison_active_rebal_freq" not in st.session_state:
        st.session_state["strategy_comparison_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="strategy_comparison_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced.", )
with col_freq_add:
    if "strategy_comparison_active_add_freq" not in st.session_state:
        st.session_state["strategy_comparison_active_add_freq"] = active_portfolio['added_frequency']
    st.selectbox("Added Frequency", freq_options, key="strategy_comparison_active_add_freq", on_change=update_add_freq, help="How often cash is added to the portfolio.")

with st.expander("Rebalancing and Added Frequency Explained", expanded=False):
    st.markdown("""
    **Added Frequency** is the frequency at which cash is added to the portfolio.
    
    **Rebalancing Frequency** is the frequency at which the portfolio is rebalanced to the specified allocations. It is also at this date that any additional cash from the `Added Frequency` is invested into the portfolio.
    
    *Keeping a Rebalancing Frequency to "none" will mean no additional cash is invested, even if you have an `Added Frequency` specified.*
    """)

if "strategy_comparison_active_benchmark" not in st.session_state:
    st.session_state["strategy_comparison_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker", key="strategy_comparison_active_benchmark", on_change=update_benchmark)

st.subheader("Strategy")
if "strategy_comparison_active_use_momentum" not in st.session_state:
    st.session_state["strategy_comparison_active_use_momentum"] = active_portfolio['use_momentum']
st.checkbox("Use Momentum Strategy", key="strategy_comparison_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")

if st.session_state.get('strategy_comparison_active_use_momentum', active_portfolio.get('use_momentum', True)):
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative"],
            index=["Classic", "Relative"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=f"strategy_comparison_momentum_strategy_{st.session_state.strategy_comparison_active_portfolio_index}"
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=f"strategy_comparison_negative_momentum_strategy_{st.session_state.strategy_comparison_active_portfolio_index}"
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "strategy_comparison_active_calc_beta" not in st.session_state:
            st.session_state["strategy_comparison_active_calc_beta"] = active_portfolio['calc_beta']
        st.checkbox("Include Beta in momentum weighting", key="strategy_comparison_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        # Reset Beta button
        if st.button("Reset Beta", key=f"strategy_comparison_reset_beta_btn_{st.session_state.strategy_comparison_active_portfolio_index}", on_click=reset_beta_callback):
            pass
        if st.session_state.get('strategy_comparison_active_calc_beta', False):
            if "strategy_comparison_active_beta_window" not in st.session_state:
                st.session_state["strategy_comparison_active_beta_window"] = active_portfolio['beta_window_days']
            if "strategy_comparison_active_beta_exclude" not in st.session_state:
                st.session_state["strategy_comparison_active_beta_exclude"] = active_portfolio['exclude_days_beta']
            st.number_input("Beta Lookback (days)", min_value=1, key="strategy_comparison_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="strategy_comparison_active_beta_exclude", on_change=update_beta_exclude)
        if "strategy_comparison_active_calc_vol" not in st.session_state:
            st.session_state["strategy_comparison_active_calc_vol"] = active_portfolio['calc_volatility']
        st.checkbox("Include Volatility in momentum weighting", key="strategy_comparison_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        # Reset Volatility button
        if st.button("Reset Volatility", key=f"strategy_comparison_reset_vol_btn_{st.session_state.strategy_comparison_active_portfolio_index}", on_click=reset_vol_callback):
            pass
        if st.session_state.get('strategy_comparison_active_calc_vol', False):
            if "strategy_comparison_active_vol_window" not in st.session_state:
                st.session_state["strategy_comparison_active_vol_window"] = active_portfolio['vol_window_days']
            if "strategy_comparison_active_vol_exclude" not in st.session_state:
                st.session_state["strategy_comparison_active_vol_exclude"] = active_portfolio['exclude_days_vol']
            st.number_input("Volatility Lookback (days)", min_value=1, key="strategy_comparison_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="strategy_comparison_active_vol_exclude", on_change=update_vol_exclude)
    st.markdown("---")
    st.subheader("Momentum Windows")
    col_reset, col_norm, col_addrem = st.columns([0.4, 0.4, 0.2])
    with col_reset:
        if st.button("Reset Momentum Windows", on_click=reset_momentum_windows_callback):
            pass
    with col_norm:
        if st.button("Normalize Weights to 100%", on_click=normalize_momentum_weights_callback):
            pass
    with col_addrem:
        if st.button("Add Window", on_click=add_momentum_window_callback):
            pass
        if st.button("Remove Window", on_click=remove_momentum_window_callback):
            pass

    total_weight = sum(w['weight'] for w in active_portfolio['momentum_windows'])
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"Current total weight is {total_weight*100:.2f}%, not 100%. Click 'Normalize Weights' to fix.")
    else:
        st.success(f"Total weight is {total_weight*100:.2f}%.")

    def update_momentum_lookback(index):
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'][index]['lookback'] = st.session_state[f"strategy_comparison_lookback_active_{index}"]

    def update_momentum_exclude(index):
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'][index]['exclude'] = st.session_state[f"strategy_comparison_exclude_active_{index}"]
    
    def update_momentum_weight(index):
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['momentum_windows'][index]['weight'] = st.session_state[f"strategy_comparison_weight_input_active_{index}"] / 100.0

    # Create lambda functions for on_change callbacks
    def create_momentum_lookback_callback(index):
        return lambda: update_momentum_lookback(index)
    
    def create_momentum_exclude_callback(index):
        return lambda: update_momentum_exclude(index)
    
    def create_momentum_weight_callback(index):
        return lambda: update_momentum_weight(index)

    # Allow the user to remove momentum windows down to zero.
    # Previously the UI forced a minimum of 3 windows which prevented removing them.
    # If no windows exist, show an informational message and allow adding via the button.
    if len(active_portfolio.get('momentum_windows', [])) == 0:
        st.info("No momentum windows configured. Click 'Add Window' to create momentum lookback windows.")
    col_headers = st.columns(3)
    with col_headers[0]:
        st.markdown("**Lookback (days)**")
    with col_headers[1]:
        st.markdown("**Exclude (days)**")
    with col_headers[2]:
        st.markdown("**Weight %**")

    for j in range(len(active_portfolio['momentum_windows'])):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"strategy_comparison_lookback_active_{j}"
            exclude_key = f"strategy_comparison_exclude_active_{j}"
            weight_key = f"strategy_comparison_weight_input_active_{j}"
            if lookback_key not in st.session_state:
                # Convert lookback to integer to match min_value type
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
                # Convert exclude to integer to match min_value type
                st.session_state[exclude_key] = int(active_portfolio['momentum_windows'][j]['exclude'])
            if weight_key not in st.session_state:
                # Sanitize weight to prevent StreamlitValueAboveMaxError
                weight = active_portfolio['momentum_windows'][j]['weight']
                if isinstance(weight, (int, float)):
                    # If weight is already a percentage (e.g., 50 for 50%), use it directly
                    if weight > 1.0:
                        # Cap at 100% and use as percentage
                        weight_percentage = min(weight, 100.0)
                    else:
                        # Convert decimal to percentage
                        weight_percentage = weight * 100.0
                else:
                    # Invalid weight, set to default
                    weight_percentage = 10.0
                st.session_state[weight_key] = int(weight_percentage)
            with col_mw1:
                st.number_input(f"Lookback {j+1}", value=st.session_state[lookback_key], min_value=1, key=lookback_key, on_change=create_momentum_lookback_callback(j), label_visibility="collapsed")
            with col_mw2:
                st.number_input(f"Exclude {j+1}", value=st.session_state[exclude_key], min_value=0, key=exclude_key, on_change=create_momentum_exclude_callback(j), label_visibility="collapsed")
            with col_mw3:
                st.number_input(f"Weight {j+1}", value=st.session_state[weight_key], min_value=0, max_value=100, step=1, format="%d", key=weight_key, on_change=create_momentum_weight_callback(j), label_visibility="collapsed")
else:
    active_portfolio['use_relative_momentum'] = False
    active_portfolio['equal_if_all_negative'] = False
    active_portfolio['momentum_windows'] = []
    active_portfolio['calc_beta'] = False
    active_portfolio['calc_volatility'] = False

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    config_json = json.dumps(active_portfolio, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    st.text_area("Paste JSON Here to Update Portfolio", key="strategy_comparison_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)

# Run backtests when triggered from sidebar
if st.session_state.get('strategy_comparison_run_backtest', False):
    st.session_state.strategy_comparison_run_backtest = False
    
    # Pre-backtest validation check for all portfolios
    configs_to_run = st.session_state.strategy_comparison_portfolio_configs
    valid_configs = True
    for cfg in configs_to_run:
        if cfg['use_momentum']:
            total_momentum_weight = sum(w['weight'] for w in cfg['momentum_windows'])
            if abs(total_momentum_weight - 1.0) > 0.001:
                st.error(f"Portfolio '{cfg['name']}' has momentum enabled but the total momentum weight is not 100%. Please fix and try again.")
                valid_configs = False
        else:
            valid_stocks_for_cfg = [s for s in cfg['stocks'] if s['ticker']]
            total_stock_allocation = sum(s['allocation'] for s in valid_stocks_for_cfg)
            if abs(total_stock_allocation - 1.0) > 0.001:
                st.warning(f"Portfolio '{cfg['name']}' is not using momentum, but the total stock allocation is not 100%. Click 'Normalize Stocks %' to fix.")
                
    if not valid_configs:
        st.stop()

    progress_bar = st.empty()
    progress_bar.progress(0, text="Starting backtest...")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        all_tickers = sorted(list(set(s['ticker'] for cfg in st.session_state.strategy_comparison_portfolio_configs for s in cfg['stocks'] if s['ticker']) | set(cfg['benchmark_ticker'] for cfg in st.session_state.strategy_comparison_portfolio_configs if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        print("Downloading data for all tickers...")
        data = {}
        for i, t in enumerate(all_tickers):
            try:
                progress_text = f"Downloading data for {t} ({i+1}/{len(all_tickers)})..."
                progress_bar.progress((i + 1) / (len(all_tickers) + len(st.session_state.strategy_comparison_portfolio_configs)), text=progress_text)
                ticker = yf.Ticker(t)
                hist = ticker.history(period="max", auto_adjust=False)[["Close", "Dividends"]]
                if hist.empty:
                    print(f"No data available for {t}")
                    continue
                hist.index = pd.to_datetime(hist.index)
                hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                data[t] = hist
                print(f"Data loaded for {t} from {data[t].index[0].date()}")
            except Exception as e:
                print(f"Error loading {t}: {e}")
        if not data:
            print("No data downloaded; aborting.")
            st.warning("No data downloaded; aborting.")
            progress_bar.empty()
            st.session_state.strategy_comparison_all_results = None
            st.session_state.strategy_comparison_all_allocations = None
            st.session_state.strategy_comparison_all_metrics = None
        else:
            # Persist raw downloaded price data so later recomputations can access benchmark series
            st.session_state.strategy_comparison_raw_data = data
            common_start = max(df.first_valid_index() for df in data.values())
            common_end = min(df.last_valid_index() for df in data.values())
            print()
            all_results = {}
            all_drawdowns = {}
            all_stats = {}
            all_allocations = {}
            all_metrics = {}
            # Map portfolio index (0-based) to the unique key used in the result dicts
            portfolio_key_map = {}
            for i, cfg in enumerate(st.session_state.strategy_comparison_portfolio_configs, start=1):
                progress_text = f"Running backtest for {cfg.get('name', f'Backtest {i}')} ({i}/{len(st.session_state.strategy_comparison_portfolio_configs)})..."
                progress_bar.progress((len(all_tickers) + i) / (len(all_tickers) + len(st.session_state.strategy_comparison_portfolio_configs)), text=progress_text)
                name = cfg.get('name', f'Backtest {i}')
                # Ensure unique key for storage to avoid overwriting when duplicate names exist
                base_name = name
                unique_name = base_name
                suffix = 1
                while unique_name in all_results or unique_name in all_allocations:
                    unique_name = f"{base_name} ({suffix})"
                    suffix += 1
                print(f"\nRunning backtest {i}/{len(st.session_state.strategy_comparison_portfolio_configs)}: {name}")
                
                # Tickers for config includes all assets for which data is needed (portfolio stocks + benchmark)
                tickers_for_config = [s['ticker'] for s in cfg['stocks'] if s['ticker']] + ([cfg['benchmark_ticker']] if cfg['benchmark_ticker'] else [])
                tickers_for_config = [t for t in tickers_for_config if t in data and t is not None]
                
                # Portfolio stocks only (excluding benchmark) for start date calculation
                portfolio_stocks_only = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                portfolio_stocks_only = [t for t in portfolio_stocks_only if t in data and t is not None]
                
                if not portfolio_stocks_only:
                    print(f"  No valid portfolio stocks for {name}; skipping backtest for this portfolio.")
                    continue
                
                # Override the portfolio's start_with with the global selection from sidebar
                global_start_with = st.session_state.get('strategy_comparison_start_with', 'all')
                print(f"  Using global start_with: {global_start_with}")
                if global_start_with == 'all':
                    final_start = max(data[t].first_valid_index() for t in portfolio_stocks_only)
                    print(f"  All portfolio assets start date: {final_start.date()}")
                else: # global_start_with == 'oldest'
                    final_start = min(data[t].first_valid_index() for t in portfolio_stocks_only)
                    print(f"  Oldest portfolio asset start date: {final_start.date()}")
                if cfg.get('start_date_user'):
                    user_start = pd.to_datetime(cfg['start_date_user'])
                    final_start = max(final_start, user_start)
                if cfg.get('end_date_user'):
                    final_end = min(pd.to_datetime(cfg['end_date_user']), min(data[t].last_valid_index() for t in tickers_for_config))
                else:
                    final_end = min(data[t].last_valid_index() for t in tickers_for_config)
                if final_start > final_end:
                    print(f"  Start date {final_start.date()} is after end date {final_end.date()}. Skipping {name}.")
                    continue
                simulation_index = pd.date_range(start=final_start, end=final_end, freq='D')
                print(f"  Simulation period for {name}: {final_start.date()} to {final_end.date()}\n")
                data_reindexed_for_config = {}
                for t in tickers_for_config:
                    df = data[t].reindex(simulation_index)
                    df["Close"] = df["Close"].ffill()
                    df["Dividends"] = df["Dividends"].fillna(0)
                    df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                    data_reindexed_for_config[t] = df
                total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed_for_config)
                # Store both series under the unique key for later use
                # compute today_weights_map (target weights as-if rebalanced at final snapshot date)
                today_weights_map = {}
                try:
                    alloc_dates = sorted(list(historical_allocations.keys()))
                    final_d = alloc_dates[-1]
                    metrics_local = historical_metrics
                    
                    # Check if momentum is used for this portfolio
                    use_momentum = cfg.get('use_momentum', True)
                    
                    if final_d in metrics_local:
                        if use_momentum:
                            # extract Calculated_Weight if present (momentum-based)
                            weights = {t: v.get('Calculated_Weight', 0) for t, v in metrics_local[final_d].items()}
                            # normalize (ensure sums to 1 excluding CASH)
                            sumw = sum(w for k, w in weights.items() if k != 'CASH')
                            if sumw > 0:
                                norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                            else:
                                norm = weights
                            today_weights_map = norm
                        else:
                            # When momentum is not used, use user-defined allocations from portfolio config
                            today_weights_map = {}
                            for stock in cfg.get('stocks', []):
                                ticker = stock.get('ticker', '').strip()
                                if ticker:
                                    today_weights_map[ticker] = stock.get('allocation', 0)
                            # Add CASH if needed
                            total_alloc = sum(today_weights_map.values())
                            if total_alloc < 1.0:
                                today_weights_map['CASH'] = 1.0 - total_alloc
                            else:
                                today_weights_map['CASH'] = 0
                    else:
                        # fallback: use allocation snapshot at final date but convert market-value alloc to target weights (exclude CASH then renormalize)
                        final_alloc = historical_allocations.get(final_d, {})
                        noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                        s = sum(noncash.values())
                        if s > 0:
                            norm = {k: (v / s) for k, v in noncash.items()}
                            norm['CASH'] = final_alloc.get('CASH', 0)
                        else:
                            norm = final_alloc
                        today_weights_map = norm
                except Exception as e:
                    # If computation fails, use user-defined allocations as fallback
                    today_weights_map = {}
                    for stock in cfg.get('stocks', []):
                        ticker = stock.get('ticker', '').strip()
                        if ticker:
                            today_weights_map[ticker] = stock.get('allocation', 0)
                    # Add CASH if needed
                    total_alloc = sum(today_weights_map.values())
                    if total_alloc < 1.0:
                        today_weights_map['CASH'] = 1.0 - total_alloc
                    else:
                        today_weights_map['CASH'] = 0

                all_results[unique_name] = {
                    'no_additions': total_series_no_additions,
                    'with_additions': total_series,
                    'today_weights_map': today_weights_map
                }
                all_allocations[unique_name] = historical_allocations
                all_metrics[unique_name] = historical_metrics
                # Remember mapping from portfolio index (0-based) to unique key
                portfolio_key_map[i-1] = unique_name
                # --- PATCHED CASH FLOW LOGIC ---
                # Track cash flows as pandas Series indexed by date
                cash_flows = pd.Series(0.0, index=total_series.index)
                # Initial investment: negative cash flow on first date
                if len(total_series.index) > 0:
                    cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # Periodic additions: negative cash flow on their respective dates
                dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
                for d in dates_added:
                    if d in cash_flows.index and d != cash_flows.index[0]:
                        cash_flows.loc[d] -= cfg.get('added_amount', 0)
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    cash_flows.iloc[-1] += total_series.iloc[-1]
                # Get benchmark returns for stats calculation
                benchmark_returns = None
                if cfg['benchmark_ticker'] and cfg['benchmark_ticker'] in data_reindexed_for_config:
                    benchmark_returns = data_reindexed_for_config[cfg['benchmark_ticker']]['Price_change']
                # Ensure benchmark_returns is a pandas Series aligned to total_series
                if benchmark_returns is not None:
                    benchmark_returns = pd.Series(benchmark_returns, index=total_series.index).dropna()
                # Ensure cash_flows is a pandas Series indexed by date, with initial investment and additions
                cash_flows = pd.Series(cash_flows, index=total_series.index)
                # Align for stats calculation
                # Track cash flows for MWRR exactly as in app.py
                # Initial investment: negative cash flow on first date
                mwrr_cash_flows = pd.Series(0.0, index=total_series.index)
                if len(total_series.index) > 0:
                    mwrr_cash_flows.iloc[0] = -cfg.get('initial_value', 0)
                # Periodic additions: negative cash flow on their respective dates
                dates_added = get_dates_by_freq(cfg.get('added_frequency'), total_series.index[0], total_series.index[-1], total_series.index)
                for d in dates_added:
                    if d in mwrr_cash_flows.index and d != mwrr_cash_flows.index[0]:
                        mwrr_cash_flows.loc[d] -= cfg.get('added_amount', 0)
                # Final value: positive cash flow on last date for MWRR
                if len(total_series.index) > 0:
                    mwrr_cash_flows.iloc[-1] += total_series.iloc[-1]

                # Use the no-additions series returned by single_backtest (do NOT reconstruct it here)
                # total_series_no_additions is returned by single_backtest and already represents the portfolio value without added cash.

                # Calculate statistics
                # Use total_series_no_additions for all stats except MWRR
                stats_values = total_series_no_additions.values
                stats_dates = total_series_no_additions.index
                stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                cagr = calculate_cagr(stats_values, stats_dates)
                max_dd, drawdowns = calculate_max_drawdown(stats_values)
                vol = calculate_volatility(stats_returns)
                # Avoid division by zero; if std is zero set Sharpe to NaN
                try:
                    sharpe = np.nan if stats_returns.std() == 0 else (stats_returns.mean() * 252 / (stats_returns.std() * np.sqrt(252)))
                except Exception:
                    sharpe = np.nan
                sortino = calculate_sortino(stats_returns)
                ulcer = calculate_ulcer_index(stats_values)
                upi = calculate_upi(cagr, ulcer)
                # --- Beta calculation (copied from app.py) ---
                beta = np.nan
                if benchmark_returns is not None:
                    portfolio_returns = stats_returns.copy()
                    benchmark_returns_series = pd.Series(benchmark_returns, index=stats_dates).dropna()
                    common_idx = portfolio_returns.index.intersection(benchmark_returns_series.index)
                    if len(common_idx) >= 2:
                        pr = portfolio_returns.reindex(common_idx).dropna()
                        br = benchmark_returns_series.reindex(common_idx).dropna()
                        common_idx2 = pr.index.intersection(br.index)
                        if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                            cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                            var = br.loc[common_idx2].var()
                            beta = cov / var
                # MWRR uses the full backtest with additions
                mwrr = calculate_mwrr(total_series, mwrr_cash_flows, total_series.index)
                def scale_pct(val):
                    if val is None or np.isnan(val):
                        return np.nan
                    # Only scale if value is between -1 and 1 (decimal)
                    if -1.5 < val < 1.5:
                        return val * 100
                    return val

                def clamp_stat(val, stat_type):
                    if val is None or np.isnan(val):
                        return "N/A"
                    v = scale_pct(val)
                    # Clamp ranges for each stat type
                    if stat_type in ["CAGR", "Volatility", "MWRR", "Total Return"]:
                        if v < 0 or v > 100:
                            return "N/A"
                    if stat_type == "MaxDrawdown":
                        if v < -100 or v > 0:
                            return "N/A"
                    return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

                # Calculate total return (no additions)
                total_return = None
                if len(stats_values) > 0:
                    initial_val = stats_values[0]
                    final_val = stats_values[-1]
                    if initial_val > 0:
                        total_return = (final_val / initial_val - 1)  # Return as decimal, not percentage

                stats = {
                    "Total Return": clamp_stat(total_return, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": clamp_stat(mwrr, "MWRR"),
                }
                all_stats[unique_name] = stats
                all_drawdowns[unique_name] = pd.Series(drawdowns, index=stats_dates)
            progress_bar.progress(100, text="Backtests complete!")
            progress_bar.empty()
            print("\n" + "="*80)
            print(" " * 25 + "FINAL PERFORMANCE STATISTICS")
            print("="*80 + "\n")
            stats_df = pd.DataFrame(all_stats).T
            def fmt_pct(x):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x*100:.2f}%"
                if isinstance(x, str):
                    return x
                return "N/A"
            def fmt_num(x, prec=3):
                if isinstance(x, (int, float)) and pd.notna(x):
                    return f"{x:.3f}"
                if isinstance(x, str):
                    return x
                return "N/A"
            if not stats_df.empty:
                stats_df_display = stats_df.copy()
                stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
                stats_df_display['Total Return'] = stats_df_display['Total Return'].apply(lambda x: fmt_pct(x))
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it, Total Return at the very end
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns and 'Total Return' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta, MWRR, and Total Return
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    total_return_col = cols.pop(cols.index('Total Return'))
                    # Insert Beta before MWRR, then Total Return at the very end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
                    cols.append(total_return_col)
                    stats_df_display = stats_df_display[cols]
                stats_df_display['MWRR'] = stats_df_display['MWRR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Sharpe'] = stats_df_display['Sharpe'].apply(lambda x: fmt_num(x))
                stats_df_display['Sortino'] = stats_df_display['Sortino'].apply(lambda x: fmt_num(x))
                stats_df_display['Ulcer Index'] = stats_df_display['Ulcer Index'].apply(lambda x: fmt_num(x))
                stats_df_display['UPI'] = stats_df_display['UPI'].apply(lambda x: fmt_num(x))
                if 'Beta' in stats_df_display.columns:
                    stats_df_display['Beta'] = stats_df_display['Beta'].apply(lambda x: fmt_num(x))
                print(stats_df_display.to_string())
            else:
                print("No stats to display.")
            # Yearly performance section (interactive table below)
            all_years = {}
            for name, ser in all_results.items():
                # Use the with-additions series for yearly performance (user requested)
                yearly = ser['with_additions'].resample('YE').last()
                all_years[name] = yearly
            years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
            names = list(all_years.keys())
            
            # Print console log yearly table correctly
            col_width = 22
            header_format = "{:<6} |" + "".join([" {:^" + str(col_width*2+1) + "} |" for _ in names])
            row_format = "{:<6} |" + "".join([" {:>" + str(col_width) + "} {:>" + str(col_width) + "} |" for _ in names])
            
            print(header_format.format("Year", *names))
            print("-" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            print(row_format.format(" ", *[item for pair in [('% Change', 'Final Value')] * len(names) for item in pair]))
            print("=" * (6 + 3 + (col_width*2+1 + 3)*len(names)))
            
            for y in years:
                row_items = [f"{y}"]
                for nm in names:
                    ser = all_years[nm]
                    ser_year = ser[ser.index.year == y]
                    
                    # Corrected logic for yearly performance calculation
                    start_val_for_year = None
                    if y == min(years):
                        config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == nm), None)
                        if config_for_name:
                            initial_val_of_config = config_for_name['initial_value']
                            if initial_val_of_config > 0:
                                start_val_for_year = initial_val_of_config
                    else:
                        prev_year = y - 1
                        prev_ser_year = all_years[nm][all_years[nm].index.year == prev_year]
                        if not prev_ser_year.empty:
                            start_val_for_year = prev_ser_year.iloc[-1]
                        
                    if not ser_year.empty and start_val_for_year is not None:
                        end_val = ser_year.iloc[-1]
                        if start_val_for_year > 0:
                            pct = f"{(end_val - start_val_for_year) / start_val_for_year * 100:.2f}%"
                            final_val = f"${end_val:,.2f}"
                        else:
                            pct = "N/A"
                            final_val = "N/A"
                    else:
                        pct = "N/A"
                        final_val = "N/A"
                        
                    row_items.extend([pct, final_val])
                print(row_format.format(*row_items))
            print("\n" + "="*80)
    
            # console output captured previously is no longer shown on the page
            st.session_state.strategy_comparison_all_results = all_results
            st.session_state.strategy_comparison_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.strategy_comparison_stats_df_display = stats_df_display
            st.session_state.strategy_comparison_all_years = all_years
            # Save a snapshot used by the allocations UI so charts/tables remain static until rerun
            try:
                # Create today_weights_map for all portfolios
                today_weights_map = {}
                for unique_name, results in all_results.items():
                    if isinstance(results, dict) and 'today_weights_map' in results:
                        today_weights_map[unique_name] = results['today_weights_map']
                
                st.session_state.strategy_comparison_snapshot_data = {
                    'raw_data': data,
                    'portfolio_configs': st.session_state.strategy_comparison_portfolio_configs,
                    'all_allocations': all_allocations,
                    'all_metrics': all_metrics,
                    'today_weights_map': today_weights_map
                }
            except Exception:
                pass
            
            st.session_state.strategy_comparison_all_allocations = all_allocations
            st.session_state.strategy_comparison_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.strategy_comparison_portfolio_key_map = portfolio_key_map
            st.session_state.strategy_comparison_ran = True

if 'strategy_comparison_ran' in st.session_state and st.session_state.strategy_comparison_ran:
    if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
        # Use the no-additions series for all display and calculations
        first_date = min(series['no_additions'].index.min() for series in st.session_state.strategy_comparison_all_results.values())
        last_date = max(series['no_additions'].index.max() for series in st.session_state.strategy_comparison_all_results.values())
        st.subheader(f"Results for Backtest Period: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")

        fig1 = go.Figure()
        for name, series_dict in st.session_state.strategy_comparison_all_results.items():
            # Plot the series that includes added cash (with_additions) for comparison
            series_to_plot = series_dict['with_additions'] if isinstance(series_dict, dict) and 'with_additions' in series_dict else series_dict
            fig1.add_trace(go.Scatter(x=series_to_plot.index, y=series_to_plot.values, mode='lines', name=name))
        fig1.update_layout(
            title="Backtest Comparison — Portfolio Value (with cash additions)",
            xaxis_title="Date",
            yaxis=dict(title="Portfolio Value ($)", title_standoff=20),
            legend_title="Portfolios",
            hovermode="x unified",
            template="plotly_dark",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            margin=dict(l=100, r=20, t=80, b=60)
        )
        st.plotly_chart(fig1, use_container_width=True, key="multi_performance_chart")

        fig2 = go.Figure()
        for name, series in st.session_state.strategy_comparison_all_drawdowns.items():
            fig2.add_trace(go.Scatter(x=series.index, y=series.values * 100, mode='lines', name=name))
        fig2.update_layout(
            title="Backtest Comparison (Max Drawdown)",
            xaxis_title="Date",
            yaxis=dict(title="Drawdown (%)", title_standoff=20),
            legend_title="Portfolios",
            hovermode="x unified",
            template="plotly_dark",
            margin=dict(l=100, r=20, t=80, b=60)
        )
        st.plotly_chart(fig2, use_container_width=True, key="multi_drawdown_chart")

        # --- Variation summary chart: compares total return, CAGR, volatility and max drawdown across portfolios ---
        try:
            def get_no_additions_series(obj):
                return obj['no_additions'] if isinstance(obj, dict) and 'no_additions' in obj else obj if isinstance(obj, pd.Series) else None

            metrics_summary = {}
            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_no = get_no_additions_series(series_obj)
                if ser_no is None or len(ser_no) < 2:
                    continue
                vals = ser_no.values
                dates = ser_no.index
                # Total return over the period
                try:
                    total_return = (vals[-1] / vals[0] - 1) * 100 if vals[0] and not np.isnan(vals[0]) else np.nan
                except Exception:
                    total_return = np.nan

                # CAGR, volatility, max drawdown (convert to percent for display)
                try:
                    cagr = calculate_cagr(vals, dates)
                except Exception:
                    cagr = np.nan
                try:
                    returns = pd.Series(vals, index=dates).pct_change().fillna(0)
                    vol = calculate_volatility(returns)
                except Exception:
                    vol = np.nan
                try:
                    max_dd, _ = calculate_max_drawdown(vals)
                except Exception:
                    max_dd = np.nan

                metrics_summary[name] = {
                    'Total Return': total_return,
                    'CAGR': (cagr * 100) if isinstance(cagr, (int, float)) and not np.isnan(cagr) else np.nan,
                    'Volatility': (vol * 100) if isinstance(vol, (int, float)) and not np.isnan(vol) else np.nan,
                    'Max Drawdown': (max_dd * 100) if isinstance(max_dd, (int, float)) and not np.isnan(max_dd) else np.nan,
                }

            if metrics_summary:
                df_metrics = pd.DataFrame(metrics_summary).T
                # Ensure numeric columns
                for c in df_metrics.columns:
                    df_metrics[c] = pd.to_numeric(df_metrics[c], errors='coerce')

                # Create grouped bar chart
                fig_metrics = go.Figure()
                metric_order = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                for i, metric in enumerate(metric_order):
                    if metric in df_metrics.columns:
                        fig_metrics.add_trace(go.Bar(
                            x=df_metrics.index,
                            y=df_metrics[metric].values,
                            name=metric,
                            marker_color=colors[i % len(colors)],
                            text=[f"{v:.2f}%" if not pd.isna(v) else 'N/A' for v in df_metrics[metric].values],
                            textposition='auto'
                        ))

                fig_metrics.update_layout(
                    title='Portfolio Variation Summary (percent)',
                    barmode='group',
                    template='plotly_dark',
                    yaxis=dict(title='Percent', ticksuffix='%'),
                    legend_title='Metric',
                    height=520,
                    margin=dict(l=60, r=40, t=80, b=120),
                )

                st.plotly_chart(fig_metrics, use_container_width=True, key="multi_metrics_chart")
        except Exception as e:
            print(f"[VARIATION CHART DEBUG] Failed to build metrics summary chart: {e}")

        # --- Monthly returns heatmap: rows = portfolios, columns = Year-Month, values = monthly % change ---
        try:
            # Build a DataFrame of monthly returns for each portfolio
            monthly_returns = {}
            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_no = series_obj['no_additions'] if isinstance(series_obj, dict) and 'no_additions' in series_obj else series_obj if isinstance(series_obj, pd.Series) else None
                if ser_no is None or len(ser_no) < 2:
                    continue
                # Resample to month-end and compute percent change
                try:
                    # Use month-end resample with 'ME' alias to avoid FutureWarning; keep as DatetimeIndex
                    ser_month = ser_no.resample('ME').last()
                    pct_month = ser_month.pct_change().dropna() * 100
                    # label months as 'YYYY-MM' using DatetimeIndex to avoid PeriodArray conversion
                    pct_month.index = pct_month.index.strftime('%Y-%m')
                    monthly_returns[name] = pct_month
                except Exception:
                    continue

            if monthly_returns:
                # Align indexes (months) across portfolios
                all_months = sorted(list({m for ser in monthly_returns.values() for m in ser.index}))
                heat_data = pd.DataFrame(index=list(monthly_returns.keys()), columns=all_months)
                for name, ser in monthly_returns.items():
                    for m, v in ser.items():
                        heat_data.at[name, m] = v
                heat_data = heat_data.astype(float)

                # Create heatmap with Plotly
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat_data.values,
                    x=heat_data.columns.astype(str),
                    y=heat_data.index.astype(str),
                    colorscale='RdYlGn',
                    colorbar=dict(title='Monthly %'),
                    hovertemplate='Portfolio: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
                ))
                fig_heat.update_layout(
                    title='Monthly Returns Heatmap (rows=portfolios, columns=year-month)',
                    xaxis_nticks=20,
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_heat, use_container_width=True, key="multi_heatmap_chart")
        except Exception as e:
            print(f"[MONTHLY HEATMAP DEBUG] Failed to build monthly heatmap: {e}")

        # Recompute Final Performance Statistics from stored results to ensure they use the no-additions series
        if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
            # Helper to extract no-additions series whether stored as dict or Series
            def get_no_additions(series_or_dict):
                return series_or_dict['no_additions'] if isinstance(series_or_dict, dict) and 'no_additions' in series_or_dict else series_or_dict

            # Try to preserve existing MWRR values (these depend on cash flows and are not easily recomputed here)
            preserved_mwrr = {}
            existing_stats = st.session_state.get('strategy_comparison_stats_df_display')
            if existing_stats is not None and 'MWRR' in existing_stats.columns:
                for name, val in existing_stats['MWRR'].items():
                    preserved_mwrr[name] = val

            recomputed_stats = {}

            def scale_pct(val):
                if val is None or np.isnan(val):
                    return np.nan
                if -1.5 < val < 1.5:
                    return val * 100
                return val

            def clamp_stat(val, stat_type):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                v = scale_pct(val)
                
                # Apply specific scaling for Total Return before clamping
                if stat_type == "Total Return":
                    v = v * 100
                
                # Clamping logic - separate Total Return from other percentage stats
                if stat_type in ["CAGR", "Volatility", "MWRR"]:
                    if isinstance(v, (int, float)) and (v < 0 or v > 100):
                        return "N/A"
                elif stat_type == "Total Return":
                    if isinstance(v, (int, float)) and v < 0:  # Only check for negative values
                        return "N/A"
                elif stat_type == "MaxDrawdown":
                    if isinstance(v, (int, float)) and (v < -100 or v > 0):
                        return "N/A"
                
                return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR", "Total Return"] else f"{v:.3f}" if isinstance(v, float) else v

            for name, series_obj in st.session_state.strategy_comparison_all_results.items():
                ser_noadd = get_no_additions(series_obj)
                if ser_noadd is None or len(ser_noadd) < 2:
                    recomputed_stats[name] = {
                        "Total Return": "N/A",
                        "CAGR": "N/A",
                        "MaxDrawdown": "N/A",
                        "Volatility": "N/A",
                        "Sharpe": "N/A",
                        "Sortino": "N/A",
                        "UlcerIndex": "N/A",
                        "UPI": "N/A",
                        "Beta": "N/A",
                        "MWRR": preserved_mwrr.get(name, "N/A"),
                        # Final values with and without additions (if available)
                        "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else "N/A"),
                        "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else "N/A")
                    }
                    continue

                stats_values = ser_noadd.values
                stats_dates = ser_noadd.index
                stats_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                
                # Calculate total return (no additions)
                total_return = None
                if len(stats_values) > 0:
                    initial_val = stats_values[0]
                    final_val = stats_values[-1]
                    if initial_val > 0:
                        total_return = (final_val / initial_val - 1)  # Return as decimal, not percentage
                
                cagr = calculate_cagr(stats_values, stats_dates)
                max_dd, drawdowns = calculate_max_drawdown(stats_values)
                vol = calculate_volatility(stats_returns)
                # Avoid division by zero; if std is zero set Sharpe to NaN
                try:
                    sharpe = np.nan if stats_returns.std() == 0 else (stats_returns.mean() * 252 / (stats_returns.std() * np.sqrt(252)))
                except Exception:
                    sharpe = np.nan
                sortino = calculate_sortino(stats_returns)
                ulcer = calculate_ulcer_index(stats_values)
                upi = calculate_upi(cagr, ulcer)
                # Compute Beta based on the no-additions portfolio returns and the portfolio's benchmark (if available)
                beta = np.nan
                # Find the portfolio config to get benchmark ticker
                cfg_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                if cfg_for_name:
                    bench_ticker = cfg_for_name.get('benchmark_ticker')
                    raw_data = st.session_state.get('strategy_comparison_raw_data')
                    if bench_ticker and raw_data and bench_ticker in raw_data:
                        # get benchmark price_change series aligned to ser_noadd index
                        try:
                            bench_df = raw_data[bench_ticker].reindex(ser_noadd.index)
                            if 'Price_change' in bench_df.columns:
                                bench_returns = bench_df['Price_change'].fillna(0)
                            else:
                                bench_returns = bench_df['Close'].pct_change().fillna(0)

                            portfolio_returns = pd.Series(stats_values, index=stats_dates).pct_change().fillna(0)
                            common_idx = portfolio_returns.index.intersection(bench_returns.index)
                            if len(common_idx) >= 2:
                                pr = portfolio_returns.reindex(common_idx).dropna()
                                br = bench_returns.reindex(common_idx).dropna()
                                common_idx2 = pr.index.intersection(br.index)
                                if len(common_idx2) >= 2 and br.loc[common_idx2].var() != 0:
                                    cov = pr.loc[common_idx2].cov(br.loc[common_idx2])
                                    var = br.loc[common_idx2].var()
                                    beta = cov / var
                        except Exception as e:
                            print(f"[BETA DEBUG] Failed to compute beta for {name}: {e}")
                mwrr_val = preserved_mwrr.get(name, "N/A")

                recomputed_stats[name] = {
                    "Total Return": clamp_stat(total_return, "Total Return"),
                    "CAGR": clamp_stat(cagr, "CAGR"),
                    "MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
                    "Volatility": clamp_stat(vol, "Volatility"),
                    "Sharpe": clamp_stat(sharpe / 100 if isinstance(sharpe, (int, float)) and pd.notna(sharpe) else sharpe, "Sharpe"),
                    "Sortino": clamp_stat(sortino / 100 if isinstance(sortino, (int, float)) and pd.notna(sortino) else sortino, "Sortino"),
                    "UlcerIndex": clamp_stat(ulcer, "UlcerIndex"),
                    "UPI": clamp_stat(upi / 100 if isinstance(upi, (int, float)) and pd.notna(upi) else upi, "UPI"),
                    "Beta": clamp_stat(beta / 100 if isinstance(beta, (int, float)) and pd.notna(beta) else beta, "Beta"),
                    "MWRR": mwrr_val,
                    # Final values with and without additions
                    "Final Value (with)": (series_obj['with_additions'].iloc[-1] if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions'])>0 else "N/A"),
                    "Final Value (no_additions)": (ser_noadd.iloc[-1] if isinstance(ser_noadd, pd.Series) and len(ser_noadd)>0 else "N/A")
                }

            stats_df_display = pd.DataFrame(recomputed_stats).T
            # Move final value columns to the front and format them as currency
            cols = list(stats_df_display.columns)
            fv_with = 'Final Value (with)'
            fv_no = 'Final Value (no_additions)'
            front = [c for c in [fv_with, fv_no] if c in cols]
            for c in front:
                cols.remove(c)
            cols = front + cols
            stats_df_display = stats_df_display[cols]
            # Rename and format columns similarly to prior display code
            stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
            # Ensure ordering: Beta then MWRR at end, Total Return at the very end
            cols = list(stats_df_display.columns)
            if 'Beta' in cols and 'MWRR' in cols and 'Total Return' in cols:
                cols.remove('Beta'); cols.remove('MWRR'); cols.remove('Total Return')
                cols.extend(['Beta','MWRR','Total Return'])
                stats_df_display = stats_df_display[cols]

            st.subheader("Final Performance Statistics")
            # Format currency for final value columns if present
            fmt_map_display = {}
            if fv_with in stats_df_display.columns:
                fmt_map_display[fv_with] = '${:,.2f}'
            if fv_no in stats_df_display.columns:
                fmt_map_display[fv_no] = '${:,.2f}'
            if fmt_map_display:
                try:
                    st.dataframe(stats_df_display.style.format(fmt_map_display), use_container_width=True)
                except Exception:
                    # Fallback to raw dataframe if styling fails
                    st.dataframe(stats_df_display, use_container_width=True)
            else:
                st.dataframe(stats_df_display, use_container_width=True)

        st.subheader("Yearly Performance (Interactive Table)")
        all_years = st.session_state.strategy_comparison_all_years
        years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs if cfg.get('name') in all_years]

        # Corrected yearly table creation
        df_yearly_pct_data = {}
        df_yearly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions yearly series (used for final values)
            ser_with = all_years.get(name) if isinstance(all_years, dict) else None
            # no-additions yearly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.strategy_comparison_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('YE').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('YE').last()
            except Exception:
                ser_noadd = None

            for y in years:
                # get year slices
                ser_year_with = ser_with[ser_with.index.year == y] if ser_with is not None else pd.Series()
                ser_year_no = ser_noadd[ser_noadd.index.year == y] if ser_noadd is not None else pd.Series()

                start_val_for_year = None
                if y == min(years):
                    config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_year = initial_val_of_config
                else:
                    prev_year = y - 1
                    # Use no-additions previous year end as the start value for pct change
                    prev_ser_year_no = ser_noadd[ser_noadd.index.year == prev_year] if ser_noadd is not None else pd.Series()
                    if not prev_ser_year_no.empty:
                        start_val_for_year = prev_ser_year_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_year_no.empty and start_val_for_year is not None:
                    end_val_no = ser_year_no.iloc[-1]
                    if start_val_for_year > 0:
                        pct_change = (end_val_no - start_val_for_year) / start_val_for_year * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_year_with.empty:
                    final_value = ser_year_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_yearly_pct_data[f'{name} % Change'] = pct_list
            df_yearly_final_data[f'{name} Final Value'] = final_list

        df_yearly_pct = pd.DataFrame(df_yearly_pct_data, index=years)
        df_yearly_final = pd.DataFrame(df_yearly_final_data, index=years)
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined = pd.concat([df_yearly_pct, df_yearly_final], axis=1)
        ordered_cols = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined.columns:
                ordered_cols.append(pct_col)
            if val_col in temp_combined.columns:
                ordered_cols.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols:
            combined_df = temp_combined
        else:
            combined_df = temp_combined[ordered_cols]

        def color_gradient_stock(val):
            if isinstance(val, (int, float)):
                if val > 50:
                    return 'background-color: #004d00'
                elif val > 20:
                    return 'background-color: #1e8449'
                elif val > 5:
                    return 'background-color: #388e3c'
                elif val > 0:
                    return 'background-color: #66bb6a'
                elif val < -50:
                    return 'background-color: #7b0000'
                elif val < -20:
                    return 'background-color: #b22222'
                elif val < -5:
                    return 'background-color: #d32f2f'
                elif val < 0:
                    return 'background-color: #ef5350'
            return ''
        
        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df.columns.duplicated().any():
            cols = list(combined_df.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df.columns = new_cols

        if combined_df.index.duplicated().any():
            idx = list(map(str, combined_df.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols = [col for col in combined_df.columns if '% Change' in col]
        final_val_cols = [col for col in combined_df.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols:
            if col in combined_df.columns:
                try:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df[col] = pd.to_numeric(pd.Series(combined_df[col]), errors='coerce')
                    except Exception:
                        combined_df[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map = {col: '{:,.2f}%' for col in pct_cols if col in combined_df.columns}
        fmt_map.update({col: '${:,.2f}' for col in final_val_cols if col in combined_df.columns})

        styler = combined_df.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols:
            try:
                # Styler.map is the supported replacement for applymap
                styler = styler.map(color_gradient_stock, subset=pct_cols)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map:
            styler = styler.format(fmt_map, na_rep='N/A')

        st.dataframe(styler, use_container_width=True, hide_index=False)

        st.markdown("---")
        st.markdown("**Detailed Portfolio Information**")
        # Make the selector visually prominent
        st.markdown(
            "<div style='background:#0b1221;padding:12px;border-radius:8px;margin-bottom:8px;'>"
            "<div style='font-size:16px;font-weight:700;color:#ffffff;margin-bottom:6px;'>Select a portfolio for detailed view</div>"
            "</div>", unsafe_allow_html=True)

        # NUCLEAR APPROACH: Store selection by portfolio name, not display index
        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
        
        # Get all available portfolio names
        available_portfolio_names = [cfg.get('name', 'Portfolio') for cfg in portfolio_configs]
        extra_names = [n for n in st.session_state.get('strategy_comparison_all_results', {}).keys() if n not in available_portfolio_names]
        all_portfolio_names = available_portfolio_names + extra_names
        
        # Initialize persistent selection by name
        if "strategy_comparison_selected_portfolio_name" not in st.session_state:
            st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[0] if all_portfolio_names else "No portfolios"
        
        # Ensure the selected name is still valid
        if st.session_state["strategy_comparison_selected_portfolio_name"] not in all_portfolio_names and all_portfolio_names:
            st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[0]
        
        # Create display options with index prefixes for uniqueness
        display_options = [f"{i} - {name}" for i, name in enumerate(all_portfolio_names)]
        
        # Find the current selection index
        current_selection_index = 0
        if st.session_state["strategy_comparison_selected_portfolio_name"] in all_portfolio_names:
            current_selection_index = all_portfolio_names.index(st.session_state["strategy_comparison_selected_portfolio_name"])
        
        # Place the selectbox in its own column to make it larger/centered
        # Build a prominent action row: selector + colored 'View' button
        left_col, mid_col, right_col = st.columns([1, 3, 1])
        with mid_col:
            st.markdown("<div style='display:flex; gap:8px; align-items:center;'>", unsafe_allow_html=True)
            selected_display = st.selectbox(
                "Select portfolio for details", 
                options=display_options, 
                index=current_selection_index,
                key="strategy_comparison_detail_portfolio_selector_temp", 
                help='Choose which portfolio to inspect in detail', 
                label_visibility='collapsed'
            )
            
            # Update the persistent selection when the widget changes
            if selected_display:
                try:
                    prefix, rest = selected_display.split(' - ', 1)
                    if prefix.startswith('extra_'):
                        # extra entries use the rest as the name
                        st.session_state["strategy_comparison_selected_portfolio_name"] = rest
                    else:
                        idx = int(prefix)
                        st.session_state["strategy_comparison_selected_portfolio_name"] = all_portfolio_names[idx]
                except Exception:
                    st.session_state["strategy_comparison_selected_portfolio_name"] = selected_display
            # Add a prominent view button with a professional color
            view_clicked = st.button("View Details", key='strategy_comparison_view_details_btn')
            st.markdown("</div>", unsafe_allow_html=True)

        # Map display label back to actual portfolio name
        selected_portfolio_detail = st.session_state["strategy_comparison_selected_portfolio_name"]

        if selected_portfolio_detail:
            # Highlight the selected portfolio and optionally expand details when the View button is used
            st.markdown(f"<div style='padding:8px 12px;background:#04293a;border-radius:6px;margin-top:8px;'><strong style='color:#bde0fe;'>Showing details for:</strong> <span style='font-size:16px;color:#ffffff;margin-left:8px;'>{selected_portfolio_detail}</span></div>", unsafe_allow_html=True)
            if view_clicked:
                # No-op here; the detail panels below will render based on selected_portfolio_detail. Keep a small indicator
                st.success(f"Loaded details for {selected_portfolio_detail}")
            # Table 1: Historical Allocations
            if selected_portfolio_detail in st.session_state.strategy_comparison_all_allocations:
                st.markdown("---")
                st.markdown(f"**Historical Allocations for {selected_portfolio_detail}**")
                # Ensure proper DataFrame structure with explicit column names
                allocations_df_raw = pd.DataFrame(st.session_state.strategy_comparison_all_allocations[selected_portfolio_detail]).T
                
                # Handle case where only CASH exists - ensure column name is preserved
                if allocations_df_raw.empty or (len(allocations_df_raw.columns) == 1 and allocations_df_raw.columns[0] is None):
                    # Reconstruct DataFrame with proper column names
                    processed_data = {}
                    for date, alloc_dict in st.session_state.strategy_comparison_all_allocations[selected_portfolio_detail].items():
                        processed_data[date] = {}
                        for ticker, value in alloc_dict.items():
                            if ticker is None:
                                processed_data[date]['CASH'] = value
                            else:
                                processed_data[date][ticker] = value
                    allocations_df_raw = pd.DataFrame(processed_data).T
                
                allocations_df_raw.index.name = "Date"
                
                # Corrected styling logic for alternating row colors
                def highlight_rows_by_index(s):
                    is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
                    bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
                    return [bg_color] * len(s)

                styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
                styler.format('{:,.0f}%', na_rep='N/A')
                st.dataframe(styler, use_container_width=True)


            # Table 2: Momentum Metrics and Calculated Weights
            if selected_portfolio_detail in st.session_state.strategy_comparison_all_metrics:
                st.markdown("---")
                st.markdown(f"**Momentum Metrics and Calculated Weights for {selected_portfolio_detail}**")

                metrics_records = []
                for date, tickers_data in st.session_state.strategy_comparison_all_metrics[selected_portfolio_detail].items():
                    # Add all asset lines
                    asset_weights = []
                    for ticker, data in tickers_data.items():
                        # Handle None ticker as CASH
                        display_ticker = 'CASH' if ticker is None else ticker
                        if display_ticker != 'CASH':
                            asset_weights.append(data.get('Calculated_Weight', 0))
                        # Filter out any internal-only keys (e.g., 'Composite') so they don't show in the UI
                        filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                        
                        # Check if momentum is used for this portfolio
                        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                        use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                        
                        # If momentum is not used, replace Calculated_Weight with target_allocation
                        if not use_momentum:
                            if 'target_allocation' in filtered_data:
                                filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                            else:
                                # If target_allocation is not available, use the entered allocations from portfolio_cfg
                                ticker_name = display_ticker if display_ticker != 'CASH' else None
                                if ticker_name and portfolio_cfg:
                                    # Find the stock in portfolio_cfg and use its allocation
                                    for stock in portfolio_cfg.get('stocks', []):
                                        if stock.get('ticker', '').strip() == ticker_name:
                                            filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                            break
                                elif display_ticker == 'CASH' and portfolio_cfg:
                                    # For CASH, calculate the remaining allocation
                                    total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                    filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                        
                        record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                        metrics_records.append(record)
                    
                    # Ensure CASH line is added if there's non-zero cash in allocations
                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                    if allocs_for_portfolio and date in allocs_for_portfolio:
                        cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                        if cash_alloc > 0:
                            # Check if CASH is already in metrics_records for this date
                            cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                            if not cash_exists:
                                # Add CASH line to metrics
                                # Check if momentum is used to determine which weight to show
                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                                
                                if not use_momentum:
                                    # When momentum is not used, calculate CASH allocation from entered allocations
                                    total_alloc = sum(stock.get('allocation', 0) for stock in portfolio_cfg.get('stocks', []))
                                    cash_weight = max(0, 1.0 - total_alloc)
                                    cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                                else:
                                    cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
                                metrics_records.append(cash_record)
                    
                    # Add CASH line if fully allocated to cash (100%) or all asset weights are 0% (fallback logic)
                    cash_line_needed = False
                    if 'CASH' in tickers_data or None in tickers_data:
                        cash_data = tickers_data.get('CASH', tickers_data.get(None, {}))
                        cash_weight = cash_data.get('Calculated_Weight', 0)
                        if abs(cash_weight - 1.0) < 1e-6:  # 100% in decimal
                            cash_line_needed = True
                    if all(w == 0 for w in asset_weights) and asset_weights:
                        cash_line_needed = True
                    if cash_line_needed and 'CASH' not in [r['Ticker'] for r in metrics_records if r['Date'] == date]:
                        # If no explicit CASH data, create a default line
                        cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': 1.0}
                        metrics_records.append(cash_record)

                if metrics_records:
                    metrics_df = pd.DataFrame(metrics_records)
                    
                    # Filter out CASH lines where Calculated_Weight is 0 for the last date
                    if 'Calculated_Weight' in metrics_df.columns:
                        # Get the last date
                        last_date = metrics_df['Date'].max()
                        # Remove CASH records where Calculated_Weight is 0 for the last date
                        mask = ~((metrics_df['Ticker'] == 'CASH') & (metrics_df['Date'] == last_date) & (metrics_df['Calculated_Weight'] == 0))
                        metrics_df = metrics_df[mask].reset_index(drop=True)
                    
                    if not metrics_df.empty:
                        # Ensure unique index by adding a counter if needed
                        if metrics_df.duplicated(subset=['Date', 'Ticker']).any():
                            # Add a counter to make indices unique
                            metrics_df['Counter'] = metrics_df.groupby(['Date', 'Ticker']).cumcount()
                            metrics_df['Ticker_Unique'] = metrics_df['Ticker'] + metrics_df['Counter'].astype(str)
                            metrics_df.set_index(['Date', 'Ticker_Unique'], inplace=True)
                        else:
                            metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        
                    metrics_df_display = metrics_df.copy()

                    # Ensure Momentum column exists and normalize to percent when present
                    if 'Momentum' in metrics_df_display.columns:
                        metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                    else:
                        metrics_df_display['Momentum'] = np.nan

                    def color_momentum(val):
                        if isinstance(val, (int, float)):
                            color = 'green' if val > 0 else 'red'
                            return f'color: {color}'
                        return ''
                    
                    def highlight_metrics_rows(s):
                        date_str = s.name[0]
                        ticker_str = s.name[1]
                        # If this is the CASH row, use dark green background
                        if 'CASH' in ticker_str:
                            return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                        # Otherwise, alternate row colors by date
                        unique_dates = list(metrics_df_display.index.get_level_values(0).unique())
                        is_even = unique_dates.index(date_str) % 2 == 0
                        bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                        return [bg_color] * len(s)

                    # Format Calculated_Weight as a percentage if present
                    if 'Calculated_Weight' in metrics_df_display.columns:
                        metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                    # Convert Volatility from decimal (e.g., 0.20) to percent (20.0)
                    if 'Volatility' in metrics_df_display.columns:
                        metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                    # Corrected styling logic for alternating row colors and momentum color
                    styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
                    if 'Momentum' in metrics_df_display.columns:
                        styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])

                    fmt_dict = {}
                    if 'Momentum' in metrics_df_display.columns:
                        fmt_dict['Momentum'] = '{:,.0f}%'
                    if 'Beta' in metrics_df_display.columns:
                        fmt_dict['Beta'] = '{:,.2f}'
                    if 'Volatility' in metrics_df_display.columns:
                        fmt_dict['Volatility'] = '{:,.2f}%'
                    if 'Calculated_Weight' in metrics_df_display.columns:
                        fmt_dict['Calculated_Weight'] = '{:,.0f}%'

                    if fmt_dict:
                        styler_metrics = styler_metrics.format(fmt_dict)

                    st.dataframe(styler_metrics, use_container_width=True)

                    # --- Allocation plots: Final allocation and last rebalance allocation ---
                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                    if allocs_for_portfolio:
                        try:
                            # Sort allocation dates
                            alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                            if len(alloc_dates) == 0:
                                st.info("No allocation history available to plot.")
                            else:
                                final_date = alloc_dates[-1]
                                last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]

                                final_alloc = allocs_for_portfolio.get(final_date, {})
                                rebal_alloc = allocs_for_portfolio.get(last_rebal_date, {})

                                # Helper to prepare bar data
                                def prepare_bar_data(d):
                                    labels = []
                                    values = []
                                    for k, v in sorted(d.items(), key=lambda x: (-x[1], x[0])):
                                        labels.append(k)
                                        try:
                                            values.append(float(v) * 100)
                                        except Exception:
                                            values.append(0.0)
                                    return labels, values

                                labels_final, vals_final = prepare_bar_data(final_alloc)
                                labels_rebal, vals_rebal = prepare_bar_data(rebal_alloc)

                                # Add timer for next rebalance date
                                try:
                                    # Get the last rebalance date from allocation history
                                    if len(alloc_dates) > 1:
                                        last_rebal_date_for_timer = alloc_dates[-2]  # Second to last date (excluding today/yesterday)
                                    else:
                                        last_rebal_date_for_timer = alloc_dates[-1] if alloc_dates else None
                                    
                                    # Get rebalancing frequency from portfolio config
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                                    # Convert to lowercase and map to function expectations
                                    rebalancing_frequency = rebalancing_frequency.lower()
                                    # Map frequency names to what the function expects
                                    frequency_mapping = {
                                        'monthly': 'month',
                                        'weekly': 'week',
                                        'bi-weekly': '2weeks',
                                        'biweekly': '2weeks',
                                        'quarterly': '3months',
                                        'semi-annually': '6months',
                                        'semiannually': '6months',
                                        'annually': 'year',
                                        'yearly': 'year',
                                        'market_day': 'market_day',
                                        'calendar_day': 'calendar_day',
                                        'never': 'none',
                                        'none': 'none'
                                    }
                                    rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                                    
                                    if last_rebal_date_for_timer and rebalancing_frequency != 'none':
                                        # Ensure last_rebal_date_for_timer is a naive datetime object
                                        if isinstance(last_rebal_date_for_timer, str):
                                            last_rebal_date_for_timer = pd.to_datetime(last_rebal_date_for_timer)
                                        if hasattr(last_rebal_date_for_timer, 'tzinfo') and last_rebal_date_for_timer.tzinfo is not None:
                                            last_rebal_date_for_timer = last_rebal_date_for_timer.replace(tzinfo=None)
                                        next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                            rebalancing_frequency, last_rebal_date_for_timer
                                        )
                                        
                                        if next_date and time_until:
                                            st.markdown("---")
                                            st.markdown("**⏰ Next Rebalance Timer**")
                                            
                                            # Create columns for timer display
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric(
                                                    label="Time Until Next Rebalance",
                                                    value=format_time_until(time_until),
                                                    delta=None
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    label="Target Rebalance Date",
                                                    value=next_date.strftime("%B %d, %Y"),
                                                    delta=None
                                                )
                                            
                                            with col3:
                                                st.metric(
                                                    label="Rebalancing Frequency",
                                                    value=rebalancing_frequency.replace('_', ' ').title(),
                                                    delta=None
                                                )
                                            
                                            # Add a progress bar showing progress to next rebalance
                                            if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                                # Calculate progress percentage
                                                if hasattr(last_rebal_date_for_timer, 'to_pydatetime'):
                                                    last_rebal_datetime = last_rebal_date_for_timer.to_pydatetime()
                                                else:
                                                    last_rebal_datetime = last_rebal_date_for_timer
                                                
                                                total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                                elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                                progress = min(max(elapsed_period / total_period, 0), 1)
                                                
                                                st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                                except Exception as e:
                                    pass  # Silently ignore timer calculation errors

                                # Main "Rebalance as of today" plot and table - this should be the main rebalancing representation
                                st.markdown("---")
                                st.markdown("**🔄 Rebalance as of Today**")
                                
                                # Get momentum-based calculated weights for today's rebalancing from stored snapshot
                                today_weights = {}
                                
                                # Get the stored today_weights_map from snapshot data
                                snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                                today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
                                
                                if selected_portfolio_detail in today_weights_map:
                                    today_weights = today_weights_map.get(selected_portfolio_detail, {})
                                else:
                                    # Fallback to current allocation if no stored weights found
                                    today_weights = final_alloc
                                
                                # Create labels and values for the plot
                                labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                                vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                                
                                # Create a larger plot for the main rebalancing representation
                                st.markdown(f"**Target Allocation if Rebalanced Today**")
                                fig_today = go.Figure()
                                fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                                fig_today.update_traces(textinfo='percent+label')
                                fig_today.update_layout(
                                    template='plotly_dark', 
                                    margin=dict(t=30),
                                    height=600,  # Make it even bigger
                                    showlegend=True
                                )
                                st.plotly_chart(fig_today, use_container_width=True, key=f"multi_today_{selected_portfolio_detail}")
                                
                                # Table moved under the plot
                                # Add the "Rebalance as of today" table
                                try:
                                        # Get portfolio configuration for calculations
                                        portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                        portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                        
                                        if portfolio_cfg:
                                            portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                                            
                                            # Get raw data for price calculations
                                            raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                            
                                            def _price_on_or_before(df, target_date):
                                                try:
                                                    idx = df.index[df.index <= pd.to_datetime(target_date)]
                                                    if len(idx) == 0:
                                                        return None
                                                    return float(df.loc[idx[-1], 'Close'])
                                                except Exception:
                                                    return None

                                            def build_table_from_alloc(alloc_dict, price_date, label):
                                                rows = []
                                                for tk in sorted(alloc_dict.keys()):
                                                    alloc_pct = float(alloc_dict.get(tk, 0))
                                                    if tk == 'CASH':
                                                        price = None
                                                        shares = 0
                                                        total_val = portfolio_value * alloc_pct
                                                    else:
                                                        df = raw_data.get(tk)
                                                        price = None
                                                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                            if price_date is None:
                                                                # use latest price
                                                                try:
                                                                    price = float(df['Close'].iloc[-1])
                                                                except Exception:
                                                                    price = None
                                                            else:
                                                                price = _price_on_or_before(df, price_date)
                                                        try:
                                                            if price and price > 0:
                                                                allocation_value = portfolio_value * alloc_pct
                                                                # allow fractional shares shown to 1 decimal place
                                                                shares = round(allocation_value / price, 1)
                                                                total_val = shares * price
                                                            else:
                                                                shares = 0.0
                                                                total_val = portfolio_value * alloc_pct
                                                        except Exception:
                                                            shares = 0
                                                            total_val = portfolio_value * alloc_pct

                                                    pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                                    rows.append({
                                                        'Ticker': tk,
                                                        'Allocation %': alloc_pct * 100,
                                                        'Price ($)': price if price is not None else float('nan'),
                                                        'Shares': shares,
                                                        'Total Value ($)': total_val,
                                                        '% of Portfolio': pct_of_port,
                                                    })

                                                df_table = pd.DataFrame(rows).set_index('Ticker')
                                                # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                                df_display = df_table.copy()
                                                show_cash = False
                                                if 'CASH' in df_display.index:
                                                    cash_val = None
                                                    if 'Total Value ($)' in df_display.columns:
                                                        cash_val = df_display.at['CASH', 'Total Value ($)']
                                                    elif 'Shares' in df_display.columns:
                                                        cash_val = df_display.at['CASH', 'Shares']
                                                    try:
                                                        show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                                    except Exception:
                                                        show_cash = False
                                                    if not show_cash:
                                                        df_display = df_display.drop('CASH')

                                                # formatting for display
                                                fmt = {
                                                    'Allocation %': '{:,.1f}%',
                                                    'Price ($)': '${:,.2f}',
                                                    'Shares': '{:,.1f}',
                                                    'Total Value ($)': '${:,.2f}',
                                                    '% of Portfolio': '{:,.2f}%'
                                                }
                                                try:
                                                    st.markdown(f"**{label}**")
                                                    sty = df_display.style.format(fmt)
                                                    if 'CASH' in df_table.index and show_cash:
                                                        def _highlight_cash_row(s):
                                                            if s.name == 'CASH':
                                                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                        sty = sty.apply(_highlight_cash_row, axis=1)
                                                    st.dataframe(sty, use_container_width=True)
                                                except Exception:
                                                    st.dataframe(df_display, use_container_width=True)
                                            
                                            # "Rebalance as of today" table (use momentum-based calculated weights)
                                            build_table_from_alloc(today_weights, None, f"Target Allocation if Rebalanced Today")
                                            
                                except Exception as e:
                                    print(f"[REBALANCE TODAY TABLE DEBUG] Failed to render rebalance today table for {selected_portfolio_detail}: {e}")

                                # Other rebalancing plots (smaller, placed after the main one)
                                st.markdown("---")
                                st.markdown("**📊 Historical Rebalancing Comparison**")
                                
                                col_plot1, col_plot2 = st.columns(2)
                                with col_plot1:
                                    st.markdown(f"**Last Rebalance Allocation (as of {last_rebal_date.date()})**")
                                    fig_rebal = go.Figure()
                                    fig_rebal.add_trace(go.Pie(labels=labels_rebal, values=vals_rebal, hole=0.3))
                                    fig_rebal.update_traces(textinfo='percent+label')
                                    fig_rebal.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                                    st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_{selected_portfolio_detail}")
                                with col_plot2:
                                    st.markdown(f"**Current Allocation (as of {final_date.date()})**")
                                    fig_final = go.Figure()
                                    fig_final.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.3))
                                    fig_final.update_traces(textinfo='percent+label')
                                    fig_final.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                                    st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_{selected_portfolio_detail}")
                                
                                # Add the three allocation tables from Allocations page
                                try:
                                    # Get portfolio configuration for calculations
                                    portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                    portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                    
                                    if portfolio_cfg:
                                        portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                                        
                                        # Get raw data for price calculations
                                        raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                        
                                        def _price_on_or_before(df, target_date):
                                            try:
                                                idx = df.index[df.index <= pd.to_datetime(target_date)]
                                                if len(idx) == 0:
                                                    return None
                                                return float(df.loc[idx[-1], 'Close'])
                                            except Exception:
                                                return None

                                        def build_table_from_alloc(alloc_dict, price_date, label):
                                            rows = []
                                            for tk in sorted(alloc_dict.keys()):
                                                alloc_pct = float(alloc_dict.get(tk, 0))
                                                if tk == 'CASH':
                                                    price = None
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct
                                                else:
                                                    df = raw_data.get(tk)
                                                    price = None
                                                    if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                        if price_date is None:
                                                            # use latest price
                                                            try:
                                                                price = float(df['Close'].iloc[-1])
                                                            except Exception:
                                                                price = None
                                                        else:
                                                            price = _price_on_or_before(df, price_date)
                                                    try:
                                                        if price and price > 0:
                                                            allocation_value = portfolio_value * alloc_pct
                                                            # allow fractional shares shown to 1 decimal place
                                                            shares = round(allocation_value / price, 1)
                                                            total_val = shares * price
                                                        else:
                                                            shares = 0.0
                                                            total_val = portfolio_value * alloc_pct
                                                    except Exception:
                                                        shares = 0
                                                        total_val = portfolio_value * alloc_pct

                                                pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                                rows.append({
                                                    'Ticker': tk,
                                                    'Allocation %': alloc_pct * 100,
                                                    'Price ($)': price if price is not None else float('nan'),
                                                    'Shares': shares,
                                                    'Total Value ($)': total_val,
                                                    '% of Portfolio': pct_of_port,
                                                })

                                            df_table = pd.DataFrame(rows).set_index('Ticker')
                                            # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                            df_display = df_table.copy()
                                            show_cash = False
                                            if 'CASH' in df_display.index:
                                                cash_val = None
                                                if 'Total Value ($)' in df_display.columns:
                                                    cash_val = df_display.at['CASH', 'Total Value ($)']
                                                elif 'Shares' in df_display.columns:
                                                    cash_val = df_display.at['CASH', 'Shares']
                                                try:
                                                    show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                                except Exception:
                                                    show_cash = False
                                                if not show_cash:
                                                    df_display = df_display.drop('CASH')

                                            # formatting for display
                                            fmt = {
                                                'Allocation %': '{:,.1f}%',
                                                'Price ($)': '${:,.2f}',
                                                'Shares': '{:,.1f}',
                                                'Total Value ($)': '${:,.2f}',
                                                '% of Portfolio': '{:,.2f}%'
                                            }
                                            try:
                                                st.markdown(f"**{label}**")
                                                sty = df_display.style.format(fmt)
                                                if 'CASH' in df_table.index and show_cash:
                                                    def _highlight_cash_row(s):
                                                        if s.name == 'CASH':
                                                            return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                    sty = sty.apply(_highlight_cash_row, axis=1)
                                                st.dataframe(sty, use_container_width=True)
                                            except Exception:
                                                st.dataframe(df_display, use_container_width=True)
                                        
                                        # Last rebalance table (use last_rebal_date)
                                        build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date.date()})")
                                        # Current / Today table (use final_date's latest available prices as of now)
                                        build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
                                        
                                except Exception as e:
                                    print(f"[ALLOC TABLE DEBUG] Failed to render allocation tables for {selected_portfolio_detail}: {e}")
                                    
                        except Exception as e:
                            print(f"[ALLOC PLOT DEBUG] Failed to render allocation plots for {selected_portfolio_detail}: {e}")
                    else:
                        st.info("No allocation history available for this portfolio to show allocation plots.")
                else:
                    # Fallback: show table and plots based on last known allocations so UI stays visible
                    allocs_for_portfolio = st.session_state.strategy_comparison_all_allocations.get(selected_portfolio_detail) if 'strategy_comparison_all_allocations' in st.session_state else None
                    if not allocs_for_portfolio:
                        st.info("No allocation or momentum metrics available for this portfolio.")
                    else:
                        alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                        last_date = alloc_dates[-1]
                        last_alloc = allocs_for_portfolio.get(last_date, {})
                        metrics_records_fb = []
                        for ticker, alloc in last_alloc.items():
                            record = {'Date': last_date, 'Ticker': ticker, 'Momentum': np.nan, 'Beta': np.nan, 'Volatility': np.nan, 'Calculated_Weight': alloc}
                            metrics_records_fb.append(record)

                        metrics_df = pd.DataFrame(metrics_records_fb)
                        metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                        metrics_df_display = metrics_df.copy()
                        if 'Momentum' in metrics_df_display.columns:
                            metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
                            metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

                        def color_momentum(val):
                            if isinstance(val, (int, float)):
                                color = 'green' if val > 0 else 'red'
                                return f'color: {color}'
                            return ''

                        def highlight_metrics_rows(s):
                            date_str = s.name[0]
                            if s.name[1] == 'CASH':
                                return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                            unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                            is_even = unique_dates.index(date_str) % 2 == 0
                            bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                            return [bg_color] * len(s)

                        fmt_map = {}
                        if 'Momentum' in metrics_df_display.columns:
                            fmt_map['Momentum'] = '{:,.0f}%'
                        if 'Beta' in metrics_df_display.columns:
                            fmt_map['Beta'] = '{:,.2f}'
                        if 'Volatility' in metrics_df_display.columns:
                            fmt_map['Volatility'] = '{:,.2f}%'
                        if 'Calculated_Weight' in metrics_df_display.columns:
                            fmt_map['Calculated_Weight'] = '{:,.0f}%'

                        styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
                        if 'Momentum' in metrics_df_display.columns:
                            styler_metrics = styler_metrics.map(color_momentum, subset=['Momentum'])
                        if fmt_map:
                            styler_metrics = styler_metrics.format(fmt_map)
                        st.dataframe(styler_metrics, use_container_width=True)

                        # Add timer for next rebalance date (fallback scenario)
                        try:
                            # Get rebalancing frequency from portfolio config
                            portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                            portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                            rebalancing_frequency = portfolio_cfg.get('rebalancing_frequency', 'none') if portfolio_cfg else 'none'
                            # Convert to lowercase and map to function expectations
                            rebalancing_frequency = rebalancing_frequency.lower()
                            # Map frequency names to what the function expects
                            frequency_mapping = {
                                'monthly': 'month',
                                'weekly': 'week',
                                'bi-weekly': '2weeks',
                                'biweekly': '2weeks',
                                'quarterly': '3months',
                                'semi-annually': '6months',
                                'semiannually': '6months',
                                'annually': 'year',
                                'yearly': 'year',
                                'market_day': 'market_day',
                                'calendar_day': 'calendar_day',
                                'never': 'none',
                                'none': 'none'
                            }
                            rebalancing_frequency = frequency_mapping.get(rebalancing_frequency, rebalancing_frequency)
                            
                            if last_date and rebalancing_frequency != 'none':
                                # Ensure last_date is a naive datetime object
                                if isinstance(last_date, str):
                                    last_date_for_timer = pd.to_datetime(last_date)
                                else:
                                    last_date_for_timer = last_date
                                if hasattr(last_date_for_timer, 'tzinfo') and last_date_for_timer.tzinfo is not None:
                                    last_date_for_timer = last_date_for_timer.replace(tzinfo=None)
                                next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                                    rebalancing_frequency, last_date_for_timer
                                )
                                
                                if next_date and time_until:
                                    st.markdown("---")
                                    st.markdown("**⏰ Next Rebalance Timer**")
                                    
                                    # Create columns for timer display
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            label="Time Until Next Rebalance",
                                            value=format_time_until(time_until),
                                            delta=None
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            label="Target Rebalance Date",
                                            value=next_date.strftime("%B %d, %Y"),
                                            delta=None
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            label="Rebalancing Frequency",
                                            value=rebalancing_frequency.replace('_', ' ').title(),
                                            delta=None
                                        )
                                    
                                    # Add a progress bar showing progress to next rebalance
                                    if rebalancing_frequency in ['week', '2weeks', 'month', '3months', '6months', 'year']:
                                        # Calculate progress percentage
                                        if hasattr(last_date_for_timer, 'to_pydatetime'):
                                            last_rebal_datetime = last_date_for_timer.to_pydatetime()
                                        else:
                                            last_rebal_datetime = last_date_for_timer
                                        
                                        total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                                        elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                                        progress = min(max(elapsed_period / total_period, 0), 1)
                                        
                                        st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
                        except Exception as e:
                            pass  # Silently ignore timer calculation errors

                        # Main "Rebalance as of today" plot and table for fallback scenario
                        st.markdown("---")
                        st.markdown("**🔄 Rebalance as of Today**")
                        
                        # Get momentum-based calculated weights for today's rebalancing from stored snapshot (fallback scenario)
                        today_weights = {}
                        
                        # Get the stored today_weights_map from snapshot data
                        snapshot = st.session_state.get('strategy_comparison_snapshot_data', {})
                        today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
                        
                        if selected_portfolio_detail in today_weights_map:
                            today_weights = today_weights_map.get(selected_portfolio_detail, {})
                        else:
                            # Fallback to current allocation if no stored weights found
                            final_date = last_date
                            final_alloc = last_alloc
                            today_weights = final_alloc
                        
                        # Create labels and values for the plot
                        labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
                        vals_today = [float(today_weights[k]) * 100 for k in labels_today]
                        
                        # Create a larger plot for the main rebalancing representation
                        col_main_plot, col_main_table = st.columns([2, 1])
                        
                        with col_main_plot:
                            st.markdown(f"**Target Allocation if Rebalanced Today**")
                            fig_today = go.Figure()
                            fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_today.update_traces(textinfo='percent+label')
                            fig_today.update_layout(
                                template='plotly_dark', 
                                margin=dict(t=30),
                                height=500,  # Make it bigger
                                showlegend=True
                            )
                            st.plotly_chart(fig_today, use_container_width=True, key=f"multi_today_fallback_{selected_portfolio_detail}")
                        
                        with col_main_table:
                            # Add the "Rebalance as of today" table for fallback
                            try:
                                # Get portfolio configuration for calculations
                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                
                                if portfolio_cfg:
                                    portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                                    
                                    # Get raw data for price calculations
                                    raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                    
                                    def _price_on_or_before(df, target_date):
                                        try:
                                            idx = df.index[df.index <= pd.to_datetime(target_date)]
                                            if len(idx) == 0:
                                                return None
                                            return float(df.loc[idx[-1], 'Close'])
                                        except Exception:
                                            return None

                                    def build_table_from_alloc(alloc_dict, price_date, label):
                                        rows = []
                                        for tk in sorted(alloc_dict.keys()):
                                            alloc_pct = float(alloc_dict.get(tk, 0))
                                            if tk == 'CASH':
                                                price = None
                                                shares = 0
                                                total_val = portfolio_value * alloc_pct
                                            else:
                                                df = raw_data.get(tk)
                                                price = None
                                                if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                    if price_date is None:
                                                        # use latest price
                                                        try:
                                                            price = float(df['Close'].iloc[-1])
                                                        except Exception:
                                                            price = None
                                                    else:
                                                        price = _price_on_or_before(df, price_date)
                                                try:
                                                    if price and price > 0:
                                                        allocation_value = portfolio_value * alloc_pct
                                                        # allow fractional shares shown to 1 decimal place
                                                        shares = round(allocation_value / price, 1)
                                                        total_val = shares * price
                                                    else:
                                                        shares = 0.0
                                                        total_val = portfolio_value * alloc_pct
                                                except Exception:
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct

                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            rows.append({
                                                'Ticker': tk,
                                                'Allocation %': alloc_pct * 100,
                                                'Price ($)': price if price is not None else float('nan'),
                                                'Shares': shares,
                                                'Total Value ($)': total_val,
                                                '% of Portfolio': pct_of_port,
                                            })

                                        df_table = pd.DataFrame(rows).set_index('Ticker')
                                        # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                        df_display = df_table.copy()
                                        show_cash = False
                                        if 'CASH' in df_display.index:
                                            cash_val = None
                                            if 'Total Value ($)' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Total Value ($)']
                                            elif 'Shares' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Shares']
                                            try:
                                                show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                            except Exception:
                                                show_cash = False
                                            if not show_cash:
                                                df_display = df_display.drop('CASH')

                                        # formatting for display
                                        fmt = {
                                            'Allocation %': '{:,.1f}%',
                                            'Price ($)': '${:,.2f}',
                                            'Shares': '{:,.1f}',
                                            'Total Value ($)': '${:,.2f}',
                                            '% of Portfolio': '{:,.2f}%'
                                        }
                                        try:
                                            st.markdown(f"**{label}**")
                                            sty = df_display.style.format(fmt)
                                            if 'CASH' in df_table.index and show_cash:
                                                def _highlight_cash_row(s):
                                                    if s.name == 'CASH':
                                                        return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                sty = sty.apply(_highlight_cash_row, axis=1)
                                            st.dataframe(sty, use_container_width=True)
                                        except Exception:
                                            st.dataframe(df_display, use_container_width=True)
                                    
                                # "Rebalance as of today" table for fallback (use momentum-based calculated weights)
                                build_table_from_alloc(today_weights, None, f"Target Allocation if Rebalanced Today")
                                
                            except Exception as e:
                                print(f"[REBALANCE TODAY TABLE DEBUG] Failed to render fallback rebalance today table for {selected_portfolio_detail}: {e}")

                        # Other rebalancing plots (smaller, placed after the main one) for fallback
                        st.markdown("---")
                        st.markdown("**📊 Historical Rebalancing Comparison**")
                        
                        col_plot1, col_plot2 = st.columns(2)
                        with col_plot1:
                            st.markdown(f"**Last Rebalance Allocation (as of {last_date.date()})**")
                            fig_rebal = go.Figure()
                            fig_rebal.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_rebal.update_traces(textinfo='percent+label')
                            fig_rebal.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                            st.plotly_chart(fig_rebal, use_container_width=True, key=f"multi_rebal_fallback_{selected_portfolio_detail}")
                        with col_plot2:
                            st.markdown(f"**Current Allocation (as of {last_date.date()})**")
                            fig_final = go.Figure()
                            fig_final.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.3))
                            fig_final.update_traces(textinfo='percent+label')
                            fig_final.update_layout(template='plotly_dark', margin=dict(t=30), height=400)
                            st.plotly_chart(fig_final, use_container_width=True, key=f"multi_final_fallback_{selected_portfolio_detail}")
                            
                            # Add allocation tables for fallback case as well
                            try:
                                # Get portfolio configuration for calculations
                                portfolio_configs = st.session_state.get('strategy_comparison_portfolio_configs', [])
                                portfolio_cfg = next((cfg for cfg in portfolio_configs if cfg.get('name') == selected_portfolio_detail), None)
                                
                                if portfolio_cfg:
                                    portfolio_value = float(portfolio_cfg.get('initial_value', 0) or 0)
                                    
                                    # Get raw data for price calculations
                                    raw_data = st.session_state.get('strategy_comparison_raw_data', {})
                                    
                                    def _price_on_or_before(df, target_date):
                                        try:
                                            idx = df.index[df.index <= pd.to_datetime(target_date)]
                                            if len(idx) == 0:
                                                return None
                                            return float(df.loc[idx[-1], 'Close'])
                                        except Exception:
                                            return None

                                    def build_table_from_alloc(alloc_dict, price_date, label):
                                        rows = []
                                        for tk in sorted(alloc_dict.keys()):
                                            alloc_pct = float(alloc_dict.get(tk, 0))
                                            if tk == 'CASH':
                                                price = None
                                                shares = 0
                                                total_val = portfolio_value * alloc_pct
                                            else:
                                                df = raw_data.get(tk)
                                                price = None
                                                if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                                                    if price_date is None:
                                                        # use latest price
                                                        try:
                                                            price = float(df['Close'].iloc[-1])
                                                        except Exception:
                                                            price = None
                                                    else:
                                                        price = _price_on_or_before(df, price_date)
                                                try:
                                                    if price and price > 0:
                                                        allocation_value = portfolio_value * alloc_pct
                                                        # allow fractional shares shown to 1 decimal place
                                                        shares = round(allocation_value / price, 1)
                                                        total_val = shares * price
                                                    else:
                                                        shares = 0.0
                                                        total_val = portfolio_value * alloc_pct
                                                except Exception:
                                                    shares = 0
                                                    total_val = portfolio_value * alloc_pct

                                            pct_of_port = (total_val / portfolio_value * 100) if portfolio_value > 0 else 0
                                            rows.append({
                                                'Ticker': tk,
                                                'Allocation %': alloc_pct * 100,
                                                'Price ($)': price if price is not None else float('nan'),
                                                'Shares': shares,
                                                'Total Value ($)': total_val,
                                                '% of Portfolio': pct_of_port,
                                            })

                                        df_table = pd.DataFrame(rows).set_index('Ticker')
                                        # Decide whether to show CASH row: hide if Total Value is zero or Shares zero/NaN
                                        df_display = df_table.copy()
                                        show_cash = False
                                        if 'CASH' in df_display.index:
                                            cash_val = None
                                            if 'Total Value ($)' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Total Value ($)']
                                            elif 'Shares' in df_display.columns:
                                                cash_val = df_display.at['CASH', 'Shares']
                                            try:
                                                show_cash = bool(cash_val and not pd.isna(cash_val) and cash_val != 0)
                                            except Exception:
                                                show_cash = False
                                            if not show_cash:
                                                df_display = df_display.drop('CASH')

                                        # formatting for display
                                        fmt = {
                                            'Allocation %': '{:,.1f}%',
                                            'Price ($)': '${:,.2f}',
                                            'Shares': '{:,.1f}',
                                            'Total Value ($)': '${:,.2f}',
                                            '% of Portfolio': '{:,.2f}%'
                                        }
                                        try:
                                            st.markdown(f"**{label}**")
                                            sty = df_display.style.format(fmt)
                                            if 'CASH' in df_table.index and show_cash:
                                                def _highlight_cash_row(s):
                                                    if s.name == 'CASH':
                                                        return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                                                sty = sty.apply(_highlight_cash_row, axis=1)
                                            st.dataframe(sty, use_container_width=True)
                                        except Exception:
                                            st.dataframe(df_display, use_container_width=True)
                                    
                                    # Current allocation table (use final_date's latest available prices as of now)
                                    build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
                                    
                            except Exception as e:
                                print(f"[ALLOC TABLE DEBUG] Failed to render fallback allocation tables for {selected_portfolio_detail}: {e}")

    else:
        st.info("Configuration is ready. Press 'Run Backtests' to see results.")
