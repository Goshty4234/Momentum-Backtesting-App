import streamlit as st
import datetime
from datetime import timedelta, time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import contextlib
import warnings
warnings.filterwarnings('ignore')

def check_currency_warning(tickers):
    """
    Check if any tickers are non-USD and display a warning.
    """
    non_usd_suffixes = ['.TO', '.V', '.CN', '.AX', '.L', '.PA', '.AS', '.SW', '.T', '.HK', '.KS', '.TW', '.JP']
    non_usd_tickers = []
    
    for ticker in tickers:
        if any(ticker.endswith(suffix) for suffix in non_usd_suffixes):
            non_usd_tickers.append(ticker)
    
    if non_usd_tickers:
        st.warning(f"⚠️ **Currency Warning**: The following tickers are not in USD: {', '.join(non_usd_tickers)}. "
                  f"Currency conversion is not taken into account, which may affect allocation accuracy. "
                  f"Consider using USD equivalents for more accurate results.")
st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis", page_icon="📈")
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

# ==============================================================================
# PAGE-SCOPED SESSION STATE INITIALIZATION - ALLOCATIONS PAGE
# ==============================================================================
# Ensure complete independence from other pages by using page-specific session keys
if 'allocations_page_initialized' not in st.session_state:
    st.session_state.allocations_page_initialized = True

# Initialize page-specific session state with default configurations
if 'alloc_portfolio_configs' not in st.session_state:
    # Default configuration for allocations page
    st.session_state.alloc_portfolio_configs = [
        {
            'name': 'Allocation Portfolio',
            'stocks': [
                {'ticker': 'SPY', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'QQQ', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'GLD', 'allocation': 0.25, 'include_dividends': True},
                {'ticker': 'TLT', 'allocation': 0.25, 'include_dividends': True},
            ],
            'benchmark_ticker': '^GSPC',
            'initial_value': 10000,
                          'added_amount': 0,
              'added_frequency': 'none',
              'rebalancing_frequency': 'Monthly',
              'start_date_user': None,
              'end_date_user': None,
              'start_with': 'all',
              'use_momentum': True,
            'momentum_strategy': 'Classic',
            'negative_momentum_strategy': 'Cash',
            'momentum_windows': [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ],
            'calc_beta': True,
            'calc_volatility': True,
            'beta_window_days': 365,
            'exclude_days_beta': 30,
            'vol_window_days': 365,
            'exclude_days_vol': 30,
        }
    ]
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

# Clean up any existing portfolio configs to remove unused settings
if 'alloc_portfolio_configs' in st.session_state:
    for config in st.session_state.alloc_portfolio_configs:
        config.pop('use_relative_momentum', None)
        config.pop('equal_if_all_negative', None)
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""

# ==============================================================================
# END PAGE-SCOPED SESSION STATE INITIALIZATION
# ==============================================================================

# Use page-scoped active portfolio for the allocations page
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] if 'alloc_portfolio_configs' in st.session_state and 'alloc_active_portfolio_index' in st.session_state else None
if active_portfolio:
    # Removed duplicate Portfolio Name input field
    if st.session_state.get('alloc_rerun_flag', False):
        st.session_state.alloc_rerun_flag = False
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

st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis")

st.title("Portfolio Allocations")
st.markdown("Use the forms below to configure and run backtests to obtain allocation insights.")

# Portfolio Name
if 'alloc_portfolio_name' not in st.session_state:
    st.session_state.alloc_portfolio_name = "Allocation Portfolio"
alloc_portfolio_name = st.text_input("Portfolio Name", value=st.session_state.alloc_portfolio_name, key="alloc_portfolio_name_input")
st.session_state.alloc_portfolio_name = alloc_portfolio_name

# Sync portfolio name with active portfolio configuration
if 'alloc_active_portfolio_index' in st.session_state:
    active_idx = st.session_state.alloc_active_portfolio_index
    if 'alloc_portfolio_configs' in st.session_state:
        if active_idx < len(st.session_state.alloc_portfolio_configs):
            st.session_state.alloc_portfolio_configs[active_idx]['name'] = alloc_portfolio_name

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
    'start_with': 'oldest',
        'use_momentum': False,
        'momentum_windows': [],
    'calc_beta': True,
    'calc_volatility': True,
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
    elif freq == "Annually":
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

# -----------------------
# Single-backtest core (adapted from your code, robust)
# -----------------------
def single_backtest(config, sim_index, reindexed_data):
    stocks_list = config.get('stocks', [])
    raw_tickers = [s.get('ticker') for s in stocks_list if s.get('ticker')]
    # Filter out tickers not present in reindexed_data to avoid crashes for invalid tickers
    if reindexed_data:
        tickers = [t for t in raw_tickers if t in reindexed_data]
    else:
        tickers = raw_tickers[:]
    missing_tickers = [t for t in raw_tickers if t not in tickers]
    if missing_tickers:
        # Log a warning and ignore unknown tickers
        print(f"[ALLOC WARN] Ignoring unknown or missing tickers: {missing_tickers}")
    # Handle duplicate tickers by summing their allocations
    allocations = {}
    include_dividends = {}
    for s in stocks_list:
        if s.get('ticker') and s.get('ticker') in tickers:
            ticker = s.get('ticker')
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
    benchmark_ticker = config.get('benchmark_ticker')
    initial_value = config.get('initial_value', 0)
    # Allocation tracker: ignore added cash for this mode. Use initial_value as current portfolio value.
    added_amount = 0
    added_frequency = 'none'
    # Map frequency to ensure compatibility with get_dates_by_freq
    raw_rebalancing_frequency = config.get('rebalancing_frequency', 'none')
    def map_frequency_for_backtest(freq):
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
    
    rebalancing_frequency = map_frequency_for_backtest(raw_rebalancing_frequency)
    use_momentum = config.get('use_momentum', True)
    momentum_windows = config.get('momentum_windows', [])
    calc_beta = config.get('calc_beta', False)
    calc_volatility = config.get('calc_volatility', False)
    beta_window_days = config.get('beta_window_days', 365)
    exclude_days_beta = config.get('exclude_days_beta', 30)
    vol_window_days = config.get('vol_window_days', 365)
    exclude_days_vol = config.get('exclude_days_vol', 30)
    current_data = {t: reindexed_data[t] for t in tickers + [benchmark_ticker] if t in reindexed_data}
    # Respect start_with setting: 'all' (default) or 'oldest' (add assets over time)
    start_with = config.get('start_with', 'all')
    # Precompute first-valid dates for each ticker to decide availability
    start_dates_config = {}
    for t in tickers:
        if t in reindexed_data and isinstance(reindexed_data.get(t), pd.DataFrame):
            fd = reindexed_data[t].first_valid_index()
            start_dates_config[t] = fd if fd is not None else pd.NaT
        else:
            start_dates_config[t] = pd.NaT
    dates_added = set()
    dates_rebal = sorted(get_dates_by_freq(rebalancing_frequency, sim_index[0], sim_index[-1], sim_index))

    # Dictionaries to store historical data for new tables
    historical_allocations = {}
    historical_metrics = {}

    def calculate_momentum(date, current_assets, momentum_windows):
        cumulative_returns, valid_assets = {}, []
        filtered_windows = [w for w in momentum_windows if w.get("weight", 0) > 0]
        # Normalize weights so they sum to 1 (same as app.py)
        total_weight = sum(w.get("weight", 0) for w in filtered_windows)
        if total_weight == 0:
            normalized_weights = [0 for _ in filtered_windows]
        else:
            normalized_weights = [w.get("weight", 0) / total_weight for w in filtered_windows]
        # Only consider assets that exist in current_data (filtered earlier)
        candidate_assets = [t for t in current_assets if t in current_data]
        for t in candidate_assets:
            is_valid, asset_returns = True, 0.0
            df_t = current_data.get(t)
            if not (isinstance(df_t, pd.DataFrame) and 'Close' in df_t.columns and not df_t['Close'].dropna().empty):
                # no usable data for this ticker
                continue
            for idx, window in enumerate(filtered_windows):
                lookback, exclude = window.get("lookback", 0), window.get("exclude", 0)
                weight = normalized_weights[idx]
                start_mom = date - pd.Timedelta(days=lookback)
                end_mom = date - pd.Timedelta(days=exclude)
                sd = start_dates_config.get(t, pd.NaT)
                # If no start date or asset starts after required lookback, mark invalid
                if pd.isna(sd) or sd > start_mom:
                    is_valid = False
                    break
                try:
                    price_start_index = df_t.index.asof(start_mom)
                    price_end_index = df_t.index.asof(end_mom)
                except Exception:
                    is_valid = False
                    break
                if pd.isna(price_start_index) or pd.isna(price_end_index):
                    is_valid = False
                    break
                price_start = df_t.loc[price_start_index, "Close"]
                price_end = df_t.loc[price_end_index, "Close"]
                if pd.isna(price_start) or pd.isna(price_end) or price_start == 0:
                    is_valid = False
                    break
                ret = (price_end - price_start) / price_start
                asset_returns += ret * weight
            if is_valid:
                cumulative_returns[t] = asset_returns
                valid_assets.append(t)
        return cumulative_returns, valid_assets

    def calculate_momentum_weights(returns, valid_assets, date, momentum_strategy='Classic', negative_momentum_strategy='Cash'):
        if not valid_assets: return {}, {}
        rets = {t: returns[t] for t in valid_assets if not pd.isna(returns[t])}
        if not rets: return {}, {}
        beta_vals, vol_vals = {}, {}
        metrics = {t: {} for t in tickers}
        if calc_beta or calc_volatility:
            df_bench = current_data.get(benchmark_ticker)
            if calc_beta:
                start_beta = date - pd.Timedelta(days=beta_window_days)
                end_beta = date - pd.Timedelta(days=exclude_days_beta)
            if calc_volatility:
                start_vol = date - pd.Timedelta(days=vol_window_days)
                end_vol = date - pd.Timedelta(days=exclude_days_vol)
            for t in valid_assets:
                df_t = current_data[t]
                if calc_beta and df_bench is not None:
                    mask_beta = (df_t.index >= start_beta) & (df_t.index <= end_beta)
                    returns_t_beta = df_t.loc[mask_beta, "Price_change"]
                    mask_bench_beta = (df_bench.index >= start_beta) & (df_bench.index <= end_beta)
                    returns_bench_beta = df_bench.loc[mask_bench_beta, "Price_change"]
                    if len(returns_t_beta) < 2 or len(returns_bench_beta) < 2:
                        beta_vals[t] = np.nan
                    else:
                        covariance = np.cov(returns_t_beta, returns_bench_beta)[0,1]
                        variance = np.var(returns_bench_beta)
                        beta_vals[t] = covariance/variance if variance>0 else np.nan
                    metrics[t]['Beta'] = beta_vals[t]
                if calc_volatility:
                    mask_vol = (df_t.index >= start_vol) & (df_t.index <= end_vol)
                    returns_t_vol = df_t.loc[mask_vol, "Price_change"]
                    if len(returns_t_vol) < 2:
                        vol_vals[t] = np.nan
                    else:
                        vol_vals[t] = returns_t_vol.std() * np.sqrt(252)
                    metrics[t]['Volatility'] = vol_vals[t]
        
        for t in rets:
            metrics[t]['Momentum'] = rets[t]

        # Compute initial weights from raw momentum scores (relative/classic) then apply
        # post-filtering by inverse volatility and inverse absolute beta (app.py approach).
        weights = {}
        # raw momentum values
        rets_keys = list(rets.keys())
        all_negative = all(r <= 0 for r in rets.values())

        # Helper: detect relative mode from momentum_strategy string
        relative_mode = isinstance(momentum_strategy, str) and momentum_strategy.lower().startswith('relat')

        if all_negative:
            if negative_momentum_strategy == 'Cash':
                weights = {t: 0 for t in rets_keys}
            elif negative_momentum_strategy == 'Equal weight':
                weights = {t: 1 / len(rets_keys) for t in rets_keys}
            elif negative_momentum_strategy == 'Relative momentum':
                min_score = min(rets.values())
                offset = -min_score + 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
        else:
            if relative_mode:
                min_score = min(rets.values())
                offset = -min_score + 0.01 if min_score < 0 else 0.01
                shifted = {t: max(0.01, rets[t] + offset) for t in rets_keys}
                ssum = sum(shifted.values())
                weights = {t: shifted[t] / ssum for t in shifted}
            else:
                positive_scores = {t: s for t, s in rets.items() if s > 0}
                if positive_scores:
                    sum_positive = sum(positive_scores.values())
                    weights = {t: positive_scores[t] / sum_positive for t in positive_scores}
                    for t in [t for t in rets_keys if rets.get(t, 0) <= 0]:
                        weights[t] = 0
                else:
                    weights = {t: 0 for t in rets_keys}

        # Apply post-filtering using inverse volatility and inverse absolute beta (like app.py)
        fallback_mode = all_negative and negative_momentum_strategy == 'Equal weight'
        if weights and (calc_volatility or calc_beta) and not fallback_mode:
            filtered_weights = {}
            for t, w in weights.items():
                if w > 0:
                    score = 1.0
                    if calc_volatility:
                        v = vol_vals.get(t, np.nan)
                        if not pd.isna(v) and v > 0:
                            score *= (1.0 / v)
                        else:
                            score *= 0
                    if calc_beta:
                        b = beta_vals.get(t, np.nan)
                        if not pd.isna(b):
                            abs_beta = abs(b)
                            if abs_beta > 0:
                                score *= (1.0 / abs_beta)
                            else:
                                score *= 1.0
                    filtered_weights[t] = w * score
            total_filtered = sum(filtered_weights.values())
            if total_filtered > 0:
                weights = {t: v / total_filtered for t, v in filtered_weights.items()}
            else:
                weights = {}

        for t in weights:
            metrics[t]['Calculated_Weight'] = weights.get(t, 0)

        # Debug: print metrics summary for this rebal date when beta/vol modifiers are active
        if calc_beta or calc_volatility:
            try:
                debug_lines = [
                    f"[MOM DEBUG] Date: {date} | Ticker: {t} | Momentum: {metrics[t].get('Momentum')} | Beta: {metrics[t].get('Beta')} | Vol: {metrics[t].get('Volatility')} | Weight: {weights.get(t, metrics[t].get('Calculated_Weight'))}"
                    for t in rets_keys
                ]
                for ln in debug_lines:
                    print(ln)
            except Exception as e:
                print(f"[MOM DEBUG] Error printing debug metrics: {e}")

        return weights, metrics
        # --- MODIFIED LOGIC END ---

    values = {t: [0.0] for t in tickers}
    unallocated_cash = [0.0]
    unreinvested_cash = [0.0]
    portfolio_no_additions = [initial_value]
    
    # Initial allocation and metric storage
    if not use_momentum:
        # If start_with is 'oldest', only allocate to tickers that are available at the simulation start
        if start_with == 'oldest':
            available_at_start = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= sim_index[0]]
            current_allocations = {t: allocations.get(t, 0) if t in available_at_start else 0 for t in tickers}
        else:
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
                # Non-momentum rebalancing: respect 'start_with' option
                if start_with == 'oldest':
                    # Only consider tickers that have data by this rebalancing date
                    available = [t for t in tickers if start_dates_config.get(t, pd.Timestamp.max) <= date]
                    sum_alloc_avail = sum(allocations.get(t,0) for t in available)
                    if sum_alloc_avail > 0:
                        for t in tickers:
                            if t in available:
                                weight = allocations.get(t,0)/sum_alloc_avail
                                values[t][-1] = current_total * weight
                            else:
                                values[t][-1] = 0
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = 0
                    else:
                        # No assets available yet — keep everything as cash
                        for t in tickers:
                            values[t][-1] = 0
                        unreinvested_cash[-1] = 0
                        unallocated_cash[-1] = current_total
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
# Main App Logic
# -----------------------

from copy import deepcopy
# Use page-scoped session keys so this page does not share state with other pages
if 'alloc_portfolio_configs' not in st.session_state:
    # initialize from existing global configs if present, but deep-copy to avoid shared references
            st.session_state.alloc_portfolio_configs = deepcopy(st.session_state.get('alloc_portfolio_configs', default_configs))
if 'alloc_active_portfolio_index' not in st.session_state:
    st.session_state.alloc_active_portfolio_index = 0
if 'alloc_paste_json_text' not in st.session_state:
    st.session_state.alloc_paste_json_text = ""
if 'alloc_rerun_flag' not in st.session_state:
    st.session_state.alloc_rerun_flag = False

def add_portfolio_callback():
    new_portfolio = default_configs[1].copy()
    new_portfolio['name'] = f"New Portfolio {len(st.session_state.alloc_portfolio_configs) + 1}"
    st.session_state.alloc_portfolio_configs.append(new_portfolio)
    st.session_state.alloc_active_portfolio_index = len(st.session_state.alloc_portfolio_configs) - 1
    st.session_state.alloc_rerun_flag = True

def remove_portfolio_callback():
    if len(st.session_state.alloc_portfolio_configs) > 1:
        st.session_state.alloc_portfolio_configs.pop(st.session_state.alloc_active_portfolio_index)
        st.session_state.alloc_active_portfolio_index = max(0, st.session_state.alloc_active_portfolio_index - 1)
        st.session_state.alloc_rerun_flag = True

def add_stock_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'].append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
    st.session_state.alloc_rerun_flag = True

def remove_stock_callback(ticker):
    """Immediate stock removal callback"""
    try:
        active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
        stocks = active_portfolio['stocks']
        
        # Find and remove the stock with matching ticker
        for i, stock in enumerate(stocks):
            if stock['ticker'] == ticker:
                stocks.pop(i)
                # If this was the last stock, add an empty one
                if len(stocks) == 0:
                    stocks.append({'ticker': '', 'allocation': 0.0, 'include_dividends': True})
                st.session_state.alloc_rerun_flag = True
                break
    except (IndexError, KeyError):
        pass

def normalize_stock_allocations_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    total_alloc = sum(s['allocation'] for s in valid_stocks)
    if total_alloc > 0:
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] /= total_alloc
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(s['allocation'] * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True

def equal_stock_allocation_callback():
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    stocks = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks']
    valid_stocks = [s for s in stocks if s['ticker']]
    if valid_stocks:
        equal_weight = 1.0 / len(valid_stocks)
        for idx, s in enumerate(stocks):
            if s['ticker']:
                s['allocation'] = equal_weight
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = int(equal_weight * 100)
            else:
                s['allocation'] = 0.0
                alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{idx}"
                st.session_state[alloc_key] = 0
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = stocks
    st.session_state.alloc_rerun_flag = True
    
def reset_portfolio_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
        default_cfg_found['name'] = current_name
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = default_cfg_found
    st.session_state.alloc_rerun_flag = True

def reset_stock_selection_callback():
    current_name = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name']
    default_cfg_found = next((cfg for cfg in default_configs if cfg['name'] == current_name), None)
    if default_cfg_found is None:
        default_cfg_found = default_configs[1].copy()
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] = default_cfg_found['stocks']
    st.session_state.alloc_rerun_flag = True

def reset_momentum_windows_callback():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
        {"lookback": 365, "exclude": 30, "weight": 0.5},
        {"lookback": 180, "exclude": 30, "weight": 0.3},
        {"lookback": 120, "exclude": 30, "weight": 0.2},
    ]
    st.session_state.alloc_rerun_flag = True

def reset_beta_callback():
    # Reset beta lookback/exclude to defaults and enable beta calculation for alloc page
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['beta_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_beta'] = 30
    # Ensure checkbox state reflects enabled
    st.session_state.alloc_portfolio_configs[idx]['calc_beta'] = True
    st.session_state['alloc_active_calc_beta'] = True
    st.session_state.alloc_rerun_flag = True

def reset_vol_callback():
    # Reset volatility lookback/exclude to defaults and enable volatility calculation
    idx = st.session_state.alloc_active_portfolio_index
    st.session_state.alloc_portfolio_configs[idx]['vol_window_days'] = 365
    st.session_state.alloc_portfolio_configs[idx]['exclude_days_vol'] = 30
    st.session_state.alloc_portfolio_configs[idx]['calc_volatility'] = True
    st.session_state['alloc_active_calc_vol'] = True
    st.session_state.alloc_rerun_flag = True

def add_momentum_window_callback():
    # Append a new momentum window with modest defaults (alloc page)
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' not in cfg:
        cfg['momentum_windows'] = []
    # default new window
    cfg['momentum_windows'].append({"lookback": 90, "exclude": 30, "weight": 0.1})
    st.session_state.alloc_portfolio_configs[idx] = cfg
    st.session_state.alloc_rerun_flag = True

def remove_momentum_window_callback():
    idx = st.session_state.alloc_active_portfolio_index
    cfg = st.session_state.alloc_portfolio_configs[idx]
    if 'momentum_windows' in cfg and cfg['momentum_windows']:
        cfg['momentum_windows'].pop()
        st.session_state.alloc_portfolio_configs[idx] = cfg
        st.session_state.alloc_rerun_flag = True

def normalize_momentum_weights_callback():
    # Use page-scoped configs for allocations page
    if 'alloc_portfolio_configs' not in st.session_state or 'alloc_active_portfolio_index' not in st.session_state:
        return
    active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
    total_weight = sum(w['weight'] for w in active_portfolio.get('momentum_windows', []))
    if total_weight > 0:
        for idx, w in enumerate(active_portfolio.get('momentum_windows', [])):
            w['weight'] /= total_weight
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{idx}"
            # Sanitize weight to prevent StreamlitValueAboveMaxError
            weight = w['weight']
            if isinstance(weight, (int, float)):
                # Convert decimal to percentage, ensuring it's within bounds
                weight_percentage = max(0.0, min(weight * 100.0, 100.0))
            else:
                # Invalid weight, set to default
                weight_percentage = 10.0
            st.session_state[weight_key] = int(weight_percentage)
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = active_portfolio.get('momentum_windows', [])
    st.session_state.alloc_rerun_flag = True

def paste_json_callback():
    try:
        json_data = json.loads(st.session_state.get('alloc_paste_json_text', '{}'))
        
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
        elif momentum_strategy == 'Relative momentum':
            momentum_strategy = 'Relative Momentum'
        elif momentum_strategy not in ['Classic', 'Relative Momentum']:
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
        
        # Map frequency values from app.py format to Allocations format
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
        
        # Allocations page specific: ensure all required fields are present
        # and ignore fields that are specific to other pages
        allocations_config = {
            'name': json_data.get('name', 'Allocation Portfolio'),
            'stocks': stocks,
            'benchmark_ticker': json_data.get('benchmark_ticker', '^GSPC'),
            'initial_value': json_data.get('initial_value', 10000),
            'added_amount': json_data.get('added_amount', 0),  # Allocations page typically doesn't use additions
            'added_frequency': map_frequency(json_data.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
            'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': json_data.get('start_date_user'),
            'end_date_user': json_data.get('end_date_user'),
            'start_with': json_data.get('start_with', 'all'),
            'use_momentum': json_data.get('use_momentum', True),
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
        
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index] = allocations_config
        st.success("Portfolio configuration updated from JSON (Allocations page).")
        st.info(f"Final stocks list: {[s['ticker'] for s in allocations_config['stocks']]}")
        st.info(f"Final momentum windows: {allocations_config['momentum_windows']}")
        st.info(f"Final use_momentum: {allocations_config['use_momentum']}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check the text and try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.session_state.alloc_rerun_flag = True

def update_active_portfolio_index():
    # Allocation page: keep a page-scoped index. If a selector exists, respect it; otherwise default to 0
    selected_name = st.session_state.get('alloc_portfolio_selector', None)
    portfolio_configs = st.session_state.get('alloc_portfolio_configs', [])
    portfolio_names = [cfg.get('name', '') for cfg in portfolio_configs]
    if selected_name and selected_name in portfolio_names:
        st.session_state.alloc_active_portfolio_index = portfolio_names.index(selected_name)
    else:
        st.session_state.alloc_active_portfolio_index = 0 if portfolio_names else None
    st.session_state.alloc_rerun_flag = True

def update_name():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['name'] = st.session_state.get('alloc_active_name', '')

def update_initial():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['initial_value'] = st.session_state.get('alloc_active_initial', 0)

def update_added_amount():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_amount'] = st.session_state.get('alloc_active_added_amount', 0)

def update_add_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['added_frequency'] = st.session_state.get('alloc_active_add_freq', 'none')

def update_rebal_freq():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['rebalancing_frequency'] = st.session_state.get('alloc_active_rebal_freq', 'none')

def update_benchmark():
    # Convert benchmark ticker to uppercase
    benchmark_val = st.session_state.get('alloc_active_benchmark', '')
    upper_benchmark = benchmark_val.upper()
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['benchmark_ticker'] = upper_benchmark
    # Update the widget to show uppercase value
    st.session_state['alloc_active_benchmark'] = upper_benchmark

def update_use_momentum():
    current_val = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('use_momentum', True)
    new_val = st.session_state.get('alloc_active_use_momentum', True)
    if current_val != new_val:
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['use_momentum'] = new_val
        if new_val:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = [
                {"lookback": 365, "exclude": 30, "weight": 0.5},
                {"lookback": 180, "exclude": 30, "weight": 0.3},
                {"lookback": 120, "exclude": 30, "weight": 0.2},
            ]
        else:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['momentum_windows'] = []
        st.session_state.alloc_rerun_flag = True



def update_calc_beta():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_beta'] = st.session_state.get('alloc_active_calc_beta', True)

def update_beta_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['beta_window_days'] = st.session_state.get('alloc_active_beta_window', 365)

def update_beta_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_beta'] = st.session_state.get('alloc_active_beta_exclude', 30)

def update_calc_vol():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['calc_volatility'] = st.session_state.get('alloc_active_calc_vol', True)

def update_vol_window():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['vol_window_days'] = st.session_state.get('alloc_active_vol_window', 365)

def update_vol_exclude():
    st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['exclude_days_vol'] = st.session_state.get('alloc_active_vol_exclude', 30)

# Sidebar simplified for single-portfolio allocation tracker
st.sidebar.title("Allocation Tracker")


# Work with the first portfolio as active (single-portfolio mode). Keep inputs accessible.
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
# Do not show portfolio name in allocation tracker. Keep a page-scoped session key for compatibility.
if "alloc_active_name" not in st.session_state:
    st.session_state["alloc_active_name"] = active_portfolio['name']

col_left, col_right = st.columns([1, 1])
with col_left:
    if "alloc_active_initial" not in st.session_state:
        # Treat this as the current portfolio value (not a backtest initial cash)
        st.session_state["alloc_active_initial"] = int(active_portfolio.get('initial_value', 0))
    st.number_input("Portfolio Value ($)", min_value=0, step=1000, format="%d", key="alloc_active_initial", on_change=update_initial, help="Current total portfolio value used to compute required shares.")
# Removed Added Amount / Added Frequency UI - allocation tracker is not running periodic additions

# Swap positions: show Rebalancing Frequency first, then Added Frequency.
# Use two equal-width columns and make selectboxes use the container width so they match visually.
col_freq_rebal, col_freq_add = st.columns([1, 1])
freq_options = ["Never", "Weekly", "Biweekly", "Monthly", "Quarterly", "Semiannually", "Annually"]
with col_freq_rebal:
    if "alloc_active_rebal_freq" not in st.session_state:
        st.session_state["alloc_active_rebal_freq"] = active_portfolio['rebalancing_frequency']
    st.selectbox("Rebalancing Frequency", freq_options, key="alloc_active_rebal_freq", on_change=update_rebal_freq, help="How often the portfolio is rebalanced.", )
# Note: Added Frequency removed for allocation tracker

# Rebalancing and Added Frequency explanation removed for allocation tracker UI

if "alloc_active_benchmark" not in st.session_state:
    st.session_state["alloc_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, used for beta calculation)", key="alloc_active_benchmark", on_change=update_benchmark)

st.subheader("Stocks")
col_stock_buttons = st.columns([0.3, 0.3, 0.3, 0.1])
with col_stock_buttons[0]:
    if st.button("Normalize Stocks %", on_click=normalize_stock_allocations_callback, use_container_width=True):
        pass
with col_stock_buttons[1]:
    if st.button("Equal Allocation %", on_click=equal_stock_allocation_callback, use_container_width=True):
        pass
with col_stock_buttons[2]:
    if st.button("Reset Stocks", on_click=reset_stock_selection_callback, use_container_width=True):
        pass

# Calculate live total stock allocation
valid_stocks = [s for s in st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'] if s['ticker']]
total_stock_allocation = sum(s['allocation'] for s in valid_stocks)

if active_portfolio['use_momentum']:
    st.info("Stock allocations are not used directly for Momentum strategies.")
else:
    if abs(total_stock_allocation - 1.0) > 0.001:
        st.warning(f"Total stock allocation is {total_stock_allocation*100:.2f}%, not 100%. Click 'Normalize' to fix.")
    else:
        st.success(f"Total stock allocation is {total_stock_allocation*100:.2f}%.")

def update_stock_allocation(index):
    try:
        key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['allocation'] = float(val) / 100.0
    except Exception:
        # Ignore transient errors (e.g., active_portfolio_index changed); UI will reflect state on next render
        return


def update_stock_ticker(index):
    try:
        key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        
        # Convert the input value to uppercase
        upper_val = val.upper()

        # Update the portfolio configuration with the uppercase value
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['ticker'] = upper_val
        
        # Update the text box's state to show the uppercase value
        st.session_state[key] = upper_val

    except Exception:
        # Defensive: if portfolio index or structure changed, skip silently
        return


def update_stock_dividends(index):
    try:
        key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][index]['include_dividends'] = bool(val)
    except Exception:
        return

# Update active_portfolio
active_portfolio = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]
 
for i in range(len(active_portfolio['stocks'])):
    stock = active_portfolio['stocks'][i]
    col_t, col_a, col_d, col_b = st.columns([0.2, 0.2, 0.3, 0.15])
    with col_t:
        ticker_key = f"alloc_ticker_{st.session_state.alloc_active_portfolio_index}_{i}"
        if ticker_key not in st.session_state:
            st.session_state[ticker_key] = stock['ticker']
        st.text_input("Ticker", key=ticker_key, label_visibility="visible", on_change=update_stock_ticker, args=(i,))
    with col_a:
        use_mom = st.session_state.get('alloc_active_use_momentum', active_portfolio.get('use_momentum', True))
        if not use_mom:
            alloc_key = f"alloc_input_alloc_{st.session_state.alloc_active_portfolio_index}_{i}"
            if alloc_key not in st.session_state:
                st.session_state[alloc_key] = int(stock['allocation'] * 100)
            st.number_input("Allocation %", min_value=0, step=1, format="%d", key=alloc_key, label_visibility="visible", on_change=update_stock_allocation, args=(i,))
            if st.session_state[alloc_key] != int(stock['allocation'] * 100):
                st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['allocation'] = st.session_state[alloc_key] / 100.0
        else:
            st.write("")
    with col_d:
        div_key = f"alloc_div_{st.session_state.alloc_active_portfolio_index}_{i}"
        if div_key not in st.session_state:
            st.session_state[div_key] = stock['include_dividends']
        st.checkbox("Include Dividends", key=div_key)
        if st.session_state[div_key] != stock['include_dividends']:
            st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index]['stocks'][i]['include_dividends'] = st.session_state[div_key]
    with col_b:
        st.write("")
        if st.button("Remove", key=f"alloc_rem_stock_{st.session_state.alloc_active_portfolio_index}_{i}_{stock['ticker']}_{id(stock)}", on_click=remove_stock_callback, args=(stock['ticker'],)):
            pass

if st.button("Add Stock", on_click=add_stock_callback):
    pass

st.subheader("Strategy")
if "alloc_active_use_momentum" not in st.session_state:
    st.session_state["alloc_active_use_momentum"] = active_portfolio['use_momentum']
st.checkbox("Use Momentum Strategy", key="alloc_active_use_momentum", on_change=update_use_momentum, help="Enables momentum-based weighting of stocks.")

if active_portfolio['use_momentum']:
    st.markdown("---")
    col_mom_options, col_beta_vol = st.columns(2)
    with col_mom_options:
        st.markdown("**Momentum Strategy Options**")
        momentum_strategy = st.selectbox(
            "Momentum strategy when NOT all negative:",
            ["Classic", "Relative Momentum"],
            index=["Classic", "Relative Momentum"].index(active_portfolio.get('momentum_strategy', 'Classic')),
            key=f"momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        )
        negative_momentum_strategy = st.selectbox(
            "Strategy when ALL momentum scores are negative:",
            ["Cash", "Equal weight", "Relative momentum"],
            index=["Cash", "Equal weight", "Relative momentum"].index(active_portfolio.get('negative_momentum_strategy', 'Cash')),
            key=f"negative_momentum_strategy_{st.session_state.alloc_active_portfolio_index}"
        )
        active_portfolio['momentum_strategy'] = momentum_strategy
        active_portfolio['negative_momentum_strategy'] = negative_momentum_strategy
        st.markdown("💡 **Note:** These options control how weights are assigned based on momentum scores.")

    with col_beta_vol:
        if "alloc_active_calc_beta" not in st.session_state:
            st.session_state["alloc_active_calc_beta"] = active_portfolio.get('calc_beta', True)
        st.checkbox("Include Beta in momentum weighting", key="alloc_active_calc_beta", on_change=update_calc_beta, help="Incorporates a stock's Beta (volatility relative to the benchmark) into its momentum score.")
        if st.session_state.get('alloc_active_calc_beta', False):
            if "alloc_active_beta_window" not in st.session_state:
                st.session_state["alloc_active_beta_window"] = active_portfolio['beta_window_days']
            if "alloc_active_beta_exclude" not in st.session_state:
                st.session_state["alloc_active_beta_exclude"] = active_portfolio['exclude_days_beta']
            st.number_input("Beta Lookback (days)", min_value=1, key="alloc_active_beta_window", on_change=update_beta_window)
            st.number_input("Beta Exclude (days)", min_value=0, key="alloc_active_beta_exclude", on_change=update_beta_exclude)
            if st.button("Reset Beta", on_click=reset_beta_callback):
                pass
        if "alloc_active_calc_vol" not in st.session_state:
            st.session_state["alloc_active_calc_vol"] = active_portfolio.get('calc_volatility', True)
        st.checkbox("Include Volatility in momentum weighting", key="alloc_active_calc_vol", on_change=update_calc_vol, help="Incorporates a stock's volatility (standard deviation of returns) into its momentum score.")
        if st.session_state.get('alloc_active_calc_vol', False):
            if "alloc_active_vol_window" not in st.session_state:
                st.session_state["alloc_active_vol_window"] = active_portfolio['vol_window_days']
            if "alloc_active_vol_exclude" not in st.session_state:
                st.session_state["alloc_active_vol_exclude"] = active_portfolio['exclude_days_vol']
            st.number_input("Volatility Lookback (days)", min_value=1, key="alloc_active_vol_window", on_change=update_vol_window)
            st.number_input("Volatility Exclude (days)", min_value=0, key="alloc_active_vol_exclude", on_change=update_vol_exclude)
            if st.button("Reset Volatility", on_click=reset_vol_callback):
                pass
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
        key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['lookback'] = st.session_state.get(key, None)

    def update_momentum_exclude(index):
        key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['exclude'] = st.session_state.get(key, None)
    
    def update_momentum_weight(index):
        key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{index}"
        val = st.session_state.get(key, None)
        if val is None:
            return
        momentum_windows = st.session_state.alloc_portfolio_configs[st.session_state.alloc_active_portfolio_index].get('momentum_windows', [])
        if index < len(momentum_windows):
            momentum_windows[index]['weight'] = val / 100.0

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

    for j in range(len(active_portfolio.get('momentum_windows', []))):
        with st.container():
            col_mw1, col_mw2, col_mw3 = st.columns(3)
            lookback_key = f"alloc_lookback_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            exclude_key = f"alloc_exclude_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            weight_key = f"alloc_weight_input_active_{st.session_state.alloc_active_portfolio_index}_{j}"
            
            # Initialize session state values if not present
            if lookback_key not in st.session_state:
                st.session_state[lookback_key] = int(active_portfolio['momentum_windows'][j]['lookback'])
            if exclude_key not in st.session_state:
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
                st.number_input(f"Lookback {j+1}", min_value=1, key=lookback_key, label_visibility="collapsed", on_change=update_momentum_lookback, args=(j,))
            with col_mw2:
                st.number_input(f"Exclude {j+1}", min_value=0, key=exclude_key, label_visibility="collapsed", on_change=update_momentum_exclude, args=(j,))
            with col_mw3:
                st.number_input(f"Weight {j+1}", min_value=0, max_value=100, step=1, format="%d", key=weight_key, label_visibility="collapsed", on_change=update_momentum_weight, args=(j,))
else:
    
    active_portfolio['momentum_windows'] = []

with st.expander("JSON Configuration (Copy & Paste)", expanded=False):
    # Clean portfolio config for export by removing unused settings
    cleaned_config = active_portfolio.copy()
    cleaned_config.pop('use_relative_momentum', None)
    cleaned_config.pop('equal_if_all_negative', None)
    config_json = json.dumps(cleaned_config, indent=4)
    st.code(config_json, language='json')
    # Fixed JSON copy button
    import streamlit.components.v1 as components
    copy_html = f"""
    <button onclick='navigator.clipboard.writeText({json.dumps(config_json)});' style='margin-bottom:10px;'>Copy to Clipboard</button>
    """
    components.html(copy_html, height=40)
    st.text_area("Paste JSON Here to Update Portfolio", key="alloc_paste_json_text", height=200)
    st.button("Update with Pasted JSON", on_click=paste_json_callback)

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0

# Move Run Backtests to the left sidebar to make it conspicuous and separate from config
if st.sidebar.button("Run Backtests", type='primary'):
    
    # Pre-backtest validation check for all portfolios
    # Prefer the allocations page configs when present so this page's edits are included
    configs_to_run = st.session_state.get('alloc_portfolio_configs', [])
    # Local alias used throughout the run block
    portfolio_list = configs_to_run
    # Set flag to show metrics after running backtest
    st.session_state['alloc_backtest_run'] = True
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
        # Don't run the backtest, but continue showing the UI
        pass
    else:
        progress_bar = st.empty()
    progress_bar.progress(0, text="Starting backtest...")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        all_tickers = sorted(list(set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker']) | set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)))
        all_tickers = [t for t in all_tickers if t]
        print("Downloading data for all tickers...")
        data = {}
        invalid_tickers = []
        for i, t in enumerate(all_tickers):
            try:
                progress_text = f"Downloading data for {t} ({i+1}/{len(all_tickers)})..."
                progress_bar.progress((i + 1) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
                ticker = yf.Ticker(t)
                hist = ticker.history(period="max", auto_adjust=False)[["Close", "Dividends"]]
                if hist.empty:
                    print(f"No data available for {t}")
                    invalid_tickers.append(t)
                    continue
                # Force tz-naive for hist (like Backtest_Engine.py)
                hist = hist.copy()
                hist.index = hist.index.tz_localize(None)
                
                hist["Price_change"] = hist["Close"].pct_change(fill_method=None).fillna(0)
                data[t] = hist
                print(f"Data loaded for {t} from {data[t].index[0].date()}")
            except Exception as e:
                print(f"Error loading {t}: {e}")
                invalid_tickers.append(t)
        
        # Display invalid ticker warnings in Streamlit UI
        if invalid_tickers:
            # Separate portfolio tickers from benchmark tickers
            portfolio_tickers = set(s['ticker'] for cfg in portfolio_list for s in cfg['stocks'] if s['ticker'])
            benchmark_tickers = set(cfg.get('benchmark_ticker') for cfg in portfolio_list if 'benchmark_ticker' in cfg)
            
            portfolio_invalid = [t for t in invalid_tickers if t in portfolio_tickers]
            benchmark_invalid = [t for t in invalid_tickers if t in benchmark_tickers]
            
            if portfolio_invalid:
                st.warning(f"The following portfolio tickers are invalid and will be skipped: {', '.join(portfolio_invalid)}")
            if benchmark_invalid:
                st.warning(f"The following benchmark tickers are invalid and will be skipped: {', '.join(benchmark_invalid)}")
        
        # BULLETPROOF VALIDATION: Check for valid tickers and stop gracefully if none
        if not data:
            if invalid_tickers and len(invalid_tickers) == len(all_tickers):
                st.error(f"❌ **No valid tickers found!** All tickers are invalid: {', '.join(invalid_tickers)}. Please check your ticker symbols and try again.")
            else:
                st.error("❌ **No valid tickers found!** No data downloaded; aborting.")
            progress_bar.empty()
            st.session_state.alloc_all_results = None
            st.session_state.alloc_all_allocations = None
            st.session_state.alloc_all_metrics = None
            st.stop()
        else:
            # Check if any portfolio has valid tickers
            all_portfolio_tickers = set()
            for cfg in portfolio_list:
                portfolio_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                all_portfolio_tickers.update(portfolio_tickers)
            
            # Check for non-USD tickers and display currency warning
            check_currency_warning(list(all_portfolio_tickers))
            
            valid_portfolio_tickers = [t for t in all_portfolio_tickers if t in data]
            if not valid_portfolio_tickers:
                st.error(f"❌ **No valid tickers found!** No valid portfolio tickers found. Invalid tickers: {', '.join(all_portfolio_tickers)}. Please check your ticker symbols and try again.")
                progress_bar.empty()
                st.session_state.alloc_all_results = None
                st.session_state.alloc_all_allocations = None
                st.session_state.alloc_all_metrics = None
                st.stop()
            else:
                # Persist raw downloaded price data so later recomputations can access benchmark series
                st.session_state.alloc_raw_data = data
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
                
                for i, cfg in enumerate(portfolio_list, start=1):
                    progress_text = f"Running backtest for {cfg.get('name', f'Backtest {i}')} ({i}/{len(portfolio_list)})..."
                    progress_bar.progress((len(all_tickers) + i) / (len(all_tickers) + len(portfolio_list)), text=progress_text)
                    name = cfg.get('name', f'Backtest {i}')
                    # Ensure unique key for storage to avoid overwriting when duplicate names exist
                    base_name = name
                    unique_name = base_name
                    suffix = 1
                    while unique_name in all_results or unique_name in all_allocations:
                        unique_name = f"{base_name} ({suffix})"
                        suffix += 1
                    print(f"\nRunning backtest {i}/{len(portfolio_list)}: {name}")
                    # Separate asset tickers from benchmark. Do NOT use benchmark when
                    # computing start/end/simulation dates or available-rebalance logic.
                    asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                    asset_tickers = [t for t in asset_tickers if t in data and t is not None]
                    benchmark_local = cfg.get('benchmark_ticker')
                    benchmark_in_data = benchmark_local if benchmark_local in data else None
                    tickers_for_config = asset_tickers
                    # Build the list of tickers whose data we will reindex (include benchmark if present)
                    data_tickers = list(asset_tickers)
                    if benchmark_in_data:
                        data_tickers.append(benchmark_in_data)
                    if not tickers_for_config:
                        # Check if this is because all tickers are invalid
                        original_asset_tickers = [s['ticker'] for s in cfg['stocks'] if s['ticker']]
                        missing_tickers = [t for t in original_asset_tickers if t not in data]
                        if missing_tickers:
                            print(f"  No available asset tickers for {name}; invalid tickers: {missing_tickers}. Skipping.")
                        else:
                            print(f"  No available asset tickers for {name}; skipping.")
                        continue
                    if cfg.get('start_with') == 'all':
                        # Start only when all asset tickers have data
                        final_start = max(data[t].first_valid_index() for t in tickers_for_config)
                    else:
                        # 'oldest' -> start at the earliest asset ticker date so assets can be added over time
                        final_start = min(data[t].first_valid_index() for t in tickers_for_config)
                    if cfg.get('start_date_user'):
                        user_start = pd.to_datetime(cfg['start_date_user'])
                        final_start = max(final_start, user_start)
                    # Preserve previous global alignment only for 'all' mode; do NOT force 'oldest' back to global latest
                    if cfg.get('start_with') == 'all':
                        final_start = max(final_start, common_start)
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
                    invalid_tickers = []
                    for t in data_tickers:
                        if t in data:  # Only process tickers that have data
                            df = data[t].reindex(simulation_index)
                            df["Close"] = df["Close"].ffill()
                            df["Dividends"] = df["Dividends"].fillna(0)
                            df["Price_change"] = df["Close"].pct_change(fill_method=None).fillna(0)
                            data_reindexed_for_config[t] = df
                        else:
                            invalid_tickers.append(t)
                            print(f"Warning: Invalid ticker '{t}' - no data available, skipping reindexing")
                    
                    # Display invalid ticker warnings in Streamlit UI
                    if invalid_tickers:
                        st.warning(f"The following tickers are invalid and will be skipped: {', '.join(invalid_tickers)}")
                    total_series, total_series_no_additions, historical_allocations, historical_metrics = single_backtest(cfg, simulation_index, data_reindexed_for_config)
                    # Store both series under the unique key for later use
                    all_results[unique_name] = {
                        'no_additions': total_series_no_additions,
                        'with_additions': total_series
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
                # No periodic additions for allocation tracker
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
                sharpe = np.nan if stats_returns.std() == 0 else stats_returns.mean() * 252 / (stats_returns.std() * np.sqrt(252))
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
                        common_idx = pr.index.intersection(br.index)
                        if len(common_idx) >= 2 and br.loc[common_idx].var() != 0:
                            cov = pr.loc[common_idx].cov(br.loc[common_idx])
                            var = br.loc[common_idx].var()
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
                    if stat_type in ["CAGR", "Volatility", "MWRR"]:
                        if v < 0 or v > 100:
                            return "N/A"
                    if stat_type == "MaxDrawdown":
                        if v < -100 or v > 0:
                            return "N/A"
                    return f"{v:.2f}%" if stat_type in ["CAGR", "MaxDrawdown", "Volatility", "MWRR"] else f"{v:.3f}" if isinstance(v, float) else v

                stats = {
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
                all_stats[name] = stats
                all_drawdowns[name] = pd.Series(drawdowns, index=stats_dates)
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
                stats_df_display['CAGR'] = stats_df_display['CAGR'].apply(lambda x: fmt_pct(x))
                stats_df_display['Max Drawdown'] = stats_df_display['Max Drawdown'].apply(lambda x: fmt_pct(x))
                stats_df_display['Volatility'] = stats_df_display['Volatility'].apply(lambda x: fmt_pct(x))
                # Ensure MWRR is the last column, Beta immediately before it
                if 'Beta' in stats_df_display.columns and 'MWRR' in stats_df_display.columns:
                    cols = list(stats_df_display.columns)
                    # Remove Beta and MWRR
                    beta_col = cols.pop(cols.index('Beta'))
                    mwrr_col = cols.pop(cols.index('MWRR'))
                    # Insert Beta before MWRR at the end
                    cols.append(beta_col)
                    cols.append(mwrr_col)
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
                        config_for_name = next((c for c in portfolio_list if c['name'] == nm), None)
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
            st.session_state.alloc_all_results = all_results
            st.session_state.alloc_all_drawdowns = all_drawdowns
            if 'stats_df_display' in locals():
                st.session_state.alloc_stats_df_display = stats_df_display
            st.session_state.alloc_all_years = all_years
            st.session_state.alloc_all_allocations = all_allocations
            # Save a snapshot used by the allocations UI so charts/tables remain static until rerun
            try:
                # compute today_weights_map (target weights as-if rebalanced at final snapshot date)
                today_weights_map = {}
                for pname, allocs in all_allocations.items():
                    try:
                        alloc_dates = sorted(list(allocs.keys()))
                        final_d = alloc_dates[-1]
                        metrics_local = all_metrics.get(pname, {})
                        
                        # Check if momentum is used for this portfolio
                        portfolio_cfg = next((cfg for cfg in portfolio_list if cfg.get('name') == pname), None)
                        use_momentum = portfolio_cfg.get('use_momentum', True) if portfolio_cfg else True
                        
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
                                today_weights_map[pname] = norm
                            else:
                                # When momentum is not used, use target_allocation (user-defined allocations)
                                weights = {t: v.get('target_allocation', 0) for t, v in metrics_local[final_d].items()}
                                # normalize (ensure sums to 1 excluding CASH)
                                sumw = sum(w for k, w in weights.items() if k != 'CASH')
                                if sumw > 0:
                                    norm = {k: (w / sumw) if k != 'CASH' else weights.get('CASH', 0) for k, w in weights.items()}
                                else:
                                    norm = weights
                                today_weights_map[pname] = norm
                        else:
                            # fallback: use allocation snapshot at final date but convert market-value alloc to target weights (exclude CASH then renormalize)
                            final_alloc = allocs.get(final_d, {})
                            noncash = {k: v for k, v in final_alloc.items() if k != 'CASH'}
                            s = sum(noncash.values())
                            if s > 0:
                                norm = {k: (v / s) for k, v in noncash.items()}
                                norm['CASH'] = final_alloc.get('CASH', 0)
                            else:
                                norm = final_alloc
                            today_weights_map[pname] = norm
                    except Exception:
                        today_weights_map[pname] = {}

                st.session_state.alloc_snapshot_data = {
                    'raw_data': data,
                    'portfolio_configs': portfolio_list,
                    'all_allocations': all_allocations,
                    'all_metrics': all_metrics,
                    'today_weights_map': today_weights_map
                }
            except Exception:
                pass
            st.session_state.alloc_all_metrics = all_metrics
            # Save portfolio index -> unique key mapping so UI selectors can reference results reliably
            st.session_state.alloc_portfolio_key_map = portfolio_key_map
            st.session_state.alloc_backtest_run = True

# Sidebar JSON export/import for ALL portfolios
def paste_all_json_callback():
    txt = st.session_state.get('alloc_paste_all_json_text', '')
    if not txt:
        st.warning('No JSON provided')
        return
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            # Process each portfolio configuration for Allocations page
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy == 'Relative momentum':
                    momentum_strategy = 'Relative Momentum'
                elif momentum_strategy not in ['Classic', 'Relative Momentum']:
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
            
            # Process each portfolio configuration for Allocations page (existing logic)
            processed_configs = []
            for cfg in obj:
                if not isinstance(cfg, dict) or 'name' not in cfg:
                    st.error('Invalid portfolio configuration structure.')
                    return
                
                # Handle momentum strategy value mapping from other pages
                momentum_strategy = cfg.get('momentum_strategy', 'Classic')
                if momentum_strategy == 'Classic momentum':
                    momentum_strategy = 'Classic'
                elif momentum_strategy == 'Relative momentum':
                    momentum_strategy = 'Relative Momentum'
                elif momentum_strategy not in ['Classic', 'Relative Momentum']:
                    momentum_strategy = 'Classic'  # Default fallback
                
                # Handle negative momentum strategy value mapping from other pages
                negative_momentum_strategy = cfg.get('negative_momentum_strategy', 'Cash')
                if negative_momentum_strategy == 'Go to cash':
                    negative_momentum_strategy = 'Cash'
                elif negative_momentum_strategy not in ['Cash', 'Equal weight', 'Relative momentum']:
                    negative_momentum_strategy = 'Cash'  # Default fallback
                

                
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
                
                # Map frequency values from app.py format to Allocations format
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
                
                # Allocations page specific: ensure all required fields are present
                # and ignore fields that are specific to other pages
                allocations_config = {
                    'name': cfg.get('name', 'Allocation Portfolio'),
                    'stocks': stocks,
                    'benchmark_ticker': cfg.get('benchmark_ticker', '^GSPC'),
                    'initial_value': cfg.get('initial_value', 10000),
                    'added_amount': cfg.get('added_amount', 0),  # Allocations page typically doesn't use additions
                    'added_frequency': map_frequency(cfg.get('added_frequency', 'Never')),  # Allocations page typically doesn't use additions
                                          'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                      'start_date_user': cfg.get('start_date_user'),
                      'end_date_user': cfg.get('end_date_user'),
                      'start_with': cfg.get('start_with', 'all'),
                      'use_momentum': cfg.get('use_momentum', True),
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
                processed_configs.append(allocations_config)
            
            st.session_state.alloc_portfolio_configs = processed_configs
            # Reset active selection and derived mappings so the UI reflects the new configs
            if processed_configs:
                st.session_state.alloc_active_portfolio_index = 0
                st.session_state.alloc_portfolio_selector = processed_configs[0].get('name', '')
                # Update portfolio name input field to match the first imported portfolio
                st.session_state.alloc_portfolio_name = processed_configs[0].get('name', 'Allocation Portfolio')
                # Mirror several active_* widget defaults so the UI selectboxes/inputs update
                st.session_state['alloc_active_name'] = processed_configs[0].get('name', '')
                st.session_state['alloc_active_initial'] = int(processed_configs[0].get('initial_value', 0) or 0)
                st.session_state['alloc_active_added_amount'] = int(processed_configs[0].get('added_amount', 0) or 0)
                st.session_state['alloc_active_rebal_freq'] = processed_configs[0].get('rebalancing_frequency', 'month')
                st.session_state['alloc_active_add_freq'] = processed_configs[0].get('added_frequency', 'none')
                st.session_state['alloc_active_benchmark'] = processed_configs[0].get('benchmark_ticker', '')
                st.session_state['alloc_active_use_momentum'] = bool(processed_configs[0].get('use_momentum', True))
            else:
                st.session_state.alloc_active_portfolio_index = None
                st.session_state.alloc_portfolio_selector = ''
            st.session_state.alloc_portfolio_key_map = {}
            st.session_state.alloc_backtest_run = False
            st.success('All portfolio configurations updated from JSON (Allocations page).')
            # Debug: Show final momentum windows for first portfolio
            if processed_configs:
                st.info(f"Final momentum windows for first portfolio: {processed_configs[0]['momentum_windows']}")
                st.info(f"Final use_momentum for first portfolio: {processed_configs[0]['use_momentum']}")
            # Force a rerun so widgets rebuild with the new configs
            try:
                st.experimental_rerun()
            except Exception:
                # In some environments experimental rerun may raise; setting a rerun flag is a fallback
                st.session_state.alloc_rerun_flag = True
        else:
            st.error('JSON must be a list of portfolio configurations.')
    except Exception as e:
        st.error(f'Failed to parse JSON: {e}')




# Simplified display for allocation tracker: only allocation pies and rebalancing metrics are shown
active_name = active_portfolio.get('name')
if st.session_state.get('alloc_backtest_run', False):
    st.subheader("Allocation & Rebalancing Metrics")
    allocs_for_portfolio = st.session_state.get('alloc_all_allocations', {}).get(active_name) if st.session_state.get('alloc_all_allocations') else None
    metrics_for_portfolio = st.session_state.get('alloc_all_metrics', {}).get(active_name) if st.session_state.get('alloc_all_metrics') else None

    if not allocs_for_portfolio and not metrics_for_portfolio:
        st.info("No allocation or rebalancing history available. If you have precomputed allocation snapshots, store them in session state keys `alloc_all_allocations` and `alloc_all_metrics` under this portfolio name.")
    else:
        # --- Rebalance as of Today (static snapshot from last Run Backtests) ---
        snapshot = st.session_state.get('alloc_snapshot_data', {})
        today_weights_map = snapshot.get('today_weights_map', {}) if snapshot else {}
        
        # Check if momentum is used for this portfolio
        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
        
        if snapshot and active_name in today_weights_map:
            today_weights = today_weights_map.get(active_name, {})
            
            # If momentum is not used and today_weights is empty, use the entered allocations
            if not use_momentum and (not today_weights or all(v == 0 for v in today_weights.values())):
                # Use the entered allocations from active_portfolio
                today_weights = {}
                for stock in active_portfolio.get('stocks', []):
                    ticker = stock.get('ticker', '').strip()
                    if ticker:
                        today_weights[ticker] = stock.get('allocation', 0)
                # Add CASH if needed
                total_alloc = sum(today_weights.values())
                if total_alloc < 1.0:
                    today_weights['CASH'] = 1.0 - total_alloc
                else:
                    today_weights['CASH'] = 0
            
            labels_today = [k for k, v in sorted(today_weights.items(), key=lambda x: (-x[1], x[0])) if v > 0]
            vals_today = [float(today_weights[k]) * 100 for k in labels_today]
            
            if labels_today and vals_today:
                st.markdown("## Rebalance as of Today")
                fig_today = go.Figure()
                fig_today.add_trace(go.Pie(labels=labels_today, values=vals_today, hole=0.35))
                fig_today.update_traces(textinfo='percent+label')
                fig_today.update_layout(template='plotly_dark', margin=dict(t=10), height=600)
                st.plotly_chart(fig_today, use_container_width=True, key=f"alloc_today_chart_{active_name}")
            
            # static shares table

            # Define build_table_from_alloc before usage
            def build_table_from_alloc(alloc_dict, price_date, label):
                rows = []
                # Use portfolio_value from session state or active_portfolio (current portfolio value)
                try:
                    portfolio_value = float(st.session_state.get('alloc_active_initial', active_portfolio.get('initial_value', 0) or 0))
                except Exception:
                    portfolio_value = active_portfolio.get('initial_value', 0) or 0
                # Use raw_data from snapshot or session state
                snapshot = st.session_state.get('alloc_snapshot_data', {})
                raw_data = snapshot.get('raw_data') if snapshot and snapshot.get('raw_data') is not None else st.session_state.get('alloc_raw_data', {})
                def _price_on_or_before(df, target_date):
                    try:
                        idx = df.index[df.index <= pd.to_datetime(target_date)]
                        if len(idx) == 0:
                            return None
                        return float(df.loc[idx[-1], 'Close'])
                    except Exception:
                        return None
                for tk in sorted(alloc_dict.keys()):
                    alloc_pct = float(alloc_dict.get(tk, 0))
                    if tk == 'CASH':
                        price = None
                        shares = 0.0
                        total_val = portfolio_value * alloc_pct
                    else:
                        df = raw_data.get(tk)
                        price = None
                        # Ensure df is a valid DataFrame with Close prices before accessing
                        if isinstance(df, pd.DataFrame) and 'Close' in df.columns and not df['Close'].dropna().empty:
                            if price_date is None:
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
                            shares = 0.0
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
            
        # Add timer for next rebalance date
        if allocs_for_portfolio and active_portfolio:
            try:
                # Get the last rebalance date from allocation history
                alloc_dates = sorted(list(allocs_for_portfolio.keys()))
                if len(alloc_dates) > 1:
                    last_rebal_date = alloc_dates[-2]  # Second to last date (excluding today/yesterday)
                else:
                    last_rebal_date = alloc_dates[-1] if alloc_dates else None
                
                # Get rebalancing frequency from active portfolio
                rebalancing_frequency = active_portfolio.get('rebalancing_frequency', 'none')
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
                
                if last_rebal_date and rebalancing_frequency != 'none':
                    # Ensure last_rebal_date is a naive datetime object
                    import pandas as pd
                    if isinstance(last_rebal_date, str):
                        last_rebal_date = pd.to_datetime(last_rebal_date)
                    if hasattr(last_rebal_date, 'tzinfo') and last_rebal_date.tzinfo is not None:
                        last_rebal_date = last_rebal_date.replace(tzinfo=None)
                    
                    next_date, time_until, next_rebalance_datetime = calculate_next_rebalance_date(
                        rebalancing_frequency, last_rebal_date
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
                            if hasattr(last_rebal_date, 'to_pydatetime'):
                                last_rebal_datetime = last_rebal_date.to_pydatetime()
                            else:
                                last_rebal_datetime = last_rebal_date
                            
                            total_period = (next_rebalance_datetime - last_rebal_datetime).total_seconds()
                            elapsed_period = (datetime.now() - last_rebal_datetime).total_seconds()
                            progress = min(max(elapsed_period / total_period, 0), 1)
                            
                            st.progress(progress, text=f"Progress to next rebalance: {progress:.1%}")
            except Exception as e:
                pass  # Silently ignore timer calculation errors
        
        build_table_from_alloc({**today_weights, 'CASH': today_weights.get('CASH', 0)}, None, f"Shares if Rebalanced Today (snapshot)")

    if allocs_for_portfolio:
        st.markdown("**Historical Allocations**")
        # Ensure proper DataFrame structure with explicit column names
        allocations_df_raw = pd.DataFrame(allocs_for_portfolio).T
        
        # Handle case where only CASH exists - ensure column name is preserved
        if allocations_df_raw.empty or (len(allocations_df_raw.columns) == 1 and allocations_df_raw.columns[0] is None):
            # Reconstruct DataFrame with proper column names
            processed_data = {}
            for date, alloc_dict in allocs_for_portfolio.items():
                processed_data[date] = {}
                for ticker, value in alloc_dict.items():
                    if ticker is None:
                        processed_data[date]['CASH'] = value
                    else:
                        processed_data[date][ticker] = value
            allocations_df_raw = pd.DataFrame(processed_data).T
        
        allocations_df_raw.index.name = "Date"

        def highlight_rows_by_index(s):
            is_even_row = allocations_df_raw.index.get_loc(s.name) % 2 == 0
            bg_color = 'background-color: #0e1117' if is_even_row else 'background-color: #262626'
            return [bg_color] * len(s)

        styler = allocations_df_raw.mul(100).style.apply(highlight_rows_by_index, axis=1)
        styler.format('{:,.0f}%', na_rep='N/A')
        st.dataframe(styler, use_container_width=True)

    if metrics_for_portfolio:
        st.markdown("---")
        st.markdown("**Rebalancing Metrics & Calculated Weights**")
        metrics_records = []
        for date, tickers_data in metrics_for_portfolio.items():
            for ticker, data in tickers_data.items():
                # Handle None ticker as CASH
                display_ticker = 'CASH' if ticker is None else ticker
                filtered_data = {k: v for k, v in (data or {}).items() if k != 'Composite'}
                
                # Check if momentum is used for this portfolio
                use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                
                # If momentum is not used, replace Calculated_Weight with target_allocation
                if not use_momentum:
                    if 'target_allocation' in filtered_data:
                        filtered_data['Calculated_Weight'] = filtered_data['target_allocation']
                    else:
                        # If target_allocation is not available, use the entered allocations from active_portfolio
                        ticker_name = display_ticker if display_ticker != 'CASH' else None
                        if ticker_name:
                            # Find the stock in active_portfolio and use its allocation
                            for stock in active_portfolio.get('stocks', []):
                                if stock.get('ticker', '').strip() == ticker_name:
                                    filtered_data['Calculated_Weight'] = stock.get('allocation', 0)
                                    break
                        else:
                            # For CASH, calculate the remaining allocation
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            filtered_data['Calculated_Weight'] = max(0, 1.0 - total_alloc)
                
                record = {'Date': date, 'Ticker': display_ticker, **filtered_data}
                metrics_records.append(record)
            
            # Ensure CASH line is added if there's non-zero cash in allocations
            if allocs_for_portfolio and date in allocs_for_portfolio:
                cash_alloc = allocs_for_portfolio[date].get('CASH', 0)
                if cash_alloc > 0:
                    # Check if CASH is already in metrics_records for this date
                    cash_exists = any(record['Date'] == date and record['Ticker'] == 'CASH' for record in metrics_records)
                    if not cash_exists:
                        # Add CASH line to metrics
                        # Check if momentum is used to determine which weight to show
                        use_momentum = active_portfolio.get('use_momentum', True) if active_portfolio else True
                        if not use_momentum:
                            # When momentum is not used, calculate CASH allocation from entered allocations
                            total_alloc = sum(stock.get('allocation', 0) for stock in active_portfolio.get('stocks', []))
                            cash_weight = max(0, 1.0 - total_alloc)
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_weight}
                        else:
                            cash_record = {'Date': date, 'Ticker': 'CASH', 'Calculated_Weight': cash_alloc}
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
                metrics_df.set_index(['Date', 'Ticker'], inplace=True)
                metrics_df_display = metrics_df.copy()
            if 'Momentum' in metrics_df_display.columns:
                metrics_df_display['Momentum'] = metrics_df_display['Momentum'].fillna(0) * 100
            if 'Calculated_Weight' in metrics_df_display.columns:
                metrics_df_display['Calculated_Weight'] = metrics_df_display['Calculated_Weight'].fillna(0) * 100
            if 'Volatility' in metrics_df_display.columns:
                metrics_df_display['Volatility'] = metrics_df_display['Volatility'].fillna(np.nan) * 100

            def highlight_metrics_rows(s):
                if s.name[1] == 'CASH':
                    return ['background-color: #006400; color: white; font-weight: bold;' for _ in s]
                unique_dates = list(metrics_df_display.index.get_level_values('Date').unique())
                is_even = unique_dates.index(s.name[0]) % 2 == 0
                bg_color = 'background-color: #0e1117' if is_even else 'background-color: #262626'
                return [bg_color] * len(s)

            styler_metrics = metrics_df_display.style.apply(highlight_metrics_rows, axis=1)
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

    # Allocation pie charts (last rebalance vs current)
    if allocs_for_portfolio:
        try:
            alloc_dates = sorted(list(allocs_for_portfolio.keys()))
            final_date = alloc_dates[-1]
            last_rebal_date = alloc_dates[-2] if len(alloc_dates) > 1 else alloc_dates[-1]
            final_alloc = allocs_for_portfolio.get(final_date, {})
            rebal_alloc = allocs_for_portfolio.get(last_rebal_date, {})

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
            # prepare helpers used by the 'Rebalance Today' UI
            # Prefer the snapshot saved when backtests were run so this UI is static until rerun
            snapshot = st.session_state.get('alloc_snapshot_data', {})
            snapshot_raw = snapshot.get('raw_data')
            snapshot_portfolios = snapshot.get('portfolio_configs')

            # select raw_data and portfolio config from snapshot if available, otherwise fall back to live state
            raw_data = snapshot_raw if snapshot_raw is not None else st.session_state.get('alloc_raw_data', {})
            # find snapshot portfolio config by name if present
            snapshot_cfg = None
            if snapshot_portfolios:
                try:
                    snapshot_cfg = next((c for c in snapshot_portfolios if c.get('name') == active_name), None)
                except Exception:
                    snapshot_cfg = None
            portfolio_cfg_for_today = snapshot_cfg if snapshot_cfg is not None else active_portfolio

            try:
                portfolio_value = float(portfolio_cfg_for_today.get('initial_value', 0) or 0)
            except Exception:
                portfolio_value = portfolio_cfg_for_today.get('initial_value', 0) or 0
            


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
            
            # Render small pies for Last Rebalance and Current Allocation
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Target Allocation at Last Rebalance ({last_rebal_date.date()})**")
                    fig_rebal_small = go.Figure()
                    fig_rebal_small.add_trace(go.Pie(labels=labels_rebal, values=vals_rebal, hole=0.35))
                    fig_rebal_small.update_traces(textinfo='percent+label')
                    fig_rebal_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_rebal_small, use_container_width=True, key=f"alloc_rebal_small_{active_name}")
                with col2:
                    st.markdown(f"**Portfolio Evolution (Current Allocation)**")
                    fig_today_small = go.Figure()
                    fig_today_small.add_trace(go.Pie(labels=labels_final, values=vals_final, hole=0.35))
                    fig_today_small.update_traces(textinfo='percent+label')
                    fig_today_small.update_layout(template='plotly_dark', margin=dict(t=10))
                    st.plotly_chart(fig_today_small, use_container_width=True, key=f"alloc_today_small_{active_name}")
            except Exception:
                # If plotting fails, continue and still render the tables below
                pass

            # Last rebalance table (use last_rebal_date)
            build_table_from_alloc(rebal_alloc, last_rebal_date, f"Target Allocation at Last Rebalance ({last_rebal_date.date()})")
            # Current / Today table (use final_date's latest available prices as of now)
            build_table_from_alloc(final_alloc, None, f"Portfolio Evolution (Current Allocation)")
        except Exception as e:
            print(f"[ALLOC PLOT DEBUG] Failed to render allocation plots for {active_name}: {e}")
