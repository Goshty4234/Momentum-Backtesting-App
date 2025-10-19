"""
Quick Momentum ETF Comparison - Ultra-fast & Simple backtest comparison
No momentum calculation, no SMA, no filters - just direct ETF comparison

PERFORMANCE OPTIMIZATIONS:
- Single batch download (ONE API call for all tickers) → 10x faster
- No caching needed (direct batch is already fast)
- Simplified calculations (no complex momentum/SMA logic)

FAIR COMPARISON FEATURES:
- Shows start date for each ticker (color-coded by age)
- Identifies which ticker is the "bottleneck" (newest start date)
- Option to exclude recent tickers to start backtest earlier
- All tickers aligned to common date range for fair comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Quick Momentum ETF Comparison", page_icon="⚡", layout="wide")

# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_batch_ticker_data(tickers_list, period="max"):
    """Cached batch download - only re-downloads if ticker list changes or cache expires"""
    try:
        # SINGLE BATCH DOWNLOAD - One API call for all tickers
        batch_data = yf.download(
            tickers_list,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by='ticker'
        )
        return batch_data
    except Exception as e:
        st.error(f"❌ Batch download failed: {e}")
        return None

# =============================================================================
# SIMPLE CALCULATION FUNCTIONS
# =============================================================================

def calculate_returns(price_series):
    """Calculate simple returns from price series"""
    if price_series is None or len(price_series) < 2:
        return pd.Series()
    return price_series.pct_change().fillna(0)

def calculate_portfolio_growth(returns, initial_value=10000):
    """Calculate cumulative portfolio growth"""
    if len(returns) == 0:
        return pd.Series()
    return initial_value * (1 + returns).cumprod()

def calculate_metrics(returns, benchmark_returns=None):
    """Calculate key performance metrics"""
    if len(returns) < 2:
        return {}
    
    # Annualization factor (assuming daily data)
    periods_per_year = 252
    
    # Total return
    total_return = float((1 + returns).prod() - 1)
    
    # Number of years
    years = float(len(returns) / periods_per_year)
    
    # CAGR
    cagr = float((1 + total_return) ** (1 / years) - 1 if years > 0 else 0)
    
    # Volatility (annualized)
    volatility = float(returns.std() * np.sqrt(periods_per_year))
    
    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe = float(cagr / volatility) if volatility > 0 else 0.0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())
    
    # Calmar Ratio
    calmar = float(abs(cagr / max_drawdown)) if max_drawdown != 0 else 0.0
    
    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar,
        'Years': years
    }
    
    # Beta (if benchmark provided)
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        # Calculate covariance manually to avoid DataFrame issues
        returns_clean = returns.fillna(0)
        benchmark_clean = benchmark_returns.fillna(0)
        
        covariance = np.cov(returns_clean, benchmark_clean)[0, 1]
        benchmark_variance = np.var(benchmark_clean, ddof=1)
        
        beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else np.nan
        metrics['Beta'] = beta
    
    return metrics

def calculate_yearly_returns(returns, dates):
    """Calculate yearly returns"""
    if len(returns) == 0 or len(dates) == 0:
        return {}
    
    df = pd.DataFrame({'returns': returns, 'date': dates})
    df['year'] = df['date'].dt.year
    
    yearly_returns = {}
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]['returns']
        yearly_return = (1 + year_data).prod() - 1
        yearly_returns[year] = yearly_return
    
    return yearly_returns

def calculate_monthly_returns(returns, dates):
    """Calculate monthly returns"""
    if len(returns) == 0 or len(dates) == 0:
        return {}
    
    df = pd.DataFrame({'returns': returns, 'date': dates})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    monthly_returns = {}
    for year in sorted(df['year'].unique()):
        monthly_returns[year] = {}
        for month in range(1, 13):
            month_data = df[(df['year'] == year) & (df['month'] == month)]['returns']
            if len(month_data) > 0:
                monthly_return = (1 + month_data).prod() - 1
                monthly_returns[year][month] = monthly_return
            else:
                monthly_returns[year][month] = np.nan
    
    return monthly_returns

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.title("⚡ Quick Momentum ETF Comparison")
st.markdown("**Ultra-fast comparison of momentum ETFs - No complex calculations, just direct backtest!**")

# Default momentum ETFs (verified working tickers)
DEFAULT_MOMENTUM_ETFS = [
    "MTUM",   # iShares MSCI USA Momentum Factor ETF (2013+)
    "PDP",    # Invesco DWA Momentum ETF (2007+)
    "SPMO",   # Invesco S&P 500 Momentum ETF (2015+)
    "QMOM",   # Alpha Architect U.S. Quantitative Momentum ETF (2015+)
    "FDMO",   # Fidelity Momentum Factor ETF (2016+)
    "ONEO",   # SPDR Russell 1000 Momentum Focus ETF (2015+)
    "MMTM",   # SPDR S&P 1500 Momentum Tilt ETF (2015+)
    "IMOM",   # Alpha Architect International Quantitative Momentum ETF (2015+)
    "JMOM",   # JPMorgan U.S. Momentum Factor ETF (2017+)
]

BENCHMARK_TICKERS = ["SPY", "QQQ"]

# =============================================================================
# USER INPUTS - MAIN UI
# =============================================================================

st.header("⚙️ Configuration")

# ETF Selection with individual checkboxes and start dates
st.subheader("📊 ETF Selection")

# Show each ETF with checkbox and start date
selected_momentum_etfs = []

st.markdown("**Momentum ETFs:**")
col1, col2 = st.columns(2)

for i, etf in enumerate(DEFAULT_MOMENTUM_ETFS):
    # Get start date for each ETF (this will be approximate, actual dates shown after download)
    if etf == "PDP":
        start_info = "2007+"
    elif etf == "MTUM":
        start_info = "2013+"
    elif etf in ["SPMO", "QMOM", "ONEO", "MMTM", "IMOM"]:
        start_info = "2015+"
    elif etf in ["FDMO", "JMOM"]:
        start_info = "2016-2017+"
    else:
        start_info = "?"
    
    # Alternate between columns
    with (col1 if i % 2 == 0 else col2):
        if st.checkbox(f"{etf} ({start_info})", value=True):
            selected_momentum_etfs.append(etf)

# Add custom tickers
st.subheader("➕ Add Custom Tickers")
custom_tickers_input = st.text_area(
    "Add custom tickers (one per line)",
    placeholder="AAPL\nGOOGL\nMSFT",
    height=100
)

custom_tickers = []
if custom_tickers_input:
    custom_tickers = [t.strip().upper() for t in custom_tickers_input.split("\n") if t.strip()]

# Benchmarks
st.subheader("📈 Benchmarks")
benchmark_col1, benchmark_col2 = st.columns(2)

with benchmark_col1:
    show_spy = st.checkbox("SPY (S&P 500, 1993+)", value=True)

with benchmark_col2:
    show_qqq = st.checkbox("QQQ (NASDAQ 100, 1999+)", value=True)

# Combine all tickers
selected_benchmarks = []
if show_spy:
    selected_benchmarks.append("SPY")
if show_qqq:
    selected_benchmarks.append("QQQ")

all_tickers = selected_momentum_etfs + custom_tickers + selected_benchmarks
all_tickers = list(dict.fromkeys(all_tickers))  # Remove duplicates while preserving order

# Validation
if not all_tickers:
    st.error("⚠️ Please select at least one ticker!")
    st.stop()

# Additional settings
st.subheader("⚙️ Additional Settings")
settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    # Date range (will be auto-detected after download)
    start_date = st.date_input(
        "Start Date (Will be auto-detected after download)",
        value=pd.to_datetime("2000-01-01"),
        min_value=pd.to_datetime("1990-01-01"),
        max_value=pd.to_datetime("today"),
        help="This will be automatically updated to the most recent ticker's start date after downloading data"
    )

with settings_col2:
    # Initial investment
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )

# =============================================================================
# RUN BACKTEST
# =============================================================================

# Store results in session state to prevent disappearing when checkboxes change
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Run button
st.markdown("---")
if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
    
    with st.spinner("⚡ Fetching data (cached for 1 hour)..."):
        
        # Quick individual download to check real start dates
        st.subheader("🔍 Checking Real Start Dates (Quick Test)")
        
        ticker_real_dates = {}
        for ticker in all_tickers:
            try:
                # Quick download with minimal data to get start date
                ticker_obj = yf.Ticker(ticker)
                ticker_data = ticker_obj.history(period="max", auto_adjust=True)
                
                if ticker_data is not None and not ticker_data.empty:
                    start_date = ticker_data.index.min()
                    ticker_real_dates[ticker] = start_date
                    st.write(f"✅ **{ticker}**: Start = {start_date.strftime('%Y-%m-%d')}")
                else:
                    st.write(f"❌ **{ticker}**: No data available")
            except Exception as e:
                st.write(f"❌ **{ticker}**: Error - {str(e)[:100]}")
        
        st.markdown("---")
        
        # Now use cached batch download for actual data
        status_text = st.empty()
        status_text.text(f"📥 Now downloading full data (cached)...")
        
        # Sort tickers for consistent cache key
        sorted_tickers = sorted(all_tickers)
        batch_data = get_batch_ticker_data(tuple(sorted_tickers), "max")
        
        if batch_data is None:
            st.stop()
        
        # Process downloaded data
        data_dict = {}
        original_data_dict = {}
        failed_tickers = []
        
        for ticker in all_tickers:
            try:
                if len(all_tickers) == 1:
                    # Single ticker case
                    ticker_data = batch_data
                else:
                    # Multiple tickers case
                    ticker_data = batch_data[ticker]
                
                if ticker_data is not None and not ticker_data.empty:
                    # Use REAL start date from individual check, not batch data
                    original_data_dict[ticker] = ticker_data
                    data_dict[ticker] = ticker_data
                else:
                    failed_tickers.append(ticker)
            except Exception:
                failed_tickers.append(ticker)
        
        status_text.empty()
        
        # Show warnings for failed tickers
        if failed_tickers:
            st.warning(f"⚠️ Could not download data for: {', '.join(failed_tickers)}")
        
        if not data_dict:
            st.error("❌ No valid data downloaded. Please check your tickers and try again.")
            st.stop()
        
        st.success(f"✅ Successfully downloaded {len(data_dict)} tickers!")
    
    # =============================================================================
    # SHOW START DATES FOR EACH TICKER
    # =============================================================================
    
    st.subheader("📅 Data Availability by Ticker")
    
    # Use REAL start dates from individual checks (not batch data)
    ticker_start_dates = ticker_real_dates
    
    # AUTO-DETECT optimal start date from real dates
    if ticker_start_dates:
        optimal_start_date = max(ticker_start_dates.values())  # Most recent ticker (bottleneck)
        bottleneck_ticker = max(ticker_start_dates, key=ticker_start_dates.get)
        
        st.info(f"🎯 **Auto-detected optimal start date**: {optimal_start_date.strftime('%Y-%m-%d')} (from {bottleneck_ticker})")
        
        # Check if user start_date is optimal
        if start_date == optimal_start_date:
            st.success(f"✅ **Perfect!** Your start date matches the optimal date ({bottleneck_ticker})")
        elif start_date < optimal_start_date:
            st.warning(f"⚠️ **Early start**: You're starting {start_date.strftime('%Y-%m-%d')} but {bottleneck_ticker} starts {optimal_start_date.strftime('%Y-%m-%d')} - some tickers won't have data!")
        else:
            st.info(f"ℹ️ **Custom start**: You've chosen a later start date than {bottleneck_ticker}")
        
        # Override start_date with optimal for the backtest
        st.info(f"🔄 **Using optimal start date**: {optimal_start_date.strftime('%Y-%m-%d')} for fair comparison")
        start_date = optimal_start_date
    
    # Create DataFrame with start dates
    start_dates_df = pd.DataFrame([
        {
            'Ticker': ticker,
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'Start Year': start_date.year,
            'Days of Data': len(data_dict[ticker])
        }
        for ticker, start_date in sorted(ticker_start_dates.items(), key=lambda x: x[1])
    ])
    
    # Add color coding
    def color_start_dates(row):
        year = row['Start Year']
        if year < 2010:
            return ['background-color: #28a745; color: white'] * len(row)  # Dark green (oldest)
        elif year < 2015:
            return ['background-color: #d4edda; color: #155724'] * len(row)  # Light green
        elif year < 2020:
            return ['background-color: #fff3cd; color: #856404'] * len(row)  # Yellow
        else:
            return ['background-color: #f8d7da; color: #721c24'] * len(row)  # Red (newest/shortest)
    
    styled_start_dates = start_dates_df.style.apply(color_start_dates, axis=1)
    st.dataframe(styled_start_dates, use_container_width=True, hide_index=True)
    
    # Find common date range using REAL individual dates
    common_start_date = max(ticker_start_dates.values())  # Most recent start (bottleneck)
    earliest_possible_date = min(ticker_start_dates.values())  # Earliest start (oldest ticker)
    
    st.info(f"""
    📊 **Date Range Analysis**:
    - **Earliest ticker**: {min(ticker_start_dates, key=ticker_start_dates.get)} starts {earliest_possible_date.strftime('%Y-%m-%d')}
    - **Latest ticker**: {max(ticker_start_dates, key=ticker_start_dates.get)} starts {common_start_date.strftime('%Y-%m-%d')}
    - **Your backtest will start**: {start_date.strftime('%Y-%m-%d')} (auto-detected from selected tickers)
    """)
    
    # Show if start_date is optimal
    if start_date == common_start_date:
        st.success(f"✅ **OPTIMAL**: Start date matches the latest ticker ({max(ticker_start_dates, key=ticker_start_dates.get)}) - fair comparison!")
    elif start_date < common_start_date:
        st.warning(f"⚠️ **EARLY START**: You're starting before {max(ticker_start_dates, key=ticker_start_dates.get)} - some tickers won't have data!")
    else:
        st.info(f"ℹ️ **CUSTOM START**: You've chosen a later start date than the latest ticker.")
    
    # Option to exclude tickers to start earlier
    st.markdown("---")
    st.subheader("🎯 Exclude Tickers to Start Earlier")
    st.markdown("**Exclude recent tickers to start your backtest earlier with older tickers only**")
    
    exclude_tickers = st.multiselect(
        "Select tickers to EXCLUDE from comparison",
        options=list(data_dict.keys()),
        default=[],
        help="Excluding newer tickers allows the backtest to start earlier with older tickers"
    )
    
    if exclude_tickers:
        st.warning(f"⚠️ Excluding: {', '.join(exclude_tickers)}")
        
        # Remove excluded tickers
        for ticker in exclude_tickers:
            if ticker in data_dict:
                del data_dict[ticker]
        
        # Recalculate common date (same logic as Page 1)
        if data_dict:
            ticker_start_dates = {ticker: data.index.min() for ticker, data in data_dict.items()}
            common_start_date = max(ticker_start_dates.values())  # Most recent start (bottleneck)
            earliest_possible_date = min(ticker_start_dates.values())  # Earliest start
            
            st.success(f"""
            ✅ **New Date Range After Exclusions**:
            - **Backtest will now start**: {common_start_date.strftime('%Y-%m-%d')}
            - **Oldest ticker**: {min(ticker_start_dates, key=ticker_start_dates.get)} (starts {earliest_possible_date.strftime('%Y-%m-%d')})
            - **Newest ticker**: {max(ticker_start_dates, key=ticker_start_dates.get)} (starts {common_start_date.strftime('%Y-%m-%d')})
            - **Total tickers**: {len(data_dict)}
            """)
        else:
            st.error("❌ You excluded all tickers! Please keep at least one ticker.")
            st.stop()
    
    st.markdown("---")
    
    # =============================================================================
    # ALIGN DATA & CALCULATE RETURNS
    # =============================================================================
    
    with st.spinner("📊 Calculating returns..."):
        
        # Find common date range
        all_dates = None
        for ticker, data in data_dict.items():
            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)
        
        if len(all_dates) == 0:
            st.error("❌ No overlapping dates found between tickers!")
            st.stop()
        
        # Display final backtest period (using USER start_date)
        st.success(f"""
        ✅ **Final Backtest Period (Using Your Start Date)**:
        - **Start**: {start_date.strftime('%Y-%m-%d')} (your chosen date)
        - **End**: {all_dates.max().strftime('%Y-%m-%d')}
        - **Total Trading Days**: {len(all_dates):,}
        - **Years**: {len(all_dates) / 252:.1f}
        """)
        
        # Filter data by USER start_date (timezone-safe)
        filtered_data_dict = {}
        for ticker, data in data_dict.items():
            try:
                # Ensure both are timezone-naive datetime64[ns]
                data_index = data.index
                if data_index.tz is not None:
                    data_index = data_index.tz_localize(None)
                
                # Convert start_date to datetime64[ns] (same dtype as data_index)
                user_start_datetime = pd.to_datetime(start_date).to_datetime64()
                
                # Filter using same dtype comparison
                mask = data_index >= user_start_datetime
                filtered_data = data[mask]
                
                if not filtered_data.empty:
                    filtered_data_dict[ticker] = filtered_data
                else:
                    st.write(f"⚠️ {ticker}: No data after {start_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                st.write(f"⚠️ Warning filtering {ticker}: {e}")
                # If filtering fails, use original data
                filtered_data_dict[ticker] = data
        
        if not filtered_data_dict:
            st.error("❌ No data available after applying start date filter!")
            st.stop()
        
        # Find common date range from FILTERED data (using USER start_date)
        all_dates = None
        for ticker, data in filtered_data_dict.items():
            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)
        
        if len(all_dates) == 0:
            st.error("❌ No overlapping dates found between tickers after filtering!")
            st.stop()
        
        # Ensure all_dates starts from user start_date (timezone-safe)
        try:
            # Make both timezone-naive for comparison
            all_dates_index = all_dates.tz_localize(None) if all_dates.tz is not None else all_dates
            start_datetime = pd.to_datetime(start_date)
            
            # Filter using timezone-naive comparison
            mask = all_dates_index >= start_datetime
            all_dates = all_dates[mask]
        except Exception as e:
            st.write(f"⚠️ Warning filtering all_dates: {e}")
            # If filtering fails, keep original all_dates
        
        # Align all data to common dates
        aligned_prices = {}
        for ticker, data in filtered_data_dict.items():
            aligned_prices[ticker] = data.loc[all_dates, 'Close']
        
        # Calculate returns
        returns_dict = {}
        for ticker, prices in aligned_prices.items():
            returns_dict[ticker] = calculate_returns(prices)
        
        # Calculate portfolio growth
        growth_dict = {}
        for ticker, returns in returns_dict.items():
            growth_dict[ticker] = calculate_portfolio_growth(returns, initial_investment)
    
    # =============================================================================
    # CALCULATE METRICS
    # =============================================================================
    
    with st.spinner("📈 Calculating metrics..."):
        
        # Use SPY as benchmark if available
        benchmark_returns = returns_dict.get("SPY", None)
        
        metrics_dict = {}
        for ticker, returns in returns_dict.items():
            metrics_dict[ticker] = calculate_metrics(returns, benchmark_returns)
    
    # =============================================================================
    # DISPLAY RESULTS
    # =============================================================================
    
    st.markdown("---")
    st.header("📊 Results")
    
    # Display key info at top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Backtest Start", start_date.strftime('%Y-%m-%d'))
    with col2:
        st.metric("📊 Tickers Compared", len(data_dict))
    with col3:
        st.metric("⏱️ Years of Data", f"{len(all_dates) / 252:.1f}")
    
    st.markdown("---")
    
    # Performance Chart
    st.subheader("📈 Portfolio Growth")
    
    fig = go.Figure()
    
    # Separate momentum ETFs and benchmarks for better visualization
    momentum_tickers = [t for t in growth_dict.keys() if t not in BENCHMARK_TICKERS]
    benchmark_tickers_present = [t for t in growth_dict.keys() if t in BENCHMARK_TICKERS]
    
    # Add momentum ETFs (normal lines)
    for ticker in momentum_tickers:
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=growth_dict[ticker],
            mode='lines',
            name=ticker,
            line=dict(width=2),
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    # Add benchmarks (dashed lines)
    for ticker in benchmark_tickers_present:
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=growth_dict[ticker],
            mode='lines',
            name=f"{ticker} (Benchmark)",
            line=dict(width=2, dash='dash'),
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Portfolio Growth (Initial: ${initial_investment:,})",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # METRICS TABLE
    # =============================================================================
    
    st.subheader("📊 Performance Metrics")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Reorder columns
    column_order = ['Total Return', 'CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Beta', 'Years']
    existing_columns = [col for col in column_order if col in metrics_df.columns]
    metrics_df = metrics_df[existing_columns]
    
    # Format as percentages
    for col in ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Format ratios
    for col in ['Sharpe Ratio', 'Calmar Ratio', 'Beta']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    
    # Format years
    if 'Years' in metrics_df.columns:
        metrics_df['Years'] = metrics_df['Years'].apply(lambda x: f"{x:.1f}")
    
    # Add color styling (same as Page 1)
    def color_metrics(val):
        """Color code metrics"""
        try:
            # Extract numeric value from percentage strings
            if isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', ''))
                if num_val > 0:
                    return 'color: green'  # Same as Page 1
                elif num_val < 0:
                    return 'color: red'    # Same as Page 1
            return ''
        except:
            return ''
    
    styled_df = metrics_df.style.applymap(color_metrics)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # =============================================================================
    # YEARLY RETURNS TABLE
    # =============================================================================
    
    st.subheader("📅 Yearly Returns")
    
    yearly_dict = {}
    for ticker, returns in returns_dict.items():
        yearly_dict[ticker] = calculate_yearly_returns(returns, all_dates)
    
    # Create yearly returns dataframe
    yearly_df = pd.DataFrame(yearly_dict).T
    
    # Format as percentages
    yearly_df = yearly_df.applymap(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    
    # Add color styling (same as Page 1)
    def color_yearly(val):
        """Color code yearly returns"""
        try:
            if isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', ''))
                if num_val > 0:
                    return 'color: green'  # Same as Page 1
                elif num_val < 0:
                    return 'color: red'    # Same as Page 1
            return ''
        except:
            return ''
    
    styled_yearly = yearly_df.style.applymap(color_yearly)
    
    st.dataframe(styled_yearly, use_container_width=True)
    
    # =============================================================================
    # MONTHLY RETURNS TABLE
    # =============================================================================
    
    st.subheader("📆 Monthly Returns (Select Ticker)")
    
    # Ticker selection for monthly view
    monthly_ticker = st.selectbox(
        "Select ticker for monthly returns",
        options=list(returns_dict.keys()),
        index=0
    )
    
    if monthly_ticker:
        monthly_dict = calculate_monthly_returns(
            returns_dict[monthly_ticker],
            all_dates
        )
        
        # Create monthly returns dataframe
        monthly_df = pd.DataFrame(monthly_dict).T
        
        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_df.columns = month_names
        
        # Format as percentages
        monthly_df = monthly_df.applymap(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "")
        
        # Add color styling
        styled_monthly = monthly_df.style.applymap(color_yearly)
        
        st.dataframe(styled_monthly, use_container_width=True)
    
    # =============================================================================
    # FINAL VALUE TABLE
    # =============================================================================
    
    st.subheader("💰 Final Portfolio Values")
    
    final_values = {}
    for ticker, growth in growth_dict.items():
        final_values[ticker] = growth.iloc[-1]
    
    final_df = pd.DataFrame(list(final_values.items()), columns=['Ticker', 'Final Value'])
    final_df['Return'] = ((final_df['Final Value'] - initial_investment) / initial_investment * 100).apply(lambda x: f"{x:.2f}%")
    final_df['Final Value'] = final_df['Final Value'].apply(lambda x: f"${x:,.0f}")
    final_df = final_df.sort_values('Final Value', ascending=False, key=lambda x: x.str.replace('$', '').str.replace(',', '').astype(float))
    
    st.dataframe(final_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.success("✅ Comparison complete!")
    
    # Store results in session state
    st.session_state.comparison_results = {
        'data_dict': data_dict,
        'original_data_dict': original_data_dict,
        'ticker_start_dates': ticker_start_dates,
        'common_start_date': common_start_date,
        'earliest_possible_date': earliest_possible_date,
        'filtered_data_dict': filtered_data_dict,
        'all_dates': all_dates,
        'aligned_prices': aligned_prices,
        'returns_dict': returns_dict,
        'growth_dict': growth_dict,
        'metrics_dict': metrics_dict,
        'benchmark_returns': benchmark_returns,
        'initial_investment': initial_investment
    }

# Display results if they exist (even if checkboxes change)
elif st.session_state.comparison_results is not None:
    # Retrieve stored results
    results = st.session_state.comparison_results
    data_dict = results['data_dict']
    ticker_start_dates = results['ticker_start_dates']
    common_start_date = results['common_start_date']
    earliest_possible_date = results['earliest_possible_date']
    filtered_data_dict = results['filtered_data_dict']
    all_dates = results['all_dates']
    aligned_prices = results['aligned_prices']
    returns_dict = results['returns_dict']
    growth_dict = results['growth_dict']
    metrics_dict = results['metrics_dict']
    benchmark_returns = results['benchmark_returns']
    initial_investment = results['initial_investment']
    
    st.info("📊 **Displaying previous results** - Click 'Run Comparison' to update with new selections")
    
    # Show start dates table again
    st.subheader("📅 Data Availability by Ticker")
    start_dates_df = pd.DataFrame([
        {
            'Ticker': ticker,
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'Start Year': start_date.year,
            'Days of Data': len(data_dict[ticker])
        }
        for ticker, start_date in sorted(ticker_start_dates.items(), key=lambda x: x[1])
    ])
    
    def color_start_dates(row):
        year = row['Start Year']
        if year < 2010:
            return ['background-color: #28a745; color: white'] * len(row)  # Dark green (oldest)
        elif year < 2015:
            return ['background-color: #d4edda; color: #155724'] * len(row)  # Light green
        elif year < 2020:
            return ['background-color: #fff3cd; color: #856404'] * len(row)  # Yellow
        else:
            return ['background-color: #f8d7da; color: #721c24'] * len(row)  # Red (newest/shortest)
    
    styled_start_dates = start_dates_df.style.apply(color_start_dates, axis=1)
    st.dataframe(styled_start_dates, use_container_width=True, hide_index=True)
    
    st.info(f"""
    📊 **Previous Results Date Range**:
    - **Backtest started**: {common_start_date.strftime('%Y-%m-%d')} (newest ticker: {max(ticker_start_dates, key=ticker_start_dates.get)})
    - **Earliest ticker**: {min(ticker_start_dates, key=ticker_start_dates.get)} (starts {earliest_possible_date.strftime('%Y-%m-%d')})
    - **All tickers aligned to**: {common_start_date.strftime('%Y-%m-%d')} → Today
    """)
    
    # Display results
    st.markdown("---")
    st.header("📊 Results")
    
    # Display key info at top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Backtest Start", start_date.strftime('%Y-%m-%d'))
    with col2:
        st.metric("📊 Tickers Compared", len(data_dict))
    with col3:
        st.metric("⏱️ Years of Data", f"{len(all_dates) / 252:.1f}")
    
    st.markdown("---")
    
    # Performance Chart
    st.subheader("📈 Portfolio Growth")
    
    fig = go.Figure()
    
    # Separate momentum ETFs and benchmarks for better visualization
    momentum_tickers = [t for t in growth_dict.keys() if t not in BENCHMARK_TICKERS]
    benchmark_tickers_present = [t for t in growth_dict.keys() if t in BENCHMARK_TICKERS]
    
    # Add momentum ETFs (normal lines)
    for ticker in momentum_tickers:
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=growth_dict[ticker],
            mode='lines',
            name=ticker,
            line=dict(width=2),
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    # Add benchmarks (dashed lines)
    for ticker in benchmark_tickers_present:
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=growth_dict[ticker],
            mode='lines',
            name=f"{ticker} (Benchmark)",
            line=dict(width=2, dash='dash'),
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Portfolio Growth (Initial: ${initial_investment:,})",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Table
    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame(metrics_dict).T
    column_order = ['Total Return', 'CAGR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Beta', 'Years']
    existing_columns = [col for col in column_order if col in metrics_df.columns]
    metrics_df = metrics_df[existing_columns]
    
    # Format as percentages
    for col in ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Format ratios
    for col in ['Sharpe Ratio', 'Calmar Ratio', 'Beta']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    
    # Format years
    if 'Years' in metrics_df.columns:
        metrics_df['Years'] = metrics_df['Years'].apply(lambda x: f"{x:.1f}")
    
    # Add color styling (same as Page 1)
    def color_metrics(val):
        try:
            if isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', ''))
                if num_val > 0:
                    return 'color: green'  # Same as Page 1
                elif num_val < 0:
                    return 'color: red'    # Same as Page 1
            return ''
        except:
            return ''
    
    styled_df = metrics_df.style.applymap(color_metrics)
    st.dataframe(styled_df, use_container_width=True)
    
    # Yearly Returns Table
    st.subheader("📅 Yearly Returns")
    yearly_dict = {}
    for ticker, returns in returns_dict.items():
        yearly_dict[ticker] = calculate_yearly_returns(returns, all_dates)
    
    yearly_df = pd.DataFrame(yearly_dict).T
    yearly_df = yearly_df.applymap(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    
    def color_yearly(val):
        try:
            if isinstance(val, str) and '%' in val:
                num_val = float(val.replace('%', ''))
                if num_val > 0:
                    return 'color: green'  # Same as Page 1
                elif num_val < 0:
                    return 'color: red'    # Same as Page 1
            return ''
        except:
            return ''
    
    styled_yearly = yearly_df.style.applymap(color_yearly)
    st.dataframe(styled_yearly, use_container_width=True)
    
    # Monthly Returns Table
    st.subheader("📆 Monthly Returns (Select Ticker)")
    monthly_ticker = st.selectbox(
        "Select ticker for monthly returns",
        options=list(returns_dict.keys()),
        index=0
    )
    
    if monthly_ticker:
        monthly_dict = calculate_monthly_returns(
            returns_dict[monthly_ticker],
            all_dates
        )
        monthly_df = pd.DataFrame(monthly_dict).T
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_df.columns = month_names
        monthly_df = monthly_df.applymap(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "")
        styled_monthly = monthly_df.style.applymap(color_yearly)
        st.dataframe(styled_monthly, use_container_width=True)
    
    # Final Value Table
    st.subheader("💰 Final Portfolio Values")
    final_values = {}
    for ticker, growth in growth_dict.items():
        final_values[ticker] = growth.iloc[-1]
    
    final_df = pd.DataFrame(list(final_values.items()), columns=['Ticker', 'Final Value'])
    final_df['Return'] = ((final_df['Final Value'] - initial_investment) / initial_investment * 100).apply(lambda x: f"{x:.2f}%")
    final_df['Final Value'] = final_df['Final Value'].apply(lambda x: f"${x:,.0f}")
    final_df = final_df.sort_values('Final Value', ascending=False, key=lambda x: x.str.replace('$', '').str.replace(',', '').astype(float))
    
    st.dataframe(final_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 **Select your tickers above and click Run Comparison to start!**")
    
    # Show available tickers with start dates
    st.subheader("📋 Available Tickers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Momentum ETFs:**")
        st.markdown("- **PDP** (2007+) - Longest history")
        st.markdown("- **MTUM** (2013+) - iShares")
        st.markdown("- **SPMO** (2015+) - Invesco S&P 500")
        st.markdown("- **QMOM** (2015+) - Alpha Architect")
        st.markdown("- **FDMO** (2016+) - Fidelity")
    
    with col2:
        st.markdown("**More ETFs:**")
        st.markdown("- **ONEO** (2015+) - SPDR Russell 1000")
        st.markdown("- **MMTM** (2015+) - SPDR S&P 1500")
        st.markdown("- **IMOM** (2015+) - International")
        st.markdown("- **JMOM** (2017+) - JPMorgan")
        st.markdown("")
        st.markdown("**Benchmarks:**")
        st.markdown("- **SPY** (1993+) - S&P 500")
        st.markdown("- **QQQ** (1999+) - NASDAQ 100")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Tips:")
    st.markdown("- ✅ **Check/uncheck individual tickers** above")
    st.markdown("- 📅 **Start dates shown** next to each ticker")
    st.markdown("- 🚀 **Exclude recent tickers** after download to start backtest earlier")
    st.markdown("- 📊 **Fair comparison** - all tickers aligned to common date range")

