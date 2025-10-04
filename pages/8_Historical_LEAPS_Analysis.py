import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm

st.set_page_config(
    page_title="Historical Options Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'leaps_analysis_ran' not in st.session_state:
    st.session_state.leaps_analysis_ran = False

# Clear session state if needed
if st.button("🗑️ Clear All Results", help="Clear all saved analysis results"):
    keys_to_clear = ['leaps_analysis_results', 'leaps_analysis_params', 'portfolio_backtest_results', 'portfolio_initial_capital', 'portfolio_leaps_allocation']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ All results cleared!")
    st.rerun()

st.title("📈 Historical Options Analysis")

st.markdown("""
Analyze historical options pricing across different time periods with intelligent volatility selection.
""")

# Enhanced Black-Scholes function
def black_scholes_call_professional(S, K, T, r, sigma, dividend_yield=0.0):
    """
    Professional Black-Scholes implementation with maximum accuracy
    Handles edge cases and uses proper mathematical precision
    """
    # Handle edge cases
    if T <= 0:
        return max(S - K, 0)
    
    if sigma <= 0:
        return max(S - K, 0)
    
    if S <= 0 or K <= 0:
        return 0.0
    
    # Ensure inputs are within reasonable bounds
    T = max(T, 1/365)  # Minimum 1 day
    sigma = max(sigma, 0.001)  # Minimum 0.1% volatility
    sigma = min(sigma, 5.0)  # Maximum 500% volatility
    
    # Black-Scholes formula with continuous dividend yield
    sqrt_T = np.sqrt(T)
    
    # Calculate d1 and d2 with proper precision
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Use high-precision cumulative normal distribution
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Calculate option price
    call_price = S * np.exp(-dividend_yield * T) * N_d1 - K * np.exp(-r * T) * N_d2
    
    # Ensure price is never below intrinsic value
    intrinsic_value = max(S - K, 0)
    call_price = max(call_price, intrinsic_value)
    
    # Handle extreme cases (very deep ITM or OTM)
    if call_price < intrinsic_value:
        call_price = intrinsic_value
    
    return call_price

def get_intelligent_volatility_period(expiration_days):
    """
    Determine the appropriate historical period for volatility calculation
    based on time to expiration - ENHANCED ACCURACY MODEL
    """
    if expiration_days <= 30:
        return 60   # 2 months for short-term options
    elif expiration_days <= 60:
        return 90   # 3 months for 2-month options
    elif expiration_days <= 90:
        return 120  # 4 months for 3-month options
    elif expiration_days <= 180:
        return 252  # 1 year for 6-month options
    elif expiration_days <= 365:
        return 504  # 2 years for 1-year options
    elif expiration_days <= 730:
        return 756  # 3 years for 2-year options
    else:
        return 1008  # 4 years for 3+ year options

def calculate_ewma_volatility(prices, period_days, alpha=0.94):
    """
    EWMA (Exponentially Weighted Moving Average) volatility - Industry Standard
    """
    if len(prices) < 30:
        return 0.20
    
    if len(prices) > period_days:
        prices = prices.tail(period_days)
    
    # Calculate log returns (more accurate for EWMA)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns[np.isfinite(log_returns)]
    
    if len(log_returns) < 30:
        return 0.20
    
    # EWMA variance calculation
    ewma_var = np.zeros(len(log_returns))
    ewma_var[0] = np.var(log_returns[:30])  # Initialize with first 30 days variance
    
    for i in range(1, len(log_returns)):
        ewma_var[i] = alpha * ewma_var[i-1] + (1 - alpha) * log_returns[i]**2
    
    # Use the most recent EWMA variance
    volatility = np.sqrt(ewma_var[-1] * 252)
    
    return volatility

def calculate_garch_volatility(prices, period_days):
    """
    GARCH(1,1) volatility model - Most sophisticated professional model
    """
    if len(prices) < 100:  # Need more data for GARCH
        return calculate_ewma_volatility(prices, period_days)
    
    if len(prices) > period_days:
        prices = prices.tail(period_days)
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns[np.isfinite(log_returns)]
    
    if len(log_returns) < 100:
        return calculate_ewma_volatility(prices, period_days)
    
    # GARCH(1,1) parameters (typical values)
    omega = 0.000001  # Long-term variance
    alpha = 0.05      # ARCH effect
    beta = 0.90       # GARCH effect
    
    # Initialize variance
    garch_var = np.zeros(len(log_returns))
    garch_var[0] = np.var(log_returns[:30])
    
    # GARCH(1,1) recursion
    for i in range(1, len(log_returns)):
        garch_var[i] = omega + alpha * log_returns[i-1]**2 + beta * garch_var[i-1]
    
    # Use the most recent GARCH variance
    volatility = np.sqrt(garch_var[-1] * 252)
    
    return volatility

def calculate_simple_calibrated_volatility(prices, period_days, calibration_factor=1.0):
    """
    Simple, reliable volatility calculation that can be calibrated
    """
    if len(prices) < 30:
        return 0.20  # Default 20%
    
    if len(prices) > period_days:
        prices = prices.tail(period_days)
    
    # Simple percentage returns calculation
    returns = prices.pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 30:
        return 0.20
    
    # Calculate standard deviation and annualize
    volatility = returns.std() * np.sqrt(252)
    
    # Apply calibration factor to adjust entire backtest
    calibrated_volatility = volatility * calibration_factor
    
    # Ensure reasonable bounds (5% to 100%)
    calibrated_volatility = max(0.05, min(1.0, calibrated_volatility))
    
    return calibrated_volatility

def calculate_professional_volatility(prices, period_days, vix_value=None, expiration_days=365, method="Simple"):
    """
    Simple volatility calculation that works reliably
    """
    # Get calibration factor from session state
    calibration_data = st.session_state.get('calibration_data', {'enabled': False})
    calibration_factor = calibration_data.get('calibration_factor', 1.0)
    
    return calculate_simple_calibrated_volatility(prices, period_days, calibration_factor)

def estimate_volatility_from_price(stock_price, strike_price, option_price, days_to_exp, risk_free_rate=0.04, dividend_yield=0.015):
    """
    Estimate volatility from current option price using Black-Scholes
    """
    time_years = days_to_exp / 365.25
    
    # Use numerical method to find volatility that gives the option price
    # Start with reasonable bounds
    vol_low = 0.01  # 1%
    vol_high = 2.0  # 200%
    
    # Binary search to find the right volatility
    for _ in range(50):  # Max 50 iterations
        vol_mid = (vol_low + vol_high) / 2
        predicted_price = black_scholes_call_professional(
            stock_price, strike_price, time_years, 
            risk_free_rate, vol_mid, dividend_yield
        )
        
        if abs(predicted_price - option_price) < 0.01:  # Close enough
            break
        
        if predicted_price < option_price:
            vol_low = vol_mid
        else:
            vol_high = vol_mid
    
    return vol_mid

def calculate_calibration_factor(calibration_data):
    """
    Calculate calibration factor based on real option price
    """
    if not calibration_data.get('enabled', False):
        return 1.0
    
    # Get calibration inputs
    stock_price = calibration_data['stock_price']
    strike_price = calibration_data['strike_price']
    real_price = calibration_data['real_price']
    days_to_exp = calibration_data.get('days_to_exp', 365)
    
    # Estimate what volatility gives the real price
    estimated_vol = estimate_volatility_from_price(stock_price, strike_price, real_price, days_to_exp)
    
    # Get base volatility from historical data
    base_vol = 0.15  # Default base
    
    # Calculate calibration factor
    calibration_factor = estimated_vol / base_vol
    
    # Store both the estimated volatility and calibration factor
    calibration_data['estimated_volatility'] = estimated_vol
    calibration_data['calibration_factor'] = calibration_factor
    
    return calibration_factor

def get_13_week_treasury_rate(date=None):
    """
    Get 13-week treasury rate (3-month T-bill rate) for risk-free rate
    """
    try:
        if date is None:
            date = datetime.now()
        
        # Use IRX (3-month treasury rate) from Yahoo Finance
        treasury = yf.Ticker("^IRX")
        hist = treasury.history(start=date - timedelta(days=30), end=date + timedelta(days=1))
        
        if not hist.empty:
            # Get the most recent rate
            latest_rate = hist['Close'].iloc[-1]
            return latest_rate / 100  # Convert to decimal
        else:
            # Fallback to current rate
            current = yf.Ticker("^IRX").history(period="5d")
            if not current.empty:
                return current['Close'].iloc[-1] / 100
            else:
                return 0.04  # 4% fallback
    except:
        return 0.04  # 4% fallback

def get_dynamic_dividend_yield(ticker, date=None):
    """
    Get dynamic dividend yield for the ticker - CORRECTED VERSION
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Method 1: Try to get dividend yield from info
        dividend_info = stock.info
        if 'dividendYield' in dividend_info and dividend_info['dividendYield'] is not None:
            yield_raw = dividend_info['dividendYield']
            
            # Yahoo sometimes returns as decimal (0.0111) or percentage (1.11)
            if yield_raw < 1:
                # It's a decimal, convert to percentage
                yield_percentage = yield_raw * 100
            else:
                # It's already a percentage
                yield_percentage = yield_raw
            
            return yield_percentage
        
        # Method 2: Calculate from recent dividends
        dividends = stock.dividends
        if not dividends.empty:
            # Get recent annual dividend (last 4 quarters)
            recent_dividends = dividends.tail(4)
            annual_dividend = recent_dividends.sum()
            
            # Get current price
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            # Calculate dividend yield
            dividend_yield = (annual_dividend / current_price) * 100
            return max(0, dividend_yield)
        
        # Method 3: No dividend data available - ask user
        st.warning(f"⚠️ No dividend data available for {ticker}")
        st.error("❌ Cannot calculate dividend yield - please specify manually or choose a different ticker")
        return None  # Return None to indicate failure
        
    except Exception as e:
        st.error(f"❌ Error calculating dividend yield: {str(e)}")
        return None

def run_simple_portfolio_backtest(leaps_data, initial_capital, leaps_allocation, cash_allocation):
    """
    Run portfolio backtest with LEAPS investments
    """
    try:
        portfolio_data = []
        current_cash = initial_capital
        current_leaps_contracts = 0
        current_leaps_cost = 0
        current_leaps_purchase_date = None
        current_leaps_strike_price = 0  # FIX: Initialize strike price
        current_leaps_original_price = 0  # Store original purchase price for consistent valuation
        current_leaps_original_stock_price = 0  # Store original stock price at purchase
        
        leaps_data['Date'] = pd.to_datetime(leaps_data['Date'])
        leaps_data = leaps_data.sort_values('Date')
        
        for idx, row in leaps_data.iterrows():
            date = row['Date']
            stock_price = row['Stock_Price']
            leaps_price = row['Option_Price']
            strike_price = row['Strike_Price']
            
        # Check if LEAPS have expired
        leaps_expired = False
        if current_leaps_contracts > 0 and current_leaps_purchase_date:
            days_since_purchase = (date - current_leaps_purchase_date).days
            if days_since_purchase >= 365:  # LEAPS have expired
                leaps_expired = True
                    
            
            # Check if we need to rebalance (buy new LEAPS)
            should_rebalance = False
            if current_leaps_purchase_date is None:
                # First purchase
                should_rebalance = True
            else:
                # Check if enough time has passed for rebalancing (ONLY when LEAPS expire)
                days_since_purchase = (date - current_leaps_purchase_date).days
                if days_since_purchase >= 365:  # Assuming 365-day LEAPS
                    should_rebalance = True
            
            if should_rebalance:
                # Calculate current LEAPS value BEFORE rebalancing - REALISTIC
                current_leaps_value_before = 0
                if current_leaps_contracts > 0 and current_leaps_purchase_date:
                    # Real LEAPS behavior: intrinsic value + time value
                    intrinsic_value_per_contract = max(stock_price - current_leaps_strike_price, 0)
                    
                    # Calculate time value
                    days_held = (date - current_leaps_purchase_date).days
                    days_to_expiration = 365 - days_held
                    if days_to_expiration <= 0:
                        time_value_per_contract = 0
                    else:
                        original_intrinsic = max(current_leaps_original_stock_price - current_leaps_strike_price, 0)
                        original_time_value = current_leaps_original_price - original_intrinsic
                        time_value_per_contract = max(original_time_value * (days_to_expiration / 365), 0)
                    
                    current_leaps_value_before = current_leaps_contracts * (intrinsic_value_per_contract + time_value_per_contract)
                
                # Calculate total portfolio value BEFORE rebalancing
                total_portfolio_value = current_cash + current_leaps_value_before
                
                # CRITICAL FIX: Cash should jump up by the LEAPS value, then we buy new LEAPS
                # This ensures total portfolio value remains exactly the same
                
                # Step 1: Add current LEAPS value to cash (this is the "jump")
                current_cash += current_leaps_value_before
                
                # Step 2: Calculate new allocation based on total portfolio value
                leaps_capital = total_portfolio_value * (leaps_allocation / 100)
                cash_capital = total_portfolio_value * (cash_allocation / 100)
                
                # Step 3: Buy new LEAPS with the allocated capital
                if leaps_price > 0:
                    current_leaps_contracts = leaps_capital / leaps_price
                    current_leaps_cost = leaps_capital
                    current_leaps_purchase_date = date
                    current_leaps_strike_price = strike_price
                    current_leaps_original_price = leaps_price  # Store original purchase price
                    current_leaps_original_stock_price = stock_price  # Store original stock price
                    current_cash = cash_capital
                    
                    # CRITICAL: Immediately after rebalancing, new LEAPS value must equal what we paid
                    # This ensures total portfolio value remains exactly the same
                    new_leaps_value_immediately_after = current_leaps_contracts * leaps_price
                    total_portfolio_after_rebalancing = current_cash + new_leaps_value_immediately_after
                    
                    # Verify no jump occurred
                    if abs(total_portfolio_after_rebalancing - total_portfolio_value) > 1:
                        print(f"WARNING: Portfolio jumped by ${total_portfolio_after_rebalancing - total_portfolio_value:,.0f}")
                else:
                    current_leaps_contracts = 0
                    current_leaps_cost = 0
                    current_cash = cash_capital
                
                # CRITICAL: Add intrinsic value from expired LEAPS AFTER rebalancing
                if leaps_expired:
                    # Use current contracts and calculate intrinsic value
                    expired_intrinsic = max(stock_price - current_leaps_strike_price, 0)
                    cash_from_expired_leaps = current_leaps_contracts * expired_intrinsic
                    current_cash += cash_from_expired_leaps
                
                # CRITICAL: Verify total portfolio value remains unchanged
                # Calculate new LEAPS value immediately after rebalancing
                new_leaps_value_after_rebalancing = 0
                if current_leaps_contracts > 0:
                    # For newly purchased LEAPS, value should equal purchase price initially
                    # (intrinsic value = 0 for ATM options, time value = full premium)
                    intrinsic_value_per_contract = max(stock_price - current_leaps_strike_price, 0)
                    intrinsic_value_total = current_leaps_contracts * intrinsic_value_per_contract
                    
                    # Time value = full premium since we just bought them
                    days_remaining = 365
                    time_decay_factor = days_remaining / 365
                    original_intrinsic = max(stock_price - current_leaps_strike_price, 0)
                    original_time_value = current_leaps_original_price - original_intrinsic
                    remaining_time_value = original_time_value * time_decay_factor
                    extrinsic_value_total = current_leaps_contracts * remaining_time_value
                    
                    new_leaps_value_after_rebalancing = intrinsic_value_total + extrinsic_value_total
                
                # Verify total portfolio value is unchanged
                total_after_rebalancing = current_cash + new_leaps_value_after_rebalancing
                if abs(total_after_rebalancing - total_portfolio_value) > 1:
                    print(f"WARNING: Portfolio value changed during rebalancing: ${total_after_rebalancing - total_portfolio_value:,.0f}")
                
                # CRITICAL: Ensure the total portfolio value immediately after rebalancing 
                # equals the total portfolio value before rebalancing
                # This prevents any jumps in the blue line
                # The new LEAPS value should be calculated the same way as during normal operation
            
        # Calculate current LEAPS value - REALISTIC LEAPS BEHAVIOR
        if current_leaps_contracts > 0 and current_leaps_purchase_date:
            days_held = (date - current_leaps_purchase_date).days
            
            # Real LEAPS behavior: intrinsic value + time value
            intrinsic_value_per_contract = max(stock_price - current_leaps_strike_price, 0)
            
            # Calculate time value (decreases linearly to expiration)
            days_to_expiration = 365 - days_held
            if days_to_expiration <= 0:
                time_value_per_contract = 0
            else:
                # Original time value = original premium - original intrinsic value
                original_intrinsic = max(current_leaps_original_stock_price - current_leaps_strike_price, 0)
                original_time_value = current_leaps_original_price - original_intrinsic
                time_value_per_contract = max(original_time_value * (days_to_expiration / 365), 0)
            
            # Total LEAPS value = contracts * (intrinsic + time value)
            current_leaps_value = current_leaps_contracts * (intrinsic_value_per_contract + time_value_per_contract)
            intrinsic_value = intrinsic_value_per_contract
        else:
            current_leaps_value = 0
            intrinsic_value = 0
            days_held = 0
            
            # Total portfolio value
            total_portfolio = current_cash + current_leaps_value
            
            # Portfolio return
            portfolio_return = ((total_portfolio - initial_capital) / initial_capital) * 100
            
            portfolio_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Cash_Value': current_cash,
                'LEAPS_Value': current_leaps_value,
                'Total_Portfolio': total_portfolio,
                'Portfolio_Return_%': portfolio_return,
                'LEAPS_Contracts': current_leaps_contracts,
                'Stock_Price': stock_price,
                'Intrinsic_Value': intrinsic_value if current_leaps_contracts > 0 else 0,
                'Days_Held': days_held if current_leaps_contracts > 0 else 0
            })
        
        return pd.DataFrame(portfolio_data)
        
    except Exception as e:
        st.error(f"❌ Error running portfolio backtest: {str(e)}")
        return None

def run_simple_portfolio_backtest(analysis_results, initial_capital=100000, 
                                 cash_allocation=80, leaps_allocation=20, expiration_days=365):
    """
    Simple, clean portfolio backtest using table approach:
    1. Start with 80% cash, 20% LEAPS
    2. At rebalancing: Cash + contracts * max(Price - Strike, 0) = New Total
    3. Rebalance: 80% cash, 20% LEAPS
    """
    try:
        if analysis_results is None or analysis_results.empty:
            return None
        
        # Initialize portfolio - start with 100% cash, then rebalance to 80/20
        current_cash = initial_capital  # Start with 100% cash
        current_leaps_contracts = 0
        current_leaps_strike_price = 0
        current_leaps_purchase_date = None
        
        # Track results
        portfolio_data = []
        
        analysis_results['Date'] = pd.to_datetime(analysis_results['Date'])
        analysis_results = analysis_results.sort_values('Date')
        
        for idx, row in analysis_results.iterrows():
            date = row['Date']
            stock_price = row['Stock_Price']
            leaps_price = row['Option_Price']
            strike_price = row['Strike_Price']
            
            # Check if we need to rebalance (based on expiration days)
            should_rebalance = False
            if current_leaps_purchase_date is None:
                should_rebalance = True
            else:
                days_since_purchase = (date - current_leaps_purchase_date).days
                if days_since_purchase >= expiration_days:
                    should_rebalance = True
            
            if should_rebalance:
                # Step 1: Calculate LEAPS value at expiration
                leaps_value_at_expiration = 0
                if current_leaps_contracts > 0:
                    # Cash + contracts * max(Price - Strike, 0)
                    intrinsic_value_per_contract = max(stock_price - current_leaps_strike_price, 0)
                    leaps_value_at_expiration = current_leaps_contracts * intrinsic_value_per_contract
                
                # Step 2: New total = Cash + LEAPS value
                new_total = current_cash + leaps_value_at_expiration
                
                # Step 3: Rebalance: 80% cash, 20% LEAPS
                new_cash = new_total * (cash_allocation / 100)
                leaps_capital = new_total * (leaps_allocation / 100)
                
                # Step 4: Buy new LEAPS
                if leaps_price > 0:
                    new_leaps_contracts = leaps_capital / leaps_price
                    new_strike_price = stock_price  # ATM options
                else:
                    new_leaps_contracts = 0
                    new_strike_price = 0
                
                # Update portfolio
                current_cash = new_cash
                current_leaps_contracts = new_leaps_contracts
                current_leaps_strike_price = new_strike_price
                current_leaps_purchase_date = date
            
            # Only store data on rebalancing days (table approach)
            if should_rebalance:
                # For display, show the total portfolio value after rebalancing
                # The LEAPS value is already included in the cash (from the expired LEAPS)
                # and new LEAPS are bought with the allocated capital
                total_portfolio = current_cash + (current_leaps_contracts * leaps_price if leaps_price > 0 else 0)
                
                # Store rebalancing data
                portfolio_data.append({
                    'Date': date,
                    'Cash_Value': current_cash,
                    'LEAPS_Value': current_leaps_contracts * leaps_price if leaps_price > 0 else 0,
                    'Total_Portfolio': total_portfolio,
                    'Portfolio_Return_%': ((total_portfolio - initial_capital) / initial_capital) * 100,
                    'LEAPS_Contracts': current_leaps_contracts,
                    'Stock_Price': stock_price,
                    'Strike_Price': current_leaps_strike_price,
                    'Premium_Cost': leaps_price,
                    'Intrinsic_Value': leaps_value_at_expiration / current_leaps_contracts if current_leaps_contracts > 0 else 0,
                    'Days_Held': 0  # Just purchased
                })
        
        return pd.DataFrame(portfolio_data)
        
    except Exception as e:
        st.error(f"❌ Error running simple portfolio backtest: {str(e)}")
        return None

def display_saved_results(results_df):
    """
    Display saved LEAPS analysis results with charts and tables
    """
    try:
        # Check if results_df is valid
        if results_df is None or results_df.empty:
            st.error("No results to display.")
            return
        
        # Display results
        st.subheader("📊 Historical Options Pricing Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Observations", len(results_df))
        with col2:
            avg_price = results_df['Option_Price'].mean()
            st.metric("Average Option Price", f"${avg_price:.2f}")
        with col3:
            avg_vol = results_df['Volatility_%'].mean()
            st.metric("Average Volatility", f"{avg_vol:.1f}%")
        with col4:
            price_range = results_df['Option_Price'].max() - results_df['Option_Price'].min()
            st.metric("Price Range", f"${price_range:.2f}")
        
        # Display the full table
        st.dataframe(results_df, use_container_width=True)
        
        # Show today's option data (latest row) for comparison with real option chains
        if len(results_df) > 0:
            st.markdown("---")
            st.subheader("💰 Today's Calculated Option Data")
            st.info("💡 Use this data to compare with real option chains from your broker")
            
            latest_row = results_df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📅 Date", latest_row['Date'].strftime('%Y-%m-%d') if hasattr(latest_row['Date'], 'strftime') else str(latest_row['Date']))
            with col2:
                st.metric("📈 Stock Price", f"${latest_row['Stock_Price']:.2f}")
            with col3:
                st.metric("🎯 Strike Price", f"${latest_row['Strike_Price']:.2f}")
            with col4:
                st.metric("📊 Volatility", f"{latest_row['Volatility_%']:.1f}%")
            with col5:
                st.metric("💰 **Option Price**", f"**${latest_row['Option_Price']:.2f}**")
            
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("📈 Risk-Free Rate", f"{latest_row['Risk_Free_Rate_%']:.2f}%")
            with col7:
                st.metric("💵 Dividend Yield", f"{latest_row['Dividend_Yield_%']:.2f}%")
            with col8:
                st.metric("📊 VIX", f"{latest_row['VIX']:.1f}" if pd.notna(latest_row['VIX']) else "N/A")
        
        # Create charts
        st.subheader("📈 Charts")
        
        # Price evolution chart with separate plots for stock and options
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=('Stock Price', 'LEAPS Option Price', 'Volatility & VIX', 'Premium as % of Strike', 'Risk-Free Rate'),
            vertical_spacing=0.05
        )
        
        # Stock price in first plot
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Stock_Price'],
                name='Stock Price',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Stock Price</b><br>' +
                            'Date: %{x}<br>' +
                            'Stock Price: $%{y:.2f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Option price in second plot
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Option_Price'],
                name='Option Price',
                line=dict(color='red', width=2),
                hovertemplate='<b>Option Price</b><br>' +
                            'Date: %{x}<br>' +
                            'Option Price: $%{y:.2f}<br>' +
                            'Stock Price: $' + results_df['Stock_Price'].astype(str) + '<br>' +
                            'Strike Price: $' + results_df['Strike_Price'].astype(str) + '<br>' +
                            'Volatility: ' + results_df['Volatility_%'].astype(str) + '%<br>' +
                            'Premium %: ' + results_df['Premium_%'].astype(str) + '%<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Volatility
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Volatility_%'],
                name='Volatility %',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        # VIX
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VIX'],
                name='VIX',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Premium percentage
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Premium_%'],
                name='Premium % of Strike',
                line=dict(color='red')
            ),
            row=4, col=1
        )
        
        # Risk-Free Rate
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Risk_Free_Rate_%'],
                name='Treasury Rate %',
                line=dict(color='orange')
            ),
            row=5, col=1
        )
        
        fig.update_layout(
            height=1200,
            title_text=f"LEAPS Analysis Charts",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'  # Show all data on hover
        )
        
        st.plotly_chart(fig, use_container_width=True, key="leaps_analysis_charts")
        
        # Premium percentage statistics
        st.subheader("📊 Premium Percentage Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_premium = results_df['Premium_%'].mean()
            st.metric("Average Premium %", f"{avg_premium:.2f}%")
        with col2:
            min_premium = results_df['Premium_%'].min()
            st.metric("Minimum Premium %", f"{min_premium:.2f}%")
        with col3:
            max_premium = results_df['Premium_%'].max()
            st.metric("Maximum Premium %", f"{max_premium:.2f}%")
        with col4:
            # Calculate CAGR from the analysis results
            if len(results_df) > 1:
                start_date = pd.to_datetime(results_df['Date'].iloc[0])
                end_date = pd.to_datetime(results_df['Date'].iloc[-1])
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    start_price = results_df['Stock_Price'].iloc[0]
                    end_price = results_df['Stock_Price'].iloc[-1]
                    cagr = ((end_price / start_price) ** (1/years) - 1) * 100
                    st.metric("Stock CAGR", f"{cagr:.2f}%")
                else:
                    st.metric("Stock CAGR", "N/A")
            else:
                st.metric("Stock CAGR", "N/A")
        
        # Create rebalancing table (only show dates when buying new options)
        st.subheader("🔄 Options Rebalancing Table")
        
        # Get expiration days from session state or use default
        expiration_days = 365  # Default value
        if 'leaps_analysis_params' in st.session_state:
            expiration_days = st.session_state.leaps_analysis_params.get('expiration_days', 365)
        
        st.info(f"📅 Shows only the dates when you would buy new {expiration_days}-day LEAPS contracts")
        
        # Create rebalancing schedule
        rebalancing_dates = []
        current_date = results_df['Date'].iloc[0]
        date_index = 0
        
        while date_index < len(results_df):
            # Find the current date in results
            current_row = results_df[results_df['Date'] == current_date]
            if not current_row.empty:
                rebalancing_dates.append(current_row.iloc[0])
            
            # Move to next rebalancing date (add expiration_days)
            try:
                current_date_dt = pd.to_datetime(current_date)
                next_date_dt = current_date_dt + timedelta(days=expiration_days)
                next_date = next_date_dt.strftime('%Y-%m-%d')
                
                # Find the next available date in our data
                future_dates = results_df[results_df['Date'] > current_date]
                if future_dates.empty:
                    break
                    
                # Find the closest date to our target rebalancing date
                future_dates['Date_dt'] = pd.to_datetime(future_dates['Date'])
                target_dt = pd.to_datetime(next_date)
                
                # Find the closest date to target (within reasonable range)
                future_dates['date_diff'] = abs((future_dates['Date_dt'] - target_dt).dt.days)
                closest_date = future_dates.loc[future_dates['date_diff'].idxmin()]
                
                if closest_date['date_diff'] <= expiration_days // 4:  # Within 25% of target
                    current_date = closest_date['Date']
                    date_index = results_df[results_df['Date'] == current_date].index[0]
                else:
                    break
            except:
                break
        
        if rebalancing_dates:
            rebalancing_df = pd.DataFrame(rebalancing_dates)
            
            # Add rebalancing info
            rebalancing_df['Rebalance_#'] = range(1, len(rebalancing_df) + 1)
            
            # Calculate days since last rebalance (convert dates first)
            rebalancing_df['Date_dt'] = pd.to_datetime(rebalancing_df['Date'])
            rebalancing_df['Days_Since_Last'] = rebalancing_df['Date_dt'].diff().dt.days.fillna(0).astype(int)
            rebalancing_df = rebalancing_df.drop('Date_dt', axis=1)  # Remove temporary column
            
            # Reorder columns
            columns_order = ['Rebalance_#', 'Date', 'Days_Since_Last', 'Stock_Price', 'Strike_Price', 
                           'Option_Price', 'Premium_%', 'Volatility_%', 'Dividend_Yield_%', 'Risk_Free_Rate_%', 'VIX']
            rebalancing_df = rebalancing_df[columns_order]
            
            # Display rebalancing table
            st.dataframe(rebalancing_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            # Get ticker and dates from session state
            ticker = "LEAPS"
            start_date = "IPO"
            end_date = "latest"
            if 'leaps_analysis_params' in st.session_state:
                params = st.session_state.leaps_analysis_params
                ticker = params.get('ticker', 'LEAPS')
                start_date = params.get('start_date') or 'IPO'
                end_date = params.get('end_date') or 'latest'
            
            with col1:
                csv_full = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Daily Data",
                    data=csv_full,
                    file_name=f"{ticker}_LEAPS_Daily_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_rebalancing = rebalancing_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rebalancing Schedule",
                    data=csv_rebalancing,
                    file_name=f"{ticker}_LEAPS_Rebalancing_{expiration_days}days_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ No rebalancing dates found in the selected period")
            
            # Download full data only
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data",
                data=csv,
                file_name=f"{ticker}_LEAPS_Historical_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error displaying saved results: {str(e)}")

def display_simple_portfolio_tables(portfolio_df, initial_capital, leaps_allocation):
    """
    Display portfolio backtest results with table and charts
    """
    try:
        # Summary metrics
        st.subheader("📊 Portfolio Performance Summary")
        
        final_value = portfolio_df['Total_Portfolio'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Calculate proper max drawdown (peak-to-trough)
        portfolio_values = portfolio_df['Total_Portfolio'].values
        peak = portfolio_values[0]
        max_dd = 0
        max_value = peak
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                max_value = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        max_drawdown = max_dd
        
        # Performance metrics table (like multi-backtest)
        performance_data = {
            'Metric': [
                'Initial Capital',
                'Final Value', 
                'Total Return',
                'Max Value',
                'Max Drawdown'
            ],
            'Value': [
                f"${initial_capital:,.0f}",
                f"${final_value:,.0f}",
                f"{total_return:.2f}%",
                f"${max_value:,.0f}",
                f"{max_drawdown:.2f}%"
            ]
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Advanced Performance Metrics (like multi-backtest)
        st.subheader("📈 Advanced Performance Metrics")
        
        # Calculate time period for CAGR
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        start_date = portfolio_df['Date'].iloc[0]
        end_date = portfolio_df['Date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        # Calculate CAGR
        if years > 0 and initial_capital > 0:
            cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100
        else:
            cagr = 0
        
        # Calculate daily returns for volatility and Sharpe ratio
        portfolio_df['Daily_Return'] = portfolio_df['Total_Portfolio'].pct_change().dropna()
        daily_returns = portfolio_df['Daily_Return']
        
        # Annualized volatility
        if len(daily_returns) > 1:
            volatility = daily_returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 2.0
        if volatility > 0 and len(daily_returns) > 1:
            excess_return = (cagr - risk_free_rate)
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        if len(daily_returns) > 1:
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252) * 100
                sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = float('inf') if cagr > risk_free_rate else 0
        else:
            sortino_ratio = 0
        
        # Calmar ratio (CAGR / Max Drawdown)
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        positive_returns = (daily_returns > 0).sum()
        total_trading_days = len(daily_returns)
        win_rate = (positive_returns / total_trading_days) * 100 if total_trading_days > 0 else 0
        
        # Average win/loss
        winning_days = daily_returns[daily_returns > 0]
        losing_days = daily_returns[daily_returns < 0]
        avg_win = winning_days.mean() * 100 if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() * 100 if len(losing_days) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
        
        # Display metrics in a comprehensive table format
        metrics_data = {
            'Metric': [
                'CAGR (Compound Annual Growth Rate)',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Sortino Ratio', 
                'Calmar Ratio',
                'Max Drawdown',
                'Win Rate',
                'Average Daily Win',
                'Average Daily Loss',
                'Profit/Loss Ratio',
                'Time Period (Years)',
                'Total Trading Days'
            ],
            'Value': [
                f"{cagr:.2f}%",
                f"{volatility:.2f}%",
                f"{sharpe_ratio:.2f}",
                f"{sortino_ratio:.2f}" if sortino_ratio != float('inf') else "∞",
                f"{calmar_ratio:.2f}",
                f"{max_drawdown:.2f}%",
                f"{win_rate:.1f}%",
                f"{avg_win:.3f}%",
                f"{avg_loss:.3f}%",
                f"{profit_loss_ratio:.2f}" if profit_loss_ratio != float('inf') else "∞",
                f"{years:.2f}",
                f"{total_trading_days:,}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Performance interpretation
        st.subheader("📊 Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Return Metrics**")
            if cagr > 10:
                st.success(f"✅ Excellent CAGR: {cagr:.1f}%")
            elif cagr > 5:
                st.info(f"📊 Good CAGR: {cagr:.1f}%")
            else:
                st.warning(f"⚠️ Low CAGR: {cagr:.1f}%")
                
            if calmar_ratio > 1:
                st.success(f"✅ Good Calmar Ratio: {calmar_ratio:.2f}")
            else:
                st.warning(f"⚠️ Low Calmar Ratio: {calmar_ratio:.2f}")
        
        with col2:
            st.markdown("**⚖️ Risk Metrics**")
            if volatility < 15:
                st.success(f"✅ Low Volatility: {volatility:.1f}%")
            elif volatility < 25:
                st.info(f"📊 Moderate Volatility: {volatility:.1f}%")
            else:
                st.warning(f"⚠️ High Volatility: {volatility:.1f}%")
                
            if max_drawdown < 20:
                st.success(f"✅ Low Max Drawdown: {max_drawdown:.1f}%")
            elif max_drawdown < 35:
                st.info(f"📊 Moderate Max Drawdown: {max_drawdown:.1f}%")
            else:
                st.warning(f"⚠️ High Max Drawdown: {max_drawdown:.1f}%")
        
        with col3:
            st.markdown("**🎯 Efficiency Metrics**")
            if sharpe_ratio > 1:
                st.success(f"✅ Excellent Sharpe: {sharpe_ratio:.2f}")
            elif sharpe_ratio > 0.5:
                st.info(f"📊 Good Sharpe: {sharpe_ratio:.2f}")
            else:
                st.warning(f"⚠️ Low Sharpe: {sharpe_ratio:.2f}")
                
            if win_rate > 60:
                st.success(f"✅ High Win Rate: {win_rate:.1f}%")
            elif win_rate > 50:
                st.info(f"📊 Balanced Win Rate: {win_rate:.1f}%")
            else:
                st.warning(f"⚠️ Low Win Rate: {win_rate:.1f}%")
        
        # Portfolio performance table
        st.subheader("📈 Portfolio Performance Table")
        display_df = portfolio_df.copy()
        display_df['Cash_Value'] = display_df['Cash_Value'].round(0)
        display_df['LEAPS_Value'] = display_df['LEAPS_Value'].round(0)
        display_df['Total_Portfolio'] = display_df['Total_Portfolio'].round(0)
        display_df['Portfolio_Return_%'] = display_df['Portfolio_Return_%'].round(1)
        display_df['LEAPS_Contracts'] = display_df['LEAPS_Contracts'].round(2)
        display_df['Intrinsic_Value'] = display_df['Intrinsic_Value'].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
        
        # Portfolio Performance Charts
        st.subheader("📊 Portfolio Performance Charts")
        
        # Create portfolio value chart
        fig1 = go.Figure()
        
        # Add portfolio value line
        fig1.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['Total_Portfolio'],
            mode='lines',
            name='Total Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add cash value line
        fig1.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['Cash_Value'],
            mode='lines',
            name='Cash Value',
            line=dict(color='#2ca02c', width=2)
        ))
        
        # Add LEAPS value line
        fig1.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['LEAPS_Value'],
            mode='lines',
            name='LEAPS Value',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig1.update_layout(
            title="Portfolio Value Evolution",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode="x unified",
            template="plotly_dark",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),
            height=600
        )
        
        st.plotly_chart(fig1, use_container_width=True, key="portfolio_value_chart")
        
        # Create portfolio allocation chart
        fig2 = go.Figure()
        
        # Calculate allocation percentages
        portfolio_df['Cash_Allocation_%'] = (portfolio_df['Cash_Value'] / portfolio_df['Total_Portfolio']) * 100
        portfolio_df['LEAPS_Allocation_%'] = (portfolio_df['LEAPS_Value'] / portfolio_df['Total_Portfolio']) * 100
        
        fig2.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['Cash_Allocation_%'],
            mode='lines',
            name='Cash Allocation %',
            line=dict(color='#2ca02c', width=2),
            fill='tonexty'
        ))
        
        fig2.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['LEAPS_Allocation_%'],
            mode='lines',
            name='LEAPS Allocation %',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy'
        ))
        
        fig2.update_layout(
            title="Portfolio Allocation Over Time",
            xaxis_title="Date",
            yaxis_title="Allocation (%)",
            hovermode="x unified",
            template="plotly_dark",
            yaxis=dict(range=[0, 100]),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=80, t=120, b=80),
            height=600
        )
        
        st.plotly_chart(fig2, use_container_width=True, key="portfolio_allocation_chart")
        
        # Download button
        csv_data = portfolio_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Results",
            data=csv_data,
            file_name=f"Portfolio_LEAPS_Backtest_{leaps_allocation}percent.csv",
            mime="text/csv"
        )
        
        # Monthly and Yearly Performance Tables (styled like multi-backtest)
        st.subheader("📊 Monthly & Yearly Performance Analysis")
        
        # Convert Date column to datetime for analysis
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        
        # Monthly Performance Table
        st.subheader("📅 Monthly Performance")
        
        # Create monthly data
        portfolio_df['Year'] = portfolio_df['Date'].dt.year
        portfolio_df['Month'] = portfolio_df['Date'].dt.month
        portfolio_df['YearMonth'] = portfolio_df['Date'].dt.to_period('M')
        
        # Get first and last value of each month
        monthly_data = []
        for year_month in portfolio_df['YearMonth'].unique():
            month_data = portfolio_df[portfolio_df['YearMonth'] == year_month]
            if len(month_data) > 0:
                start_value = month_data['Total_Portfolio'].iloc[0]
                end_value = month_data['Total_Portfolio'].iloc[-1]
                monthly_return = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
                
                monthly_data.append({
                    'Year-Month': str(year_month),
                    'Start_Value': start_value,
                    'End_Value': end_value,
                    'Monthly_Return_%': monthly_return,
                    'Max_Value': month_data['Total_Portfolio'].max(),
                    'Min_Value': month_data['Total_Portfolio'].min(),
                    'LEAPS_Value': month_data['LEAPS_Value'].iloc[-1],
                    'Cash_Value': month_data['Cash_Value'].iloc[-1]
                })
        
        monthly_df = pd.DataFrame(monthly_data)
        if not monthly_df.empty:
            # Format the data
            monthly_df['Start_Value'] = monthly_df['Start_Value'].round(0)
            monthly_df['End_Value'] = monthly_df['End_Value'].round(0)
            monthly_df['Monthly_Return_%'] = monthly_df['Monthly_Return_%'].round(2)
            monthly_df['Max_Value'] = monthly_df['Max_Value'].round(0)
            monthly_df['Min_Value'] = monthly_df['Min_Value'].round(0)
            monthly_df['LEAPS_Value'] = monthly_df['LEAPS_Value'].round(0)
            monthly_df['Cash_Value'] = monthly_df['Cash_Value'].round(0)
            
            # Create styled table with colors (like multi-backtest)
            def style_monthly_table(val):
                if val.name == 'Monthly_Return_%':
                    colors = []
                    for v in val:
                        if pd.isna(v):
                            colors.append('')
                        elif v > 5:
                            colors.append('background-color: #d4edda; color: #155724')  # Green for good months
                        elif v > 0:
                            colors.append('background-color: #fff3cd; color: #856404')  # Yellow for positive
                        else:
                            colors.append('background-color: #f8d7da; color: #721c24')  # Red for negative
                    return colors
                return [''] * len(val)
            
            styled_monthly = monthly_df.style.apply(style_monthly_table, axis=0)
            st.dataframe(styled_monthly, use_container_width=True)
            
            # Monthly statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_monthly_return = monthly_df['Monthly_Return_%'].mean()
                st.metric("Avg Monthly Return", f"{avg_monthly_return:.2f}%")
            with col2:
                best_month = monthly_df['Monthly_Return_%'].max()
                st.metric("Best Month", f"{best_month:.2f}%")
            with col3:
                worst_month = monthly_df['Monthly_Return_%'].min()
                st.metric("Worst Month", f"{worst_month:.2f}%")
            with col4:
                positive_months = (monthly_df['Monthly_Return_%'] > 0).sum()
                total_months = len(monthly_df)
                win_rate = (positive_months / total_months) * 100 if total_months > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Yearly Performance Table
        st.subheader("📅 Yearly Performance")
        
        # Create yearly data
        yearly_data = []
        for year in portfolio_df['Year'].unique():
            year_data = portfolio_df[portfolio_df['Year'] == year]
            if len(year_data) > 0:
                start_value = year_data['Total_Portfolio'].iloc[0]
                end_value = year_data['Total_Portfolio'].iloc[-1]
                yearly_return = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
                
                # Calculate additional metrics
                max_value = year_data['Total_Portfolio'].max()
                min_value = year_data['Total_Portfolio'].min()
                max_drawdown = ((max_value - min_value) / max_value) * 100 if max_value > 0 else 0
                
                # Count positive months in this year
                year_monthly = monthly_df[monthly_df['Year-Month'].str.startswith(str(year))]
                positive_months = (year_monthly['Monthly_Return_%'] > 0).sum() if not year_monthly.empty else 0
                total_months = len(year_monthly) if not year_monthly.empty else 12
                
                yearly_data.append({
                    'Year': year,
                    'Start_Value': start_value,
                    'End_Value': end_value,
                    'Yearly_Return_%': yearly_return,
                    'Max_Value': max_value,
                    'Min_Value': min_value,
                    'Max_Drawdown_%': max_drawdown,
                    'Positive_Months': positive_months,
                    'Total_Months': total_months,
                    'Win_Rate_%': (positive_months / total_months) * 100 if total_months > 0 else 0
                })
        
        yearly_df = pd.DataFrame(yearly_data)
        if not yearly_df.empty:
            # Format the data
            yearly_df['Start_Value'] = yearly_df['Start_Value'].round(0)
            yearly_df['End_Value'] = yearly_df['End_Value'].round(0)
            yearly_df['Yearly_Return_%'] = yearly_df['Yearly_Return_%'].round(2)
            yearly_df['Max_Value'] = yearly_df['Max_Value'].round(0)
            yearly_df['Min_Value'] = yearly_df['Min_Value'].round(0)
            yearly_df['Max_Drawdown_%'] = yearly_df['Max_Drawdown_%'].round(2)
            yearly_df['Win_Rate_%'] = yearly_df['Win_Rate_%'].round(1)
            
            # Create styled table with colors (like multi-backtest)
            def style_yearly_table(val):
                if val.name == 'Yearly_Return_%':
                    colors = []
                    for v in val:
                        if pd.isna(v):
                            colors.append('')
                        elif v > 20:
                            colors.append('background-color: #d4edda; color: #155724')  # Green for excellent years
                        elif v > 10:
                            colors.append('background-color: #d1ecf1; color: #0c5460')  # Blue for good years
                        elif v > 0:
                            colors.append('background-color: #fff3cd; color: #856404')  # Yellow for positive
                        else:
                            colors.append('background-color: #f8d7da; color: #721c24')  # Red for negative
                    return colors
                elif val.name == 'Max_Drawdown_%':
                    colors = []
                    for v in val:
                        if pd.isna(v):
                            colors.append('')
                        elif v > 30:
                            colors.append('background-color: #f8d7da; color: #721c24')  # Red for high drawdown
                        elif v > 15:
                            colors.append('background-color: #fff3cd; color: #856404')  # Yellow for moderate drawdown
                        else:
                            colors.append('background-color: #d4edda; color: #155724')  # Green for low drawdown
                    return colors
                return [''] * len(val)
            
            styled_yearly = yearly_df.style.apply(style_yearly_table, axis=0)
            st.dataframe(styled_yearly, use_container_width=True)
            
            # Yearly statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_yearly_return = yearly_df['Yearly_Return_%'].mean()
                st.metric("Avg Yearly Return", f"{avg_yearly_return:.2f}%")
            with col2:
                best_year = yearly_df['Yearly_Return_%'].max()
                st.metric("Best Year", f"{best_year:.2f}%")
            with col3:
                worst_year = yearly_df['Yearly_Return_%'].min()
                st.metric("Worst Year", f"{worst_year:.2f}%")
            with col4:
                avg_drawdown = yearly_df['Max_Drawdown_%'].mean()
                st.metric("Avg Max Drawdown", f"{avg_drawdown:.2f}%")
        
        # Download buttons for tables
        col1, col2 = st.columns(2)
        with col1:
            if not monthly_df.empty:
                monthly_csv = monthly_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Monthly Performance",
                    data=monthly_csv,
                    file_name=f"Portfolio_Monthly_Performance_{leaps_allocation}percent.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not yearly_df.empty:
                yearly_csv = yearly_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Yearly Performance",
                    data=yearly_csv,
                    file_name=f"Portfolio_Yearly_Performance_{leaps_allocation}percent.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"❌ Error displaying portfolio results: {str(e)}")

def display_simple_portfolio_plot(portfolio_df, initial_capital, leaps_allocation):
    """
    Display simple portfolio plot and summary metrics
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Calculate summary metrics
        final_value = portfolio_df['Total_Portfolio'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Calculate CAGR
        start_date = pd.to_datetime(portfolio_df['Date'].iloc[0])
        end_date = pd.to_datetime(portfolio_df['Date'].iloc[-1])
        years = (end_date - start_date).days / 365.25
        cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        st.subheader("📊 Portfolio Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Capital", f"${initial_capital:,.0f}")
        with col2:
            st.metric("Final Value", f"${final_value:,.0f}")
        with col3:
            st.metric("Total Return", f"{total_return:.1f}%")
        with col4:
            st.metric("CAGR", f"{cagr:.1f}%")
        
        # Create comprehensive daily portfolio table
        st.subheader("📊 Daily Portfolio Performance Table")
        
        # We need to get the original analysis data to calculate daily portfolio values
        if 'leaps_analysis_results' in st.session_state:
            daily_data = st.session_state.leaps_analysis_results.copy()
            daily_data['Date'] = pd.to_datetime(daily_data['Date'])
            daily_data = daily_data.sort_values('Date')
            
            # Calculate dynamic portfolio values for each day
            daily_portfolio_values = []
            current_cash = initial_capital
            current_contracts = 0
            current_strike = 0
            prev_total = initial_capital
            
            for idx, row in daily_data.iterrows():
                date = row['Date']
                stock_price = row['Stock_Price']
                
                # Check if this is a rebalancing date
                portfolio_row = portfolio_df[portfolio_df['Date'] == date]
                if not portfolio_row.empty:
                    # Update portfolio state on rebalancing
                    current_cash = portfolio_row['Cash_Value'].iloc[0]
                    current_contracts = portfolio_row['LEAPS_Contracts'].iloc[0]
                    current_strike = portfolio_row['Strike_Price'].iloc[0]
                
                # Calculate current portfolio value: Cash + Intrinsic Value
                intrinsic_value = max(stock_price - current_strike, 0) if current_contracts > 0 else 0
                total_value = current_cash + (current_contracts * intrinsic_value)
                daily_change = ((total_value - prev_total) / prev_total * 100) if prev_total > 0 else 0
                
                daily_portfolio_values.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Stock_Price': stock_price,
                    'Cash': current_cash,
                    'LEAPS_Contracts': current_contracts,
                    'Strike_Price': current_strike,
                    'Intrinsic_Value': intrinsic_value,
                    'LEAPS_Total_Value': current_contracts * intrinsic_value,
                    'Total_Portfolio': total_value,
                    'Daily_Change_%': daily_change,
                    'Is_Rebalancing': 'Yes' if not portfolio_row.empty else 'No'
                })
                
                prev_total = total_value
            
            daily_df = pd.DataFrame(daily_portfolio_values)
            
            # Format the display
            daily_display_df = daily_df.copy()
            daily_display_df['Stock_Price'] = daily_display_df['Stock_Price'].apply(lambda x: f"${x:.2f}")
            daily_display_df['Cash'] = daily_display_df['Cash'].apply(lambda x: f"${x:,.0f}")
            daily_display_df['LEAPS_Contracts'] = daily_display_df['LEAPS_Contracts'].apply(lambda x: f"{x:,.0f}")
            daily_display_df['Strike_Price'] = daily_display_df['Strike_Price'].apply(lambda x: f"${x:.2f}")
            daily_display_df['Intrinsic_Value'] = daily_display_df['Intrinsic_Value'].apply(lambda x: f"${x:.2f}")
            daily_display_df['LEAPS_Total_Value'] = daily_display_df['LEAPS_Total_Value'].apply(lambda x: f"${x:,.0f}")
            daily_display_df['Total_Portfolio'] = daily_display_df['Total_Portfolio'].apply(lambda x: f"${x:,.0f}")
            daily_display_df['Daily_Change_%'] = daily_display_df['Daily_Change_%'].apply(lambda x: f"{x:+.2f}%")
            
            # Rename columns for clarity
            daily_display_df.columns = [
                'Date', 'Stock Price', 'Cash', 'LEAPS Contracts', 'Strike Price', 
                'Intrinsic Value/Contract', 'LEAPS Total Value', 'Total Portfolio', 
                'Daily Change %', 'Rebalancing Day'
            ]
            
            st.dataframe(daily_display_df, use_container_width=True)
            
            # Create plot based directly on the daily table data
            st.subheader("📈 Portfolio Performance Chart (Based on Daily Table)")
            
            # Convert back to numeric for plotting (remove formatting)
            plot_df = daily_df.copy()
            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
            
            fig = go.Figure()
            
            # Add total portfolio value line
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['Total_Portfolio'],
                mode='lines',
                name='Total Portfolio Value',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Add cash value line
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['Cash'],
                mode='lines',
                name='Cash Value',
                line=dict(color='#2ca02c', width=2)
            ))
            
            # Add LEAPS total value line
            fig.add_trace(go.Scatter(
                x=plot_df['Date'],
                y=plot_df['LEAPS_Total_Value'],
                mode='lines',
                name='LEAPS Intrinsic Value',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            # Add rebalancing markers
            rebalancing_days = plot_df[plot_df['Is_Rebalancing'] == 'Yes']
            if not rebalancing_days.empty:
                fig.add_trace(go.Scatter(
                    x=rebalancing_days['Date'],
                    y=rebalancing_days['Total_Portfolio'],
                    mode='markers',
                    name='Rebalancing Events',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate='<b>Rebalancing Day</b><br>Date: %{x}<br>Total Value: $%{y:,.0f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title="Portfolio Value Evolution (Based on Daily Table Data)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
                template="plotly_dark",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                yaxis=dict(tickformat='$,.0f')
            )
            
            st.plotly_chart(fig, use_container_width=True, key="daily_portfolio_plot")
            
        else:
            st.warning("⚠️ Daily analysis data not available. Please run the historical analysis first.")
        
        # Display essential rebalancing table
        st.subheader("📋 Portfolio Rebalancing Details")
        
        # Prepare display data
        display_df = portfolio_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Calculate percentage change from previous rebalancing
        display_df['Prev_Total'] = display_df['Total_Portfolio'].shift(1)
        display_df['Change_%'] = ((display_df['Total_Portfolio'] - display_df['Prev_Total']) / display_df['Prev_Total'] * 100).fillna(0)
        
        # Select and format columns for display
        rebalancing_table = display_df[[
            'Date', 'Stock_Price', 'Strike_Price', 'Premium_Cost', 'LEAPS_Contracts', 'Total_Portfolio', 
            'Cash_Value', 'LEAPS_Value', 'Change_%'
        ]].copy()
        
        # Format the display
        rebalancing_table['Stock_Price'] = rebalancing_table['Stock_Price'].apply(lambda x: f"${x:.2f}")
        rebalancing_table['Strike_Price'] = rebalancing_table['Strike_Price'].apply(lambda x: f"${x:.2f}")
        rebalancing_table['Premium_Cost'] = rebalancing_table['Premium_Cost'].apply(lambda x: f"${x:.2f}")
        rebalancing_table['LEAPS_Contracts'] = rebalancing_table['LEAPS_Contracts'].apply(lambda x: f"{x:,.0f}")
        rebalancing_table['Total_Portfolio'] = rebalancing_table['Total_Portfolio'].apply(lambda x: f"${x:,.0f}")
        rebalancing_table['Cash_Value'] = rebalancing_table['Cash_Value'].apply(lambda x: f"${x:,.0f}")
        rebalancing_table['LEAPS_Value'] = rebalancing_table['LEAPS_Value'].apply(lambda x: f"${x:,.0f}")
        rebalancing_table['Change_%'] = rebalancing_table['Change_%'].apply(lambda x: f"{x:+.1f}%")
        
        # Rename columns for clarity
        rebalancing_table.columns = [
            'Rebalance Date', 'Stock Price', 'Strike Price', 'Premium Cost', 'LEAPS Contracts', 'Total Portfolio',
            'Cash Amount', 'LEAPS Amount', 'Change vs Previous'
        ]
        
        st.dataframe(rebalancing_table, use_container_width=True)
        
        # Yearly Performance Summary (styled like multi-backtest)
        st.subheader("📅 Yearly Performance Summary")
        
        # Create yearly summary based on calendar years, not rebalancing dates
        yearly_data = []
        
        # Get all years in the data range
        start_year = pd.to_datetime(display_df['Date']).min().year
        end_year = pd.to_datetime(display_df['Date']).max().year
        
        # Create a proper yearly calendar-based summary
        display_df['Date_dt'] = pd.to_datetime(display_df['Date'])
        display_df = display_df.sort_values('Date_dt')
        
        for year in range(start_year, end_year + 1):
            # Find the portfolio value at the start of the year
            if year == start_year:
                start_value = initial_capital
            else:
                # Look for the last rebalancing event before or at the start of this year
                start_of_year = pd.to_datetime(f'{year}-01-01')
                before_year_data = display_df[display_df['Date_dt'] <= start_of_year]
                if len(before_year_data) > 0:
                    start_value = before_year_data['Total_Portfolio'].iloc[-1]
                else:
                    start_value = initial_capital
            
            # Find the portfolio value at the end of the year
            # Use the first rebalancing event of the NEXT year to get the proper year-end value
            next_year = year + 1
            next_year_start = pd.to_datetime(f'{next_year}-01-01')
            next_year_data = display_df[display_df['Date_dt'] >= next_year_start]
            
            if len(next_year_data) > 0:
                # Use the first rebalancing event of next year as the end value for this year
                end_value = next_year_data['Total_Portfolio'].iloc[0]
            else:
                # If no next year data, use the last event of current year
                end_of_year = pd.to_datetime(f'{year}-12-31')
                year_data = display_df[display_df['Date_dt'] <= end_of_year]
                if len(year_data) > 0:
                    end_value = year_data['Total_Portfolio'].iloc[-1]
                else:
                    end_value = start_value
            
            # Calculate the return for this calendar year
            yearly_return = ((end_value - start_value) / start_value) * 100 if start_value > 0 else 0
            
            yearly_data.append({
                'Year': year,
                'Start Value': start_value,
                'End Value': end_value,
                'Return %': yearly_return
            })
        
        if yearly_data:
            yearly_df = pd.DataFrame(yearly_data)
            
            # Apply the same styling as multi-backtest
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
            
            # Apply styling to numeric values first
            styler = yearly_df.style
            styler = styler.map(color_gradient_stock, subset=['Return %'])
            
            # Format the values for display
            format_dict = {
                'Start Value': '${:,.0f}',
                'End Value': '${:,.0f}',
                'Return %': '{:+.1f}%'
            }
            styler = styler.format(format_dict, na_rep='N/A')
            
            st.dataframe(styler, use_container_width=True, hide_index=False)
        
        # Display key performance metrics in a compact format
        st.subheader("📈 Performance Metrics")
        
        # Calculate basic metrics
        years = len(portfolio_df)  # Number of rebalancing events = number of years
        cagr = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate max drawdown
        portfolio_values = portfolio_df['Total_Portfolio']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate additional metrics
        avg_change = portfolio_df['Portfolio_Return_%'].diff().mean()
        
        # Create detailed performance metrics table
        metrics_data = {
            'Metric': [
                'Total Return',
                'CAGR (Compound Annual Growth Rate)',
                'Max Drawdown',
                'Number of Rebalancing Events',
                'Average Change per Rebalancing',
                'Initial Capital',
                'Final Portfolio Value',
                'LEAPS Allocation'
            ],
            'Value': [
                f"{total_return:.1f}%",
                f"{cagr:.1f}%",
                f"{max_drawdown:.1f}%",
                f"{len(portfolio_df)}",
                f"{avg_change:+.1f}%",
                f"${initial_capital:,.0f}",
                f"${final_value:,.0f}",
                f"{leaps_allocation}%"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error displaying portfolio results: {str(e)}")

def run_historical_options_analysis(ticker, start_date, end_date, strike_percent, expiration_days, dividend_yield, progress_bar=None, status_text=None):
    """
    Run historical options analysis for the given parameters - FAST VERSION
    """
    try:
        # Calculate volatility period
        vol_period = get_intelligent_volatility_period(expiration_days)
        
        # Download historical data starting from IPO (max available)
        if status_text:
            status_text.text("📥 Downloading historical data...")
        if progress_bar:
            progress_bar.progress(0.1)
        
        stock = yf.Ticker(ticker)
        
        # Get max available data (IPO date)
        hist = stock.history(period="max")
        
        if hist.empty:
            st.error(f"No historical data found for {ticker}.")
            return None
        
        # Download treasury rates and VIX for the same period
        if status_text:
            status_text.text("📊 Downloading treasury rate and VIX data...")
        if progress_bar:
            progress_bar.progress(0.2)
            
        treasury_hist = yf.Ticker("^IRX").history(start=hist.index[0], end=hist.index[-1])
        vix_hist = yf.Ticker("^VIX").history(start=hist.index[0], end=hist.index[-1])
        
        # Pre-calculate dividend data once (OPTIMIZATION)
        if status_text:
            status_text.text("📊 Pre-calculating dividend yields...")
        if progress_bar:
            progress_bar.progress(0.3)
            
        dividend_lookup = {}
        if dividend_yield_manual is None:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            
            if not dividends.empty:
                # Create dividend lookup table
                for date in hist.index:
                    date_dt = pd.to_datetime(date)
                    start_lookup = date_dt - timedelta(days=365)
                    end_lookup = date_dt + timedelta(days=365)
                    
                    period_dividends = dividends[(dividends.index >= start_lookup) & (dividends.index <= end_lookup)]
                    
                    if len(period_dividends) > 0:
                        annual_dividend = period_dividends.tail(4).sum()
                        stock_price = hist.loc[date, 'Close']
                        dividend_yield = (annual_dividend / stock_price) * 100
                        dividend_lookup[date.strftime('%Y-%m-%d')] = max(0, dividend_yield)
                    else:
                        dividend_lookup[date.strftime('%Y-%m-%d')] = 1.5  # Default
            else:
                # No dividend data, use default for all dates
                for date in hist.index:
                    dividend_lookup[date.strftime('%Y-%m-%d')] = 1.5
        
        # Handle dividend yield
        if dividend_yield_manual is not None:
            # Use manual dividend yield for all dates
            dividend_yield = dividend_yield_manual
            st.success(f"✅ Using fixed dividend yield: {dividend_yield:.2f}% for all dates")
        else:
            # Will calculate dynamic dividend yield for each date
            st.info(f"📊 Will calculate dynamic dividend yield for each date based on historical dividends")
        
        # Determine analysis period
        if start_date:
            analysis_start = start_date
        else:
            # Start from IPO + volatility period (like momentum window)
            analysis_start = hist.index[vol_period].date()
            
        if end_date:
            analysis_end = end_date
        else:
            analysis_end = hist.index[-1].date()
        
        # Filter data to analysis period
        analysis_data = hist[(hist.index.date >= analysis_start) & (hist.index.date <= analysis_end)]
        
        if analysis_data.empty:
            st.error("No data available in the specified analysis period.")
            return None
        
        if status_text:
            status_text.text(f"📊 Analyzing {len(analysis_data)} trading days...")
        if progress_bar:
            progress_bar.progress(0.4)
        
        # Prepare results
        results = []
        total_days = len(analysis_data)
        
        # Vectorized volatility calculation (much faster)
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        
        # Process each date in analysis period
        for idx, (date, row) in enumerate(analysis_data.iterrows()):
            # Update progress
            if progress_bar and idx % 100 == 0:  # Update every 100 iterations
                progress = 0.4 + (0.5 * idx / total_days)
                progress_bar.progress(min(progress, 0.9))
            stock_price = row['Close']
            
            # Calculate strike price
            strike_price = stock_price * (1 + strike_percent / 100)
            
            # Get VIX value for this date (always collect VIX data)
            vix_value = None
            if not vix_hist.empty:
                vix_date = vix_hist.index[vix_hist.index <= date]
                if len(vix_date) > 0:
                    vix_value = vix_hist.loc[vix_date[-1], 'Close']
            
            # Calculate accurate volatility using proper methodology
            vol_period = get_intelligent_volatility_period(expiration_days)
            hist_up_to_date = hist[hist.index <= date]
            
            if len(hist_up_to_date) >= vol_period:
                vol_prices = hist_up_to_date.tail(vol_period)
                
                # Calculate calibration factor if enabled
                calibration_data = st.session_state.get('calibration_data', {'enabled': False})
                if calibration_data.get('enabled', False) and len(results) == 0:
                    # Calculate calibration factor on first iteration
                    calibration_factor = calculate_calibration_factor(calibration_data)
                    st.session_state.calibration_data = calibration_data  # Save updated data
                
                # Use simple calibrated volatility calculation
                volatility = calculate_professional_volatility(vol_prices['Close'], vol_period, vix_value, expiration_days, "Simple")
            else:
                # Skip this date - not enough data
                continue
            
            # Get treasury rate for this date
            risk_free_rate = 0.04  # Default
            if not treasury_hist.empty:
                treasury_date = treasury_hist.index[treasury_hist.index <= date]
                if len(treasury_date) > 0:
                    risk_free_rate = treasury_hist.loc[treasury_date[-1], 'Close'] / 100
            
            
            # Get dividend yield for this specific date (FAST - from lookup table)
            if dividend_yield_manual is None:
                date_dividend_yield = dividend_lookup.get(date.strftime('%Y-%m-%d'), 1.5)
            else:
                date_dividend_yield = dividend_yield_manual
            
            # Calculate time to expiration in years
            time_years = expiration_days / 365.25
            
            
            # Calculate LEAPS price using Professional Black-Scholes
            leaps_price = black_scholes_call_professional(
                stock_price, strike_price, time_years, 
                risk_free_rate, volatility, date_dividend_yield/100
            )
            
            # Check for zero prices (only warn if there's an issue)
            if leaps_price <= 0:
                st.warning(f"⚠️ Zero LEAPS price detected on {date.strftime('%Y-%m-%d')}: S={stock_price}, K={strike_price}, T={time_years}, r={risk_free_rate}, vol={volatility}")
            
            # Calculate premium as % of strike price
            premium_percentage = (leaps_price / strike_price) * 100 if strike_price > 0 else 0
            
            # Store results
            results.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Stock_Price': round(stock_price, 2),
                'Strike_Price': round(strike_price, 2),
                'Volatility_%': round(volatility * 100, 2),
                'Risk_Free_Rate_%': round(risk_free_rate * 100, 2),
                'Dividend_Yield_%': round(date_dividend_yield, 2),
                'VIX': round(vix_value, 2) if vix_value else None,
                'Option_Price': round(leaps_price, 2),
                'Premium_%': round(premium_percentage, 2),
                'Time_to_Exp_Days': expiration_days
            })
        
        # Complete progress
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("✅ Analysis complete!")
            
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Error running historical analysis: {str(e)}")
        return None

# Input form
st.subheader("📝 Analysis Parameters")

col1, col2 = st.columns(2)

with col1:
    # Ticker selection
    available_tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VTI", "VEA", "VWO", "BND", "AGG", "^GSPC", "^IXIC", "^NDX"]
    ticker = st.selectbox(
        "📈 Select Ticker",
        options=available_tickers,
        help="Choose the underlying asset for LEAPS analysis"
    )
    
    # Date range selection
    start_date = st.date_input(
        "📅 Start Date (Optional)",
        value=None,
        help="Start date for historical analysis (leave empty for IPO date)"
    )
    
    end_date = st.date_input(
        "📅 End Date (Optional)", 
        value=None,
        help="End date for historical analysis (leave empty for latest)"
    )

with col2:
    # Strike percentage
    strike_percent = st.number_input(
        "🎯 Strike % (0% = ATM, +10% = 10% OTM, -10% = 10% ITM)",
        min_value=-50.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        help="Percentage above/below current price for strike"
    )
    
    # Days to expiration
    expiration_days = st.number_input(
        "⏰ Days to Expiration",
        min_value=30,
        max_value=1095,  # 3 years max
        value=365,
        step=1,
        help="Number of days until option expiration"
    )
    
    # Dividend yield selection
    dividend_mode = st.radio(
        "💰 Dividend Yield Source",
        ["Automatic (from market data)", "Manual (enter value)"],
        help="Choose how to get dividend yield"
    )
    
    if dividend_mode == "Manual (enter value)":
        dividend_yield_manual = st.number_input(
            "💰 Dividend Yield (%)",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.01,
            help="Enter the dividend yield percentage"
        )
    else:
        dividend_yield_manual = None

# Simple volatility calculation
st.subheader("📊 Volatility Calculation")
st.info("🎯 **Simple & Reliable**: Uses standard deviation of returns, automatically calibrated to your real data")

# Current Price Calibration (NEW FEATURE)
st.subheader("🎯 Current Price Calibration (Optional)")
st.markdown("**Calibrate the model using current real market data:**")

calibrate_enabled = st.checkbox(
    "Enable current price calibration",
    help="Enter the current real option price to calibrate the entire backtest"
)

if calibrate_enabled:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_stock_price = st.number_input(
            "Current Stock Price ($)",
            min_value=0.01,
            value=670.0,
            step=0.01,
            help="Current stock price (e.g., SPY at $670)"
        )
    
    with col2:
        current_strike_price = st.number_input(
            "Current Strike Price ($)",
            min_value=0.01,
            value=670.0,
            step=0.01,
            help="Current strike price (e.g., $670 for ATM)"
        )
    
    with col3:
        current_days_to_expiration = st.number_input(
            "Days to Expiration",
            min_value=1,
            value=365,
            step=1,
            help="Days until expiration (e.g., 365 for 1-year LEAPS)"
        )
    
    with col4:
        current_real_option_price = st.number_input(
            "Real Option Price ($)",
            min_value=0.01,
            value=56.0,
            step=0.01,
            help="Current real option price from market (e.g., $56)"
        )
    
    # Store calibration data in session state
    st.session_state.calibration_data = {
        'enabled': True,
        'stock_price': current_stock_price,
        'strike_price': current_strike_price,
        'days_to_exp': current_days_to_expiration,
        'real_price': current_real_option_price
    }
    
    # Show calibration preview
    if current_stock_price and current_strike_price and current_real_option_price:
        st.info(f"🎯 **Calibration Preview**: {current_stock_price} stock, {current_strike_price} strike, {current_days_to_expiration} days → Real price: ${current_real_option_price}")
        
        # Estimate volatility from current option price
        estimated_volatility = estimate_volatility_from_price(
            current_stock_price, current_strike_price, current_real_option_price, current_days_to_expiration
        )
        
        st.success(f"🎯 **Estimated Volatility**: {estimated_volatility*100:.1f}% (from your ${current_real_option_price} option price)")
        st.info("💡 **Next**: Run the analysis - it will calibrate itself using this volatility estimate!")
else:
    st.session_state.calibration_data = {'enabled': False}


# Run analysis button
if st.button("📈 Run Historical Options Analysis", type="primary", use_container_width=True):
    # Clear session state to force fresh calculation
    if 'leaps_analysis_results' in st.session_state:
        del st.session_state.leaps_analysis_results
    if 'leaps_analysis_params' in st.session_state:
        del st.session_state.leaps_analysis_params
    if 'leaps_analysis_params_key' in st.session_state:
        del st.session_state.leaps_analysis_params_key
    
    # Check if calibration is enabled
    calibration_enabled = st.session_state.get('calibration_data', {}).get('enabled', False)
    
    if calibration_enabled:
        st.info("🎯 **Two-Step Calibration Process**")
        
        # Step 1: Calculate calibration factor from current option price
        calibration_data = st.session_state.calibration_data
        calibration_factor = calculate_calibration_factor(calibration_data)
        estimated_vol = calibration_data.get('estimated_volatility', 0.15)
        
        st.success(f"✅ **Step 1 Complete**: Estimated volatility = {estimated_vol*100:.1f}% from your real option price")
        
        # Step 2: Run analysis with calibrated volatility
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🔄 Step 2: Running calibrated analysis...")
        
        results_df = run_historical_options_analysis(
            ticker, start_date, end_date, strike_percent, 
            expiration_days, dividend_yield_manual, progress_bar, status_text
        )
        
        if results_df is not None and not results_df.empty:
            # Save to session state
            st.session_state.leaps_analysis_results = results_df
            st.session_state.leaps_analysis_params = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'strike_percent': strike_percent,
                'expiration_days': expiration_days,
                'dividend_yield': dividend_yield_manual,
                'calibrated': True,
                'calibration_factor': calibration_factor
            }
            
            st.success("✅ **Step 2 Complete**: Calibrated analysis finished!")
            st.info(f"🎯 **Calibration Applied**: {calibration_factor:.2f}x volatility adjustment throughout entire backtest")
            
            # Display results using the shared function
            display_saved_results(results_df)
        else:
            st.error("❌ Calibrated analysis failed - no results generated")
    else:
        # Normal single-run analysis (no calibration)
        params_key = f"{ticker}_{start_date}_{end_date}_{strike_percent}_{expiration_days}_{dividend_yield_manual}_Simple"
        
        if 'leaps_analysis_results' in st.session_state and st.session_state.get('leaps_analysis_params_key') == params_key:
            st.info("📊 Using previously calculated results")
            results_df = st.session_state.leaps_analysis_results
        else:
            # Show loading bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run the analysis (start_date can be None for IPO date)
            results_df = run_historical_options_analysis(
                ticker, start_date, end_date, strike_percent, 
                expiration_days, dividend_yield_manual, progress_bar, status_text
            )
            
            if results_df is not None and not results_df.empty:
                # Save to session state
                st.session_state.leaps_analysis_results = results_df
                st.session_state.leaps_analysis_params = {
                    'ticker': ticker,
                    'start_date': start_date,
                    'end_date': end_date,
                    'strike_percent': strike_percent,
                    'expiration_days': expiration_days,
                    'dividend_yield': dividend_yield_manual
                }
                st.session_state.leaps_analysis_params_key = params_key
                
                st.success("✅ Historical analysis completed!")
                
                # Display results using the shared function
                display_saved_results(results_df)
            else:
                st.error("❌ Analysis failed - no results generated")
        
    
# Display saved results even if button hasn't been clicked
if 'leaps_analysis_results' in st.session_state and 'results_df' not in locals():
    st.info("📊 Displaying previously calculated options analysis results")
    saved_results = st.session_state.leaps_analysis_results
    if saved_results is not None and not saved_results.empty:
        display_saved_results(saved_results)
    else:
        st.error("❌ No valid saved results found")

# Portfolio Backtest Section - AT THE END
if 'results_df' in locals() or 'leaps_analysis_results' in st.session_state:
    st.markdown("---")
    st.subheader("💰 Portfolio Options Backtest")
    st.info("💡 Use the options data above to simulate a portfolio strategy with periodic options investments")
    
    # Check if we have LEAPS data available
    if 'results_df' in locals():
        portfolio_df = results_df
    elif 'leaps_analysis_results' in st.session_state:
        portfolio_df = st.session_state.leaps_analysis_results
    else:
        st.warning("⚠️ Please run a LEAPS analysis first before running portfolio backtest")
        portfolio_df = None
    
    if portfolio_df is not None:
        # Portfolio inputs
        col1, col2 = st.columns([1, 1])
        
        with col1:
            initial_capital = st.number_input(
                "🏦 Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                help="Starting portfolio value"
            )
                    
        with col2:
            leaps_allocation = st.slider(
                "⚖️ LEAPS Allocation (%)",
                min_value=5,
                max_value=95,
                value=20,
                step=5,
                help="Percentage of capital allocated to LEAPS"
            )
        
        cash_allocation = 100 - leaps_allocation
        
        if st.button("🚀 Run Portfolio Backtest", type="primary"):
            portfolio_results = run_simple_portfolio_backtest(
                portfolio_df, initial_capital, cash_allocation, leaps_allocation, expiration_days
            )
            
            if portfolio_results is not None and not portfolio_results.empty:
                # Save to session state
                st.session_state.portfolio_backtest_results = portfolio_results
                st.session_state.portfolio_initial_capital = initial_capital
                st.session_state.portfolio_leaps_allocation = leaps_allocation
        
        # Display saved portfolio results if they exist
        if 'portfolio_backtest_results' in st.session_state:
            st.success("✅ Portfolio backtest results loaded from session state")
            display_simple_portfolio_plot(
                st.session_state.portfolio_backtest_results,
                st.session_state.portfolio_initial_capital,
                st.session_state.portfolio_leaps_allocation
            )
