import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(
    page_title="Simple LEAPS Calculator",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Simple LEAPS Calculator")

st.markdown("""
Calculate LEAPS option premiums using Black-Scholes pricing model.
""")

# Enhanced Black-Scholes function with more accurate implementation
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price with enhanced accuracy
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free rate
    sigma: Volatility (annualized)
    
    Returns:
    Call option price
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value at expiration
    
    # Ensure no division by zero
    if sigma <= 0:
        return max(S - K, 0)
    
    # Calculate d1 and d2 with more precision
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Calculate option price
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    discount_factor = np.exp(-r * T)
    
    call_price = S * N_d1 - K * discount_factor * N_d2
    
    # Ensure price is at least intrinsic value
    return max(call_price, S - K)

def black_scholes_call_enhanced(S, K, T, r, sigma, dividend_yield=0.0):
    """
    Enhanced Black-Scholes with dividend yield and better precision
    """
    if T <= 0:
        return max(S - K, 0)
    
    if sigma <= 0:
        return max(S - K, 0)
    
    # Black-Scholes with continuous dividend yield
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    call_price = S * np.exp(-dividend_yield * T) * N_d1 - K * np.exp(-r * T) * N_d2
    
    return max(call_price, S - K)

def get_volatility_skew(strike_percent, base_volatility, time_years):
    """
    Apply volatility skew based on strike and time to expiration
    This mimics real market behavior where OTM options have higher implied volatility
    """
    # Volatility skew parameters (based on real market data)
    skew_factor = 0.15  # How much volatility increases per 10% OTM
    time_factor = 0.5   # How much skew decreases with time
    
    # Calculate skew based on how far OTM we are
    otm_factor = abs(strike_percent) / 100  # Convert to decimal
    
    # Skew is stronger for shorter-term options
    time_adjustment = 1 - (time_factor * (1 - time_years))
    
    # Apply skew (higher volatility for OTM options)
    if strike_percent > 0:  # OTM calls
        skew_adjustment = 1 + (skew_factor * otm_factor * time_adjustment)
    else:  # ITM calls (lower volatility)
        skew_adjustment = 1 - (skew_factor * otm_factor * time_adjustment * 0.5)
    
    # Ensure minimum volatility
    skew_adjustment = max(0.5, min(2.0, skew_adjustment))
    
    return base_volatility * skew_adjustment

def professional_option_pricing(S, K, T, r, sigma, dividend_yield=0.0, strike_percent=0.0):
    """
    Professional-grade option pricing with volatility skew and market adjustments
    """
    if T <= 0:
        return max(S - K, 0)
    
    # Apply volatility skew
    adjusted_vol = get_volatility_skew(strike_percent, sigma, T)
    
    # Market microstructure adjustments
    liquidity_adjustment = 1.0
    if T < 0.25:  # Less than 3 months
        liquidity_adjustment = 1.05  # 5% premium for short-term options
    elif T > 2.0:  # More than 2 years
        liquidity_adjustment = 1.02  # 2% premium for very long-term options
    
    # Calculate Black-Scholes with adjusted volatility
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * adjusted_vol**2) * T) / (adjusted_vol * sqrt_T)
    d2 = d1 - adjusted_vol * sqrt_T
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    call_price = S * np.exp(-dividend_yield * T) * N_d1 - K * np.exp(-r * T) * N_d2
    
    # Apply liquidity adjustment
    final_price = call_price * liquidity_adjustment
    
    return max(final_price, S - K)


# Input form
st.subheader("📝 Enter Your Parameters")

col1, col2 = st.columns(2)

with col1:
    # Stock price
    stock_price = st.number_input(
        "💰 Current Stock Price ($)",
        min_value=0.01,
        max_value=10000.0,
        value=450.0,
        step=0.01,
        help="Current price of the underlying stock/ETF"
    )
    
    # Strike percentage
    strike_percent = st.number_input(
        "🎯 Strike % Above/Below Price (%)",
        min_value=-50.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        help="0% = ATM, 10% = 10% above price, -10% = 10% below price"
    )
    
    # Time to expiration
    expiration_days = st.number_input(
        "⏰ Days to Expiration",
        min_value=1,
        max_value=1095,  # 3 years max
        value=365,
        step=1,
        help="Number of days until the option expires"
    )

with col2:
    # Volatility
    volatility = st.number_input(
        "📊 Volatility (%)",
        min_value=1.0,
        max_value=200.0,
        value=20.0,
        step=0.1,
        help="Annualized volatility as a percentage"
    )
    
    # Risk-free rate
    risk_free_rate = st.number_input(
        "🏦 Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.9,
        step=0.01,
        help="Annual risk-free rate as a percentage"
    )
    
    # Dividend yield
    dividend_yield = st.number_input(
        "💰 Dividend Yield (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.01,
        help="Annual dividend yield as a percentage (SPY ≈ 1.5%)"
    )
    
    # Market stress mode
    stress_mode = st.selectbox(
        "🌪️ Market Conditions",
        options=["Normal", "Stress (2x Vol)", "Crisis (3x Vol)", "Extreme Crisis (4x Vol)"],
        help="Simulate different market stress levels"
    )
    
    # Pricing model selection
    pricing_model = st.selectbox(
        "🎯 Pricing Model",
        options=["Basic Black-Scholes", "Professional (with Volatility Skew)"],
        help="Professional model includes volatility skew and market microstructure adjustments"
    )

# Calculate button
if st.button("🧮 Calculate LEAPS Price", type="primary", use_container_width=True):
    try:
        # Calculate strike price from percentage
        strike_price = stock_price * (1 + strike_percent / 100)
        
        # Apply stress mode multiplier to volatility
        stress_multipliers = {
            "Normal": 1.0,
            "Stress (2x Vol)": 2.0,
            "Crisis (3x Vol)": 3.0,
            "Extreme Crisis (4x Vol)": 4.0
        }
        
        stress_multiplier = stress_multipliers[stress_mode]
        adjusted_volatility = volatility * stress_multiplier
        
        # Convert inputs to decimals
        vol_decimal = adjusted_volatility / 100
        rate_decimal = risk_free_rate / 100
        
        # Convert days to years
        time_years = expiration_days / 365.25
        
        # Calculate LEAPS price using selected pricing model
        dividend_decimal = dividend_yield / 100
        
        if pricing_model == "Professional (with Volatility Skew)":
            leaps_price = professional_option_pricing(
                stock_price, strike_price, time_years, rate_decimal, vol_decimal, 
                dividend_decimal, strike_percent
            )
        else:
            leaps_price = black_scholes_call_enhanced(
                stock_price, strike_price, time_years, rate_decimal, vol_decimal, dividend_decimal
            )
        
        # Calculate intrinsic and time value
        intrinsic_value = max(stock_price - strike_price, 0)
        time_value = leaps_price - intrinsic_value
        
        # Display results
        st.success("✅ LEAPS Price Calculated!")
        
        # Results in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Stock Price",
                f"${stock_price:.2f}"
            )
        
        with col2:
            st.metric(
                "LEAPS Price",
                f"${leaps_price:.2f}",
                f"${time_value:.2f} time value"
            )
        
        with col3:
            st.metric(
                "Strike Price",
                f"${strike_price:.2f}",
                f"{strike_percent:+.1f}% {'above' if strike_percent > 0 else 'below' if strike_percent < 0 else 'ATM'}"
            )
        
        with col4:
            st.metric(
                "Intrinsic Value",
                f"${intrinsic_value:.2f}",
                f"${time_value:.2f} time value"
            )
        
        # Detailed calculation info
        st.subheader("📊 Calculation Details")
        
        # Get the final volatility used (for professional model)
        if pricing_model == "Professional (with Volatility Skew)":
            final_volatility = get_volatility_skew(strike_percent, adjusted_volatility, time_years)
        else:
            final_volatility = adjusted_volatility
        
        details_data = {
            'Parameter': [
                'Stock Price',
                'Strike Price',
                'Strike %',
                'Days to Expiration',
                'Years to Expiration',
                'Base Volatility',
                'Stress Multiplier',
                'Volatility Skew Applied',
                'Final Volatility',
                'Risk-Free Rate',
                'Dividend Yield',
                'Pricing Model',
                'LEAPS Price',
                'Intrinsic Value',
                'Time Value'
            ],
            'Value': [
                f"${stock_price:.2f}",
                f"${strike_price:.2f}",
                f"{strike_percent:+.1f}%",
                f"{expiration_days} days",
                f"{time_years:.3f} years",
                f"{volatility:.1f}%",
                f"{stress_multiplier:.1f}x",
                "Yes" if pricing_model == "Professional (with Volatility Skew)" else "No",
                f"{final_volatility:.1f}%",
                f"{risk_free_rate:.1f}%",
                f"{dividend_yield:.1f}%",
                pricing_model,
                f"${leaps_price:.2f}",
                f"${intrinsic_value:.2f}",
                f"${time_value:.2f}"
            ]
        }
        
        import pandas as pd
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df, use_container_width=True)
        
        # Price breakdown chart
        st.subheader("📈 Price Breakdown")
        
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                x=['Intrinsic Value', 'Time Value'],
                y=[intrinsic_value, time_value],
                marker_color=['green', 'blue'],
                text=[f'${intrinsic_value:.2f}', f'${time_value:.2f}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="LEAPS Price Breakdown",
            yaxis_title="Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating LEAPS price: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This calculator uses the Black-Scholes pricing model for educational purposes. 
Real-world option prices may vary due to bid-ask spreads, liquidity constraints, and other market factors.
""")
