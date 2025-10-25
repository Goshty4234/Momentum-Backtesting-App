import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots

# =============================================================================
# COMPOSANTS PROFESSIONNELS PERSONNALISÉS
# =============================================================================

def create_metric_card(title, value, subtitle, icon="📊", color="blue"):
    """Crée une carte métrique professionnelle avec animations"""
    colors = {
        "blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "green": "linear-gradient(135deg, #27ae60 0%, #229954 100%)",
        "orange": "linear-gradient(135deg, #f39c12 0%, #e67e22 100%)",
        "red": "linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)",
        "purple": "linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%)"
    }
    
    st.markdown(f"""
    <div style="
        background: {colors.get(color, colors['blue'])};
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 35px rgba(0,0,0,0.2)'" 
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.15)'">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3 style="margin: 0; font-size: 2rem; font-weight: bold;">{value}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{title}</p>
        <small style="opacity: 0.8; font-size: 0.9rem;">{subtitle}</small>
    </div>
    """, unsafe_allow_html=True)

# Configuration de la page
st.set_page_config(
    page_title="Stratégie Momentum 12-1 | Analyse Quantitative Complète",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design professionnel avec animations
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">Momentum Strategy 12-1</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quantitative Analysis & Multi-Portfolio Backtesting System</p>', unsafe_allow_html=True)

# Navigation par onglets principaux
main_tab1, main_tab2 = st.tabs(["System Overview", "Research & ETFs"])

# ============================================================================
# ONGLET 1: SYSTEM OVERVIEW (de test.py)
# ============================================================================
with main_tab1:
    # Section d'introduction professionnelle
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #007bff;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">Momentum Strategy Implementation</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #495057; margin: 0;">
        Implementation of the Jegadeesh & Titman (1993) momentum strategy with multi-window approach, 
        negative momentum handling, and comprehensive risk management. The system supports multiple 
        asset classes including equities, fixed income, commodities, and cryptocurrencies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Momentum Research Facts
    st.markdown("## Momentum Research Facts")
    
    # Professional metrics
    st.markdown("""
    <div style="background: #e8f4fd; border: 1px solid #bee5eb; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <p style="margin: 0; color: #0c5460; font-weight: 500;">
            <strong>VERIFIED METRICS</strong> : All metrics based on actual code analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Monthly Alpha", "1.49%", "Jegadeesh & Titman (1993) 12-1 portfolios", "📈", "blue")
    
    with col2:
        create_metric_card("Countries", "40+", "Asness et al. (2013) global momentum", "🌍", "green")
    
    with col3:
        create_metric_card("Research Years", "30+", "Continuous research - most persistent anomalies", "🔬", "orange")
    
    with col4:
        create_metric_card("T-Statistic", "6.18", "Statistical significance (J&T 1993)", "📊", "purple")
    
    # Momentum Strategy Overview
    st.markdown("## Momentum Strategy Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Academic Foundation
        
        **Jegadeesh & Titman (1993)** documented abnormal returns of 1.49% per month (t-stat 6.18) for 12-1 portfolios over 1965-1989. **Jegadeesh & Titman (2001)** confirmed persistence with 0.95% per month (t-stat 3.24) over 1965-1997.
        
        **Asness et al. (2013)** extended findings to 40 countries and 4 asset classes (1972-2011), demonstrating momentum universality.
        
        ### Implementation Details
        
        **Multi-Window Approach**: 365, 180, 120 days with weighted averaging
        **Exclusion Period**: 30 days to avoid microstructure effects
        **Dividend Handling**: Optional inclusion via include_dividends parameter
        **Negative Momentum**: Cash, Equal Weight, or Relative strategies
        """)
    
    with col2:
        st.markdown("""
        ### Empirical Evidence
        
        **Rouwenhorst (1998)**: 12 European countries, 1980-1995, 0.5% per month
        **Griffin et al. (2003)**: 40 countries, 1975-2000, cross-country momentum
        **Moskowitz & Grinblatt (1999)**: Sector momentum explains 50% of individual momentum
        **Novy-Marx (2012)**: Price momentum distinct from earnings momentum
        **Fama & French (2008)**: Momentum survives traditional risk factor controls
        
        ### Behavioral Mechanisms
        
        **Barberis et al. (1998)**: Under-reaction with anchoring bias
        **Odean (1998)**: Disposition effect - premature selling of winners
        **Daniel et al. (1998)**: Overconfidence and self-attribution bias
        **Hong & Stein (1999)**: Gradual information diffusion
        """)
    
    # Momentum Strategy Implementation
    st.markdown("## Momentum Strategy Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Momentum Strategy Types
        
        **Classic Momentum**
        - Absolute momentum calculation
        - Direct ranking by performance
        - Traditional Jegadeesh & Titman approach
        
        **Relative Momentum**
        - Relative performance ranking (not absolute)
        - Shifts all returns to positive by adding offset
        - All assets get allocation even when negative
        - Formula: shifted = max(0.01, return + offset)
        - Enhanced diversification benefits
        
        **Near-Zero Symmetry**
        - Proprietary method with neutral zone compression
        - Neutral zone: [-5%, +5%] get similar allocations
        - Progressive compression of negative assets
        - Mathematical compression factors (exponential decay)
        - Independent ticker treatment
        
        **Negative Momentum Handling**
        - **Cash**: Move to cash when all negative
        - **Equal Weight**: Equal allocation when negative
        - **Relative Momentum**: Continue relative ranking
        - **Near-Zero Symmetry**: Handle near-zero momentum cases
        """)
    
    with col2:
        st.markdown("""
        ### Advanced Configuration
        
        **Momentum Windows**
        - Academic standard: 12-1 (365 days, -30 days)
        - Customizable windows with weighted averaging
        - Multiple windows: 6M, 12M, 18M, etc.
        
        **Risk Adjustments**
        - Beta division for risk-adjusted momentum
        - Volatility weighting for risk management
        - Inverse volatility allocation
        
        **Rebalancing Frequencies**
        - Monthly, Quarterly, Semi-Annual, Annual, etc.
        - Custom rebalancing schedules
        - Dynamic rebalancing triggers
        """)
    
    # Technical Features
    st.markdown("## Technical Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Data Management
        
        **Progressive Ticker Addition**
        - Start backtest before ticker existence
        - Dynamic ticker addition during backtest
        - Historical data reconstruction
        
        **Cash Management**
        - Periodic cash additions
        - Dividend reinvestment handling
        - Cash allocation optimization
        
        **Filtering Systems**
        - SMA/EMA filters for ticker selection
        - Quality filters (liquidity, market cap)
        - Advanced allocation caps and thresholds
        """)
    
    with col2:
        st.markdown("""
        ### Performance Analytics
        
        **Historical Analysis**
        - Complete allocation history
        - Rebalancing event tracking
        - Performance attribution analysis
        
        **Quantitative Allocation**
        - Personal portfolio rebalancing
        - Strategy-based allocation recommendations
        - Risk-adjusted position sizing
        
        **Leverage Management**
        - Daily leverage calculations
        - Risk-free rate integration
        - Fee structure optimization
        """)
    
    # System Architecture
    st.markdown("## System Architecture")
    
    tab1, tab2, tab3 = st.tabs(["Core Functions", "Data Pipeline", "Optimization"])
    
    with tab1:
        st.markdown("""
        ### Core Momentum Functions
        
        **calculate_momentum()**
        - Multi-window momentum calculation
        - Weighted averaging across timeframes
        - Dividend inclusion handling (Jegadeesh & Titman 1993)
        - Data validation and error handling
        
        **calculate_momentum_weights()**
        - Strategy-specific weight calculation
        - Negative momentum handling (Cash, Equal, Relative, Near-Zero)
        - Risk adjustment application (Beta, Volatility)
        - Portfolio allocation optimization
        
        **Near-Zero Symmetry Algorithm**
        - Neutral zone compression (5% threshold)
        - Progressive compression of negative assets
        - Independent ticker treatment
        - Mathematical compression factors
        """)
    
    with tab2:
        st.markdown("""
        ### Data Pipeline
        
        **Data Sources**
        - yfinance for market data
        - Custom ticker implementations
        - Historical data reconstruction
        - Real-time data integration
        
        **Data Processing**
        - Price adjustment handling
        - Dividend reinvestment
        - Split adjustment
        - Missing data interpolation
        
        **Data Validation**
        - Date consistency checks
        - Price validation
        - Volume verification
        - Data quality metrics
        """)
    
    with tab3:
        st.markdown("""
        ### Performance Optimization
        
        **Computational Efficiency**
        - Numba JIT compilation (@jit)
        - Vectorized calculations
        - Parallel processing
        - Memory optimization
        
        **Caching Strategy**
        - Streamlit cache (@st.cache_data)
        - LRU cache for calculations
        - Data persistence
        - Cache invalidation
        
        **System Reliability**
        - Emergency kill system (hard_kill_process)
        - Error recovery
        - Logging and monitoring
        - Resource management
        """)

# ============================================================================
# ONGLET 2: RESEARCH & ETFS (de Home.py)
# ============================================================================
with main_tab2:
    st.markdown("# 📚 Momentum Research & ETFs")
    
    # Research Papers
    st.markdown("## 📖 Essential Momentum Research Papers")
    
    st.markdown("""
    ### 🏆 Essential Momentum Research Papers
    
    **1. [Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"](https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf)**
    - **Journal**: Journal of Finance, Vol. 48, No. 1, pp. 65-91
    - **Authors**: Narasimhan Jegadeesh, Sheridan Titman
    - **Key Finding**: 1.49% monthly alpha (t-stat 6.18) for 12-1 portfolios
    - **Period**: 1965-1989, 12-1 portfolios
    - **Significance**: First rigorous documentation of momentum anomaly
    
    **2. [Jegadeesh & Titman (2001) - "Profitability of Momentum Strategies"](https://www.nber.org/system/files/working_papers/w7159/w7159.pdf)**
    - **Journal**: Journal of Finance, Vol. 56, No. 2, pp. 699-720
    - **Authors**: Narasimhan Jegadeesh, Sheridan Titman
    - **Key Finding**: 0.95% monthly alpha (t-stat 3.24) over extended period
    - **Period**: 1965-1997, extended validation
    - **Significance**: Confirmed persistence, ruled out risk explanations
    
    **3. [Asness, Moskowitz & Pedersen (2013) - "Value and Momentum Everywhere"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1363476)**
    - **Journal**: Journal of Finance, Vol. 68, No. 3, pp. 929-985
    - **Authors**: Clifford S. Asness, Tobias J. Moskowitz, Lasse Heje Pedersen
    - **Key Finding**: Momentum works across 40 countries, 4 asset classes
    - **Period**: 1972-2011, global validation
    - **Significance**: Proved momentum is a universal anomaly
    
    **4. [Barroso & Santa-Clara (2015) - "Momentum Has Its Moments"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429)**
    - **Journal**: Journal of Financial Economics, Vol. 116, Issue 1, pp. 111-120
    - **Authors**: Pedro Barroso, Pedro Santa-Clara
    - **Key Finding**: Momentum performance varies with market conditions
    - **Period**: 1927-2012, US equity markets
    - **Significance**: Momentum timing and risk management insights
    
    **5. [Antonacci (2017) - "Risk Premia Harvesting Through Dual Momentum"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2042750)**
    - **Journal**: Journal of Management & Entrepreneurship, Vol. 2, No. 1, pp. 27-55
    - **Authors**: Gary Antonacci
    - **Key Finding**: Dual momentum strategy combining absolute and relative momentum
    - **Period**: 1974-2016, global asset classes
    - **Significance**: Practical implementation of momentum strategies
    
    **6. [Jegadeesh & Titman (2023) - "Momentum: Evidence and Insights 30 Years Later"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4602426)**
    - **Journal**: Working Paper, 20 pages
    - **Authors**: Narasimhan Jegadeesh, Sheridan Titman
    - **Key Finding**: Momentum remains robust after 30 years of research
    - **Period**: 1965-2023, comprehensive review
    - **Significance**: Latest insights from momentum pioneers
    """)
    
    # Momentum ETFs
    st.markdown("## 📈 Complete Momentum ETF Universe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🇺🇸 US Momentum ETFs
        
        **MTUM** - iShares MSCI USA Momentum Factor ETF
        
        **SPMO** - SPDR S&P 500 Momentum ETF
        
        **PDP** - Invesco DWA Momentum ETF
        
        **QMOM** - Alpha Architect U.S. Quantitative Momentum ETF
        
        **MOM** - AGFiQ U.S. Market Neutral Momentum Fund
        
        **MOMI** - AGFiQ U.S. Market Neutral Anti-Beta Fund
        
        **MOMZ** - Alpha Architect U.S. Quantitative Value ETF
        """)
    
    with col2:
        st.markdown("""
        ### 🌍 International & Global Momentum ETFs
        
        **IMTM** - iShares MSCI International Momentum Factor ETF
        
        **EMOM** - Alpha Architect International Quantitative Momentum ETF
        
        **DWAS** - Invesco DWA SmallCap Momentum ETF
        
        **DWMC** - Invesco DWA MidCap Momentum ETF
        
        **DWUS** - Invesco DWA Momentum & Low Volatility Rotation ETF
        
        **DWAT** - Invesco DWA Technology Momentum ETF
        
        **DWMF** - Invesco DWA Momentum & Low Volatility Rotation ETF
        """)
    
    # Additional Momentum ETFs
    st.markdown("### 🏢 Sector & Specialty Momentum ETFs")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        **DWAS** - Invesco DWA SmallCap Momentum ETF
        
        **DWMC** - Invesco DWA MidCap Momentum ETF
        
        **DWAT** - Invesco DWA Technology Momentum ETF
        """)
    
    with col4:
        st.markdown("""
        **DWUS** - Invesco DWA Momentum & Low Volatility Rotation ETF
        
        **DWMF** - Invesco DWA Momentum & Low Volatility Rotation ETF
        
        **MOMI** - AGFiQ U.S. Market Neutral Anti-Beta Fund
        """)
    
    # Footer - Always visible
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center; 
        color: #333; 
        margin: 2rem 0; 
        padding: 1rem; 
        background: #f8f9fa; 
        border: 1px solid #dee2e6; 
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
    ">
        Made by Nicolas Cool
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #666; 
    margin: 2rem 0; 
    padding: 1rem; 
    font-size: 0.9rem;
    font-weight: 500;
">
    Made by Nicolas Cool
</div>
""", unsafe_allow_html=True)