import streamlit as st
import datetime
from datetime import timedelta, time
from datetime import timedelta, time, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
@@ -1494,6 +1494,8 @@ def reset_beta_callback():
# Update UI widget values to reflect reset
st.session_state['strategy_comparison_active_beta_window'] = 365
st.session_state['strategy_comparison_active_beta_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def reset_vol_callback():
# Reset volatility lookback/exclude to defaults and enable volatility calculation
@@ -1505,6 +1507,8 @@ def reset_vol_callback():
# Update UI widget values to reflect reset
st.session_state['strategy_comparison_active_vol_window'] = 365
st.session_state['strategy_comparison_active_vol_exclude'] = 30
    # Trigger rerun to update UI
    st.session_state.strategy_comparison_rerun_flag = True

def sync_cashflow_from_first_portfolio_callback():
"""Sync initial value, added amount, and added frequency from first portfolio to all others"""
@@ -1517,21 +1521,38 @@ def sync_cashflow_from_first_portfolio_callback():
added_amount = first_portfolio.get('added_amount', 1000)
added_frequency = first_portfolio.get('added_frequency', 'Monthly')

            # Update all other portfolios
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
for i in range(1, len(st.session_state.strategy_comparison_portfolio_configs)):
                st.session_state.strategy_comparison_portfolio_configs[i]['initial_value'] = initial_value
                st.session_state.strategy_comparison_portfolio_configs[i]['added_amount'] = added_amount
                st.session_state.strategy_comparison_portfolio_configs[i]['added_frequency'] = added_frequency
                portfolio = st.session_state.strategy_comparison_portfolio_configs[i]
                if not portfolio.get('exclude_from_cashflow_sync', False):
                    # Only update if values are actually different
                    if (portfolio.get('initial_value') != initial_value or 
                        portfolio.get('added_amount') != added_amount or 
                        portfolio.get('added_frequency') != added_frequency):
                        portfolio['initial_value'] = initial_value
                        portfolio['added_amount'] = added_amount
                        portfolio['added_frequency'] = added_frequency
                        updated_count += 1

            # Update UI widget session states to reflect the changes
            st.session_state['strategy_comparison_active_initial'] = initial_value
            st.session_state['strategy_comparison_active_added_amount'] = added_amount
            st.session_state['strategy_comparison_active_add_freq'] = added_frequency
            
            st.session_state.strategy_comparison_rerun_flag = True
            st.session_state.strategy_comparison_sync_success = True
    except Exception:
        pass
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from cash flow sync
                current_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
                if not current_portfolio.get('exclude_from_cashflow_sync', False):
                    # Update UI widget session states to reflect the changes
                    st.session_state['strategy_comparison_active_initial'] = initial_value
                    st.session_state['strategy_comparison_active_added_amount'] = added_amount
                    st.session_state['strategy_comparison_active_add_freq'] = added_frequency
                
                # Force immediate rerun to show changes
                st.session_state.strategy_comparison_rerun_flag = True
                st.session_state.strategy_comparison_sync_success = True
                st.rerun()
            else:
                st.info("No portfolios were updated (all were excluded or already had matching values)")
    except Exception as e:
        st.error(f"Error during cash flow sync: {str(e)}")

def sync_rebalancing_from_first_portfolio_callback():
"""Sync rebalancing frequency from first portfolio to all others"""
@@ -1542,17 +1563,32 @@ def sync_rebalancing_from_first_portfolio_callback():
# Get rebalancing frequency from first portfolio
rebalancing_frequency = first_portfolio.get('rebalancing_frequency', 'Monthly')

            # Update all other portfolios
            # Update all other portfolios (skip those excluded from sync)
            updated_count = 0
for i in range(1, len(st.session_state.strategy_comparison_portfolio_configs)):
                st.session_state.strategy_comparison_portfolio_configs[i]['rebalancing_frequency'] = rebalancing_frequency
            
            # Update UI widget session state to reflect the change
            st.session_state['strategy_comparison_active_rebal_freq'] = rebalancing_frequency
                portfolio = st.session_state.strategy_comparison_portfolio_configs[i]
                if not portfolio.get('exclude_from_rebalancing_sync', False):
                    # Only update if value is actually different
                    if portfolio.get('rebalancing_frequency') != rebalancing_frequency:
                        portfolio['rebalancing_frequency'] = rebalancing_frequency
                        updated_count += 1

            st.session_state.strategy_comparison_rerun_flag = True
            st.session_state.strategy_comparison_sync_success = True
    except Exception:
        pass
            # Only update UI and rerun if something actually changed
            if updated_count > 0:
                # Only update UI widgets if the current portfolio is NOT excluded from rebalancing sync
                current_portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
                if not current_portfolio.get('exclude_from_rebalancing_sync', False):
                    # Update UI widget session state to reflect the change
                    st.session_state['strategy_comparison_active_rebal_freq'] = rebalancing_frequency
                
                # Force immediate rerun to show changes
                st.session_state.strategy_comparison_rerun_flag = True
                st.session_state.strategy_comparison_sync_success = True
                st.rerun()
            else:
                st.info("No portfolios were updated (all were excluded or already had matching values)")
    except Exception as e:
        st.error(f"Error during rebalancing sync: {str(e)}")

def add_momentum_window_callback():
st.session_state.strategy_comparison_add_momentum_window_flag = True
@@ -1584,6 +1620,15 @@ def paste_json_callback():
try:
json_data = json.loads(st.session_state.strategy_comparison_paste_json_text)

        # Clear widget keys to force re-initialization
        widget_keys_to_clear = [
            "strategy_comparison_active_calc_beta", "strategy_comparison_active_beta_window", "strategy_comparison_active_beta_exclude",
            "strategy_comparison_active_calc_vol", "strategy_comparison_active_vol_window", "strategy_comparison_active_vol_exclude"
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        


# Handle momentum strategy value mapping from other pages
@@ -1687,8 +1732,8 @@ def map_frequency(freq):
'added_amount': json_data.get('added_amount', 1000),
'added_frequency': map_frequency(json_data.get('added_frequency', 'Monthly')),
'rebalancing_frequency': map_frequency(json_data.get('rebalancing_frequency', 'Monthly')),
            'start_date_user': json_data.get('start_date_user'),
            'end_date_user': json_data.get('end_date_user'),
            'start_date_user': parse_date_from_json(json_data.get('start_date_user')),
            'end_date_user': parse_date_from_json(json_data.get('end_date_user')),
'start_with': json_data.get('start_with', 'all'),
'first_rebalance_strategy': json_data.get('first_rebalance_strategy', 'rebalancing_date'),
'use_momentum': json_data.get('use_momentum', True),
@@ -1703,9 +1748,23 @@ def map_frequency(freq):
'exclude_days_vol': json_data.get('exclude_days_vol', 30),
'collect_dividends_as_cash': json_data.get('collect_dividends_as_cash', False),
'saved_momentum_settings': json_data.get('saved_momentum_settings', {}),
            # Preserve sync exclusion settings from imported JSON
            'exclude_from_cashflow_sync': json_data.get('exclude_from_cashflow_sync', False),
            'exclude_from_rebalancing_sync': json_data.get('exclude_from_rebalancing_sync', False),
# Note: Ignoring Backtest Engine specific fields like 'portfolio_drag_pct', 'use_custom_dates', etc.
}

        # Fix: ensure proper defaults for beta/volatility windows regardless of calc_beta/calc_volatility state
        if strategy_comparison_config['beta_window_days'] <= 1:
            strategy_comparison_config['beta_window_days'] = 365
        if strategy_comparison_config['exclude_days_beta'] <= 0:
            strategy_comparison_config['exclude_days_beta'] = 30
        if strategy_comparison_config['vol_window_days'] <= 1:
            strategy_comparison_config['vol_window_days'] = 365
        if strategy_comparison_config['exclude_days_vol'] <= 0:
            strategy_comparison_config['exclude_days_vol'] = 30
        
        # Update the configuration with corrected values
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = strategy_comparison_config

# UPDATE UI WIDGET STATES TO REFLECT IMPORTED SETTINGS
@@ -1773,6 +1832,9 @@ def map_frequency(freq):
# DON'T sync global tickers to all portfolios - this would overwrite the imported settings
# Instead, just update the current portfolio's stocks to match global tickers
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['stocks'] = st.session_state.strategy_comparison_global_tickers.copy()
        
        # Sync date widgets with the updated portfolio
        sync_date_widgets_with_portfolio()
except json.JSONDecodeError:
st.error("Invalid JSON format. Please check the text and try again.")
except Exception as e:
@@ -1794,7 +1856,9 @@ def paste_all_json_callback():
"strategy_comparison_active_added_amount", "strategy_comparison_active_rebal_freq",
"strategy_comparison_active_add_freq", "strategy_comparison_active_benchmark",
"strategy_comparison_active_use_momentum", "strategy_comparison_active_collect_dividends_as_cash",
                "strategy_comparison_start_with_radio", "strategy_comparison_first_rebalance_strategy_radio"
                "strategy_comparison_start_with_radio", "strategy_comparison_first_rebalance_strategy_radio",
                "strategy_comparison_active_calc_beta", "strategy_comparison_active_beta_window", "strategy_comparison_active_beta_exclude",
                "strategy_comparison_active_calc_vol", "strategy_comparison_active_vol_window", "strategy_comparison_active_vol_exclude"
]
for key in widget_keys_to_clear:
if key in st.session_state:
@@ -1897,8 +1961,8 @@ def map_frequency(freq):
'added_amount': cfg.get('added_amount', 1000),
'added_frequency': map_frequency(cfg.get('added_frequency', 'Monthly')),
'rebalancing_frequency': map_frequency(cfg.get('rebalancing_frequency', 'Monthly')),
                    'start_date_user': cfg.get('start_date_user'),
                    'end_date_user': cfg.get('end_date_user'),
                                    'start_date_user': parse_date_from_json(cfg.get('start_date_user')),
                'end_date_user': parse_date_from_json(cfg.get('end_date_user')),
'start_with': cfg.get('start_with', 'all'),
'first_rebalance_strategy': cfg.get('first_rebalance_strategy', 'rebalancing_date'),
'use_momentum': cfg.get('use_momentum', True),
@@ -1913,6 +1977,9 @@ def map_frequency(freq):
'exclude_days_vol': cfg.get('exclude_days_vol', 30),
'collect_dividends_as_cash': cfg.get('collect_dividends_as_cash', False),
'saved_momentum_settings': cfg.get('saved_momentum_settings', {}),
                    # Preserve sync exclusion settings from imported JSON
                    'exclude_from_cashflow_sync': cfg.get('exclude_from_cashflow_sync', False),
                    'exclude_from_rebalancing_sync': cfg.get('exclude_from_rebalancing_sync', False),
# Note: Ignoring Backtest Engine specific fields like 'portfolio_drag_pct', 'use_custom_dates', etc.
}
processed_configs.append(strategy_comparison_config)
@@ -1973,6 +2040,12 @@ def map_frequency(freq):
st.session_state.strategy_comparison_portfolio_key_map = {}
st.session_state.strategy_comparison_ran = False
st.success('All portfolio configurations updated from JSON.')
            if processed_configs:
                st.info(f"Sync exclusions for first portfolio - Cash Flow: {processed_configs[0].get('exclude_from_cashflow_sync', False)}, Rebalancing: {processed_configs[0].get('exclude_from_rebalancing_sync', False)}")
            
            # Sync date widgets with the updated portfolio
            sync_date_widgets_with_portfolio()
            
# Force a rerun so widgets rebuild with the new configs
try:
st.experimental_rerun()
@@ -1994,6 +2067,9 @@ def update_active_portfolio_index():
else:
# default to first portfolio if selector is missing or value not found
st.session_state.strategy_comparison_active_portfolio_index = 0 if portfolio_names else None
    
    # Sync date widgets with the new portfolio
    sync_date_widgets_with_portfolio()
st.session_state.strategy_comparison_rerun_flag = True

def update_name():
@@ -2005,6 +2081,56 @@ def update_initial():
def update_added_amount():
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_amount'] = st.session_state.strategy_comparison_active_added_amount

def clear_dates_callback():
    """Clear the date inputs and reset to None"""
    st.session_state.strategy_comparison_start_date = None
    st.session_state.strategy_comparison_end_date = date.today()
    st.session_state.strategy_comparison_use_custom_dates = False
    # Also clear from the portfolio config
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['start_date_user'] = None
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['end_date_user'] = None

def parse_date_from_json(date_value):
    """Parse date from JSON string format back to date object"""
    if date_value is None:
        return None
    if isinstance(date_value, date):
        return date_value
    if isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        except ValueError:
            try:
                # Try parsing as ISO format
                return datetime.fromisoformat(date_value).date()
            except ValueError:
                return None
    return None

def sync_date_widgets_with_portfolio():
    """Sync date widgets with current portfolio configuration"""
    from datetime import date
    if st.session_state.strategy_comparison_active_portfolio_index is not None:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        # Sync start date
        portfolio_start_date = portfolio.get('start_date_user')
        if portfolio_start_date is not None:
            st.session_state["strategy_comparison_start_date"] = portfolio_start_date
        else:
            st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
        
        # Sync end date
        portfolio_end_date = portfolio.get('end_date_user')
        if portfolio_end_date is not None:
            st.session_state["strategy_comparison_end_date"] = portfolio_end_date
        else:
            st.session_state["strategy_comparison_end_date"] = date.today()
        
        # Sync custom dates checkbox
        has_custom_dates = portfolio_start_date is not None or portfolio_end_date is not None
        st.session_state["strategy_comparison_use_custom_dates"] = has_custom_dates

def update_add_freq():
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['added_frequency'] = st.session_state.strategy_comparison_active_add_freq

@@ -2082,7 +2208,59 @@ def update_use_momentum():


def update_calc_beta():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_beta'] = st.session_state.strategy_comparison_active_calc_beta
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_beta']
    new_val = st.session_state.strategy_comparison_active_calc_beta
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling beta - restore saved settings or use defaults
            if 'saved_beta_settings' in portfolio:
                # Restore previously saved beta settings
                saved_settings = portfolio['saved_beta_settings']
                portfolio['beta_window_days'] = saved_settings.get('beta_window_days', 365)
                portfolio['exclude_days_beta'] = saved_settings.get('exclude_days_beta', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_beta_window'] = portfolio['beta_window_days']
                st.session_state['strategy_comparison_active_beta_exclude'] = portfolio['exclude_days_beta']
            else:
                # No saved settings, use defaults
                portfolio['beta_window_days'] = 365
                portfolio['exclude_days_beta'] = 30
                st.session_state['strategy_comparison_active_beta_window'] = 365
                st.session_state['strategy_comparison_active_beta_exclude'] = 30
        else:
            # Disabling beta - save current settings before clearing
            saved_settings = {
                'beta_window_days': portfolio.get('beta_window_days', 365),
                'exclude_days_beta': portfolio.get('exclude_days_beta', 30),
            }
            portfolio['saved_beta_settings'] = saved_settings
        
        portfolio['calc_beta'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_sync_exclusion(sync_type):
    """Update sync exclusion settings when checkboxes change"""
    try:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if sync_type == 'cashflow':
            key = f"strategy_comparison_exclude_cashflow_sync_{st.session_state.strategy_comparison_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_cashflow_sync'] = st.session_state[key]
        elif sync_type == 'rebalancing':
            key = f"strategy_comparison_exclude_rebalancing_sync_{st.session_state.strategy_comparison_active_portfolio_index}"
            if key in st.session_state:
                portfolio['exclude_from_rebalancing_sync'] = st.session_state[key]
        
        # Force immediate update to session state
        st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = portfolio
        st.session_state.strategy_comparison_rerun_flag = True
    except Exception:
        pass

def update_beta_window():
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['beta_window_days'] = st.session_state.strategy_comparison_active_beta_window
@@ -2091,7 +2269,39 @@ def update_beta_exclude():
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['exclude_days_beta'] = st.session_state.strategy_comparison_active_beta_exclude

def update_calc_vol():
    st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_volatility'] = st.session_state.strategy_comparison_active_calc_vol
    current_val = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['calc_volatility']
    new_val = st.session_state.strategy_comparison_active_calc_vol
    
    if current_val != new_val:
        portfolio = st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]
        
        if new_val:
            # Enabling volatility - restore saved settings or use defaults
            if 'saved_vol_settings' in portfolio:
                # Restore previously saved volatility settings
                saved_settings = portfolio['saved_vol_settings']
                portfolio['vol_window_days'] = saved_settings.get('vol_window_days', 365)
                portfolio['exclude_days_vol'] = saved_settings.get('exclude_days_vol', 30)
                
                # Update UI widgets to reflect restored values
                st.session_state['strategy_comparison_active_vol_window'] = portfolio['vol_window_days']
                st.session_state['strategy_comparison_active_vol_exclude'] = portfolio['exclude_days_vol']
            else:
                # No saved settings, use defaults
                portfolio['vol_window_days'] = 365
                portfolio['exclude_days_vol'] = 30
                st.session_state['strategy_comparison_active_vol_window'] = 365
                st.session_state['strategy_comparison_active_vol_exclude'] = 30
        else:
            # Disabling volatility - save current settings before clearing
            saved_settings = {
                'vol_window_days': portfolio.get('vol_window_days', 365),
                'exclude_days_vol': portfolio.get('exclude_days_vol', 30),
            }
            portfolio['saved_vol_settings'] = saved_settings
        
        portfolio['calc_volatility'] = new_val
        st.session_state.strategy_comparison_rerun_flag = True

def update_vol_window():
st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['vol_window_days'] = st.session_state.strategy_comparison_active_vol_window
@@ -2222,6 +2432,79 @@ def update_collect_dividends_as_cash():
if st.button("x", key=f"remove_global_stock_{i}", help="Remove this ticker", on_click=remove_global_stock_callback, args=(stock['ticker'],)):
pass

# Bulk ticker input section
with st.sidebar.expander("📝 Bulk Ticker Input", expanded=False):
    st.markdown("**Enter multiple tickers separated by spaces or commas:**")
    
    # Initialize bulk ticker input in session state
    if 'strategy_comparison_bulk_tickers' not in st.session_state:
        st.session_state.strategy_comparison_bulk_tickers = ""
    
    # Auto-populate bulk ticker input with current tickers
    current_tickers = [stock['ticker'] for stock in st.session_state.strategy_comparison_global_tickers if stock['ticker']]
    if current_tickers:
        current_ticker_string = ' '.join(current_tickers)
        if st.session_state.strategy_comparison_bulk_tickers != current_ticker_string:
            st.session_state.strategy_comparison_bulk_tickers = current_ticker_string
    
    # Text area for bulk ticker input
    bulk_tickers = st.text_area(
        "Tickers (e.g., SPY QQQ GLD TLT or SPY,QQQ,GLD,TLT)",
        value=st.session_state.strategy_comparison_bulk_tickers,
        key="strategy_comparison_bulk_ticker_input",
        height=100,
        help="Enter ticker symbols separated by spaces or commas. Click 'Fill Tickers' to replace tickers (keeps existing allocations)."
    )
    
    if st.button("Fill Tickers", key="strategy_comparison_fill_tickers_btn"):
        if bulk_tickers.strip():
            # Parse tickers (split by comma or space)
            ticker_list = []
            for ticker in bulk_tickers.replace(',', ' ').split():
                ticker = ticker.strip().upper()
                if ticker:
                    ticker_list.append(ticker)
            
            if ticker_list:
                current_stocks = st.session_state.strategy_comparison_global_tickers.copy()
                
                # Replace tickers - new ones get 0% allocation
                new_stocks = []
                
                for i, ticker in enumerate(ticker_list):
                    if i < len(current_stocks):
                        # Use existing allocation if available
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': current_stocks[i]['allocation'],
                            'include_dividends': current_stocks[i]['include_dividends']
                        })
                    else:
                        # New tickers get 0% allocation
                        new_stocks.append({
                            'ticker': ticker,
                            'allocation': 0.0,
                            'include_dividends': True
                        })
                
                # Update the global tickers
                st.session_state.strategy_comparison_global_tickers = new_stocks
                
                # Clear any existing session state keys for individual ticker inputs to force refresh
                for key in list(st.session_state.keys()):
                    if key.startswith("strategy_comparison_global_ticker_") or key.startswith("strategy_comparison_global_alloc_"):
                        del st.session_state[key]
                
                st.success(f"✅ Replaced tickers with: {', '.join(ticker_list)}")
                st.info("💡 **Note:** Existing allocations preserved. Adjust allocations manually if needed.")
                
                # Force immediate rerun
                st.rerun()
            else:
                st.error("❌ No valid tickers found. Please enter ticker symbols separated by spaces or commas.")
        else:
            st.error("❌ Please enter ticker symbols.")

# Validation constants
_TOTAL_TOL = 1.0
_ALLOC_TOL = 1.0
@@ -2288,6 +2571,40 @@ def update_collect_dividends_as_cash():
on_change=lambda: setattr(st.session_state, 'strategy_comparison_first_rebalance_strategy', st.session_state.strategy_comparison_first_rebalance_strategy_radio)
)

# Date range options
st.sidebar.markdown("---")
st.sidebar.subheader("Date Range Options")
use_custom_dates = st.sidebar.checkbox("Use custom date range", key="strategy_comparison_use_custom_dates", help="Enable to set custom start and end dates for the backtest")

if use_custom_dates:
    col_start_date, col_end_date, col_clear_dates = st.sidebar.columns([1, 1, 1])
    with col_start_date:
        # Initialize widget key with session state value
        if "strategy_comparison_start_date" not in st.session_state:
            st.session_state["strategy_comparison_start_date"] = date(2010, 1, 1)
        # Let Streamlit manage the session state automatically
        start_date = st.date_input("Start Date", min_value=date(1900, 1, 1), key="strategy_comparison_start_date")
        # Update portfolio config when date changes
        if start_date != st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index].get('start_date_user'):
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['start_date_user'] = start_date
    
    with col_end_date:
        # Initialize widget key with session state value
        if "strategy_comparison_end_date" not in st.session_state:
            st.session_state["strategy_comparison_end_date"] = date.today()
        # Let Streamlit manage the session state automatically
        end_date = st.date_input("End Date", key="strategy_comparison_end_date")
        # Update portfolio config when date changes
        if end_date != st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index].get('end_date_user'):
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index]['end_date_user'] = end_date
    
    with col_clear_dates:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
        st.button("Clear Dates", on_click=clear_dates_callback)
else:
    st.session_state["strategy_comparison_start_date"] = None
    st.session_state["strategy_comparison_end_date"] = None

# JSON section for all portfolios
st.sidebar.markdown("---")
with st.sidebar.expander('All Portfolios JSON (Export / Import)', expanded=False):
@@ -2302,6 +2619,13 @@ def clean_portfolio_configs_for_export(configs):
# Update global settings from session state
cleaned_config['start_with'] = st.session_state.get('strategy_comparison_start_with', 'all')
cleaned_config['first_rebalance_strategy'] = st.session_state.get('strategy_comparison_first_rebalance_strategy', 'rebalancing_date')
            
            # Convert date objects to strings for JSON serialization
            if cleaned_config.get('start_date_user') is not None:
                cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
            if cleaned_config.get('end_date_user') is not None:
                cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
            
cleaned_configs.append(cleaned_config)
return cleaned_configs

@@ -2381,6 +2705,51 @@ def clean_portfolio_configs_for_export(configs):
st.success("Portfolio settings synchronized successfully!")
st.session_state.strategy_comparison_sync_success = False

# Sync exclusion options (only show if there are multiple portfolios and not for the first portfolio)
if len(st.session_state.strategy_comparison_portfolio_configs) > 1 and st.session_state.strategy_comparison_active_portfolio_index > 0:
    st.markdown("**🔄 Sync Exclusion Options:**")
    col_sync1, col_sync2 = st.columns(2)
    
    with col_sync1:
        # Initialize sync exclusion settings if not present
        if 'exclude_from_cashflow_sync' not in active_portfolio:
            active_portfolio['exclude_from_cashflow_sync'] = False
        if 'exclude_from_rebalancing_sync' not in active_portfolio:
            active_portfolio['exclude_from_rebalancing_sync'] = False
        
        # Rebalancing sync exclusion - use direct portfolio value to avoid caching issues
        exclude_rebalancing = st.checkbox(
            "Exclude from Rebalancing Sync", 
            value=active_portfolio['exclude_from_rebalancing_sync'],
            key=f"strategy_comparison_exclude_rebalancing_sync_{st.session_state.strategy_comparison_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Rebalancing' button",
            on_change=lambda: update_sync_exclusion('rebalancing')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_rebalancing != active_portfolio['exclude_from_rebalancing_sync']:
            active_portfolio['exclude_from_rebalancing_sync'] = exclude_rebalancing
            # Force immediate update to session state
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = active_portfolio
            st.session_state.strategy_comparison_rerun_flag = True
    
    with col_sync2:
        # Cash flow sync exclusion - use direct portfolio value to avoid caching issues
        exclude_cashflow = st.checkbox(
            "Exclude from Cash Flow Sync", 
            value=active_portfolio['exclude_from_cashflow_sync'],
            key=f"strategy_comparison_exclude_cashflow_sync_{st.session_state.strategy_comparison_active_portfolio_index}",
            help="When checked, this portfolio will not be affected by 'Sync ALL Portfolios Cashflow' button",
            on_change=lambda: update_sync_exclusion('cashflow')
        )
        
        # Update portfolio config when checkbox changes
        if exclude_cashflow != active_portfolio['exclude_from_cashflow_sync']:
            active_portfolio['exclude_from_cashflow_sync'] = exclude_cashflow
            # Force immediate update to session state
            st.session_state.strategy_comparison_portfolio_configs[st.session_state.strategy_comparison_active_portfolio_index] = active_portfolio
            st.session_state.strategy_comparison_rerun_flag = True

if "strategy_comparison_active_benchmark" not in st.session_state:
st.session_state["strategy_comparison_active_benchmark"] = active_portfolio['benchmark_ticker']
st.text_input("Benchmark Ticker (default: ^GSPC, used for beta calculation)", key="strategy_comparison_active_benchmark", on_change=update_benchmark)
@@ -2569,6 +2938,13 @@ def create_momentum_weight_callback(index):
# Update global settings from session state
cleaned_config['start_with'] = st.session_state.get('strategy_comparison_start_with', 'all')
cleaned_config['first_rebalance_strategy'] = st.session_state.get('strategy_comparison_first_rebalance_strategy', 'rebalancing_date')
    
    # Convert date objects to strings for JSON serialization
    if cleaned_config.get('start_date_user') is not None:
        cleaned_config['start_date_user'] = cleaned_config['start_date_user'].isoformat() if hasattr(cleaned_config['start_date_user'], 'isoformat') else str(cleaned_config['start_date_user'])
    if cleaned_config.get('end_date_user') is not None:
        cleaned_config['end_date_user'] = cleaned_config['end_date_user'].isoformat() if hasattr(cleaned_config['end_date_user'], 'isoformat') else str(cleaned_config['end_date_user'])
    
config_json = json.dumps(cleaned_config, indent=4)
st.code(config_json, language='json')
# Fixed JSON copy button
@@ -3175,21 +3551,48 @@ def get_no_additions_series(obj):
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for i, metric in enumerate(metric_order):
if metric in df_metrics.columns:
                        y_values = df_metrics[metric].values
                        
                        # Transform values for symmetric log-like display
                        # Positive values: log scale, Negative values: -log(abs(value))
                        transformed_values = []
                        for val in y_values:
                            if pd.isna(val):
                                transformed_values.append(np.nan)
                            elif val > 0:
                                transformed_values.append(np.log10(val + 1))  # +1 to handle 0
                            elif val < 0:
                                transformed_values.append(-np.log10(abs(val) + 1))  # Negative log for negative values
                            else:  # val == 0
                                transformed_values.append(0)
                        
fig_metrics.add_trace(go.Bar(
x=df_metrics.index,
                            y=df_metrics[metric].values,
                            y=transformed_values,
name=metric,
marker_color=colors[i % len(colors)],
                            text=[f"{v:.2f}%" if not pd.isna(v) else 'N/A' for v in df_metrics[metric].values],
                            textposition='auto'
                            text=[f"{v:.2f}%" if not pd.isna(v) else 'N/A' for v in y_values],
                            textposition='auto',
                            showlegend=True
))

fig_metrics.update_layout(
title='Portfolio Variation Summary (percent)',
barmode='group',
template='plotly_dark',
                    yaxis=dict(title='Percent', ticksuffix='%'),
                    legend_title='Metric',
                    yaxis=dict(
                        title='Percent (Log Scale)', 
                        ticksuffix='%',
                        # Linear scale since we transformed the values manually
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10)
                    ),
height=520,
margin=dict(l=60, r=40, t=80, b=120),
)
@@ -3371,8 +3774,16 @@ def clamp_stat(val, stat_type):
if 'total_money_added_map' in snapshot_data and name in snapshot_data['total_money_added_map']:
total_money_added = snapshot_data['total_money_added_map'][name]

                # Calculate total return based on total money contributed
                total_return_contributed = "N/A"
                if isinstance(series_obj, dict) and 'with_additions' in series_obj and len(series_obj['with_additions']) > 0:
                    final_value_with_additions = series_obj['with_additions'].iloc[-1]
                    if isinstance(total_money_added, (int, float)) and total_money_added > 0:
                        total_return_contributed = (final_value_with_additions / total_money_added - 1)  # Return as decimal
                
recomputed_stats[name] = {
"Total Return": clamp_stat(total_return, "Total Return"),
                    "Total Return (Contributed)": clamp_stat(total_return_contributed, "Total Return"),
"CAGR": clamp_stat(cagr, "CAGR"),
"MaxDrawdown": clamp_stat(max_dd, "MaxDrawdown"),
"Volatility": clamp_stat(vol, "Volatility"),
@@ -3398,33 +3809,129 @@ def clamp_stat(val, stat_type):
cols.remove(c)
cols = front + cols
stats_df_display = stats_df_display[cols]
            # Rename and format columns similarly to prior display code
            stats_df_display.rename(columns={'MaxDrawdown': 'Max Drawdown', 'UlcerIndex': 'Ulcer Index'}, inplace=True)
            # Ensure ordering: Beta then MWRR at end, Total Return and Total Money Added at the very end
            # Rename and format columns to be more descriptive
            stats_df_display.rename(columns={
                'MaxDrawdown': 'Max Drawdown', 
                'UlcerIndex': 'Ulcer Index',
                'Final Value (with)': 'Final Portfolio Value',
                'Final Value (no_additions)': 'Final Value (No Contributions)',
                'Total Return (Contributed)': 'Total Return (All Money)'
            }, inplace=True)
            # Ensure ordering: Beta then MWRR at end, Total Return columns and Total Money Added at the very end
cols = list(stats_df_display.columns)
            if 'Beta' in cols and 'MWRR' in cols and 'Total Return' in cols and 'Total Money Added' in cols:
                cols.remove('Beta'); cols.remove('MWRR'); cols.remove('Total Return'); cols.remove('Total Money Added')
                cols.extend(['Beta','MWRR','Total Return','Total Money Added'])
            if 'Beta' in cols and 'MWRR' in cols and 'Total Return' in cols and 'Total Return (All Money)' in cols and 'Total Money Added' in cols:
                cols.remove('Beta'); cols.remove('MWRR'); cols.remove('Total Return'); cols.remove('Total Return (All Money)'); cols.remove('Total Money Added')
                cols.extend(['Beta','MWRR','Total Return','Total Return (All Money)','Total Money Added'])
stats_df_display = stats_df_display[cols]

            st.subheader("Final Performance Statistics")
            # Display start and end dates next to the title
            col_title, col_dates = st.columns([2, 1])
            with col_title:
                st.subheader("Final Performance Statistics")
            with col_dates:
                if 'strategy_comparison_all_results' in st.session_state and st.session_state.strategy_comparison_all_results:
                    # Get the first portfolio's dates (they should all be the same)
                    first_portfolio = next(iter(st.session_state.strategy_comparison_all_results.values()))
                    if isinstance(first_portfolio, dict) and 'no_additions' in first_portfolio:
                        series = first_portfolio['no_additions']
                        if hasattr(series, 'index') and len(series.index) > 0:
                            start_date = series.index[0].strftime('%Y-%m-%d')
                            end_date = series.index[-1].strftime('%Y-%m-%d')
                            st.markdown(f"**📅 Period:** {start_date} to {end_date}")
                        else:
                            st.markdown("**📅 Period:** N/A")
                    else:
                        st.markdown("**📅 Period:** N/A")
                else:
                    st.markdown("**📅 Period:** N/A")
# Format currency for final value columns if present
fmt_map_display = {}
            if fv_with in stats_df_display.columns:
                fmt_map_display[fv_with] = '${:,.2f}'
            if fv_no in stats_df_display.columns:
                fmt_map_display[fv_no] = '${:,.2f}'
            if 'Final Portfolio Value' in stats_df_display.columns:
                fmt_map_display['Final Portfolio Value'] = '${:,.2f}'
            if 'Final Value (No Contributions)' in stats_df_display.columns:
                fmt_map_display['Final Value (No Contributions)'] = '${:,.2f}'
if 'Total Money Added' in stats_df_display.columns:
fmt_map_display['Total Money Added'] = '${:,.2f}'
            # Format MWRR as percentage
            if 'MWRR' in stats_df_display.columns:
                fmt_map_display['MWRR'] = '{:.2f}%'
            
            # Create tooltips for each column
            tooltip_data = {
                'Total Return': 'Return based on initial investment only. Formula: (Final Value / Initial Investment) - 1',
                'Total Return (All Money)': 'Return based on all money contributed. Formula: (Final Portfolio Value / Total Money Added) - 1',
                'CAGR': 'Compound Annual Growth Rate. Average annual return over the entire period.',
                'Max Drawdown': 'Largest peak-to-trough decline. Shows the worst loss from a peak.',
                'Volatility': 'Standard deviation of returns. Measures price variability.',
                'Sharpe': 'Excess return per unit of total volatility. >1 good, >2 very good, >3 excellent.',
                'Sortino': 'Excess return per unit of downside volatility. >1 good, >2 very good, >3 excellent.',
                'Ulcer Index': 'Average depth of drawdowns. <5 excellent, 5-10 moderate, >10 high.',
                'UPI': 'Ulcer Performance Index. Excess return relative to Ulcer Index. >1 good, >2 very good, >3 excellent.',
                'Beta': 'Portfolio volatility relative to benchmark. <1 less volatile, >1 more volatile than market.',
                'MWRR': 'Money-Weighted Rate of Return. Accounts for timing and size of cash flows.',
                'Final Portfolio Value': 'Final value including all contributions and investment returns.',
                'Final Value (No Contributions)': 'Final value excluding additional contributions (only initial investment).',
                'Total Money Added': 'Total amount of money contributed (initial + periodic additions).'
            }
            
            # Add tooltips to the dataframe
            styled_df = stats_df_display.style.format(fmt_map_display)
            
            # Add tooltips using HTML
            tooltip_html = "<div style='background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px; font-size: 12px;'>"
            tooltip_html += "<b>Column Definitions:</b><br><br>"
            for col, tooltip in tooltip_data.items():
                if col in stats_df_display.columns:
                    tooltip_html += f"<b>{col}:</b> {tooltip}<br><br>"
            tooltip_html += "</div>"
            
            # Display tooltip info
            with st.expander("ℹ️ Column Definitions", expanded=False):
                st.markdown(tooltip_html, unsafe_allow_html=True)
            
if fmt_map_display:
try:
                    st.dataframe(stats_df_display.style.format(fmt_map_display), use_container_width=True)
                    st.dataframe(styled_df, use_container_width=True)
except Exception:
# Fallback to raw dataframe if styling fails
st.dataframe(stats_df_display, use_container_width=True)
else:
st.dataframe(stats_df_display, use_container_width=True)

        # Portfolio Configuration Comparison Table
        st.subheader("Portfolio Configuration Comparison")
        
        # Create configuration comparison dataframe
        config_data = {}
        for cfg in st.session_state.strategy_comparison_portfolio_configs:
            portfolio_name = cfg.get('name', 'Unknown')
            
            # Extract configuration details
            config_data[portfolio_name] = {
                'Initial Investment': f"${cfg.get('initial_value', 0):,.2f}",
                'Added Amount': f"${cfg.get('added_amount', 0):,.2f}",
                'Added Frequency': cfg.get('added_frequency', 'None'),
                'Rebalancing Frequency': cfg.get('rebalancing_frequency', 'None'),
                'Use Momentum': 'Yes' if cfg.get('use_momentum', False) else 'No',
                'Momentum Strategy': cfg.get('momentum_strategy', 'N/A'),
                'Negative Momentum Strategy': cfg.get('negative_momentum_strategy', 'N/A'),
                'Number of Stocks': len(cfg.get('stocks', [])),
                'Stocks': ', '.join([s.get('ticker', '') for s in cfg.get('stocks', [])]),
                'Benchmark': cfg.get('benchmark_ticker', 'N/A'),
                'Momentum Windows': str(cfg.get('momentum_windows', [])),
                'Beta Enabled': 'Yes' if cfg.get('use_beta', False) else 'No',
                'Volatility Enabled': 'Yes' if cfg.get('use_vol', False) else 'No',
                'Beta Window': f"{cfg.get('beta_window_days', 0)} days" if cfg.get('use_beta', False) else 'N/A',
                'Volatility Window': f"{cfg.get('vol_window_days', 0)} days" if cfg.get('use_vol', False) else 'N/A',
                'Beta Exclude Days': f"{cfg.get('beta_exclude_days', 0)} days" if cfg.get('use_beta', False) else 'N/A',
                'Volatility Exclude Days': f"{cfg.get('vol_exclude_days', 0)} days" if cfg.get('use_vol', False) else 'N/A'
            }
        
        config_df = pd.DataFrame(config_data).T
        
        # Format the configuration table
        st.dataframe(config_df, use_container_width=True)

st.subheader("Yearly Performance (Interactive Table)")
all_years = st.session_state.strategy_comparison_all_years
years = sorted(list(set(y.year for ser in all_years.values() for y in ser.index)))
@@ -3590,6 +4097,171 @@ def color_gradient_stock(val):

st.dataframe(styler, use_container_width=True, hide_index=False)

        # Monthly Performance Table
        st.subheader("Monthly Performance (Interactive Table)")
        # Use the original results data for monthly calculation, not the yearly resampled data
        all_results = st.session_state.strategy_comparison_all_results
        # Get all available months from the original data
        all_months_data = {}
        for name, results in all_results.items():
            if isinstance(results, dict) and 'with_additions' in results:
                all_months_data[name] = results['with_additions']
            elif isinstance(results, pd.Series):
                all_months_data[name] = results
        
        # Extract all unique year-month combinations from the original data
        months = set()
        for ser in all_months_data.values():
            if not ser.empty:
                for date in ser.index:
                    months.add((date.year, date.month))
        months = sorted(list(months))
        
        # Order portfolio columns according to the portfolio_configs order so new portfolios are added to the right
        names = [cfg['name'] for cfg in st.session_state.strategy_comparison_portfolio_configs if cfg.get('name') in all_months_data]

        # Monthly table creation
        df_monthly_pct_data = {}
        df_monthly_final_data = {}
        for name in names:
            pct_list = []
            final_list = []
            # with-additions monthly series (used for final values)
            ser_with = all_months_data.get(name) if isinstance(all_months_data, dict) else None
            # no-additions monthly series (used for percent-change to avoid skew)
            ser_noadd = None
            try:
                series_obj = st.session_state.strategy_comparison_all_results.get(name)
                if isinstance(series_obj, dict) and 'no_additions' in series_obj:
                    ser_noadd = series_obj['no_additions'].resample('M').last()
                elif isinstance(series_obj, pd.Series):
                    ser_noadd = series_obj.resample('M').last()
            except Exception:
                ser_noadd = None

            for y, m in months:
                # get month slices
                ser_month_with = ser_with[(ser_with.index.year == y) & (ser_with.index.month == m)] if ser_with is not None else pd.Series()
                ser_month_no = ser_noadd[(ser_noadd.index.year == y) & (ser_noadd.index.month == m)] if ser_noadd is not None else pd.Series()

                start_val_for_month = None
                if (y, m) == min(months):
                    config_for_name = next((c for c in st.session_state.strategy_comparison_portfolio_configs if c['name'] == name), None)
                    if config_for_name:
                        initial_val_of_config = config_for_name['initial_value']
                        if initial_val_of_config > 0:
                            start_val_for_month = initial_val_of_config
                else:
                    # Find previous month
                    prev_month_idx = months.index((y, m)) - 1
                    if prev_month_idx >= 0:
                        prev_y, prev_m = months[prev_month_idx]
                        # Use no-additions previous month end as the start value for pct change
                        prev_ser_month_no = ser_noadd[(ser_noadd.index.year == prev_y) & (ser_noadd.index.month == prev_m)] if ser_noadd is not None else pd.Series()
                        if not prev_ser_month_no.empty:
                            start_val_for_month = prev_ser_month_no.iloc[-1]

                # Percent change computed from no-additions series
                if not ser_month_no.empty and start_val_for_month is not None:
                    end_val_no = ser_month_no.iloc[-1]
                    if start_val_for_month > 0:
                        pct_change = (end_val_no - start_val_for_month) / start_val_for_month * 100
                    else:
                        pct_change = np.nan
                else:
                    pct_change = np.nan

                # Final value displayed from with-additions series (if available)
                if not ser_month_with.empty:
                    final_value = ser_month_with.iloc[-1]
                else:
                    final_value = np.nan

                pct_list.append(pct_change)
                final_list.append(final_value)

            df_monthly_pct_data[f'{name} % Change'] = pct_list
            df_monthly_final_data[f'{name} Final Value'] = final_list

        df_monthly_pct = pd.DataFrame(df_monthly_pct_data, index=[f"{y}-{m:02d}" for y, m in months])
        df_monthly_final = pd.DataFrame(df_monthly_final_data, index=[f"{y}-{m:02d}" for y, m in months])
        # Build combined dataframe but preserve the desired column order (selected portfolio first)
        temp_combined_monthly = pd.concat([df_monthly_pct, df_monthly_final], axis=1)
        ordered_cols_monthly = []
        for nm in names:
            pct_col = f'{nm} % Change'
            val_col = f'{nm} Final Value'
            if pct_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(pct_col)
            if val_col in temp_combined_monthly.columns:
                ordered_cols_monthly.append(val_col)
        # Fallback: if nothing matched, use whatever columns exist
        if not ordered_cols_monthly:
            combined_df_monthly = temp_combined_monthly
        else:
            combined_df_monthly = temp_combined_monthly[ordered_cols_monthly]

        # Ensure columns and index are unique (pandas Styler requires unique labels)
        if combined_df_monthly.columns.duplicated().any():
            cols = list(combined_df_monthly.columns)
            seen = {}
            new_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c} ({seen[c]})")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            combined_df_monthly.columns = new_cols

        if combined_df_monthly.index.duplicated().any():
            idx = list(map(str, combined_df_monthly.index))
            seen_idx = {}
            new_idx = []
            for v in idx:
                if v in seen_idx:
                    seen_idx[v] += 1
                    new_idx.append(f"{v} ({seen_idx[v]})")
                else:
                    seen_idx[v] = 0
                    new_idx.append(v)
            combined_df_monthly.index = new_idx

        # Recompute percent and final value column lists after any renaming
        pct_cols_monthly = [col for col in combined_df_monthly.columns if '% Change' in col]
        final_val_cols_monthly = [col for col in combined_df_monthly.columns if 'Final Value' in col]

        # Coerce percent columns to numeric so formatting applies correctly
        for col in pct_cols_monthly:
            if col in combined_df_monthly.columns:
                try:
                    combined_df_monthly[col] = pd.to_numeric(combined_df_monthly[col], errors='coerce')
                except TypeError:
                    # Unexpected column type (not Series/array). Try to coerce via pd.Series or fall back to NaN.
                    try:
                        combined_df_monthly[col] = pd.to_numeric(pd.Series(combined_df_monthly[col]), errors='coerce')
                    except Exception:
                        combined_df_monthly[col] = np.nan

        # Create combined format mapping: percent columns get '%' suffix, final value columns get currency
        fmt_map_monthly = {col: '{:,.2f}%' for col in pct_cols_monthly if col in combined_df_monthly.columns}
        fmt_map_monthly.update({col: '${:,.2f}' for col in final_val_cols_monthly if col in combined_df_monthly.columns})

        styler_monthly = combined_df_monthly.style
        # Color percent cells with a gradient and then apply formatting in one call
        if pct_cols_monthly:
            try:
                # Styler.map is the supported replacement for applymap
                styler_monthly = styler_monthly.map(color_gradient_stock, subset=pct_cols_monthly)
            except Exception:
                # If map still fails (edge cases), skip coloring to avoid breaking the page
                pass
        if fmt_map_monthly:
            styler_monthly = styler_monthly.format(fmt_map_monthly, na_rep='N/A')

        st.dataframe(styler_monthly, use_container_width=True, hide_index=False)

st.markdown("---")
st.markdown("**Detailed Portfolio Information**")
# Make the selector visually prominent
@@ -3968,7 +4640,7 @@ def prepare_bar_data(d):

# Main "Rebalance as of today" plot and table - this should be the main rebalancing representation
st.markdown("---")
                                st.markdown("**🔄 Rebalance as of Today**")
                                st.markdown(f"**🔄 Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})**")

# Get momentum-based calculated weights for today's rebalancing from stored snapshot
today_weights = {}
@@ -4424,7 +5096,7 @@ def highlight_metrics_rows(s):

# Main "Rebalance as of today" plot and table for fallback scenario
st.markdown("---")
                        st.markdown("**🔄 Rebalance as of Today**")
                        st.markdown(f"**🔄 Rebalance as of Today ({pd.Timestamp.now().strftime('%Y-%m-%d')})**")

# Get momentum-based calculated weights for today's rebalancing from stored snapshot (fallback scenario)
today_weights = {}
