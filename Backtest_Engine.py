console_output = io.StringIO()
logger = logging.getLogger(__name__)

def sync_date_widgets_with_imported_values():
    """Sync date widgets with imported values from JSON"""
    # Sync start date
    if st.session_state.get('start_date') is not None:
        # Ensure the widget key is set to the imported value
        st.session_state["start_date"] = st.session_state.start_date
    
    # Sync end date
    if st.session_state.get('end_date') is not None:
        # Ensure the widget key is set to the imported value
        st.session_state["end_date"] = st.session_state.end_date
    
    # Sync custom dates checkbox
    has_custom_dates = (st.session_state.get('start_date') is not None or 
                       st.session_state.get('end_date') is not None)
    st.session_state["use_custom_dates"] = has_custom_dates
    st.session_state["use_custom_dates_checkbox"] = has_custom_dates

# --- Print start dates for all tickers and benchmark at script start ---
def print_ticker_start_dates(asset_tickers, benchmark_ticker):
# All print statements removed for performance
@@ -2173,10 +2191,18 @@ def _ss_default(key, value):
st.session_state["use_custom_dates_checkbox"] = st.session_state["use_custom_dates"]
if "_import_start_date" in st.session_state:
sd = st.session_state.pop("_import_start_date")
            st.session_state["start_date"] = None if sd in (None, 'None', '') else pd.to_datetime(sd).date()
            start_date = None if sd in (None, 'None', '') else pd.to_datetime(sd).date()
            st.session_state["start_date"] = start_date
            if start_date is not None:
                st.session_state["use_custom_dates"] = True
                st.session_state["use_custom_dates_checkbox"] = True
if "_import_end_date" in st.session_state:
ed = st.session_state.pop("_import_end_date")
            st.session_state["end_date"] = None if ed in (None, 'None', '') else pd.to_datetime(ed).date()
            end_date = None if ed in (None, 'None', '') else pd.to_datetime(ed).date()
            st.session_state["end_date"] = end_date
            if end_date is not None:
                st.session_state["use_custom_dates"] = True
                st.session_state["use_custom_dates_checkbox"] = True


# Handle portfolio-specific JSON imports for main app
@@ -2281,6 +2307,36 @@ def _ss_default(key, value):
if 'collect_dividends_as_cash' in main_app_config:
st.session_state["collect_dividends_as_cash"] = bool(main_app_config['collect_dividends_as_cash'])
st.session_state["collect_dividends_as_cash_checkbox"] = bool(main_app_config['collect_dividends_as_cash'])
                if 'start_date_user' in main_app_config and main_app_config['start_date_user'] is not None:
                    # Parse date from string if needed
                    start_date = main_app_config['start_date_user']
                    if isinstance(start_date, str):
                        try:
                            start_date = pd.to_datetime(start_date).date()
                        except:
                            start_date = None
                    if start_date is not None:
                        st.session_state["start_date"] = start_date
                        st.session_state["use_custom_dates"] = True
                        st.session_state["use_custom_dates_checkbox"] = True
                if 'end_date_user' in main_app_config and main_app_config['end_date_user'] is not None:
                    # Parse date from string if needed
                    end_date = main_app_config['end_date_user']
                    if isinstance(end_date, str):
                        try:
                            end_date = pd.to_datetime(end_date).date()
                        except:
                            end_date = None
                    if end_date is not None:
                        st.session_state["end_date"] = end_date
                        st.session_state["use_custom_dates"] = True
                        st.session_state["use_custom_dates_checkbox"] = True
                
                # Ensure checkbox is enabled if either date is set
                if (main_app_config.get('start_date_user') is not None or 
                    main_app_config.get('end_date_user') is not None):
                    st.session_state["use_custom_dates"] = True
                    st.session_state["use_custom_dates_checkbox"] = True
if 'start_with' in main_app_config:
# Handle start_with value mapping from other pages
start_with = main_app_config['start_with']
@@ -2361,9 +2417,16 @@ def _ss_default(key, value):
except Exception:
# If any staging application fails, keep whatever was applied and continue
pass
    
    # Sync date widgets with imported values
    sync_date_widgets_with_imported_values()
    
# Clear the pending flag so widgets created afterwards won't be overwritten
if "_import_pending" in st.session_state:
del st.session_state["_import_pending"]
    
    # Force a rerun to update the UI widgets
    st.session_state.main_rerun_flag = True

# Handle rerun flag for smooth UI updates
if st.session_state.get('main_rerun_flag', False):
@@ -2728,6 +2791,9 @@ def paste_json_callback():
if 'first_rebalance_strategy' in json_data:
st.session_state['_import_first_rebalance_strategy'] = json_data['first_rebalance_strategy']

        # Set the pending flag to trigger import processing
        st.session_state['_import_pending'] = True
        
st.success("Portfolio configuration updated from JSON (Backtest Engine page).")
except json.JSONDecodeError:
st.error("Invalid JSON format. Please check the text and try again.")
@@ -3251,8 +3317,17 @@ def update_backtest_json():
# Time & Data Options section (only once, after portfolio settings)
st.header("Time & Data Options")
# Use the same pattern as working Multi backtest and Allocations pages
    # Ensure checkbox state is properly synchronized
if "use_custom_dates_checkbox" not in st.session_state:
st.session_state["use_custom_dates_checkbox"] = st.session_state.use_custom_dates
    else:
        # Keep checkbox in sync with the main state
        st.session_state["use_custom_dates_checkbox"] = st.session_state.use_custom_dates
    
    # Debug: Show current checkbox state
    if st.session_state.get('_import_pending', False):
        st.info(f"Debug: use_custom_dates={st.session_state.get('use_custom_dates', 'Not set')}, checkbox={st.session_state.get('use_custom_dates_checkbox', 'Not set')}")
    
st.checkbox(
"Use Custom Date Range", 
key="use_custom_dates_checkbox",
