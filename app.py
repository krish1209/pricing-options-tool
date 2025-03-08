import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from scipy import stats

# Import our modules
from models.black_scholes import black_scholes, black_scholes_greeks
from models.monte_carlo import monte_carlo_european, monte_carlo_path_generator
from models.binomial import binomial_tree_european, binomial_tree_american, get_binomial_tree_data
from models.risk_metrics import calculate_var, calculate_cvar, calculate_sharpe_ratio, calculate_beta, monte_carlo_var
from data.yahoo_finance import get_stock_data, get_option_chain, calculate_implied_volatility, get_risk_free_rate, get_market_data

# Initialize session state variables for visualizations
if 'has_calculated' not in st.session_state:
    st.session_state.has_calculated = False
if 'mc_paths' not in st.session_state:
    st.session_state.mc_paths = None
if 'binomial_data' not in st.session_state:
    st.session_state.binomial_data = None
if 'option_params' not in st.session_state:
    st.session_state.option_params = {}

# Reset visualization state when parameters change
def reset_visualization_state():
    st.session_state.has_calculated = False
    st.session_state.mc_paths = None
    st.session_state.binomial_data = None

st.set_page_config(
    page_title="Options Pricing & Risk Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Color Palette */
    :root {
        --off-white: #F7F7F7;
        --amber: #FFA726;
        --brown: #704214;
        --black: #000000;
        --dark-gray: #121212;
        --medium-gray: #272727;
        --text-color: var(--off-white);
        --accent-color: var(--amber);
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--black);
        color: var(--text-color) !important;
    }
    
    /* Force all text to be light by default */
    .stMarkdown, .stText, p, div, span, label {
        color: var(--text-color) !important;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--accent-color) !important;
        font-weight: 600 !important;
    }
    
    h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--dark-gray);
        border-right: 1px solid var(--medium-gray);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-color) !important;
    }
    
    section[data-testid="stSidebar"] button {
        background-color: var(--accent-color) !important;
        color: var(--black) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color) !important;
        color: var(--black) !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--brown) !important;
        color: var(--off-white) !important;
        box-shadow: 0 0 10px rgba(255, 167, 38, 0.5) !important;
    }
    
    /* Input text */
    .stTextInput > div > div > input {
        color: var(--off-white) !important;
        background-color: var(--medium-gray) !important;
        border: 1px solid var(--dark-gray) !important;
    }
    
    /* Fix number inputs */
    input[type="number"] {
        color: var(--off-white) !important;
        background-color: var(--medium-gray) !important;
        border: 1px solid var(--dark-gray) !important;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio label, .stCheckbox label {
        color: var(--text-color) !important;
    }
    
    /* Radio buttons color */
    .stRadio [data-baseweb="radio"] div[role="radiogroup"] div[role="radio"] div {
        border-color: var(--accent-color) !important;
    }
    
    .stRadio [data-baseweb="radio"] div[role="radiogroup"] div[role="radio"][aria-checked="true"] div::after {
        background-color: var(--accent-color) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--medium-gray) !important;
        border: 1px solid var(--dark-gray) !important;
    }
    
    /* Card-like elements */
    div.stDataFrame, div[data-testid="stTable"] {
        background-color: var(--dark-gray);
        border: 1px solid var(--medium-gray);
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: var(--dark-gray);
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--medium-gray);
        border-radius: 4px 4px 0 0;
        color: var(--off-white) !important;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: var(--black) !important;
        font-weight: 600 !important;
    }
    
    /* Tab content background */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--dark-gray);
        border-radius: 0 0 4px 4px;
        padding: 1rem;
        border: 1px solid var(--medium-gray);
        border-top: none;
    }
    
    /* Override the tab content color */
    .stTabs [data-baseweb="tab-panel"] > div {
        color: var(--text-color) !important;
    }
    
    /* Improved slider spacing */
    .stSlider {
        padding-top: 20px;
        padding-bottom: 35px;
        margin-bottom: 10px;
    }
    
    /* Improve slider label visibility */
    .stSlider label {
        margin-bottom: 8px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        color: var(--text-color) !important;
    }
    
    /* Better spacing for min/max values */
    .stSlider [data-baseweb] div[data-testid] {
        margin-top: 8px;
        color: var(--text-color) !important;
    }
    
    /* Fix slider track */
    .stSlider > div > div > div {
        background-color: var(--medium-gray) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--accent-color) !important;
    }
    
    /* Fix slider thumb value positioning */
    .stSlider [data-testid="stThumbValue"] {
        position: absolute;
        background-color: var(--accent-color) !important;
        color: var(--black) !important;
        font-weight: 600 !important;
        padding: 3px 8px !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        top: -28px !important;
        transform: translateX(-50%);
        z-index: 100;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: var(--accent-color) !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricDelta"] {
        background-color: var(--brown);
        padding: 2px 6px;
        border-radius: 4px;
        color: var(--off-white) !important;
    }
    
    /* Metric label color fix */
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    
    /* SelectBox styling */
    div[data-baseweb="select"] span {
        color: var(--text-color) !important;
    }
    
    /* MultiSelect options */
    div[role="listbox"] div {
        color: var(--text-color) !important;
        background-color: var(--medium-gray);
    }
    
    div[role="listbox"] div:hover {
        background-color: var(--dark-gray) !important;
    }
    
    /* Selected items in multiselect */
    div[data-baseweb="tag"] {
        background-color: var(--accent-color) !important;
    }
    
    div[data-baseweb="tag"] span {
        color: var(--black) !important;
    }
    
    /* Table layout improvements for better visibility */
    .dataframe {
        font-size: 13px !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
        border: none !important;
    }
    
    /* Ensure scrolling works properly */
    [data-testid="stDataFrame"] > div {
        max-height: 500px !important; /* Increased height */
        overflow: auto !important;
    }
    
    /* Ensure table cell text is light */
    .dataframe td {
        color: var(--text-color) !important;
        background-color: var(--dark-gray) !important;
    }
    
    /* Sticky header for tables */
    .dataframe thead th {
        position: sticky !important;
        top: 0 !important;
        z-index: 1 !important;
        background-color: var(--brown) !important;
        color: var(--off-white) !important;
        padding: 8px 4px !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        background-color: var(--dark-gray);
        padding: 6px 4px !important;
        border-bottom: 1px solid var(--medium-gray) !important;
    }
    
    /* Better styling for call options */
    .call-options-table th {
        background-color: var(--brown) !important;
        color: var(--off-white) !important;
        padding: 8px 4px !important;
        text-align: center !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--medium-gray) !important;
    }
    
    .call-options-table td {
        background-color: var(--dark-gray) !important;
        color: var(--text-color) !important;
        padding: 6px 4px !important;
        text-align: right !important;
        border-bottom: 1px solid var(--medium-gray) !important;
    }
    
    /* Better styling for put options */
    .put-options-table th {
        background-color: var(--brown) !important;
        color: var(--off-white) !important;
        padding: 8px 4px !important;
        text-align: center !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--medium-gray) !important;
    }
    
    .put-options-table td {
        background-color: var(--dark-gray) !important;
        color: var(--text-color) !important;
        padding: 6px 4px !important;
        text-align: right !important;
        border-bottom: 1px solid var(--medium-gray) !important;
    }
    
    /* Special highlights for in-the-money options */
    .call-options-table .itm {
        background-color: rgba(255, 167, 38, 0.2) !important;
    }
    
    .put-options-table .itm {
        background-color: rgba(255, 167, 38, 0.2) !important;
    }
    
    /* Ensure option chain columns are spaced better */
    .options-container {
        display: flex;
        gap: 20px;
    }
    
    .option-column {
        flex: 1;
    }
    
    /* Custom header bar */
    .header-container {
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        padding: 1rem 0;
        border-bottom: 2px solid var(--accent-color);
        margin-bottom: 2rem;
    }
    
    .header-title {
        color: var(--accent-color);
        margin: 0;
        font-weight: 700;
    }
    
    .header-links {
        display: flex; 
        gap: 15px; 
        align-items: center;
    }
    
    .header-links img {
        transition: transform 0.2s ease;
        filter: invert(1);
    }
    
    .header-links img:hover {
        transform: scale(1.1);
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid var(--dark-gray);
        color: var(--text-color);
        font-size: 0.9rem;
        text-align: center;
    }
    
    /* Interpretation text styling */
    .interpretation-text {
        background-color: var(--dark-gray);
        border-left: 3px solid var(--accent-color);
        padding: 0.8rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
    }
    
    /* Result box styling */
    .result-box {
        background-color: var(--dark-gray);
        border: 1px solid var(--medium-gray);
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Plots and charts styling */
    [data-testid="stDecoration"] {
        background-color: var(--dark-gray) !important;
        border: 1px solid var(--medium-gray) !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Caption and legend text fixes */
    .js-plotly-plot .plotly .legend text, 
    .js-plotly-plot .plotly .xtick text, 
    .js-plotly-plot .plotly .ytick text {
        fill: var(--text-color) !important;
    }
    
    /* Fix code blocks */
    .stCodeBlock, code {
        background-color: var(--medium-gray) !important;
        color: var(--off-white) !important;
    }
    
    /* Misc elements fixes */
    .stProgress > div > div > div {
        background-color: var(--accent-color) !important;
    }
    
    /* Make sure all progress bars have proper contrast */
    .stProgress {
        color: var(--off-white) !important;
    }
    
    /* Fix expanders */
    .stExpander details {
        background-color: var(--dark-gray) !important;
        border: 1px solid var(--medium-gray) !important;
    }
    
    .stExpander details summary {
        color: var(--text-color) !important;
    }
    
    /* Fix captions */
    .stCaption {
        color: var(--text-color) !important;
        opacity: 0.8;
    }
    
    /* Fix info boxes */
    div[data-baseweb="notification"] {
        background-color: var(--dark-gray) !important;
        border-left-color: var(--accent-color) !important;
    }
    
    div[data-baseweb="notification"] div {
        color: var(--text-color) !important;
    }
    
    /* Fix tooltip text */
    div[data-baseweb="tooltip"] {
        background-color: var(--brown) !important;
        color: var(--off-white) !important;
    }
    
    /* Make plotly background dark */
    .js-plotly-plot .plotly {
        background-color: var(--dark-gray) !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background-color: var(--dark-gray) !important;
    }
    
    /* Fix the main trace colors to be bright in dark mode */
    .js-plotly-plot .plotly .scatter .lines {
        stroke: var(--accent-color) !important;
    }
    
    /* Fix line chart grid */
    .js-plotly-plot .plotly .gridlayer path {
        stroke: var(--medium-gray) !important;
    }

    /* Fix axis lines */
    .js-plotly-plot .plotly .xaxis path.domain,
    .js-plotly-plot .plotly .yaxis path.domain {
        stroke: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# st.markdown("This app provides tools for pricing options using different models and analyzing financial risks.")
st.markdown(
    """
    <div style="padding: 1rem 0; border-bottom: 2px solid #FFA726; margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1 style="color: #FFA726; margin: 0; font-weight: 700;">Options Pricing & Risk Analysis Tool</h1>
            <div style="display: flex; gap: 15px; align-items: center;">
                <a href="github.com/krish1209/pricing-options-tool" target="_blank">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" style="filter: invert(1);" />
                </a>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="color: #F7F7F7;">Created by - Krish Bagga</span>
                    <a href="https://www.linkedin.com/in/krishbagga/" target="_blank">
                        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="20" style="filter: invert(1);" />
                    </a>
                </div>
            </div>
        </div>
        <p style="color: #F7F7F7; margin-top: 0.5rem; margin-bottom: 0;">Analyze options using various pricing models and evaluate financial risk metrics.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
# Sidebar for ticker selection and general parameters
st.sidebar.header("Settings")

# Ticker input
ticker = st.sidebar.text_input("Enter stock ticker:", "AAPL").upper()

# Create a placeholder in the sidebar for stock data status
stock_status = st.sidebar.empty()
stock_status.info("Fetching stock data...")

# Fetch stock data
stock_data = get_stock_data(ticker)
if not stock_data.empty:
    current_price = stock_data['Close'].iloc[-1]
    stock_status.success(f"Current {ticker} price: ${current_price:.2f}")
else:
    stock_status.error(f"Failed to fetch data for {ticker}")
    current_price = 100.0  # Default for demonstration

# Get risk-free rate
risk_free_rate = get_risk_free_rate()
st.sidebar.info(f"Current risk-free rate: {risk_free_rate*100:.2f}%")

# Create a placeholder for volatility status
vol_status = st.sidebar.empty()
vol_status.info("Calculating volatility...")

# Calculate implied volatility
implied_vol = calculate_implied_volatility(ticker)
if implied_vol:
    vol_status.info(f"Historical volatility (annualized): {implied_vol*100:.2f}%")
else:
    implied_vol = 0.3  # Default
    vol_status.warning(f"Using default volatility: {implied_vol*100:.2f}%")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Option Pricing", "Risk Analysis", "Market Data", "About"])

# Tab 1: Option Pricing Models
with tab1:
    st.header("Option Pricing Models")
    st.markdown("Compare different option pricing models and visualize results.")
    
    # Option parameters input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option Parameters")
        option_type = st.radio("Option Type:", ["Call", "Put"], on_change=reset_visualization_state)
        strike_price = st.number_input("Strike Price ($):", min_value=0.1, value=round(current_price, 1), step=1.0, on_change=reset_visualization_state)
        maturity_days = st.number_input("Time to Maturity (days):", min_value=1, value=30, on_change=reset_visualization_state)
        maturity = maturity_days / 365.0  # Convert to years
        volatility = st.slider("Volatility (%):", min_value=1.0, max_value=100.0, value=float(implied_vol*100), step=0.1, on_change=reset_visualization_state) / 100
        rf_rate = st.slider("Risk-Free Rate (%):", min_value=0.0, max_value=10.0, value=float(risk_free_rate*100), step=0.1, on_change=reset_visualization_state) / 100
    
    with col2:
        st.subheader("Model Settings")
        models_to_use = st.multiselect(
            "Select Models to Compare:",
            ["Black-Scholes", "Monte Carlo", "Binomial Tree"],
            default=["Black-Scholes", "Monte Carlo", "Binomial Tree"],
            on_change=reset_visualization_state
        )
        
        monte_carlo_sims = st.number_input("Monte Carlo Simulations:", min_value=1000, max_value=100000, value=10000, step=1000, on_change=reset_visualization_state)
        binomial_steps = st.number_input("Binomial Tree Steps:", min_value=10, max_value=1000, value=100, step=10, on_change=reset_visualization_state)
        
        st.markdown("---")
        st.markdown("### Stock Price")
        st.markdown(f"Current Price: **${current_price:.2f}**")
        
        # Option to manually override stock price
        override_price = st.checkbox("Override Stock Price", on_change=reset_visualization_state)
        if override_price:
            current_price = st.number_input("Stock Price ($):", min_value=0.1, value=current_price, step=1.0, on_change=reset_visualization_state)
    
    # Calculate option prices when button is clicked
    if st.button("Calculate Option Prices"):
        st.session_state.has_calculated = True
        st.session_state.option_params = {
            'current_price': current_price,
            'strike_price': strike_price,
            'maturity': maturity,
            'rf_rate': rf_rate,
            'volatility': volatility,
            'option_type': option_type.lower(),
            'models': models_to_use,
            'monte_carlo_sims': monte_carlo_sims,
            'binomial_steps': binomial_steps
        }
        
        calc_options_status = st.empty()
        calc_options_status.info("Calculating...")
        
        # Results container
        st.markdown("## Pricing Results")
        results_cols = st.columns(len(models_to_use))
        
        # Dictionary to store results for comparison
        price_results = {}
        
        # Calculate option prices for each selected model
        for i, model in enumerate(models_to_use):
            with results_cols[i]:
                st.subheader(f"{model} Model")
                
                if model == "Black-Scholes":
                    # Calculate using Black-Scholes
                    start_time = time.time()
                    bs_price = black_scholes(current_price, strike_price, maturity, rf_rate, volatility, option_type.lower())
                    end_time = time.time()
                    
                    # Calculate Greeks
                    greeks = black_scholes_greeks(current_price, strike_price, maturity, rf_rate, volatility, option_type.lower())
                    
                    # Display results
                    st.metric("Option Price", f"${bs_price:.4f}")
                    st.markdown(f"Computation Time: {(end_time - start_time)*1000:.2f} ms")
                    
                    # Display Greeks
                    st.markdown("##### Greeks:")
                    greek_cols = st.columns(3)
                    greek_cols[0].metric("Delta", f"{greeks['delta']:.4f}")
                    greek_cols[1].metric("Gamma", f"{greeks['gamma']:.4f}")
                    greek_cols[2].metric("Theta", f"{greeks['theta']:.4f}")
                    greek_cols[0].metric("Vega", f"{greeks['vega']:.4f}")
                    greek_cols[1].metric("Rho", f"{greeks['rho']:.4f}")
                    
                    # Store result
                    price_results["Black-Scholes"] = bs_price
                
                elif model == "Monte Carlo":
                    # Calculate using Monte Carlo
                    start_time = time.time()
                    mc_result = monte_carlo_european(current_price, strike_price, maturity, rf_rate, volatility, option_type.lower(), monte_carlo_sims)
                    end_time = time.time()
                    
                    # Generate Monte Carlo paths for visualization and store in session state
                    time_points, paths = monte_carlo_path_generator(
                        current_price, maturity, rf_rate, volatility, simulations=50, steps=100
                    )
                    st.session_state.mc_paths = {
                        'time_points': time_points,
                        'paths': paths
                    }
                    
                    # Display results
                    st.metric("Option Price", f"${mc_result['price']:.4f}")
                    st.markdown(f"Computation Time: {(end_time - start_time)*1000:.2f} ms")
                    
                    # Display additional info
                    st.markdown("##### Statistics:")
                    st.markdown(f"Standard Error: ${mc_result['std_error']:.4f}")
                    st.markdown(f"95% Confidence Interval: [${ mc_result['confidence_interval'][0]:.4f}, ${mc_result['confidence_interval'][1]:.4f}]")
                    st.progress(min(1.0, (100 - mc_result['accuracy']) / 100))
                    st.caption(f"Accuracy: ±{mc_result['accuracy']:.2f}%")
                    
                    # Store result
                    price_results["Monte Carlo"] = mc_result['price']
                
                elif model == "Binomial Tree":
                    # Calculate using Binomial Tree
                    start_time = time.time()
                    
                    # European option
                    bt_euro_price = binomial_tree_european(
                        current_price, strike_price, maturity, rf_rate, volatility, 
                        option_type.lower(), binomial_steps
                    )
                    
                    # American option
                    bt_amer_price = binomial_tree_american(
                        current_price, strike_price, maturity, rf_rate, volatility, 
                        option_type.lower(), binomial_steps
                    )
                    
                    # Generate tree data for visualization and store in session state
                    vis_steps = min(10, binomial_steps)  # Limit for visualization
                    st.session_state.binomial_data = get_binomial_tree_data(
                        current_price, strike_price, maturity, rf_rate, volatility,
                        option_type.lower(), vis_steps
                    )
                    
                    end_time = time.time()
                    
                    # Display results
                    euro_col, amer_col = st.columns(2)
                    
                    with euro_col:
                        st.metric("European Option", f"${bt_euro_price:.4f}")
                        
                    with amer_col:
                        st.metric("American Option", f"${bt_amer_price:.4f}")
                        early_exercise = bt_amer_price - bt_euro_price
                        if early_exercise > 0.0001:
                            st.caption(f"Early Exercise Premium: ${early_exercise:.4f}")
                    
                    st.markdown(f"Computation Time: {(end_time - start_time)*1000:.2f} ms")
                    
                    # Store result
                    price_results["Binomial Tree"] = bt_euro_price
        
        # Clear calculating status
        calc_options_status.empty()
        
        # Model comparison chart
        if len(price_results) > 1:
            st.markdown("### Model Comparison")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(price_results.keys()),
                    y=list(price_results.values()),
                    text=[f"${p:.4f}" for p in price_results.values()],
                    textposition='auto',
                    marker_color=['rgba(75, 120, 168, 0.8)'] * len(price_results)
                )
            ])
            
            fig.update_layout(
                title=f"{option_type} Option Price Comparison",
                xaxis_title="Pricing Model",
                yaxis_title="Option Price ($)",
                yaxis=dict(range=[0, max(price_results.values()) * 1.2]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # VISUALIZATION SECTION - THIS IS COMPLETELY SEPARATE FROM THE CALCULATION
    # Only show visualizations if calculation has been performed
    if st.session_state.has_calculated:
        st.markdown("---")
        st.markdown("## Visualizations")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2 = st.tabs(["Monte Carlo Paths", "Binomial Tree"])
        
        # Tab 1: Monte Carlo Paths
        with viz_tab1:
            if "Monte Carlo" in st.session_state.option_params.get('models', []) and st.session_state.mc_paths is not None:
                st.subheader("Monte Carlo Price Path Simulation")
                
                # Get parameters from session state
                params = st.session_state.option_params
                mc_data = st.session_state.mc_paths
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(min(20, mc_data['paths'].shape[0])):  # Plot up to 20 paths
                    ax.plot(mc_data['time_points'], mc_data['paths'][i], linewidth=0.8, alpha=0.6)
                
                # Plot strike price line
                ax.axhline(y=params['strike_price'], color='r', linestyle='--', linewidth=1)
                
                # Plot the mean path
                ax.plot(mc_data['time_points'], np.mean(mc_data['paths'], axis=0), color='black', linewidth=2)
                
                ax.set_xlabel('Time (years)')
                ax.set_ylabel('Stock Price ($)')
                ax.set_title(f'Monte Carlo Simulation Paths for {params["option_type"].capitalize()} Option')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.info("Please select Monte Carlo model and calculate option prices to view paths.")
        
        # Tab 2: Binomial Tree
        with viz_tab2:
            if "Binomial Tree" in st.session_state.option_params.get('models', []) and st.session_state.binomial_data is not None:
                st.subheader("Binomial Tree Visualization")
                
                # Get parameters from session state
                params = st.session_state.option_params
                tree_data = st.session_state.binomial_data
                
                # Create visualization using Plotly
                fig = go.Figure()
                
                # Add nodes
                for i, t in enumerate(tree_data['time_points']):
                    for j in range(i + 1):
                        stock_price = tree_data['stock_prices'][j, i]
                        option_price = tree_data['option_values'][j, i]
                        
                        # Node position
                        x = t
                        y = j - i/2  # Adjust y for balanced tree
                        
                        # Add node
                        fig.add_trace(go.Scatter(
                            x=[x], y=[y],
                            mode='markers+text',
                            marker=dict(size=30, color='rgba(75, 120, 168, 0.8)'),
                            text=f"${stock_price:.1f}<br>${option_price:.2f}",
                            textposition="middle center",
                            textfont=dict(color='white', size=9),
                            showlegend=False,
                            hoverinfo='text',
                            hovertext=f"Time: {t:.3f}<br>Stock: ${stock_price:.2f}<br>Option: ${option_price:.2f}"
                        ))
                        
                        # Connect with previous nodes
                        if i > 0:
                            # Connect to upper node (up factor)
                            if j > 0:
                                fig.add_trace(go.Scatter(
                                    x=[tree_data['time_points'][i-1], x],
                                    y=[j-1-(i-1)/2, y],
                                    mode='lines',
                                    line=dict(width=1, color='rgba(75, 120, 168, 0.6)'),
                                    showlegend=False
                                ))
                            
                            # Connect to lower node (down factor)
                            if j < i:
                                fig.add_trace(go.Scatter(
                                    x=[tree_data['time_points'][i-1], x],
                                    y=[j-(i-1)/2, y],
                                    mode='lines',
                                    line=dict(width=1, color='rgba(75, 120, 168, 0.6)'),
                                    showlegend=False
                                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Binomial Tree for {params['option_type'].capitalize()} Option (Simplified View)",
                    xaxis_title="Time (years)",
                    height=500,
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select Binomial Tree model and calculate option prices to view the tree.")

# Tab 2: Risk Analysis
with tab2:
    st.header("Risk Analysis")
    st.markdown("Analyze various risk metrics for selected ticker.")
    
    # Fetch extended stock data for risk analysis
    risk_data_status = st.empty()
    risk_data_status.info("Fetching extended data for risk analysis...")
    
    # Get 2 years of data for better risk analysis
    extended_data = get_stock_data(ticker, period='2y')
    risk_data_status.empty()
    
    if extended_data.empty:
        st.error("Failed to fetch extended data for risk analysis")
    else:
        # Extract returns
        returns = extended_data['Daily_Return'].dropna()
        log_returns = extended_data['Log_Return'].dropna()
        
        # Risk analysis tabs
        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Value at Risk (VaR)", "Performance Metrics", "Historical Analysis"])
        
        # VaR and CVaR tab
        with risk_tab1:
            st.subheader("Value at Risk (VaR) Analysis")
            
            # Parameters
            var_col1, var_col2 = st.columns(2)
            
            with var_col1:
                confidence_level = st.slider("Confidence Level:", min_value=0.9, max_value=0.99, value=0.95, step=0.01)
                investment_amount = st.number_input("Investment Amount ($):", min_value=1000, value=10000, step=1000)
            
            with var_col2:
                var_method = st.radio("VaR Calculation Method:", ["Historical", "Monte Carlo"])
                time_horizon = st.slider("Time Horizon (days):", min_value=1, max_value=30, value=1)
                
            # Calculate VaR and CVaR
            if st.button("Calculate Risk Metrics"):
                calc_status = st.empty()
                calc_status.info("Calculating risk metrics...")
                
                if var_method == "Historical":
                    # Historical VaR and CVaR
                    hist_var = calculate_var(returns, confidence_level)
                    hist_cvar = calculate_cvar(returns, confidence_level)
                    
                    # Calculate dollar VaR based on investment amount
                    dollar_var = investment_amount * hist_var
                    dollar_cvar = investment_amount * hist_cvar
                    
                    # Display results
                    calc_status.empty()
                    st.markdown("### Historical VaR Results")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("1-Day VaR", f"${abs(dollar_var):.2f}")
                        st.caption(f"At {confidence_level*100:.1f}% confidence level")
                        st.markdown(
                            f"""<div class="interpretation-text">
                            <strong>Interpretation:</strong> With {confidence_level*100:.1f}% confidence, we do not expect to lose 
                            more than <strong>${abs(dollar_var):.2f}</strong> in one day on a <strong>${investment_amount:,}</strong> investment.
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    with metric_col2:
                        st.metric("1-Day CVaR (Expected Shortfall)", f"${abs(dollar_cvar):.2f}")
                        st.caption(f"Expected loss exceeding VaR")
                        st.markdown(f"**Interpretation**: If the loss exceeds VaR, the expected loss amount is ${abs(dollar_cvar):.2f}.")
                    
                    # Multi-day VaR (approximation using square root of time rule)
                    if time_horizon > 1:
                        multi_day_var = dollar_var * np.sqrt(time_horizon)
                        multi_day_cvar = dollar_cvar * np.sqrt(time_horizon)
                        
                        st.markdown("---")
                        st.markdown(f"### {time_horizon}-Day VaR Forecast")
                        
                        md_col1, md_col2 = st.columns(2)
                        with md_col1:
                            st.metric(f"{time_horizon}-Day VaR", f"${abs(multi_day_var):.2f}")
                            st.caption(f"Using square root of time rule")
                        
                        with md_col2:
                            st.metric(f"{time_horizon}-Day CVaR", f"${abs(multi_day_cvar):.2f}")
                            st.caption(f"Expected shortfall over {time_horizon} days")
                    
                    # Visualize return distribution with VaR
                    st.markdown("### Return Distribution")
                    
                    fig = go.Figure()
                    
                    # Returns histogram
                    fig.add_trace(go.Histogram(
                        x=returns,
                        name="Daily Returns",
                        opacity=0.7,
                        nbinsx=50,
                        marker=dict(color="rgba(75, 120, 168, 0.6)")
                    ))
                    
                    # VaR line
                    fig.add_shape(
                        type="line",
                        x0=hist_var, y0=0,
                        x1=hist_var, y1=30,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # CVaR line
                    fig.add_shape(
                        type="line",
                        x0=hist_cvar, y0=0,
                        x1=hist_cvar, y1=30,
                        line=dict(color="darkred", width=2, dash="dot")
                    )
                    
                    # Add annotations
                    fig.add_annotation(
                        x=hist_var, y=25,
                        text=f"VaR ({confidence_level*100:.1f}%)",
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="red",
                        arrowsize=1,
                        arrowwidth=2
                    )
                    
                    fig.add_annotation(
                        x=hist_cvar, y=20,
                        text="CVaR",
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="darkred",
                        arrowsize=1,
                        arrowwidth=2
                    )
                    
                    fig.update_layout(
                        title="Return Distribution with VaR and CVaR",
                        xaxis_title="Daily Return",
                        yaxis_title="Frequency",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif var_method == "Monte Carlo":
                    # Monte Carlo VaR
                    mc_var_result = monte_carlo_var(
                        current_price, rf_rate, volatility, 
                        T=time_horizon/365, 
                        n_simulations=10000, 
                        confidence_level=confidence_level
                    )
                    
                    # Scale to investment amount
                    scaled_var = mc_var_result['VaR'] / current_price * investment_amount
                    scaled_cvar = mc_var_result['CVaR'] / current_price * investment_amount
                    
                    # Display results
                    calc_status.empty()
                    st.markdown("### Monte Carlo VaR Results")
                    
                    mc_col1, mc_col2 = st.columns(2)
                    with mc_col1:
                        st.metric(f"{time_horizon}-Day VaR", f"${scaled_var:.2f}")
                        st.caption(f"At {confidence_level*100:.1f}% confidence level")
                        st.markdown(f"**Interpretation**: With {confidence_level*100:.1f}% confidence, we do not expect to lose more than ${scaled_var:.2f} over {time_horizon} days on a ${investment_amount:,} investment.")
                    
                    with mc_col2:
                        st.metric(f"{time_horizon}-Day CVaR", f"${scaled_cvar:.2f}")
                        st.caption(f"Expected loss exceeding VaR")
                        st.markdown(f"**Interpretation**: If the loss exceeds VaR, the expected loss amount is ${scaled_cvar:.2f}.")
                    
                    # Visualize Monte Carlo simulations
                    st.markdown("### Monte Carlo Simulation")
                    
                    # Generate price paths
                    np.random.seed(42)  # For reproducibility
                    n_sims = 1000
                    n_days = time_horizon
                    
                    # Initial price for all simulations
                    simulated_returns = np.random.normal(
                        (rf_rate - 0.5 * volatility**2) / 365, 
                        volatility / np.sqrt(365), 
                        size=(n_sims, n_days)
                    )
                    
                    # Cumulative returns
                    cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
                    
                    # Scale by starting price
                    price_paths = current_price * cumulative_returns
                    
                    # Calculate final prices and determine VaR threshold
                    final_prices = price_paths[:, -1]
                    final_returns = (final_prices - current_price) / current_price
                    
                    var_threshold = np.percentile(final_returns, (1 - confidence_level) * 100)
                    cvar_threshold = final_returns[final_returns <= var_threshold].mean()
                    
                    # Create the visualization
                    fig = go.Figure()
                    
                    # Plot a sample of paths
                    for i in range(min(100, n_sims)):
                        color = 'rgba(255,0,0,0.3)' if final_returns[i] <= var_threshold else 'rgba(0,0,255,0.15)'
                        fig.add_trace(go.Scatter(
                            y=price_paths[i],
                            mode='lines',
                            line=dict(color=color, width=1),
                            showlegend=False
                        ))
                    
                    # Add VaR threshold line
                    var_price = current_price * (1 + var_threshold)
                    fig.add_shape(
                        type="line",
                        x0=0, y0=var_price,
                        x1=n_days-1, y1=var_price,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    fig.add_annotation(
                        x=n_days*0.8, y=var_price,
                        text=f"VaR Threshold: ${var_price:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="red",
                        arrowsize=1,
                        arrowwidth=2
                    )
                    
                    fig.update_layout(
                        title=f"Monte Carlo Price Simulation ({n_sims} paths, {time_horizon} days)",
                        xaxis_title="Days",
                        yaxis_title="Stock Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram of final returns
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Histogram(
                        x=final_returns * 100,  # Convert to percentage
                        opacity=0.7,
                        nbinsx=50,
                        marker=dict(color="rgba(75, 120, 168, 0.6)")
                    ))
                    
                    # VaR line
                    fig2.add_shape(
                        type="line",
                        x0=var_threshold * 100, y0=0,
                        x1=var_threshold * 100, y1=50,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # CVaR line
                    fig2.add_shape(
                        type="line",
                        x0=cvar_threshold * 100, y0=0,
                        x1=cvar_threshold * 100, y1=50,
                        line=dict(color="darkred", width=2, dash="dot")
                    )
                    
                    fig2.update_layout(
                        title=f"Distribution of {time_horizon}-Day Returns",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Performance Metrics tab
        with risk_tab2:
            st.subheader("Performance and Risk Metrics")
            
            # Fetch market data for beta calculation
            market_data = get_market_data()
            
            if not market_data.empty:
                # Align dates
                # Fix: Convert set to list before using as DataFrame indexer
                common_dates = list(set(extended_data.index) & set(market_data.index))
                stock_returns = extended_data.loc[common_dates, 'Daily_Return']
                market_returns = market_data.loc[common_dates, 'Daily_Return']
                
                # Calculate risk-free rate for Sharpe ratio
                st.markdown("Select performance evaluation period:")
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    days_options = {
                        "1 Month": 30, 
                        "3 Months": 90, 
                        "6 Months": 180, 
                        "1 Year": 365, 
                        "2 Years": 730
                    }
                    selected_period = st.selectbox("Period:", list(days_options.keys()))
                    days = days_options[selected_period]
                
                with perf_col2:
                    custom_rf = st.checkbox("Custom Risk-Free Rate")
                    if custom_rf:
                        rf_for_sharpe = st.number_input("Annual Risk-Free Rate (%):", min_value=0.0, max_value=10.0, value=float(risk_free_rate*100), step=0.1) / 100
                    else:
                        rf_for_sharpe = risk_free_rate
                
                if st.button("Calculate Performance Metrics"):
                    perf_calc_status = st.empty()
                    perf_calc_status.info("Calculating performance metrics...")
                    
                    # Filter for selected period
                    end_date = extended_data.index[-1]
                    start_date = end_date - pd.Timedelta(days=days)
                    
                    period_returns = extended_data.loc[extended_data.index >= start_date, 'Daily_Return']
                    
                    # Calculate metrics
                    total_return = (extended_data.loc[extended_data.index >= start_date, 'Close'][-1] / 
                                  extended_data.loc[extended_data.index >= start_date, 'Close'][0] - 1) * 100
                    
                    annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
                    
                    annualized_volatility = period_returns.std() * np.sqrt(252) * 100
                    
                    # Calculate Sharpe ratio
                    sharpe = calculate_sharpe_ratio(period_returns, rf_for_sharpe)
                    
                    # Calculate Beta
                    # Fix: Convert set to list before using as DataFrame indexer
                    common_period_dates = list(set(period_returns.index) & set(market_returns.index))
                    period_stock_returns = stock_returns.loc[common_period_dates]
                    period_market_returns = market_returns.loc[common_period_dates]
                    
                    beta = calculate_beta(period_stock_returns, period_market_returns)
                    
                    # Maximum drawdown calculation
                    prices = extended_data.loc[extended_data.index >= start_date, 'Close']
                    peak = prices.expanding().max()
                    drawdown = (prices / peak - 1) * 100
                    max_drawdown = drawdown.min()
                    
                    # Display metrics
                    perf_calc_status.empty()
                    st.markdown(f"### Performance Metrics for {ticker} ({selected_period})")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    metric_col1.metric("Total Return", f"{total_return:.2f}%")
                    metric_col1.metric("Annualized Return", f"{annualized_return:.2f}%")
                    
                    metric_col2.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
                    metric_col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    metric_col3.metric("Beta", f"{beta:.2f}")
                    metric_col3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    
                    interpretation = f"""
                    - **Total Return**: {ticker} has returned {total_return:.2f}% over the past {selected_period.lower()}.
                    - **Annualized Return**: The annualized return is {annualized_return:.2f}%.
                    - **Volatility**: The annualized volatility is {annualized_volatility:.2f}%, indicating the level of risk.
                    - **Sharpe Ratio**: A Sharpe ratio of {sharpe:.2f} suggests 
                      {'good risk-adjusted returns' if sharpe > 1 else 'moderate to poor risk-adjusted returns'}.
                    - **Beta**: With a beta of {beta:.2f}, this stock is 
                      {'more volatile than' if beta > 1 else 'less volatile than' if beta < 1 else 'as volatile as'} 
                      the overall market.
                    - **Maximum Drawdown**: The largest price decline from peak to trough was {abs(max_drawdown):.2f}%.
                    """
                    
                    st.markdown(interpretation)
                    
                    # Risk-Return plot
                    st.markdown("### Risk-Return Analysis")
                    
                    # Find common tickers for comparison
                    comparison_tickers = {
                        "SPY": "S&P 500 ETF",
                        "QQQ": "Nasdaq ETF",
                        "AAPL": "Apple",
                        "MSFT": "Microsoft",
                        "GOOGL": "Google",
                        "AMZN": "Amazon",
                        "META": "Meta",
                        "TSLA": "Tesla",
                        "JPM": "JPMorgan",
                        "V": "Visa"
                    }
                    
                    # Remove the current ticker from comparison if it's in the list
                    if ticker in comparison_tickers:
                        comparison_tickers.pop(ticker)
                    
                    # Add current ticker with a different name format
                    comparison_tickers[ticker] = f"► {ticker} ◄"
                    
                    # Calculate returns and volatilities for all tickers
                    risk_return_data = []
                    
                    for comp_ticker, comp_name in comparison_tickers.items():
                        try:
                            comp_data = get_stock_data(comp_ticker, period='1y')
                            if not comp_data.empty:
                                comp_returns = comp_data['Daily_Return'].dropna()
                                
                                # Calculate metrics
                                comp_annual_return = ((comp_data['Close'][-1] / comp_data['Close'][0]) ** (365/len(comp_data)) - 1) * 100
                                comp_annual_vol = comp_returns.std() * np.sqrt(252) * 100
                                
                                # Calculate Sharpe
                                comp_sharpe = comp_annual_return / comp_annual_vol
                                
                                # Add to data
                                risk_return_data.append({
                                    'Ticker': comp_name,
                                    'Annualized Return (%)': comp_annual_return,
                                    'Annualized Volatility (%)': comp_annual_vol,
                                    'Sharpe Ratio': comp_sharpe,
                                    'Is Current': comp_ticker == ticker
                                })
                        except Exception as e:
                            st.warning(f"Could not fetch data for {comp_ticker}: {e}")
                    
                    # Create DataFrame
                    risk_return_df = pd.DataFrame(risk_return_data)
                    
                    if not risk_return_df.empty:
                        # Create scatter plot
                        fig = px.scatter(
                            risk_return_df,
                            x='Annualized Volatility (%)',
                            y='Annualized Return (%)',
                            text='Ticker',
                            size=[30 if is_current else 15 for is_current in risk_return_df['Is Current']],
                            color='Sharpe Ratio',
                            color_continuous_scale='RdYlGn',
                            hover_data={
                                'Ticker': True,
                                'Annualized Return (%)': ':.2f',
                                'Annualized Volatility (%)': ':.2f',
                                'Sharpe Ratio': ':.2f',
                                'Is Current': False
                            },
                            height=600
                        )
                        
                        # Add labels for all points
                        fig.update_traces(
                            textposition='top center',
                            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                        )
                        
                        # Add a diagonal line representing Sharpe ratio = 1
                        x_range = np.linspace(0, max(risk_return_df['Annualized Volatility (%)']) * 1.1, 100)
                        y_values = x_range * (rf_for_sharpe * 100)  # Sharpe = 1 line
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_values,
                            mode='lines',
                            line=dict(color='grey', width=1, dash='dash'),
                            name='Risk-Free Rate',
                            showlegend=True
                        ))
                        
                        fig.update_layout(
                            title="Risk-Return Comparison",
                            xaxis_title="Risk (Annualized Volatility %)",
                            yaxis_title="Return (Annualized %)",
                            coloraxis_colorbar=dict(title="Sharpe Ratio"),
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("Failed to fetch market data for comparison")
        
        # Historical Analysis tab
        with risk_tab3:
            st.subheader("Historical Price and Returns Analysis")
            
            # Plot historical prices
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=extended_data.index,
                y=extended_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='rgba(75, 120, 168, 1)')
            ))
            
            # Add volume as bar chart on secondary y-axis
            fig.add_trace(go.Bar(
                x=extended_data.index,
                y=extended_data['Volume'],
                name='Volume',
                marker=dict(color='rgba(200, 200, 200, 0.5)'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"{ticker} Historical Price and Volume",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                legend=dict(x=0, y=1.1, orientation='h'),
                height=500,
                yaxis2=dict(
                    title="Volume",
                    titlefont=dict(color='gray'),
                    tickfont=dict(color='gray'),
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling volatility analysis
            st.subheader("Volatility Analysis")
            
            vol_col1, vol_col2 = st.columns(2)
            
            with vol_col1:
                volatility_window = st.slider("Rolling Window (days):", min_value=5, max_value=252, value=21, step=5)
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=volatility_window).std() * np.sqrt(252) * 100
            
            # Plot rolling volatility
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name=f'{volatility_window}-Day Rolling Volatility',
                line=dict(color='rgba(200, 80, 80, 0.8)')
            ))
            
            fig.update_layout(
                title=f"{ticker} {volatility_window}-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Return distribution
            st.subheader("Return Distribution")
            
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                return_type = st.radio("Return Type:", ["Daily", "Weekly", "Monthly"])
            
            # Calculate returns for selected frequency
            if return_type == "Daily":
                plot_returns = returns
                title_freq = "Daily"
            elif return_type == "Weekly":
                plot_returns = extended_data['Close'].resample('W').last().pct_change().dropna()
                title_freq = "Weekly"
            else:  # Monthly
                plot_returns = extended_data['Close'].resample('M').last().pct_change().dropna()
                title_freq = "Monthly"
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=plot_returns * 100,  # Convert to percentage
                opacity=0.7,
                nbinsx=30,
                marker=dict(color="rgba(75, 120, 168, 0.6)")
            ))
            
            # Add normal distribution curve for comparison
            x_range = np.linspace(min(plot_returns * 100), max(plot_returns * 100), 100)
            y_values = stats.norm.pdf(x_range, np.mean(plot_returns * 100), np.std(plot_returns * 100))
            
            # Scale the normal curve to match histogram height
            hist_values, _ = np.histogram(plot_returns * 100, bins=30)
            scaling_factor = max(hist_values) / max(y_values)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_values * scaling_factor,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ))
            
            # Calculate statistics for annotation
            mean_return = np.mean(plot_returns * 100)
            median_return = np.median(plot_returns * 100)
            std_return = np.std(plot_returns * 100)
            skewness = stats.skew(plot_returns)
            kurtosis = stats.kurtosis(plot_returns)
            
            stats_text = (
                f"Mean: {mean_return:.2f}%<br>"
                f"Median: {median_return:.2f}%<br>"
                f"Std Dev: {std_return:.2f}%<br>"
                f"Skewness: {skewness:.2f}<br>"
                f"Kurtosis: {kurtosis:.2f}"
            )
            
            fig.add_annotation(
                x=0.85,
                y=0.9,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
            
            fig.update_layout(
                title=f"{ticker} {title_freq} Return Distribution",
                xaxis_title=f"{title_freq} Return (%)",
                yaxis_title="Frequency",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Market Data
with tab3:
    st.header("Market Data")
    st.markdown("View available market data and option chains.")
    
    # Fetch option chain data
    option_data_status = st.empty()
    option_data_status.info("Fetching option chain data...")
    
    calls, puts = get_option_chain(ticker)
    option_data_status.empty()
    
    if calls.empty or puts.empty:
        st.warning(f"No option chain data available for {ticker}")
    else:
        # Display summary of available options
        expirations = sorted(calls['expirationDate'].unique())
        
        st.markdown(f"### Options Data for {ticker}")
        st.markdown(f"Current stock price: **${current_price:.2f}**")
        st.markdown(f"Available expiration dates: **{len(expirations)}**")
        
        # Select expiration date
        selected_expiry = st.selectbox("Select Expiration Date:", expirations)
        
        # Filter for selected expiration
        filtered_calls = calls[calls['expirationDate'] == selected_expiry].sort_values(by='strike')
        filtered_puts = puts[puts['expirationDate'] == selected_expiry].sort_values(by='strike')
        
        # Calculate days to expiration
        days_to_expiry = (pd.to_datetime(selected_expiry) - pd.to_datetime('today')).days
        st.markdown(f"Days to expiration: **{days_to_expiry}**")
        
        # Organize in tabs
        option_tab1, option_tab2, option_tab3 = st.tabs(["Option Chain", "Visualize", "Export"])
        
        # Option Chain tab
        with option_tab1:
            st.markdown("### Option Chain")
            
            # Add filtering capabilities
            min_strike = min(min(filtered_calls['strike']), min(filtered_puts['strike']))
            max_strike = max(max(filtered_calls['strike']), max(filtered_puts['strike']))
            
            # Create strike range slider
            selected_strike_range = st.slider(
                "Strike Price Range:",
                min_value=float(min_strike),
                max_value=float(max_strike),
                value=(float(max(min_strike, current_price * 0.8)), float(min(max_strike, current_price * 1.2))),
                step=1.0
            )
            
            # Filter by strike range
            filtered_calls = filtered_calls[
                (filtered_calls['strike'] >= selected_strike_range[0]) & 
                (filtered_calls['strike'] <= selected_strike_range[1])
            ]
            
            filtered_puts = filtered_puts[
                (filtered_puts['strike'] >= selected_strike_range[0]) & 
                (filtered_puts['strike'] <= selected_strike_range[1])
            ]
            
            # Show option chain in two columns
            call_col, put_col = st.columns(2)
            
            with call_col:
                st.markdown('<div class="option-column">', unsafe_allow_html=True)
                st.markdown('<h4 style="letter-spacing: 0.5px; color: white;">Call Options</h4>', unsafe_allow_html=True)
                
                # Select columns to display
                display_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
                
                # Format implied volatility
                formatted_calls = filtered_calls[display_columns].copy()
                formatted_calls['impliedVolatility'] = formatted_calls['impliedVolatility'] * 100
                
                # Highlight in-the-money options
                def highlight_itm_calls(row):
                    if row['strike'] < current_price:
                        return ['background-color: rgba(0, 128, 0, 0.4)'] * len(row)
                    return ['background-color: rgba(30, 30, 30, 0.5)'] * len(row)
                
                styled_calls = formatted_calls.style.apply(highlight_itm_calls, axis=1)
                
                # Rename columns for display
                styled_calls = styled_calls.format({
                    'strike': '${:.2f}', 
                    'lastPrice': '${:.2f}', 
                    'bid': '${:.2f}', 
                    'ask': '${:.2f}',
                    'impliedVolatility': '{:.2f}%'
                })
                
                st.markdown('<div class="call-options-table">', unsafe_allow_html=True)
                st.dataframe(styled_calls, height=500)  # Increased height
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True) 
            
            with put_col:
                st.markdown('<div class="option-column">', unsafe_allow_html=True)
                st.markdown('<h4 style="letter-spacing: 0.5px; color: white;">Put Options</h4>', unsafe_allow_html=True)
                
                # Format implied volatility
                formatted_puts = filtered_puts[display_columns].copy()
                formatted_puts['impliedVolatility'] = formatted_puts['impliedVolatility'] * 100
                
                # Highlight in-the-money options
                def highlight_itm_puts(row):
                    if row['strike'] > current_price:
                        return ['background-color: rgba(128, 0, 0, 0.4)'] * len(row)
                    return ['background-color: rgba(30, 30, 30, 0.5)'] * len(row)
                
                styled_puts = formatted_puts.style.apply(highlight_itm_puts, axis=1)
                
                # Rename columns for display
                styled_puts = styled_puts.format({
                    'strike': '${:.2f}', 
                    'lastPrice': '${:.2f}', 
                    'bid': '${:.2f}', 
                    'ask': '${:.2f}',
                    'impliedVolatility': '{:.2f}%'
                })
                
                st.markdown('<div class="put-options-table">', unsafe_allow_html=True)
                st.dataframe(styled_puts, height=500)  # Increased height
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)  # Close option-column div
                st.markdown('</div>', unsafe_allow_html=True) 
        
        # Visualize tab
        with option_tab2:
            st.markdown("### Options Visualization")
            
            # Option to select visualization type
            viz_type = st.radio("Visualization Type:", ["Option Prices", "Implied Volatility", "Greeks"])
            
            if viz_type == "Option Prices":
                # Create combined DataFrame for visualization
                combined_data = []
                
                for _, row in filtered_calls.iterrows():
                    combined_data.append({
                        'Strike': row['strike'],
                        'Option Type': 'Call',
                        'Last Price': row['lastPrice'],
                        'Bid': row['bid'],
                        'Ask': row['ask'],
                        'Implied Volatility': row['impliedVolatility'] * 100,
                        'Volume': row['volume'],
                        'Open Interest': row['openInterest']
                    })
                
                for _, row in filtered_puts.iterrows():
                    combined_data.append({
                        'Strike': row['strike'],
                        'Option Type': 'Put',
                        'Last Price': row['lastPrice'],
                        'Bid': row['bid'],
                        'Ask': row['ask'],
                        'Implied Volatility': row['impliedVolatility'] * 100,
                        'Volume': row['volume'],
                        'Open Interest': row['openInterest']
                    })
                
                combined_df = pd.DataFrame(combined_data)
                
                # Create price chart
                fig = go.Figure()
                
                # Add call prices
                call_data = combined_df[combined_df['Option Type'] == 'Call']
                fig.add_trace(go.Scatter(
                    x=call_data['Strike'],
                    y=call_data['Last Price'],
                    mode='lines+markers',
                    name='Call Price',
                    marker=dict(color='green', size=8),
                    line=dict(color='green', width=2)
                ))
                
                # Add put prices
                put_data = combined_df[combined_df['Option Type'] == 'Put']
                fig.add_trace(go.Scatter(
                    x=put_data['Strike'],
                    y=put_data['Last Price'],
                    mode='lines+markers',
                    name='Put Price',
                    marker=dict(color='red', size=8),
                    line=dict(color='red', width=2)
                ))
                
                # Add current stock price line
                fig.add_shape(
                    type="line",
                    x0=current_price, y0=0,
                    x1=current_price, y1=max(combined_df['Last Price']) * 1.2,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                fig.add_annotation(
                    x=current_price,
                    y=max(combined_df['Last Price']) * 1.1,
                    text=f"Current Price: ${current_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="black",
                    arrowsize=1,
                    arrowwidth=2
                )
                
                fig.update_layout(
                    title=f"{ticker} Option Prices - Expiry: {selected_expiry}",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Option Price ($)",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add volume/open interest chart
                st.markdown("### Volume and Open Interest")
                
                fig = go.Figure()
                
                # Add call volume
                fig.add_trace(go.Bar(
                    x=call_data['Strike'],
                    y=call_data['Volume'],
                    name='Call Volume',
                    marker=dict(color='rgba(0, 128, 0, 0.5)')
                ))
                
                # Add put volume
                fig.add_trace(go.Bar(
                    x=put_data['Strike'],
                    y=put_data['Volume'],
                    name='Put Volume',
                    marker=dict(color='rgba(255, 0, 0, 0.5)')
                ))
                
                # Add call OI as line
                fig.add_trace(go.Scatter(
                    x=call_data['Strike'],
                    y=call_data['Open Interest'],
                    mode='lines+markers',
                    name='Call Open Interest',
                    marker=dict(color='green', size=6),
                    line=dict(color='green', width=2)
                ))
                
                # Add put OI as line
                fig.add_trace(go.Scatter(
                    x=put_data['Strike'],
                    y=put_data['Open Interest'],
                    mode='lines+markers',
                    name='Put Open Interest',
                    marker=dict(color='red', size=6),
                    line=dict(color='red', width=2)
                ))
                
                # Add current stock price line
                fig.add_shape(
                    type="line",
                    x0=current_price, y0=0,
                    x1=current_price, y1=max(
                        max(call_data['Volume'].max(), put_data['Volume'].max()),
                        max(call_data['Open Interest'].max(), put_data['Open Interest'].max())
                    ) * 1.2,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                fig.update_layout(
                    title=f"{ticker} Option Volume & Open Interest - Expiry: {selected_expiry}",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Contracts",
                    height=500,
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Implied Volatility":
                # Create volatility smile chart
                vol_fig = go.Figure()
                
                # Filter out any zero or NaN IV values
                filtered_calls_iv = filtered_calls[filtered_calls['impliedVolatility'] > 0].copy()
                filtered_puts_iv = filtered_puts[filtered_puts['impliedVolatility'] > 0].copy()
                
                # Moneyness calculation (strike/current_price)
                filtered_calls_iv['moneyness'] = filtered_calls_iv['strike'] / current_price
                filtered_puts_iv['moneyness'] = filtered_puts_iv['strike'] / current_price
                
                # Add call IVs
                vol_fig.add_trace(go.Scatter(
                    x=filtered_calls_iv['moneyness'],
                    y=filtered_calls_iv['impliedVolatility'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Call IV',
                    marker=dict(color='green', size=8),
                    line=dict(color='green', width=2)
                ))
                
                # Add put IVs
                vol_fig.add_trace(go.Scatter(
                    x=filtered_puts_iv['moneyness'],
                    y=filtered_puts_iv['impliedVolatility'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Put IV',
                    marker=dict(color='red', size=8),
                    line=dict(color='red', width=2)
                ))
                
                # Add vertical line at moneyness = 1 (at-the-money)
                vol_fig.add_shape(
                    type="line",
                    x0=1, y0=0,
                    x1=1, y1=max(
                        filtered_calls_iv['impliedVolatility'].max() * 100,
                        filtered_puts_iv['impliedVolatility'].max() * 100
                    ) * 1.2,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                vol_fig.add_annotation(
                    x=1,
                    y=max(
                        filtered_calls_iv['impliedVolatility'].max() * 100,
                        filtered_puts_iv['impliedVolatility'].max() * 100
                    ) * 1.1,
                    text="At-the-Money",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="black",
                    arrowsize=1,
                    arrowwidth=2
                )
                
                vol_fig.update_layout(
                    title=f"{ticker} Implied Volatility Smile - Expiry: {selected_expiry}",
                    xaxis_title="Moneyness (Strike/Current Price)",
                    yaxis_title="Implied Volatility (%)",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(vol_fig, use_container_width=True)
                
                # Add interpretation
                st.markdown("### Volatility Smile Interpretation")
                
                # Calculate skew
                if not filtered_calls_iv.empty and not filtered_puts_iv.empty:
                    otm_calls = filtered_calls_iv[filtered_calls_iv['moneyness'] > 1.05]
                    otm_puts = filtered_puts_iv[filtered_puts_iv['moneyness'] < 0.95]
                    
                    if not otm_calls.empty and not otm_puts.empty:
                        avg_otm_call_iv = otm_calls['impliedVolatility'].mean() * 100
                        avg_otm_put_iv = otm_puts['impliedVolatility'].mean() * 100
                        
                        # Calculate at-the-money IV
                        atm_options = pd.concat([
                            filtered_calls_iv[(filtered_calls_iv['moneyness'] > 0.97) & (filtered_calls_iv['moneyness'] < 1.03)],
                            filtered_puts_iv[(filtered_puts_iv['moneyness'] > 0.97) & (filtered_puts_iv['moneyness'] < 1.03)]
                        ])
                        
                        if not atm_options.empty:
                            avg_atm_iv = atm_options['impliedVolatility'].mean() * 100
                            
                            # Calculate skew
                            put_skew = avg_otm_put_iv - avg_atm_iv
                            call_skew = avg_otm_call_iv - avg_atm_iv
                            
                            skew_text = f"""
                            ### Volatility Skew Analysis
                            
                            - **At-the-Money IV**: {avg_atm_iv:.2f}%
                            - **OTM Put IV (avg)**: {avg_otm_put_iv:.2f}%
                            - **OTM Call IV (avg)**: {avg_otm_call_iv:.2f}%
                            - **Put Skew**: {put_skew:.2f}% ({'Positive' if put_skew > 0 else 'Negative'})
                            - **Call Skew**: {call_skew:.2f}% ({'Positive' if call_skew > 0 else 'Negative'})
                            
                            #### Interpretation:
                            
                            """
                            
                            if put_skew > 3 and call_skew > 0:
                                skew_text += "The volatility smile shows a **pronounced skew toward OTM puts**, indicating the market is pricing in a higher probability of downside moves. This is typical during periods of market stress or when investors are seeking downside protection."
                            elif put_skew > 0 and call_skew > 0:
                                skew_text += "The volatility curve shows a **typical smile pattern** with both OTM puts and calls trading at premium volatilities compared to ATM options. This suggests the market expects potential moves in either direction but is slightly more concerned about downside risk."
                            elif put_skew > 0 and call_skew < 0:
                                skew_text += "The volatility curve shows a **traditional skew** with higher volatilities for downside protection (puts) and lower volatilities for upside calls. This is the most common pattern in equity markets."
                            elif put_skew < 0 and call_skew > 0:
                                skew_text += "The volatility curve shows a **reverse skew** with higher volatilities for OTM calls than puts. This is unusual in equity markets and may suggest expectations of a strong upside move or short squeeze."
                            elif put_skew < 0 and call_skew < 0:
                                skew_text += "Both OTM puts and calls are trading at **lower implied volatilities** than ATM options. This unusual pattern may suggest expectations of range-bound trading."
                            
                            st.markdown(skew_text)
                        else:
                            st.markdown("Not enough data near the money to calculate skew metrics.")
                    else:
                        st.markdown("Not enough OTM options to calculate skew metrics.")
                else:
                    st.markdown("Insufficient IV data to analyze the volatility smile.")
            
            elif viz_type == "Greeks":
                st.markdown("### Option Greeks Analysis")
                
                # Calculate theoretical greeks for the options
                greeks_data = []
                
                for _, row in filtered_calls.iterrows():
                    # Calculate days to expiration
                    expiry_date = pd.to_datetime(row['expirationDate'])
                    days = (expiry_date - pd.to_datetime('today')).days
                    t = days / 365.0
                    
                    # Calculate Black-Scholes Greeks
                    bs_greeks = black_scholes_greeks(
                        current_price, row['strike'], t, 
                        risk_free_rate, row['impliedVolatility'], 'call'
                    )
                    
                    greeks_data.append({
                        'Strike': row['strike'],
                        'Option Type': 'Call',
                        'Delta': bs_greeks['delta'],
                        'Gamma': bs_greeks['gamma'],
                        'Theta': bs_greeks['theta'],
                        'Vega': bs_greeks['vega'],
                        'Implied Volatility': row['impliedVolatility'] * 100
                    })
                
                for _, row in filtered_puts.iterrows():
                    # Calculate days to expiration
                    expiry_date = pd.to_datetime(row['expirationDate'])
                    days = (expiry_date - pd.to_datetime('today')).days
                    t = days / 365.0
                    
                    # Calculate Black-Scholes Greeks
                    bs_greeks = black_scholes_greeks(
                        current_price, row['strike'], t, 
                        risk_free_rate, row['impliedVolatility'], 'put'
                    )
                    
                    greeks_data.append({
                        'Strike': row['strike'],
                        'Option Type': 'Put',
                        'Delta': bs_greeks['delta'],
                        'Gamma': bs_greeks['gamma'],
                        'Theta': bs_greeks['theta'],
                        'Vega': bs_greeks['vega'],
                        'Implied Volatility': row['impliedVolatility'] * 100
                    })
                
                greeks_df = pd.DataFrame(greeks_data)
                
                # Select which greek to visualize
                selected_greek = st.selectbox("Select Greek:", ["Delta", "Gamma", "Theta", "Vega"])
                
                # Create the visualization
                greek_fig = go.Figure()
                
                # Filter data
                calls_greek = greeks_df[greeks_df['Option Type'] == 'Call']
                puts_greek = greeks_df[greeks_df['Option Type'] == 'Put']
                
                # Add call data
                greek_fig.add_trace(go.Scatter(
                    x=calls_greek['Strike'],
                    y=calls_greek[selected_greek],
                    mode='lines+markers',
                    name=f'Call {selected_greek}',
                    marker=dict(color='green', size=8),
                    line=dict(color='green', width=2)
                ))
                
                # Add put data
                greek_fig.add_trace(go.Scatter(
                    x=puts_greek['Strike'],
                    y=puts_greek[selected_greek],
                    mode='lines+markers',
                    name=f'Put {selected_greek}',
                    marker=dict(color='red', size=8),
                    line=dict(color='red', width=2)
                ))
                
                # Add current stock price line
                greek_fig.add_shape(
                    type="line",
                    x0=current_price, y0=min(
                        calls_greek[selected_greek].min(),
                        puts_greek[selected_greek].min()
                    ) * 1.2,
                    x1=current_price, y1=max(
                        calls_greek[selected_greek].max(),
                        puts_greek[selected_greek].max()
                    ) * 1.2,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                # Update layout
                greek_fig.update_layout(
                    title=f"{ticker} Option {selected_greek} - Expiry: {selected_expiry}",
                    xaxis_title="Strike Price ($)",
                    yaxis_title=selected_greek,
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(greek_fig, use_container_width=True)
                
                # Greek interpretation
                interpretation = {
                    "Delta": """
                        **Delta** represents the rate of change of the option price with respect to the underlying asset's price.
                        
                        - Call deltas range from 0 to 1, with ATM options having deltas around 0.5
                        - Put deltas range from -1 to 0, with ATM options having deltas around -0.5
                        - Delta can be interpreted as an approximate probability of the option expiring in-the-money
                        - Delta is also used for hedge ratios - a delta of 0.5 means you need 50 shares to delta-hedge one call option
                    """,
                    
                    "Gamma": """
                        **Gamma** measures the rate of change of delta with respect to the underlying asset's price.
                        
                        - Gamma is highest for at-the-money options and decreases for in or out-of-the-money options
                        - High gamma means the option's delta will change rapidly with small moves in the underlying
                        - Gamma is positive for both calls and puts
                        - Options with high gamma require more frequent rebalancing for delta-hedging
                    """,
                    
                    "Theta": """
                        **Theta** represents the rate of change of the option price with respect to time (time decay).
                        
                        - Theta is typically negative for both calls and puts (options lose value as time passes)
                        - Theta is most negative for at-the-money options near expiration
                        - Theta is expressed as the dollar value the option will lose per day
                        - Option sellers benefit from theta decay, while buyers are negatively affected
                    """,
                    
                    "Vega": """
                        **Vega** measures the rate of change of the option price with respect to the volatility of the underlying asset.
                        
                        - Vega is highest for at-the-money options with more time to expiration
                        - Vega is positive for both calls and puts (higher volatility increases option prices)
                        - Vega is expressed as the dollar value change for a 1% change in implied volatility
                        - Long options (both calls and puts) benefit from increasing volatility
                    """
                }
                
                st.markdown("### Greek Interpretation")
                st.markdown(interpretation[selected_greek])
        
        # Export tab
        with option_tab3:
            st.markdown("### Export Option Data")
            st.markdown("Download option chain data for further analysis.")
            
            # Combine data into one DataFrame for export
            export_calls = filtered_calls.copy()
            export_calls['optionType'] = 'call'
            
            export_puts = filtered_puts.copy()
            export_puts['optionType'] = 'put'
            
            export_data = pd.concat([export_calls, export_puts])
            
            # Convert to CSV
            csv = export_data.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download Option Chain CSV",
                data=csv,
                file_name=f"{ticker}_options_{selected_expiry}.csv",
                mime="text/csv"
            )
            
            # Show data preview
            st.markdown("#### Data Preview")
            st.dataframe(export_data.head(10))

# Tab 4: About the app
# Tab 4: About the app
with tab4:
    st.header("About This App")
    
    about_tab1, about_tab2, about_tab3, about_tab4 = st.tabs(["Overview", "Option Pricing Models", "Greeks", "Risk Metrics"])
    
    with about_tab1:
        st.markdown("""
        ### Options Pricing & Risk Analysis Tool
        
        This app provides a comprehensive set of tools for pricing options and analyzing financial risks. It's built using Python and Streamlit, with real-time data from Yahoo Finance.
        
        #### What are Options?
        
        Options are financial derivatives that give the buyer the right, but not the obligation, to buy (call option) or sell (put option) an underlying asset at a specified price (strike price) on or before a specified date (expiration date).
        
        **Key Terms:**
        - **Call Option**: Right to buy the underlying asset
        - **Put Option**: Right to sell the underlying asset
        - **Strike Price**: The price at which the option can be exercised
        - **Expiration Date**: When the option contract ends
        - **Premium**: The price paid to acquire an option
        
        Options are used for various purposes including:
        - **Hedging**: Protecting against adverse price movements
        - **Speculation**: Betting on future price movements with leverage
        - **Income Generation**: Collecting premium by selling options
        
        #### Features of This App:
        
        - **Option Pricing Models**: Determine the fair value of options using various mathematical models
        - **Risk Analysis Tools**: Assess potential risks and performance metrics
        - **Market Data**: Analyze real-time option chains and market conditions
        
        #### Technical Implementation:
        
        - Built with Python and Streamlit
        - Data fetched via Yahoo Finance API
        - Mathematical modeling with NumPy, SciPy, and custom implementations
        
        #### Disclaimer:
        
        This tool is for educational and informational purposes only. It is not intended to provide investment advice. Always consult with a qualified financial advisor before making investment decisions.
        """)
    
    with about_tab2:
        st.markdown("""
        ## Option Pricing Models
        
        This app implements three major option pricing models:
        
        ### 1. Black-Scholes Model
        
        The Black-Scholes model is the foundation of options pricing theory, developed by Fischer Black and Myron Scholes in 1973. It provides a closed-form solution for European options on non-dividend paying stocks.
        
        **Intuitive Idea**:
        The model conceptualizes an option's value based on the idea that the price of the underlying asset follows a geometric Brownian motion (a continuous-time random process). By constructing a risk-free portfolio that combines the option and the underlying asset, the model derives the fair price of the option.
        
        **Key Assumptions**:
        - Markets are efficient (no arbitrage opportunities)
        - No transaction costs or taxes
        - Risk-free interest rate is constant
        - Stock follows a lognormal distribution (returns are normally distributed)
        - No dividends during the option's life
        - European-style options (exercise only at expiration)
        
        **Mathematical Formula**:
        
        For a call option:
        $$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$
        
        For a put option:
        $$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$
        
        Where:
        $$d_1 = \\frac{\\ln(\\frac{S_0}{K}) + (r + \\frac{\\sigma^2}{2})T}{\\sigma\\sqrt{T}}$$
        
        $$d_2 = d_1 - \\sigma\\sqrt{T}$$
        
        - $S_0$ = Current stock price
        - $K$ = Strike price
        - $r$ = Risk-free interest rate
        - $T$ = Time to maturity (in years)
        - $\\sigma$ = Volatility of the underlying asset
        - $N(x)$ = Cumulative distribution function of the standard normal distribution
        
        **Limitations**:
        - Cannot price American options (early exercise)
        - Assumes constant volatility
        - Does not account for dividends in its basic form
        
        ### 2. Monte Carlo Simulation
        
        **Intuitive Idea**:
        Monte Carlo simulation uses random sampling to estimate the price of options. It simulates thousands of possible price paths for the underlying asset and calculates the average payoff of the option across all these scenarios, then discounts this back to present value.
        
        **How It Works**:
        1. Generate many random price paths for the underlying asset
        2. Calculate the option payoff at expiration for each path
        3. Average these payoffs and discount to present value
        
        **Mathematical Implementation**:
        
        For each simulated path:
        $$S_{t+\\Delta t} = S_t \\exp\\left((r - \\frac{\\sigma^2}{2})\\Delta t + \\sigma\\sqrt{\\Delta t}Z\\right)$$
        
        Where:
        - $S_t$ = Stock price at time $t$
        - $\\Delta t$ = Small time increment
        - $Z$ = Random draw from standard normal distribution
        
        The option price is then:
        $$C = e^{-rT} \\frac{1}{N}\\sum_{i=1}^{N} \\max(S_T^i - K, 0)$$
        
        **Advantages**:
        - Flexible - can handle complex path-dependent options
        - Can incorporate various stochastic processes
        - Easy to understand conceptually
        
        **Limitations**:
        - Computationally intensive
        - Results have statistical error
        - Convergence can be slow
        
        ### 3. Binomial Tree Model
        
        **Intuitive Idea**:
        The binomial model breaks down the time to expiration into discrete intervals. At each interval, the stock price can move up or down by certain factors, creating a "tree" of possible future stock prices. By working backward from the option's value at expiration for each final stock price, we can determine the option's current value.
        
        **How It Works**:
        1. Build a tree of possible stock prices moving forward in time
        2. Calculate option values at expiration for each final stock price
        3. Work backward through the tree, calculating option values at each node
        
        **Mathematical Implementation**:
        
        The stock price moves with up factor $u$ and down factor $d$:
        $$u = e^{\\sigma\\sqrt{\\Delta t}}$$
        $$d = e^{-\\sigma\\sqrt{\\Delta t}} = \\frac{1}{u}$$
        
        The risk-neutral probability of an up move is:
        $$p = \\frac{e^{r\\Delta t} - d}{u - d}$$
        
        For a European option, the value at each node is:
        $$V_{i,j} = e^{-r\\Delta t}(pV_{i+1,j+1} + (1-p)V_{i+1,j})$$
        
        For an American option, we also check for early exercise:
        $$V_{i,j} = \\max(e^{-r\\Delta t}(pV_{i+1,j+1} + (1-p)V_{i+1,j}), \\text{Intrinsic Value})$$
        
        **Advantages**:
        - Can price both European and American options
        - Intuitive and easy to visualize
        - Can handle early exercise and dividends
        
        **Limitations**:
        - Requires many steps for accuracy
        - Grows computationally expensive with more steps
        - Simplified model of stock price movement
        """)
    
    with about_tab3:
        st.markdown("""
        ## Option Greeks
        
        "Greeks" are sensitivity measures that describe how option prices change with respect to various factors. They're named after Greek letters and are essential tools for risk management.
        
        ### Delta (Δ)
        
        **Intuitive Explanation**: Delta measures how much an option's price changes when the underlying stock price changes by $1.
        
        **Mathematical Formula**:
        
        For call options:
        $$\\Delta_{call} = N(d_1)$$
        
        For put options:
        $$\\Delta_{put} = N(d_1) - 1$$
        
        **Practical Interpretation**:
        - Call delta ranges from 0 to 1
        - Put delta ranges from -1 to 0
        - Delta also represents the approximate probability of an option expiring in-the-money
        - Used for hedging: Delta = 0.5 means you need 50 shares to hedge 1 call option
        
        ### Gamma (Γ)
        
        **Intuitive Explanation**: Gamma measures how quickly delta changes as the stock price changes. It's the "delta of delta."
        
        **Mathematical Formula**:
        $$\\Gamma = \\frac{N'(d_1)}{S_0 \\sigma \\sqrt{T}}$$
        
        Where $N'(d_1)$ is the standard normal probability density function.
        
        **Practical Interpretation**:
        - Same for calls and puts (always positive)
        - Highest for at-the-money options close to expiration
        - High gamma means delta changes rapidly with small price movements
        - Options with high gamma require more frequent rebalancing
        
        ### Theta (Θ)
        
        **Intuitive Explanation**: Theta measures how much an option's price changes as time passes (time decay).
        
        **Mathematical Formula**:
        
        For call options:
        $$\\Theta_{call} = -\\frac{S_0 N'(d_1) \\sigma}{2\\sqrt{T}} - rKe^{-rT}N(d_2)$$
        
        For put options:
        $$\\Theta_{put} = -\\frac{S_0 N'(d_1) \\sigma}{2\\sqrt{T}} + rKe^{-rT}N(-d_2)$$
        
        **Practical Interpretation**:
        - Usually negative (options lose value over time)
        - Highest for at-the-money options close to expiration
        - Option sellers benefit from theta decay
        - Often expressed as daily decay (annual theta divided by 365)
        
        ### Vega
        
        **Intuitive Explanation**: Vega measures how much an option's price changes when volatility changes by 1%.
        
        **Mathematical Formula**:
        $$\\text{Vega} = S_0 \\sqrt{T} N'(d_1)$$
        
        **Practical Interpretation**:
        - Same for calls and puts (always positive)
        - Highest for at-the-money options with longer time to expiration
        - Higher volatility increases option prices
        - Long options benefit from increasing volatility
        
        ### Rho (ρ)
        
        **Intuitive Explanation**: Rho measures how much an option's price changes when the risk-free interest rate changes by 1%.
        
        **Mathematical Formula**:
        
        For call options:
        $$\\rho_{call} = KTe^{-rT}N(d_2)$$
        
        For put options:
        $$\\rho_{put} = -KTe^{-rT}N(-d_2)$$
        
        **Practical Interpretation**:
        - Usually positive for calls, negative for puts
        - Most significant for long-term options
        - Often the least monitored Greek due to relative stability of interest rates
        """)
    
    with about_tab4:
        st.markdown("""
        ## Risk Metrics
        
        Risk metrics help quantify potential losses and evaluate the risk-adjusted performance of investments.
        
        ### Value at Risk (VaR)
        
        **Intuitive Explanation**: VaR estimates the maximum potential loss over a specific time period at a given confidence level. For example, a 1-day 95% VaR of $1,000 means there's a 95% chance that losses won't exceed $1,000 over the next day.
        
        **Mathematical Approaches**:
        
        1. **Historical VaR**:
           - Sort historical returns
           - Find the return at the specified percentile (e.g., 5th percentile for 95% confidence)
           - VaR = Investment Amount × Return at percentile
        
        2. **Parametric VaR** (using normal distribution):
           $$\\text{VaR} = \\text{Investment Amount} × (\\mu - z_{\\alpha} × \\sigma)$$
           Where:
           - $\\mu$ = Expected return
           - $\\sigma$ = Standard deviation of returns
           - $z_{\\alpha}$ = Z-score for confidence level (e.g., 1.645 for 95%)
        
        3. **Monte Carlo VaR**:
           - Simulate many possible return scenarios
           - Find the return at the specified percentile
           - VaR = Investment Amount × Return at percentile
        
        **Multi-day VaR** can be approximated using the "square root of time" rule:
        $$\\text{VaR}_T = \\text{VaR}_1 × \\sqrt{T}$$
        
        ### Conditional Value at Risk (CVaR)
        
        **Intuitive Explanation**: Also known as Expected Shortfall, CVaR measures the expected loss given that the loss exceeds VaR. It answers the question: "If we have a really bad day (worse than VaR), how bad would it be on average?"
        
        **Mathematical Formula**:
        $$\\text{CVaR}_{\\alpha} = E[X | X ≤ \\text{VaR}_{\\alpha}]$$
        
        Where:
        - $X$ = Random variable representing returns
        - $\\text{VaR}_{\\alpha}$ = Value at Risk at confidence level $\\alpha$
        
        ### Sharpe Ratio
        
        **Intuitive Explanation**: The Sharpe ratio measures risk-adjusted return by calculating excess return (over risk-free rate) per unit of risk (standard deviation). Higher is better.
        
        **Mathematical Formula**:
        $$\\text{Sharpe Ratio} = \\frac{R_p - R_f}{\\sigma_p}$$
        
        Where:
        - $R_p$ = Portfolio return
        - $R_f$ = Risk-free return
        - $\\sigma_p$ = Portfolio standard deviation
        
        **Interpretation**:
        - Sharpe Ratio > 1: Good
        - Sharpe Ratio > 2: Very good
        - Sharpe Ratio > 3: Excellent
        
        ### Beta (β)
        
        **Intuitive Explanation**: Beta measures an asset's volatility or systematic risk relative to the overall market. A beta of 1 means the asset moves with the market. Higher beta means more volatility.
        
        **Mathematical Formula**:
        $$\\beta = \\frac{\\text{Cov}(r_i, r_m)}{\\text{Var}(r_m)}$$
        
        Where:
        - $r_i$ = Asset returns
        - $r_m$ = Market returns
        - Cov = Covariance
        - Var = Variance
        
        **Interpretation**:
        - β = 1: Moves with the market
        - β > 1: More volatile than the market
        - β < 1: Less volatile than the market
        - β < 0: Moves opposite to the market (rare)
        
        ### Maximum Drawdown
        
        **Intuitive Explanation**: Maximum drawdown measures the largest peak-to-trough decline in an asset's value, showing the worst-case scenario for an investor who bought at the peak and sold at the bottom.
        
        **Mathematical Formula**:
        $$\\text{MDD} = \\min_{t \\in (0,T)} \\left(\\frac{P_t - \\max_{s \\in (0,t)} P_s}{\\max_{s \\in (0,t)} P_s}\\right)$$
        
        Where:
        - $P_t$ = Price at time $t$
        
        **Interpretation**:
        - Smaller (less negative) is better
        - Important for understanding worst-case scenarios
        - Used to assess recovery time and drawdown risk
        
        ### Volatility
        
        **Intuitive Explanation**: Volatility measures the dispersion of returns for an asset, indicating how much its price fluctuates over time.
        
        **Types**:
        1. **Historical Volatility**: Standard deviation of past returns
        2. **Implied Volatility**: Derived from option prices in the market
        
        **Mathematical Formula** (Historical):
        $$\\sigma = \\sqrt{\\frac{\\sum_{i=1}^{n}(r_i - \\bar{r})^2}{n-1}}$$
        
        Where:
        - $r_i$ = Return in period $i$
        - $\\bar{r}$ = Average return
        - $n$ = Number of periods
        
        **Annualization**:
        $$\\sigma_{\\text{annual}} = \\sigma_{\\text{period}} × \\sqrt{\\text{periods per year}}$$
        
        For example, to annualize daily volatility:
        $$\\sigma_{\\text{annual}} = \\sigma_{\\text{daily}} × \\sqrt{252}$$
        """)
# Footer with new styling
st.markdown(
    """
    <div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #272727; text-align: center;">
        <p style="color: #F7F7F7; font-size: 0.9rem;">© 2025 Options Pricing & Risk Analysis Tool | Built with Python, Streamlit, and Yahoo Finance</p>
    </div>
    """, 
    unsafe_allow_html=True
)