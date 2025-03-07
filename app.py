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

st.set_page_config(
    page_title="Options Pricing & Risk Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
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
        color: white !important;
    }
    
    /* Better spacing for min/max values */
    .stSlider [data-baseweb] div[data-testid] {
        margin-top: 8px;
    }
    
    /* Fix slider thumb value positioning */
    .stSlider [data-testid="stThumbValue"] {
        position: absolute;
        background-color: #4162FF !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 3px 8px !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        top: -28px !important;
        transform: translateX(-50%);
        z-index: 100;
    }
    
    /* Table layout improvements for better visibility */
    .dataframe {
        font-size: 13px !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
    }
    
    /* Ensure scrolling works properly */
    [data-testid="stDataFrame"] > div {
        max-height: 500px !important; /* Increased height */
        overflow: auto !important;
    }
    
    /* Sticky header for tables */
    .dataframe thead th {
        position: sticky !important;
        top: 0 !important;
        z-index: 1 !important;
        background-color: #1E1E1E !important;
    }
    
    /* Better styling for call options */
    .call-options-table th {
        background-color: rgba(0, 100, 0, 0.4) !important;
        color: white !important;
        padding: 8px 4px !important;
        text-align: center !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #444 !important;
    }
    
    .call-options-table td {
        color: white !important;
        padding: 6px 4px !important;
        text-align: right !important;
        border-bottom: 1px solid #333 !important;
    }
    
    /* Better styling for put options */
    .put-options-table th {
        background-color: rgba(128, 0, 0, 0.4) !important;
        color: white !important;
        padding: 8px 4px !important;
        text-align: center !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #444 !important;
    }
    
    .put-options-table td {
        color: white !important;
        padding: 6px 4px !important;
        text-align: right !important;
        border-bottom: 1px solid #333 !important;
    }
    
    /* Ensure option chain columns are spaced better */
    .options-container {
        display: flex;
        gap: 20px;
    }
    
    .option-column {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Options Pricing & Risk Analysis Tool")
st.markdown("This app provides tools for pricing options using different models and analyzing financial risks.")

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
        option_type = st.radio("Option Type:", ["Call", "Put"])
        strike_price = st.number_input("Strike Price ($):", min_value=0.1, value=round(current_price, 1), step=1.0)
        maturity_days = st.number_input("Time to Maturity (days):", min_value=1, value=30)
        maturity = maturity_days / 365.0  # Convert to years
        volatility = st.slider("Volatility (%):", min_value=1.0, max_value=100.0, value=float(implied_vol*100), step=0.1) / 100
        rf_rate = st.slider("Risk-Free Rate (%):", min_value=0.0, max_value=10.0, value=float(risk_free_rate*100), step=0.1) / 100
    
    with col2:
        st.subheader("Model Settings")
        models_to_use = st.multiselect(
            "Select Models to Compare:",
            ["Black-Scholes", "Monte Carlo", "Binomial Tree"],
            default=["Black-Scholes", "Monte Carlo", "Binomial Tree"]
        )
        
        monte_carlo_sims = st.number_input("Monte Carlo Simulations:", min_value=1000, max_value=100000, value=10000, step=1000)
        binomial_steps = st.number_input("Binomial Tree Steps:", min_value=10, max_value=1000, value=100, step=10)
        
        st.markdown("---")
        st.markdown("### Stock Price")
        st.markdown(f"Current Price: **${current_price:.2f}**")
        
        # Option to manually override stock price
        override_price = st.checkbox("Override Stock Price")
        if override_price:
            current_price = st.number_input("Stock Price ($):", min_value=0.1, value=current_price, step=1.0)
    
    # Calculate option prices when button is clicked
    if st.button("Calculate Option Prices"):
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
                    
                    # Generate and plot price paths for visualization
                    if st.checkbox("Show Monte Carlo Paths", key="show_mc_paths"):
                        st.markdown("##### Price Path Simulation")
                        
                        time_points, paths = monte_carlo_path_generator(
                            current_price, maturity, rf_rate, volatility, simulations=50, steps=100
                        )
                        
                        # Create the plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for i in range(min(20, paths.shape[0])):  # Plot up to 20 paths
                            ax.plot(time_points, paths[i], linewidth=0.8, alpha=0.6)
                        
                        # Plot strike price line
                        ax.axhline(y=strike_price, color='r', linestyle='--', linewidth=1)
                        
                        # Plot the mean path
                        ax.plot(time_points, np.mean(paths, axis=0), color='black', linewidth=2)
                        
                        ax.set_xlabel('Time (years)')
                        ax.set_ylabel('Stock Price ($)')
                        ax.set_title('Monte Carlo Simulation Paths')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                
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
                    
                    # Visualize the tree
                    if st.checkbox("Show Binomial Tree", key="show_binomial"):
                        st.markdown("##### Binomial Tree Visualization (Simplified)")
                        
                        # Get tree data with reduced steps for visualization
                        vis_steps = min(10, binomial_steps)  # Limit for visualization
                        tree_data = get_binomial_tree_data(
                            current_price, strike_price, maturity, rf_rate, volatility,
                            option_type.lower(), vis_steps
                        )
                        
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
                            title="Binomial Tree (Simplified View)",
                            xaxis_title="Time (years)",
                            height=500,
                            showlegend=False,
                            hovermode='closest',
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
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
with tab4:
    st.header("About This App")
    
    st.markdown("""
    ### Options Pricing & Risk Analysis Tool
    
    This app provides a comprehensive set of tools for pricing options and analyzing financial risks. It's built using Python and Streamlit, with real-time data from Yahoo Finance.
    
    #### Features:
    
    - **Option Pricing Models**:
        - Black-Scholes model for European options
        - Monte Carlo simulation
        - Binomial tree model for both European and American options
        
    - **Risk Analysis**:
        - Value at Risk (VaR) and Conditional VaR calculation
        - Performance metrics like Sharpe ratio and Beta
        - Historical price and volatility analysis
        
    - **Market Data**:
        - Real-time stock and option chain data
        - Implied volatility visualization
        - Options greeks analysis
    
    #### Technical Implementation:
    
    - Built with Python and Streamlit
    - Data fetched via Yahoo Finance API
    - Computation accuracy targeted at 95%
    
    #### Disclaimer:
    
    This tool is for educational and informational purposes only. It is not intended to provide investment advice. Always consult with a qualified financial advisor before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Options Pricing & Risk Analysis Tool | Powered by Python, Streamlit, and Yahoo Finance")