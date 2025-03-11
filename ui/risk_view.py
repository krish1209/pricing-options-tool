import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from data.yahoo_finance import get_stock_data, get_market_data
from models.risk_metrics import calculate_var, calculate_cvar, calculate_sharpe_ratio, calculate_beta, monte_carlo_var

def render_risk_tab(ticker, current_price, risk_free_rate, implied_vol):
    """Render the risk analysis tab"""
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
                        current_price, risk_free_rate, implied_vol, 
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
                        (risk_free_rate - 0.5 * implied_vol**2) / 365, 
                        implied_vol / np.sqrt(365), 
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