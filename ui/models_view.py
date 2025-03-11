import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.heston import HestonModel

def render_models_view(ticker, current_price, risk_free_rate, implied_vol):
    """Render the advanced models tab"""
    st.header("Advanced Options Pricing Models")
    st.markdown("Explore more sophisticated options pricing models beyond Black-Scholes.")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model:", 
        ["Heston Stochastic Volatility", "Jump Diffusion (Coming Soon)", "Local Volatility (Coming Soon)"],
        key="adv_model_select"
    )
    
    if model_type == "Heston Stochastic Volatility":
        render_heston_model(ticker, current_price, risk_free_rate, implied_vol)
    else:
        st.info("This model is coming soon. Please check back later.")

def render_heston_model(ticker, current_price, risk_free_rate, implied_vol):
    """Render the Heston model interface"""
    st.subheader("Heston Stochastic Volatility Model")
    st.markdown("""
    The Heston model extends Black-Scholes by allowing volatility to vary stochastically over time,
    addressing a key limitation of the standard model. This helps capture volatility smile effects.
    """)
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option Parameters")
        option_type = st.radio("Option Type:", ["Call", "Put"], key="heston_option_type")
        strike_price = st.number_input("Strike Price ($):", min_value=0.1, value=round(current_price, 1), step=1.0, key="heston_strike")
        maturity_days = st.number_input("Time to Maturity (days):", min_value=1, value=30, key="heston_maturity_days")
        maturity = maturity_days / 365.0  # Convert to years
    
    with col2:
        st.markdown("### Heston Parameters")
        v0 = st.slider("Initial Variance (v₀):", min_value=0.01, max_value=0.25, value=implied_vol**2, format="%.3f", 
                      help="The current variance level (σ² - approximately the implied volatility squared)",
                      key="heston_v0")
        
        kappa = st.slider("Mean Reversion Rate (κ):", min_value=0.1, max_value=5.0, value=2.0, step=0.1,
                         help="Speed at which variance reverts to the long-term mean",
                         key="heston_kappa")
        
        theta = st.slider("Long-term Variance (θ):", min_value=0.01, max_value=0.25, value=implied_vol**2, format="%.3f",
                         help="The long-term variance level to which volatility reverts",
                         key="heston_theta")
        
        sigma = st.slider("Volatility of Volatility (σᵥ):", min_value=0.05, max_value=1.0, value=0.3, step=0.05,
                         help="How volatile is the volatility itself",
                         key="heston_sigma")
        
        rho = st.slider("Price-Volatility Correlation (ρ):", min_value=-0.95, max_value=0.0, value=-0.7, step=0.05,
                       help="Correlation between returns and volatility changes (typically negative for equities)",
                       key="heston_rho")
    
    # Calculation method
    method = st.radio("Calculation Method:", ["Semi-Analytical Solution", "Monte Carlo Simulation"], key="heston_calc_method")
    
    if st.button("Calculate Option Price", key="heston_calculate_btn"):
        with st.spinner("Calculating..."):
            # Create Heston model
            heston = HestonModel(
                S=current_price, 
                K=strike_price, 
                T=maturity, 
                r=risk_free_rate, 
                v0=v0, 
                kappa=kappa, 
                theta=theta, 
                sigma=sigma, 
                rho=rho
            )
            
            # Calculate option price
            if method == "Semi-Analytical Solution":
                heston_price = heston.price(option_type=option_type.lower())
                std_error = None
                confidence_interval = None
            else:  # Monte Carlo
                result = heston.simulate_and_price(option_type=option_type.lower())
                heston_price = result['price']
                std_error = result['std_error']
                confidence_interval = result['confidence_interval']
            
            # Calculate Black-Scholes price for comparison
            from models.black_scholes import black_scholes
            bs_price = black_scholes(
                S=current_price, 
                K=strike_price, 
                T=maturity, 
                r=risk_free_rate, 
                sigma=np.sqrt(v0),  # Use initial volatility for comparison
                option_type=option_type.lower()
            )
            
            # Display results
            st.subheader("Pricing Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("Heston Model Price", f"${heston_price:.4f}")
                if std_error:
                    st.caption(f"Standard Error: ${std_error:.4f}")
                    st.caption(f"95% CI: [${confidence_interval[0]:.4f}, ${confidence_interval[1]:.4f}]")
            
            with result_col2:
                st.metric("Black-Scholes Price", f"${bs_price:.4f}", delta=f"{(heston_price - bs_price):.4f}")
                st.caption(f"Difference: {((heston_price / bs_price - 1) * 100):.2f}%")
            
            # Visualization
            st.subheader("Price Path Simulation")
            st.markdown("Simulating potential stock price paths and volatility using the Heston model:")
            
            # Generate simulation data
            time_points, prices, vols = heston._simulate_path(simulations=10, steps=252)
            
            # Plot the data with Plotly
            fig = go.Figure()
            
            # First create price paths subplot
            fig = make_subplots(rows=2, cols=1, 
                               shared_xaxes=True, 
                               subplot_titles=("Stock Price Paths", "Volatility Paths"),
                               row_heights=[0.6, 0.4])
            
            # Add price paths
            for i in range(min(10, prices.shape[0])):
                fig.add_trace(
                    go.Scatter(
                        x=time_points, 
                        y=prices[i],
                        mode='lines',
                        name=f'Path {i+1}',
                        opacity=0.7,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
            
            # Add strike price line
            fig.add_shape(
                type="line",
                x0=0, y0=strike_price,
                x1=maturity, y1=strike_price,
                line=dict(color="red", width=1, dash="dash"),
                row=1, col=1
            )
            
            # Add volatility paths
            for i in range(min(10, vols.shape[0])):
                fig.add_trace(
                    go.Scatter(
                        x=time_points, 
                        y=vols[i] * 100,  # Convert to percentage
                        mode='lines',
                        name=f'Vol {i+1}',
                        opacity=0.7,
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Add long-term volatility line
            fig.add_shape(
                type="line",
                x0=0, y0=np.sqrt(theta) * 100,
                x1=maturity, y1=np.sqrt(theta) * 100,
                line=dict(color="green", width=1, dash="dash"),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                title_text="Heston Model Simulation",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(title_text="Time (years)", row=2, col=1)
            fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add volatility surface visualization if requested
            if st.checkbox("Show Implied Volatility Surface", key="heston_show_vol_surface"):
                st.subheader("Implied Volatility Surface")
                
                with st.spinner("Generating volatility surface..."):
                    # Generate volatility surface data
                    strikes_pct, tenors, vol_surface = heston.implied_volatility_surface(
                        strikes_range=(0.7, 1.3),
                        tenors_range=(0.1, 2.0),
                        num_strikes=20,
                        num_tenors=10,
                        option_type=option_type.lower()
                    )
                    
                    # Convert to percentages for better visualization
                    vol_surface = vol_surface * 100
                    
                    # Create 3D surface plot
                    fig = go.Figure(data=[go.Surface(
                        z=vol_surface,
                        x=strikes_pct,
                        y=tenors,
                        colorscale='Viridis',
                        opacity=0.8
                    )])
                    
                    fig.update_layout(
                        title='Implied Volatility Surface',
                        scene=dict(
                            xaxis_title='Moneyness (Strike/Spot)',
                            yaxis_title='Time to Maturity (Years)',
                            zaxis_title='Implied Volatility (%)',
                            xaxis=dict(showgrid=True),
                            yaxis=dict(showgrid=True),
                            zaxis=dict(showgrid=True)
                        ),
                        width=800,
                        height=700
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    ### Volatility Surface Interpretation
                    
                    The volatility surface shows how implied volatility varies with strike price (moneyness) and time to maturity.
                    
                    - **Volatility Smile**: The horizontal curve shows how volatility typically increases for deep OTM and ITM options.
                    - **Term Structure**: The vertical dimension shows how volatility changes with time to maturity.
                    - **Skew**: The asymmetry in the smile reflects market expectations about future price movements.
                    """)