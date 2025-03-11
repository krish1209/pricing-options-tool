import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time as time_module
import math

from models.black_scholes import black_scholes, black_scholes_greeks
from models.monte_carlo import monte_carlo_european, monte_carlo_path_generator
from models.binomial import binomial_tree_european, binomial_tree_american, get_binomial_tree_data

def render_pricing_tab(ticker, current_price, risk_free_rate, implied_vol):
    """Render the option pricing tab"""
    st.header("Option Pricing Models")
    st.markdown("Compare different option pricing models and visualize results.")
    
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
                    start_time = time_module.time()
                    bs_price = black_scholes(current_price, strike_price, maturity, rf_rate, volatility, option_type.lower())
                    end_time = time_module.time()
                    
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
                    start_time = time_module.time()
                    mc_result = monte_carlo_european(current_price, strike_price, maturity, rf_rate, volatility, option_type.lower(), monte_carlo_sims)
                    end_time = time_module.time()
                    
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
                    start_time = time_module.time()
                    
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
                    
                    end_time = time_module.time()
                    
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
            
            # Create columns for each model
            cols = st.columns(len(price_results))
            
            # Use app's native amber/brown colors
            for i, (model, price) in enumerate(price_results.items()):
                with cols[i]:
                    # Using st.markdown with custom CSS for background and text color
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #FFA726; 
                            padding: 20px; 
                            border-radius: 10px; 
                            text-align: center;
                            margin: 10px 0;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                            border: 2px solid #704214;
                        ">
                            <h3 style="color: #000000; font-weight: bold; margin-bottom: 15px;">{model}</h3>
                            <h2 style="color: #000000; font-weight: bold; font-size: 28px;">${price:.4f}</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Add a simple price comparison using Streamlit's metrics
            st.markdown("#### Price Comparison")
            metric_cols = st.columns(len(price_results))
            base_price = list(price_results.values())[0]
            
            for i, (model, price) in enumerate(price_results.items()):
                with metric_cols[i]:
                    if i == 0:
                        st.metric(model, f"${price:.4f}")
                    else:
                        diff = price - base_price
                        st.metric(model, f"${price:.4f}", f"{diff:+.4f}")
    
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
        # Tab 2: Binomial Tree
        # Tab 2: Binomial Tree
        with viz_tab2:
            if "Binomial Tree" in st.session_state.option_params.get('models', []) and st.session_state.binomial_data is not None:
                st.subheader("Binomial Tree Visualization")
                
                # Get parameters from session state
                params = st.session_state.option_params
                tree_data = st.session_state.binomial_data
                
                # Prepare node data
                nodes = []
                node_x = []
                node_y = []
                node_text = []
                
                # Prepare edge data
                edge_x = []
                edge_y = []
                
                # Time points define horizontal position
                time_points = tree_data['time_points']
                
                # Calculate positions for all nodes
                for i, t in enumerate(time_points):
                    # Each time step gets a column
                    x_pos = i * 100  # Scale up for better spacing
                    
                    # Calculate nodes at this level
                    for j in range(i + 1):
                        # Position nodes in a triangle formation
                        y_pos = (j - i/2) * 100  # Scale up for better spacing
                        
                        # Get values for this node
                        stock_price = tree_data['stock_prices'][j, i]
                        option_price = tree_data['option_values'][j, i]
                        
                        # Store node positions
                        node_x.append(x_pos)
                        node_y.append(y_pos)
                        node_text.append(f"Time: {t:.3f}<br>Stock: ${stock_price:.2f}<br>Option: ${option_price:.2f}")
                        
                        # Create edges to previous nodes
                        if i > 0:
                            # Connect to upper parent (if exists)
                            if j > 0:
                                # Draw edge: previous node to current node
                                edge_x.extend([i-1 * 100, x_pos, None])
                                edge_y.extend([(j-1-(i-1)/2) * 100, y_pos, None])
                            
                            # Connect to lower parent (if exists)
                            if j < i:
                                # Draw edge: previous node to current node
                                edge_x.extend([i-1 * 100, x_pos, None])
                                edge_y.extend([(j-(i-1)/2) * 100, y_pos, None])
                
                # Create a blank figure with more explicit sizing
                fig = go.Figure()
                
                # Add the nodes as scatter points
                fig.add_trace(go.Scatter(
                    x=node_x, 
                    y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color='#FFA726',
                        line=dict(width=2, color='#704214')
                    ),
                    text=[f"${tree_data['option_values'][j, i]:.2f}" for i, t in enumerate(time_points) for j in range(i + 1)],
                    textposition="middle center",
                    textfont=dict(size=12, color='black'),
                    hoverinfo='text',
                    hovertext=node_text,
                    name='Nodes'
                ))
                
                # Add the edges as lines
                for i, t in enumerate(time_points):
                    if i == 0:
                        continue  # Skip the root node
                        
                    for j in range(i + 1):
                        # Current node position
                        x1 = i * 100
                        y1 = (j - i/2) * 100
                        
                        # Connect to upper parent (if exists)
                        if j > 0:
                            x0 = (i-1) * 100
                            y0 = (j-1-(i-1)/2) * 100
                            
                            fig.add_trace(go.Scatter(
                                x=[x0, x1],
                                y=[y0, y1],
                                mode='lines',
                                line=dict(width=2, color='#704214'),
                                hoverinfo='none',
                                showlegend=False
                            ))
                        
                        # Connect to lower parent (if exists)
                        if j < i:
                            x0 = (i-1) * 100
                            y0 = (j-(i-1)/2) * 100
                            
                            fig.add_trace(go.Scatter(
                                x=[x0, x1],
                                y=[y0, y1],
                                mode='lines',
                                line=dict(width=2, color='#704214'),
                                hoverinfo='none',
                                showlegend=False
                            ))
                
                # Update layout with explicit settings
                fig.update_layout(
                    width=800,
                    height=600,
                    title="Binomial Tree Visualization",
                    title_font_color='white',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-50, max(node_x) + 50]
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[min(node_y) - 50, max(node_y) + 50],
                        scaleanchor='x', 
                        scaleratio=1
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                # Display the figure with a larger size
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanatory text
                st.markdown("""
                ### Tree Interpretation
                - Each node shows the option price at that state
                - Hover over nodes to see detailed information
                - Top branches represent upward price movements
                - Bottom branches represent downward price movements
                - Time moves from left to right
                """)