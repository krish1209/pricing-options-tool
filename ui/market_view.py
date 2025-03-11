import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data.yahoo_finance import get_option_chain

def render_market_tab(ticker, current_price):
    """Render the market data tab"""
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
                    from models.black_scholes import black_scholes_greeks
                    
                    bs_greeks = black_scholes_greeks(
                        current_price, row['strike'], t, 
                        0.04, row['impliedVolatility'], 'call'
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
                    from models.black_scholes import black_scholes_greeks
                    
                    bs_greeks = black_scholes_greeks(
                        current_price, row['strike'], t, 
                        0.04, row['impliedVolatility'], 'put'
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