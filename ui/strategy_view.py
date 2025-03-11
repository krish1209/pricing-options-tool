import streamlit as st
import pandas as pd
import numpy as np
from models.strategy_builder import StrategyBuilder, OptionType, PositionType, Option, Stock

def render_strategy_view(ticker, current_price, risk_free_rate, implied_vol):
    """Render the option strategy builder tab"""
    st.header("Options Strategy Builder (Testing Phase)")
    st.markdown("Build and analyze complex options strategies.")
    
    # Strategy selection
    strategy_types = [
        "Long Call", "Long Put", "Short Call", "Short Put",
        "Covered Call", "Protective Put", "Bull Call Spread", "Bear Put Spread",
        "Long Straddle", "Short Straddle", "Long Strangle", "Short Strangle",
        "Call Butterfly", "Put Butterfly", "Iron Condor", "Custom Strategy"
    ]
    
    selected_strategy = st.selectbox("Select Strategy:", strategy_types, key="strategy_selector")
    
    # Expiration date
    today = pd.to_datetime('today')
    expiry_options = [
        (today + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        (today + pd.Timedelta(days=60)).strftime('%Y-%m-%d'),
        (today + pd.Timedelta(days=90)).strftime('%Y-%m-%d'),
        (today + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    ]
    expiry = st.selectbox("Expiration Date:", expiry_options, key="expiry_selector")
    
    # Quantity
    quantity = st.number_input("Contracts:", min_value=1, value=1, step=1, key="contracts_quantity")
    
    # Create columns for the strategy parameters
    if selected_strategy in ["Long Call", "Short Call", "Long Put", "Short Put"]:
        col1, col2 = st.columns(2)
        
        with col1:
            strike = st.number_input("Strike Price ($):", 
                                    min_value=0.1, 
                                    value=round(current_price, 1), 
                                    step=1.0,
                                    key="single_option_strike")
        
        with col2:
            premium = st.number_input("Option Premium ($):", 
                                     min_value=0.01, 
                                     value=round(current_price * 0.05, 2), 
                                     step=0.05,
                                     key="single_option_premium")
        
        if st.button("Generate Strategy", key="single_option_generate"):
            if selected_strategy == "Long Call":
                strategy = StrategyBuilder.long_call(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            elif selected_strategy == "Long Put":
                strategy = StrategyBuilder.long_put(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            elif selected_strategy == "Short Call":
                strategy = StrategyBuilder.short_call(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            elif selected_strategy == "Short Put":
                strategy = StrategyBuilder.short_put(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            
            # Display results
            display_strategy_results(strategy)
            
    elif selected_strategy in ["Covered Call", "Protective Put"]:
        col1, col2 = st.columns(2)
        
        with col1:
            strike = st.number_input("Strike Price ($):", 
                                    min_value=0.1, 
                                    value=round(current_price * 1.05, 1) if selected_strategy == "Covered Call"
                                    else round(current_price * 0.95, 1), 
                                    step=1.0,
                                    key="stock_hedge_strike")
        
        with col2:
            premium = st.number_input("Option Premium ($):", 
                                     min_value=0.01, 
                                     value=round(current_price * 0.03, 2), 
                                     step=0.05,
                                     key="stock_hedge_premium")
        
        if st.button("Generate Strategy", key="stock_hedge_generate"):
            if selected_strategy == "Covered Call":
                strategy = StrategyBuilder.covered_call(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            else:  # Protective Put
                strategy = StrategyBuilder.protective_put(
                    underlying_price=current_price, 
                    strike=strike, 
                    premium=premium, 
                    expiration=expiry, 
                    quantity=quantity
                )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy in ["Bull Call Spread", "Bear Put Spread"]:
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_strategy == "Bull Call Spread":
                lower_strike = st.number_input("Lower Strike Price ($):", 
                                             min_value=0.1, 
                                             value=round(current_price * 0.95, 1), 
                                             step=1.0,
                                             key="bull_call_lower_strike")
                lower_premium = st.number_input("Lower Strike Premium ($):", 
                                              min_value=0.01, 
                                              value=round(current_price * 0.05, 2), 
                                              step=0.05,
                                              key="bull_call_lower_premium")
            else:  # Bear Put Spread
                higher_strike = st.number_input("Higher Strike Price ($):", 
                                              min_value=0.1, 
                                              value=round(current_price * 1.05, 1), 
                                              step=1.0,
                                              key="bear_put_higher_strike")
                higher_premium = st.number_input("Higher Strike Premium ($):", 
                                               min_value=0.01, 
                                               value=round(current_price * 0.05, 2), 
                                               step=0.05,
                                               key="bear_put_higher_premium")
        
        with col2:
            if selected_strategy == "Bull Call Spread":
                higher_strike = st.number_input("Higher Strike Price ($):", 
                                              min_value=lower_strike, 
                                              value=round(current_price * 1.05, 1), 
                                              step=1.0,
                                              key="bull_call_higher_strike")
                higher_premium = st.number_input("Higher Strike Premium ($):", 
                                               min_value=0.01, 
                                               value=round(current_price * 0.02, 2), 
                                               step=0.05,
                                               key="bull_call_higher_premium")
            else:  # Bear Put Spread
                lower_strike = st.number_input("Lower Strike Price ($):", 
                                             min_value=0.1, 
                                             max_value=higher_strike,
                                             value=round(current_price * 0.95, 1), 
                                             step=1.0,
                                             key="bear_put_lower_strike")
                lower_premium = st.number_input("Lower Strike Premium ($):", 
                                              min_value=0.01, 
                                              value=round(current_price * 0.02, 2), 
                                              step=0.05,
                                              key="bear_put_lower_premium")
        
        if st.button("Generate Strategy", key="spread_generate"):
            if selected_strategy == "Bull Call Spread":
                strategy = StrategyBuilder.bull_call_spread(
                    underlying_price=current_price, 
                    low_strike=lower_strike, 
                    low_premium=lower_premium,
                    high_strike=higher_strike, 
                    high_premium=higher_premium,
                    expiration=expiry, 
                    quantity=quantity
                )
            else:  # Bear Put Spread
                strategy = StrategyBuilder.bear_put_spread(
                    underlying_price=current_price, 
                    high_strike=higher_strike, 
                    high_premium=higher_premium,
                    low_strike=lower_strike, 
                    low_premium=lower_premium,
                    expiration=expiry, 
                    quantity=quantity
                )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy in ["Long Straddle", "Short Straddle"]:
        col1, col2 = st.columns(2)
        
        with col1:
            strike = st.number_input("Strike Price ($):", 
                                    min_value=0.1, 
                                    value=round(current_price, 1), 
                                    step=1.0,
                                    key="straddle_strike")
            call_premium = st.number_input("Call Premium ($):", 
                                          min_value=0.01, 
                                          value=round(current_price * 0.04, 2), 
                                          step=0.05,
                                          key="straddle_call_premium")
        
        with col2:
            put_premium = st.number_input("Put Premium ($):", 
                                         min_value=0.01, 
                                         value=round(current_price * 0.03, 2), 
                                         step=0.05,
                                         key="straddle_put_premium")
        
        if st.button("Generate Strategy", key="straddle_generate"):
            position = PositionType.LONG if selected_strategy == "Long Straddle" else PositionType.SHORT
            
            strategy = StrategyBuilder.straddle(
                underlying_price=current_price, 
                strike=strike, 
                call_premium=call_premium,
                put_premium=put_premium,
                expiration=expiry, 
                quantity=quantity,
                position=position
            )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy in ["Long Strangle", "Short Strangle"]:
        col1, col2 = st.columns(2)
        
        with col1:
            put_strike = st.number_input("Put Strike Price ($):", 
                                        min_value=0.1, 
                                        value=round(current_price * 0.95, 1), 
                                        step=1.0,
                                        key="strangle_put_strike")
            put_premium = st.number_input("Put Premium ($):", 
                                         min_value=0.01, 
                                         value=round(current_price * 0.03, 2), 
                                         step=0.05,
                                         key="strangle_put_premium")
        
        with col2:
            call_strike = st.number_input("Call Strike Price ($):", 
                                         min_value=put_strike, 
                                         value=round(current_price * 1.05, 1), 
                                         step=1.0,
                                         key="strangle_call_strike")
            call_premium = st.number_input("Call Premium ($):", 
                                          min_value=0.01, 
                                          value=round(current_price * 0.03, 2), 
                                          step=0.05,
                                          key="strangle_call_premium")
        
        if st.button("Generate Strategy", key="strangle_generate"):
            position = PositionType.LONG if selected_strategy == "Long Strangle" else PositionType.SHORT
            
            strategy = StrategyBuilder.strangle(
                underlying_price=current_price, 
                call_strike=call_strike, 
                call_premium=call_premium,
                put_strike=put_strike, 
                put_premium=put_premium,
                expiration=expiry, 
                quantity=quantity,
                position=position
            )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy in ["Call Butterfly", "Put Butterfly"]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lower_strike = st.number_input("Lower Strike ($):", 
                                         min_value=0.1, 
                                         value=round(current_price * 0.9, 1), 
                                         step=1.0,
                                         key="butterfly_lower_strike")
            lower_premium = st.number_input("Lower Premium ($):", 
                                          min_value=0.01, 
                                          value=round(current_price * 0.02, 2), 
                                          step=0.05,
                                          key="butterfly_lower_premium")
        
        with col2:
            middle_strike = st.number_input("Middle Strike ($):", 
                                          min_value=lower_strike, 
                                          value=round(current_price, 1), 
                                          step=1.0,
                                          key="butterfly_middle_strike")
            middle_premium = st.number_input("Middle Premium ($):", 
                                           min_value=0.01, 
                                           value=round(current_price * 0.04, 2), 
                                           step=0.05,
                                           key="butterfly_middle_premium")
        
        with col3:
            higher_strike = st.number_input("Higher Strike ($):", 
                                          min_value=middle_strike, 
                                          value=round(current_price * 1.1, 1), 
                                          step=1.0,
                                          key="butterfly_higher_strike")
            higher_premium = st.number_input("Higher Premium ($):", 
                                           min_value=0.01, 
                                           value=round(current_price * 0.02, 2), 
                                           step=0.05,
                                           key="butterfly_higher_premium")
        
        if st.button("Generate Strategy", key="butterfly_generate"):
            option_type = OptionType.CALL if selected_strategy == "Call Butterfly" else OptionType.PUT
            
            strategy = StrategyBuilder.butterfly(
                underlying_price=current_price, 
                lower_strike=lower_strike, 
                middle_strike=middle_strike, 
                higher_strike=higher_strike,
                lower_premium=lower_premium, 
                middle_premium=middle_premium, 
                higher_premium=higher_premium,
                expiration=expiry, 
                quantity=quantity,
                option_type=option_type
            )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy == "Iron Condor":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Put Spread Leg")
            put_lower_strike = st.number_input("Lower Put Strike ($):", 
                                             min_value=0.1, 
                                             value=round(current_price * 0.8, 1), 
                                             step=1.0,
                                             key="condor_put_lower_strike")
            put_lower_premium = st.number_input("Lower Put Premium ($):", 
                                              min_value=0.01, 
                                              value=round(current_price * 0.01, 2), 
                                              step=0.05,
                                              key="condor_put_lower_premium")
            
            put_higher_strike = st.number_input("Higher Put Strike ($):", 
                                              min_value=put_lower_strike, 
                                              value=round(current_price * 0.9, 1), 
                                              step=1.0,
                                              key="condor_put_higher_strike")
            put_higher_premium = st.number_input("Higher Put Premium ($):", 
                                               min_value=0.01, 
                                               value=round(current_price * 0.02, 2), 
                                               step=0.05,
                                               key="condor_put_higher_premium")
        
        with col2:
            st.subheader("Call Spread Leg")
            call_lower_strike = st.number_input("Lower Call Strike ($):", 
                                             min_value=put_higher_strike, 
                                             value=round(current_price * 1.1, 1), 
                                             step=1.0,
                                             key="condor_call_lower_strike")
            call_lower_premium = st.number_input("Lower Call Premium ($):", 
                                              min_value=0.01, 
                                              value=round(current_price * 0.02, 2), 
                                              step=0.05,
                                              key="condor_call_lower_premium")
            
            call_higher_strike = st.number_input("Higher Call Strike ($):", 
                                              min_value=call_lower_strike, 
                                              value=round(current_price * 1.2, 1), 
                                              step=1.0,
                                              key="condor_call_higher_strike")
            call_higher_premium = st.number_input("Higher Call Premium ($):", 
                                               min_value=0.01, 
                                               value=round(current_price * 0.01, 2), 
                                               step=0.05,
                                               key="condor_call_higher_premium")
        
        if st.button("Generate Strategy", key="condor_generate"):
            strategy = StrategyBuilder.iron_condor(
                underlying_price=current_price, 
                put_lower_strike=put_lower_strike, 
                put_higher_strike=put_higher_strike,
                call_lower_strike=call_lower_strike, 
                call_higher_strike=call_higher_strike,
                put_lower_premium=put_lower_premium, 
                put_higher_premium=put_higher_premium,
                call_lower_premium=call_lower_premium, 
                call_higher_premium=call_higher_premium,
                expiration=expiry, 
                quantity=quantity
            )
            
            # Display results
            display_strategy_results(strategy)
    
    elif selected_strategy == "Custom Strategy":
        st.subheader("Build Your Own Strategy")
        
        # Initialize options list
        if "custom_options" not in st.session_state:
            st.session_state.custom_options = []
        
        # Add option component
        st.subheader("Add Option Positions")
        
        option_col1, option_col2 = st.columns(2)
        
        with option_col1:
            new_option_type = st.selectbox("Option Type:", ["Call", "Put"], key="custom_option_type")
            new_position_type = st.selectbox("Position Type:", ["Long", "Short"], key="custom_position_type")
            new_strike = st.number_input("Strike Price ($):", 
                                        min_value=0.1, 
                                        value=round(current_price, 1), 
                                        step=1.0,
                                        key="custom_option_strike")
        
        with option_col2:
            new_premium = st.number_input("Premium ($):", 
                                         min_value=0.01, 
                                         value=round(current_price * 0.03, 2), 
                                         step=0.05,
                                         key="custom_option_premium")
            new_quantity = st.number_input("Quantity:", min_value=1, value=1, step=1, key="custom_option_quantity")
        
        # Add option button
        if st.button("Add Option to Strategy", key="custom_add_option_btn"):
            option = Option(
                strike=new_strike,
                option_type=OptionType.CALL if new_option_type == "Call" else OptionType.PUT,
                position_type=PositionType.LONG if new_position_type == "Long" else PositionType.SHORT,
                quantity=new_quantity,
                premium=new_premium,
                expiration=expiry
            )
            st.session_state.custom_options.append(option)
            st.success(f"Added {new_position_type} {new_quantity} {new_option_type}(s) at strike ${new_strike}")
        
        # Add stock position
        st.subheader("Add Stock Position (Optional)")
        
        add_stock = st.checkbox("Include Stock Position", key="custom_include_stock")
        
        stock = None
        if add_stock:
            stock_col1, stock_col2 = st.columns(2)
            
            with stock_col1:
                stock_position_type = st.selectbox("Stock Position:", ["Long", "Short"], key="custom_stock_position")
                stock_quantity = st.number_input("Shares:", min_value=1, value=100, step=1, key="custom_stock_quantity")
            
            with stock_col2:
                stock_price = st.number_input("Entry Price ($):", 
                                             min_value=0.1, 
                                             value=current_price, 
                                             step=1.0,
                                             key="custom_stock_price")
            
            stock = Stock(
                position_type=PositionType.LONG if stock_position_type == "Long" else PositionType.SHORT,
                quantity=stock_quantity,
                entry_price=stock_price
            )
        
        # Display current strategy components
        if st.session_state.custom_options:
            st.subheader("Current Strategy Components")
            
            for i, opt in enumerate(st.session_state.custom_options):
                st.markdown(f"{i+1}. {opt.position_type.value.capitalize()} {opt.quantity} {opt.option_type.value.capitalize()}(s) @ ${opt.strike:.2f} (Premium: ${opt.premium:.2f})")
            
            if add_stock:
                pos_type = "Long" if stock.position_type == PositionType.LONG else "Short"
                st.markdown(f"• {pos_type} {stock.quantity} shares @ ${stock.entry_price:.2f}")
            
            # Clear strategy button
            if st.button("Clear Strategy", key="custom_clear_btn"):
                st.session_state.custom_options = []
                st.success("Strategy cleared")
            
            # Generate strategy button
            if st.button("Generate Custom Strategy", key="custom_generate_btn"):
                strategy = StrategyBuilder.custom_strategy(
                    name="Custom Strategy",
                    underlying_price=current_price,
                    options=st.session_state.custom_options,
                    stock=stock
                )
                
                # Display results
                display_strategy_results(strategy)


def display_strategy_results(strategy):
    """Display the results of a strategy analysis"""
    
    # Calculate metrics
    metrics = strategy.calculate_metrics()
    
    # Display summary metrics
    st.subheader("Strategy Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Net Premium", f"${metrics['net_premium']:.2f}")
        if metrics['break_even_points']:
            be_str = ", ".join([f"${be:.2f}" for be in metrics['break_even_points']])
            st.metric("Break-even Point(s)", be_str)
    
    with col2:
        st.metric("Max Profit", f"${metrics['max_profit']:.2f}")
        st.metric("Profit at Current Price", f"${metrics['current_payoff']:.2f}")
    
    with col3:
        st.metric("Max Loss", f"${metrics['max_loss']:.2f}")
        st.metric("Risk/Reward Ratio", f"{metrics['risk_reward_ratio']:.2f}")
    
    # Generate the payoff plot
    fig = strategy.plot_payoff(use_plotly=True, show_metrics=False)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy interpretation
    st.subheader("Strategy Interpretation")
    
    interpretation = f"""
    - **Strategy**: {strategy.name}
    - **Net Premium**: {"Paid" if metrics['net_premium'] < 0 else "Received"} ${abs(metrics['net_premium']):.2f}
    - **Maximum Profit**: ${metrics['max_profit']:.2f}
    - **Maximum Loss**: ${abs(metrics['max_loss']):.2f}
    """
    
    if metrics['break_even_points']:
        be_points = ", ".join([f"${be:.2f}" for be in metrics['break_even_points']])
        interpretation += f"- **Break-even Point(s)**: {be_points}\n"
    
    if metrics['risk_reward_ratio'] < 1:
        interpretation += "- **Risk/Reward**: This strategy has a favorable risk-to-reward ratio (less than 1).\n"
    else:
        interpretation += f"- **Risk/Reward**: This strategy has a risk-to-reward ratio of {metrics['risk_reward_ratio']:.2f}. You are risking ${abs(metrics['max_loss']):.2f} to make ${metrics['max_profit']:.2f}.\n"
    
    # Add specific strategy guidance
    if strategy.name == "Long Call":
        interpretation += """
        - **Outlook**: Bullish - You expect the stock price to increase significantly.
        - **Max Profit**: Theoretically unlimited as the stock price increases.
        - **Max Loss**: Limited to the premium paid.
        - **Ideal For**: When you expect a substantial upward movement and prefer defined risk.
        """
    elif strategy.name == "Long Put":
        interpretation += """
        - **Outlook**: Bearish - You expect the stock price to decrease significantly.
        - **Max Profit**: Limited but substantial if the stock falls to zero.
        - **Max Loss**: Limited to the premium paid.
        - **Ideal For**: When you expect a substantial downward movement or as portfolio insurance.
        """
    elif strategy.name == "Short Call":
        interpretation += """
        - **Outlook**: Neutral to bearish - You expect the stock to remain below the strike price.
        - **Max Profit**: Limited to the premium received.
        - **Max Loss**: Theoretically unlimited as the stock price increases.
        - **Ideal For**: When you expect sideways or downward movement and want to generate income.
        """
    elif strategy.name == "Short Put":
        interpretation += """
        - **Outlook**: Neutral to bullish - You expect the stock to remain above the strike price.
        - **Max Profit**: Limited to the premium received.
        - **Max Loss**: Substantial but limited (strike price - premium).
        - **Ideal For**: When you expect sideways or upward movement and want to generate income.
        """
    elif strategy.name == "Covered Call":
        interpretation += """
        - **Outlook**: Neutral to slightly bullish - You don't expect price to rise much above the strike.
        - **Max Profit**: Limited to (strike - entry price + premium).
        - **Max Loss**: Substantial but limited (entry price - premium).
        - **Ideal For**: When you own stock and want to generate additional income.
        """
    elif strategy.name == "Protective Put":
        interpretation += """
        - **Outlook**: Bullish with downside protection - You own stock but want insurance.
        - **Max Profit**: Unlimited upside above break-even.
        - **Max Loss**: Limited to (entry price - strike price + premium).
        - **Ideal For**: Hedging an existing long stock position against potential losses.
        """
    elif strategy.name == "Bull Call Spread":
        interpretation += """
        - **Outlook**: Moderately bullish - You expect some upside but not a massive move.
        - **Max Profit**: Limited to (higher strike - lower strike - net premium).
        - **Max Loss**: Limited to the net premium paid.
        - **Ideal For**: When you want to reduce the cost of buying calls at the expense of capping upside.
        """
    elif strategy.name == "Bear Put Spread":
        interpretation += """
        - **Outlook**: Moderately bearish - You expect some downside but not a massive move.
        - **Max Profit**: Limited to (higher strike - lower strike - net premium).
        - **Max Loss**: Limited to the net premium paid.
        - **Ideal For**: When you want to reduce the cost of buying puts at the expense of capping downside profit.
        """
    elif strategy.name in ["Long Straddle", "Long Strangle"]:
        interpretation += """
        - **Outlook**: Volatile - You expect a significant move but are uncertain of direction.
        - **Max Profit**: Theoretically unlimited on upside, or substantial on downside.
        - **Max Loss**: Limited to the total premium paid.
        - **Ideal For**: When you anticipate high volatility (earnings reports, FDA decisions, etc.).
        """
    elif strategy.name in ["Short Straddle", "Short Strangle"]:
        interpretation += """
        - **Outlook**: Range-bound - You expect the stock to stay within a narrow range.
        - **Max Profit**: Limited to the total premium received.
        - **Max Loss**: Theoretically unlimited on upside, or substantial on downside.
        - **Ideal For**: Low-volatility environments when you expect prices to remain relatively stable.
        """
    elif "Butterfly" in strategy.name:
        interpretation += """
        - **Outlook**: Precisely neutral - You expect the stock to be very near the middle strike at expiration.
        - **Max Profit**: Limited to (wing spread - net premium).
        - **Max Loss**: Limited to the net premium paid.
        - **Ideal For**: When you expect very low volatility and can predict where the stock will be at expiration.
        """
    elif strategy.name == "Iron Condor":
        interpretation += """
        - **Outlook**: Range-bound - You expect the stock to stay between your middle strikes.
        - **Max Profit**: Limited to the net premium received.
        - **Max Loss**: Limited to (wing spread - net premium).
        - **Ideal For**: When you expect a stock to trade within a range and want to collect premium with defined risk.
        """
    
    st.markdown(interpretation)