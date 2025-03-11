import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Option:
    """
    Represents a single option contract
    """
    strike: float
    option_type: OptionType
    position_type: PositionType
    quantity: int = 1
    premium: float = 0.0
    expiration: str = ""  # ISO format date string

    def payoff(self, stock_price: float) -> float:
        """Calculate the payoff of this option at a given stock price"""
        intrinsic_value = 0

        # Calculate intrinsic value at expiration
        if self.option_type == OptionType.CALL:
            intrinsic_value = max(0, stock_price - self.strike)
        else:  # PUT
            intrinsic_value = max(0, self.strike - stock_price)

        # Apply position type and quantity
        if self.position_type == PositionType.LONG:
            return (intrinsic_value - self.premium) * self.quantity
        else:  # SHORT
            return (self.premium - intrinsic_value) * self.quantity


@dataclass
class Stock:
    """
    Represents a stock position
    """
    position_type: PositionType
    quantity: int
    entry_price: float

    def payoff(self, stock_price: float) -> float:
        """Calculate the payoff of this stock position at a given stock price"""
        if self.position_type == PositionType.LONG:
            return (stock_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - stock_price) * self.quantity


class OptionsStrategy:
    """
    Represents a collection of options and/or stock positions that form a strategy
    """
    def __init__(self, name: str, underlying_price: float):
        self.name = name
        self.underlying_price = underlying_price
        self.options: List[Option] = []
        self.stock: Optional[Stock] = None
        self._cached_metrics = None  # For caching the metrics calculation

    def add_option(self, option: Option) -> None:
        """Add an option to the strategy"""
        self.options.append(option)
        self._cached_metrics = None  # Invalidate cache

    def add_stock(self, stock: Stock) -> None:
        """Add a stock position to the strategy"""
        self.stock = stock
        self._cached_metrics = None  # Invalidate cache

    def net_premium(self) -> float:
        """Calculate the net premium paid or received for the strategy"""
        premium = 0.0
        for option in self.options:
            if option.position_type == PositionType.LONG:
                premium -= option.premium * option.quantity
            else:  # SHORT
                premium += option.premium * option.quantity
        return premium

    def payoff_at_price(self, stock_price: float) -> float:
        """Calculate the total payoff of the strategy at a given stock price"""
        total_payoff = 0.0
        
        # Add option payoffs
        for option in self.options:
            total_payoff += option.payoff(stock_price)
        
        # Add stock payoff if present
        if self.stock:
            total_payoff += self.stock.payoff(stock_price)
        
        return total_payoff

    def payoff_diagram(self, 
                     price_range: Tuple[float, float] = None, 
                     points: int = 100) -> pd.DataFrame:
        """Generate payoff data for plotting"""
        # Determine price range if not provided
        if not price_range:
            # Find min and max strikes to set a reasonable range
            all_strikes = [option.strike for option in self.options]
            if not all_strikes:
                # Default to ±20% if no options
                min_price = self.underlying_price * 0.8
                max_price = self.underlying_price * 1.2
            else:
                min_strike = min(all_strikes)
                max_strike = max(all_strikes)
                # Add padding around strikes
                padding = (max_strike - min_strike) * 0.5
                min_price = max(0.1, min_strike - padding)  # Ensure positive
                max_price = max_strike + padding
        else:
            min_price, max_price = price_range

        # Generate price points
        prices = np.linspace(min_price, max_price, points)
        payoffs = [self.payoff_at_price(price) for price in prices]
        
        # Create DataFrame for easy manipulation
        return pd.DataFrame({
            'StockPrice': prices,
            'Payoff': payoffs
        })

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key metrics for the strategy"""
        if self._cached_metrics:
            return self._cached_metrics
            
        # Get payoff data
        payoff_data = self.payoff_diagram(points=1000)
        
        # Find break-even points
        # For better accuracy, interpolate between points
        break_even_points = []
        
        for i in range(len(payoff_data) - 1):
            payoff1 = payoff_data.iloc[i]['Payoff']
            payoff2 = payoff_data.iloc[i + 1]['Payoff']
            
            # If payoff crosses zero, interpolate to find break-even
            if (payoff1 <= 0 and payoff2 >= 0) or (payoff1 >= 0 and payoff2 <= 0):
                price1 = payoff_data.iloc[i]['StockPrice']
                price2 = payoff_data.iloc[i + 1]['StockPrice']
                
                # Linear interpolation
                if payoff2 - payoff1 != 0:  # Avoid division by zero
                    break_even = price1 + (price2 - price1) * (-payoff1) / (payoff2 - payoff1)
                    break_even_points.append(break_even)
        
        # Find max profit and loss
        max_profit = payoff_data['Payoff'].max()
        max_loss = payoff_data['Payoff'].min()
        
        # Find prices at which max profit and loss occur
        max_profit_price = payoff_data.loc[payoff_data['Payoff'].idxmax(), 'StockPrice']
        max_loss_price = payoff_data.loc[payoff_data['Payoff'].idxmin(), 'StockPrice']
        
        # Calculate probability of profit (simple approximation assuming log-normal distribution)
        # This is a very simplified approach - real probability would need more sophisticated modeling
        current_payoff = self.payoff_at_price(self.underlying_price)
        prob_profit = 0.5  # Default to 50%
        
        # If any break-even exists, use distance to nearest break-even as a crude estimate
        if break_even_points:
            nearest_breakeven = min(break_even_points, key=lambda x: abs(x - self.underlying_price))
            # Crude approximation - not statistically valid but gives a feel
            distance_to_breakeven = abs(nearest_breakeven - self.underlying_price)
            prob_profit = 0.5 + 0.3 * (1 - np.exp(-distance_to_breakeven / self.underlying_price * 5))
            prob_profit = min(0.95, max(0.05, prob_profit))  # Cap between 5% and 95%
        
        # Calculate risk-reward ratio
        risk_reward_ratio = abs(max_loss / max_profit) if max_profit != 0 else float('inf')
        
        metrics = {
            'break_even_points': break_even_points,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'max_profit_price': max_profit_price,
            'max_loss_price': max_loss_price,
            'net_premium': self.net_premium(),
            'current_payoff': current_payoff,
            'prob_profit': prob_profit,
            'risk_reward_ratio': risk_reward_ratio
        }
        
        self._cached_metrics = metrics
        return metrics
        
    def plot_payoff(self, use_plotly: bool = True, 
                  highlight_be: bool = True, 
                  show_metrics: bool = True,
                  fig_size: Tuple[int, int] = (10, 6)) -> Union[plt.Figure, go.Figure]:
        """
        Plot the payoff diagram for the strategy
        
        Parameters:
        -----------
        use_plotly : bool
            Whether to use Plotly (True) or Matplotlib (False)
        highlight_be : bool
            Whether to highlight break-even points
        show_metrics : bool
            Whether to show metrics on the plot
        fig_size : tuple
            Figure size (width, height)
            
        Returns:
        --------
        Union[plt.Figure, go.Figure]
            The created figure
        """
        # Get payoff data and metrics
        payoff_data = self.payoff_diagram()
        metrics = self.calculate_metrics()
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Add payoff line
            fig.add_trace(go.Scatter(
                x=payoff_data['StockPrice'],
                y=payoff_data['Payoff'],
                mode='lines',
                name='Payoff',
                line=dict(color='blue', width=2)
            ))
            
            # Add break-even points
            if highlight_be and metrics['break_even_points']:
                for be_point in metrics['break_even_points']:
                    fig.add_trace(go.Scatter(
                        x=[be_point],
                        y=[0],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='x'),
                        name=f'Break-Even: ${be_point:.2f}'
                    ))
            
            # Add zero line
            fig.add_shape(
                type="line",
                x0=payoff_data['StockPrice'].min(),
                y0=0,
                x1=payoff_data['StockPrice'].max(),
                y1=0,
                line=dict(color="black", width=1, dash="dot")
            )
            
            # Add current price line
            fig.add_shape(
                type="line",
                x0=self.underlying_price,
                y0=min(0, metrics['max_loss'] * 1.1),
                x1=self.underlying_price,
                y1=max(0, metrics['max_profit'] * 1.1),
                line=dict(color="green", width=1, dash="dash")
            )
            
            # Add metrics as annotations if requested
            if show_metrics:
                metrics_text = (
                    f"Strategy: {self.name}<br>"
                    f"Current Price: ${self.underlying_price:.2f}<br>"
                    f"Net Premium: ${metrics['net_premium']:.2f}<br>"
                    f"Max Profit: ${metrics['max_profit']:.2f}<br>"
                    f"Max Loss: ${metrics['max_loss']:.2f}<br>"
                    f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}<br>"
                )
                
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=metrics_text,
                    showarrow=False,
                    align="left",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"{self.name} Payoff Diagram",
                xaxis_title="Stock Price at Expiration ($)",
                yaxis_title="Profit/Loss ($)",
                width=fig_size[0] * 100,  # Plotly uses pixels
                height=fig_size[1] * 100,
                hovermode="x",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        else:
            # Use Matplotlib
            fig, ax = plt.subplots(figsize=fig_size)
            
            # Plot payoff line
            ax.plot(payoff_data['StockPrice'], payoff_data['Payoff'], 'b-', linewidth=2, label='Payoff')
            
            # Add zero line
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.7)
            
            # Add current price line
            ax.axvline(x=self.underlying_price, color='g', linestyle='--', alpha=0.7,
                      label=f'Current Price: ${self.underlying_price:.2f}')
            
            # Add break-even points
            if highlight_be and metrics['break_even_points']:
                for be_point in metrics['break_even_points']:
                    ax.plot(be_point, 0, 'rx', markersize=10)
                    ax.annotate(f'BE: ${be_point:.2f}', 
                               xy=(be_point, 0),
                               xytext=(0, 10),
                               textcoords='offset points',
                               ha='center')
            
            # Add metrics text if requested
            if show_metrics:
                metrics_text = (
                    f"Strategy: {self.name}\n"
                    f"Net Premium: ${metrics['net_premium']:.2f}\n"
                    f"Max Profit: ${metrics['max_profit']:.2f}\n"
                    f"Max Loss: ${metrics['max_loss']:.2f}\n"
                    f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}\n"
                )
                
                # Position text in upper left with padding
                plt.text(0.02, 0.98, metrics_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels and title
            ax.set_xlabel('Stock Price at Expiration ($)')
            ax.set_ylabel('Profit/Loss ($)')
            ax.set_title(f"{self.name} Payoff Diagram")
            ax.grid(alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            return fig


class StrategyBuilder:
    """
    Factory class for creating common option strategies
    """
    @staticmethod
    def long_call(underlying_price: float, strike: float, premium: float, 
                expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """
        Create a long call strategy
        
        Parameters:
        -----------
        underlying_price : float
            Current price of the underlying asset
        strike : float
            Strike price of the call option
        premium : float
            Premium paid for the option
        expiration : str
            Expiration date in ISO format (optional)
        quantity : int
            Number of contracts
            
        Returns:
        --------
        OptionsStrategy
            The created strategy
        """
        strategy = OptionsStrategy("Long Call", underlying_price)
        option = Option(
            strike=strike,
            option_type=OptionType.CALL,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        return strategy
    
    @staticmethod
    def long_put(underlying_price: float, strike: float, premium: float, 
               expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a long put strategy"""
        strategy = OptionsStrategy("Long Put", underlying_price)
        option = Option(
            strike=strike,
            option_type=OptionType.PUT,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        return strategy
    
    @staticmethod
    def short_call(underlying_price: float, strike: float, premium: float, 
                 expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a short call strategy"""
        strategy = OptionsStrategy("Short Call", underlying_price)
        option = Option(
            strike=strike,
            option_type=OptionType.CALL,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        return strategy
    
    @staticmethod
    def short_put(underlying_price: float, strike: float, premium: float, 
                expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a short put strategy"""
        strategy = OptionsStrategy("Short Put", underlying_price)
        option = Option(
            strike=strike,
            option_type=OptionType.PUT,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        return strategy
    
    @staticmethod
    def covered_call(underlying_price: float, strike: float, premium: float, 
                   expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a covered call strategy"""
        strategy = OptionsStrategy("Covered Call", underlying_price)
        
        # Long stock position
        stock = Stock(
            position_type=PositionType.LONG,
            quantity=quantity * 100,  # 1 option contract = 100 shares
            entry_price=underlying_price
        )
        strategy.add_stock(stock)
        
        # Short call
        option = Option(
            strike=strike,
            option_type=OptionType.CALL,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        
        return strategy
    
    @staticmethod
    def protective_put(underlying_price: float, strike: float, premium: float, 
                     expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a protective put strategy"""
        strategy = OptionsStrategy("Protective Put", underlying_price)
        
        # Long stock position
        stock = Stock(
            position_type=PositionType.LONG,
            quantity=quantity * 100,  # 1 option contract = 100 shares
            entry_price=underlying_price
        )
        strategy.add_stock(stock)
        
        # Long put
        option = Option(
            strike=strike,
            option_type=OptionType.PUT,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=premium,
            expiration=expiration
        )
        strategy.add_option(option)
        
        return strategy
    
    @staticmethod
    def bull_call_spread(underlying_price: float, 
                       low_strike: float, low_premium: float,
                       high_strike: float, high_premium: float,
                       expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a bull call spread strategy"""
        strategy = OptionsStrategy("Bull Call Spread", underlying_price)
        
        # Long call at lower strike
        low_call = Option(
            strike=low_strike,
            option_type=OptionType.CALL,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=low_premium,
            expiration=expiration
        )
        strategy.add_option(low_call)
        
        # Short call at higher strike
        high_call = Option(
            strike=high_strike,
            option_type=OptionType.CALL,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=high_premium,
            expiration=expiration
        )
        strategy.add_option(high_call)
        
        return strategy
    
    @staticmethod
    def bear_put_spread(underlying_price: float, 
                      high_strike: float, high_premium: float,
                      low_strike: float, low_premium: float,
                      expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create a bear put spread strategy"""
        strategy = OptionsStrategy("Bear Put Spread", underlying_price)
        
        # Long put at higher strike
        high_put = Option(
            strike=high_strike,
            option_type=OptionType.PUT,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=high_premium,
            expiration=expiration
        )
        strategy.add_option(high_put)
        
        # Short put at lower strike
        low_put = Option(
            strike=low_strike,
            option_type=OptionType.PUT,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=low_premium,
            expiration=expiration
        )
        strategy.add_option(low_put)
        
        return strategy

    @staticmethod
    def straddle(underlying_price: float, strike: float, 
                call_premium: float, put_premium: float,
                expiration: str = "", quantity: int = 1,
                position: PositionType = PositionType.LONG) -> OptionsStrategy:
        """Create a straddle strategy (long or short)"""
        strategy_name = "Long Straddle" if position == PositionType.LONG else "Short Straddle"
        strategy = OptionsStrategy(strategy_name, underlying_price)
        
        # Call option
        call = Option(
            strike=strike,
            option_type=OptionType.CALL,
            position_type=position,
            quantity=quantity,
            premium=call_premium,
            expiration=expiration
        )
        strategy.add_option(call)
        
        # Put option
        put = Option(
            strike=strike,
            option_type=OptionType.PUT,
            position_type=position,
            quantity=quantity,
            premium=put_premium,
            expiration=expiration
        )
        strategy.add_option(put)
        
        return strategy
    
    @staticmethod
    def strangle(underlying_price: float, 
               call_strike: float, call_premium: float,
               put_strike: float, put_premium: float,
               expiration: str = "", quantity: int = 1,
               position: PositionType = PositionType.LONG) -> OptionsStrategy:
        """Create a strangle strategy (long or short)"""
        strategy_name = "Long Strangle" if position == PositionType.LONG else "Short Strangle"
        strategy = OptionsStrategy(strategy_name, underlying_price)
        
        # Call option (typically higher strike)
        call = Option(
            strike=call_strike,
            option_type=OptionType.CALL,
            position_type=position,
            quantity=quantity,
            premium=call_premium,
            expiration=expiration
        )
        strategy.add_option(call)
        
        # Put option (typically lower strike)
        put = Option(
            strike=put_strike,
            option_type=OptionType.PUT,
            position_type=position,
            quantity=quantity,
            premium=put_premium,
            expiration=expiration
        )
        strategy.add_option(put)
        
        return strategy
    
    @staticmethod
    def butterfly(underlying_price: float, 
                lower_strike: float, middle_strike: float, higher_strike: float,
                lower_premium: float, middle_premium: float, higher_premium: float,
                expiration: str = "", quantity: int = 1,
                option_type: OptionType = OptionType.CALL) -> OptionsStrategy:
        """Create a butterfly spread strategy"""
        strategy_type = "Call" if option_type == OptionType.CALL else "Put"
        strategy = OptionsStrategy(f"{strategy_type} Butterfly Spread", underlying_price)
        
        # Check for equidistant strikes
        if not np.isclose(middle_strike - lower_strike, higher_strike - middle_strike):
            print("Warning: Butterfly spread typically uses equidistant strikes")
        
        if option_type == OptionType.CALL:
            # Long lower strike call
            lower = Option(
                strike=lower_strike,
                option_type=OptionType.CALL,
                position_type=PositionType.LONG,
                quantity=quantity,
                premium=lower_premium,
                expiration=expiration
            )
            
            # Short middle strike calls (2x quantity)
            middle = Option(
                strike=middle_strike,
                option_type=OptionType.CALL,
                position_type=PositionType.SHORT,
                quantity=quantity * 2,
                premium=middle_premium,
                expiration=expiration
            )
            
            # Long higher strike call
            higher = Option(
                strike=higher_strike,
                option_type=OptionType.CALL,
                position_type=PositionType.LONG,
                quantity=quantity,
                premium=higher_premium,
                expiration=expiration
            )
        else:
            # Long lower strike put
            lower = Option(
                strike=lower_strike,
                option_type=OptionType.PUT,
                position_type=PositionType.LONG,
                quantity=quantity,
                premium=lower_premium,
                expiration=expiration
            )
            
            # Short middle strike puts (2x quantity)
            middle = Option(
                strike=middle_strike,
                option_type=OptionType.PUT,
                position_type=PositionType.SHORT,
                quantity=quantity * 2,
                premium=middle_premium,
                expiration=expiration
            )
            
            # Long higher strike put
            higher = Option(
                strike=higher_strike,
                option_type=OptionType.PUT,
                position_type=PositionType.LONG,
                quantity=quantity,
                premium=higher_premium,
                expiration=expiration
            )
        
        strategy.add_option(lower)
        strategy.add_option(middle)
        strategy.add_option(higher)
        
        return strategy
    
    @staticmethod
    def iron_condor(underlying_price: float,
                  put_lower_strike: float, put_higher_strike: float,
                  call_lower_strike: float, call_higher_strike: float,
                  put_lower_premium: float, put_higher_premium: float,
                  call_lower_premium: float, call_higher_premium: float,
                  expiration: str = "", quantity: int = 1) -> OptionsStrategy:
        """Create an iron condor strategy"""
        strategy = OptionsStrategy("Iron Condor", underlying_price)
        
        # Put spread - long put at lower strike
        put_lower = Option(
            strike=put_lower_strike,
            option_type=OptionType.PUT,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=put_lower_premium,
            expiration=expiration
        )
        
        # Put spread - short put at higher strike
        put_higher = Option(
            strike=put_higher_strike,
            option_type=OptionType.PUT,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=put_higher_premium,
            expiration=expiration
        )
        
        # Call spread - short call at lower strike
        call_lower = Option(
            strike=call_lower_strike,
            option_type=OptionType.CALL,
            position_type=PositionType.SHORT,
            quantity=quantity,
            premium=call_lower_premium,
            expiration=expiration
        )
        
        # Call spread - long call at higher strike
        call_higher = Option(
            strike=call_higher_strike,
            option_type=OptionType.CALL,
            position_type=PositionType.LONG,
            quantity=quantity,
            premium=call_higher_premium,
            expiration=expiration
        )
        
        strategy.add_option(put_lower)
        strategy.add_option(put_higher)
        strategy.add_option(call_lower)
        strategy.add_option(call_higher)
        
        return strategy
    
    @staticmethod
    def custom_strategy(name: str, underlying_price: float, options: List[Option], 
                      stock: Optional[Stock] = None) -> OptionsStrategy:
        """Create a custom strategy with provided options and stock position"""
        strategy = OptionsStrategy(name, underlying_price)
        
        # Add all options
        for option in options:
            strategy.add_option(option)
        
        # Add stock if provided
        if stock:
            strategy.add_stock(stock)
        
        return strategy


# Example usage
if __name__ == "__main__":
    # Create a bull call spread
    stock_price = 100
    strategy = StrategyBuilder.bull_call_spread(
        underlying_price=stock_price,
        low_strike=95,
        low_premium=7.50,
        high_strike=110,
        high_premium=2.50,
        expiration="2023-12-15"
    )
    
    # Calculate metrics
    metrics = strategy.calculate_metrics()
    print(f"Strategy: {strategy.name}")
    print(f"Net Premium: ${metrics['net_premium']:.2f}")
    print(f"Max Profit: ${metrics['max_profit']:.2f}")
    print(f"Max Loss: ${metrics['max_loss']:.2f}")
    if metrics['break_even_points']:
        print(f"Break-even point(s): {', '.join([f'${be:.2f}' for be in metrics['break_even_points']])}")
    print(f"Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
    
    # Plot the payoff diagram
    fig = strategy.plot_payoff(use_plotly=True)
    # fig.show()  # If running in an environment that supports Plotly