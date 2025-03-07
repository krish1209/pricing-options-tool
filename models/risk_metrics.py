import numpy as np
import pandas as pd
from scipy import stats

def calculate_var(returns, confidence_level=0.95, window=None):
    """
    Calculate Value at Risk (VaR)
    
    Parameters:
    -----------
    returns : array-like
        Historical returns data
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    window : int, optional
        Rolling window size. If specified, returns rolling VaR
        
    Returns:
    --------
    float or pandas.Series
        VaR value(s)
    """
    returns = pd.Series(returns)
    
    if window is None:
        # Calculate VaR for the entire series
        var = returns.quantile(1 - confidence_level)
        return var
    else:
        # Calculate rolling VaR
        rolling_var = returns.rolling(window=window).quantile(1 - confidence_level)
        return rolling_var

def calculate_cvar(returns, confidence_level=0.95, window=None):
    """
    Calculate Conditional Value at Risk (CVaR)
    
    Parameters:
    -----------
    returns : array-like
        Historical returns data
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    window : int, optional
        Rolling window size. If specified, returns rolling CVaR
        
    Returns:
    --------
    float or pandas.Series
        CVaR value(s)
    """
    returns = pd.Series(returns)
    
    if window is None:
        # Calculate VaR
        var = calculate_var(returns, confidence_level)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        return cvar
    else:
        # Initialize an empty series for rolling CVaR
        rolling_cvar = pd.Series(index=returns.index)
        
        # Calculate rolling CVaR
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i - window:i]
            var = calculate_var(window_returns, confidence_level)
            cvar = window_returns[window_returns <= var].mean()
            rolling_cvar.iloc[i - 1] = cvar
            
        return rolling_cvar

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
    """
    Calculate Sharpe Ratio
    
    Parameters:
    -----------
    returns : array-like
        Historical returns data
    risk_free_rate : float
        Risk-free rate (annualized)
    annualization_factor : int
        Number of periods in a year for annualization
        
    Returns:
    --------
    float
        Sharpe Ratio
    """
    returns = pd.Series(returns)
    
    # Convert risk-free rate to match the frequency of returns
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate mean and standard deviation of excess returns
    mean_excess_return = excess_returns.mean() * annualization_factor
    std_excess_return = excess_returns.std() * np.sqrt(annualization_factor)
    
    # Calculate Sharpe Ratio
    sharpe_ratio = mean_excess_return / std_excess_return
    
    return sharpe_ratio

def calculate_beta(returns, market_returns):
    """
    Calculate Beta (systematic risk)
    
    Parameters:
    -----------
    returns : array-like
        Asset returns
    market_returns : array-like
        Market returns
        
    Returns:
    --------
    float
        Beta value
    """
    returns = pd.Series(returns)
    market_returns = pd.Series(market_returns)
    
    # Align the series
    returns, market_returns = returns.align(market_returns, join='inner')
    
    # Calculate covariance and variance
    covariance = returns.cov(market_returns)
    market_variance = market_returns.var()
    
    # Calculate Beta
    beta = covariance / market_variance
    
    return beta

def monte_carlo_var(S, r, sigma, T=1, n_simulations=10000, confidence_level=0.95):
    """
    Calculate VaR using Monte Carlo simulation
    
    Parameters:
    -----------
    S : float
        Current asset price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time horizon in years
    n_simulations : int
        Number of simulations
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
        
    Returns:
    --------
    dict
        Dictionary containing VaR and CVaR values
    """
    # Generate random price movements
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal(n_simulations)
    
    # Calculate terminal prices
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate returns
    returns = (S_T - S) / S
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate VaR
    var_idx = int(n_simulations * (1 - confidence_level))
    var = -sorted_returns[var_idx]
    
    # Calculate CVaR
    cvar = -np.mean(sorted_returns[:var_idx])
    
    return {
        'VaR': var * S,  # Convert from return to dollar value
        'CVaR': cvar * S,  # Convert from return to dollar value
        'VaR_pct': var * 100,  # VaR as percentage
        'CVaR_pct': cvar * 100  # CVaR as percentage
    }