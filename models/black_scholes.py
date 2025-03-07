import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate European option price using Black-Scholes formula
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    option_type : str
        Type of option - 'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        # Call option price
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        # Put option price using put-call parity
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks for European options
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    option_type : str
        Type of option - 'call' or 'put'
        
    Returns:
    --------
    dict
        Dictionary containing values for Delta, Gamma, Theta, Vega, and Rho
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate common components
    N_d1 = norm.cdf(d1)
    N_neg_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(d2)
    N_neg_d2 = norm.cdf(-d2)
    n_d1 = norm.pdf(d1)
    
    # Initialize Greeks dictionary
    greeks = {}
    
    if option_type.lower() == 'call':
        # Delta - partial derivative of option price with respect to asset price
        greeks['delta'] = N_d1
        
        # Theta - partial derivative of option price with respect to time to maturity
        # Divide by 365 to get daily theta
        greeks['theta'] = (-S * sigma * n_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
        greeks['theta'] = greeks['theta'] / 365
        
        # Rho - partial derivative of option price with respect to interest rate
        greeks['rho'] = K * T * np.exp(-r * T) * N_d2
    
    elif option_type.lower() == 'put':
        # Delta for put
        greeks['delta'] = N_d1 - 1
        
        # Theta for put
        greeks['theta'] = (-S * sigma * n_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_neg_d2
        greeks['theta'] = greeks['theta'] / 365
        
        # Rho for put
        greeks['rho'] = -K * T * np.exp(-r * T) * N_neg_d2
    
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # These are the same for both call and put
    # Gamma - second partial derivative of option price with respect to asset price
    greeks['gamma'] = n_d1 / (S * sigma * np.sqrt(T))
    
    # Vega - partial derivative of option price with respect to volatility
    greeks['vega'] = S * np.sqrt(T) * n_d1
    
    return greeks