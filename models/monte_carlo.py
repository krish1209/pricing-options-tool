import numpy as np

def monte_carlo_european(S, K, T, r, sigma, option_type='call', simulations=10000, steps=252):
    """
    Calculate European option price using Monte Carlo simulation
    
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
    simulations : int
        Number of simulation paths
    steps : int
        Number of time steps per path
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'price': option price
        - 'std_error': standard error of the estimate
        - 'confidence_interval': 95% confidence interval
    """
    dt = T / steps
    nudt = (r - 0.5 * sigma ** 2) * dt
    sigsdt = sigma * np.sqrt(dt)
    
    # Initialize matrix for stock paths
    S_paths = np.zeros((simulations, steps + 1))
    S_paths[:, 0] = S
    
    # Generate random paths
    Z = np.random.standard_normal(size=(simulations, steps))
    
    # Simulate stock price paths
    for t in range(1, steps + 1):
        S_paths[:, t] = S_paths[:, t-1] * np.exp(nudt + sigsdt * Z[:, t-1])
    
    # Get terminal stock prices
    S_T = S_paths[:, -1]
    
    # Calculate option payoff
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Discount payoffs to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Calculate standard error and confidence interval
    std_error = np.std(payoffs) / np.sqrt(simulations)
    std_error_pv = np.exp(-r * T) * std_error
    confidence_interval = (
        option_price - 1.96 * std_error_pv, 
        option_price + 1.96 * std_error_pv
    )
    
    return {
        'price': option_price,
        'std_error': std_error_pv,
        'confidence_interval': confidence_interval,
        'accuracy': 1.96 * std_error_pv / option_price * 100  # Percentage error
    }

def monte_carlo_path_generator(S, T, r, sigma, simulations=100, steps=252):
    """
    Generate stock price paths using Monte Carlo simulation for visualization
    
    Parameters:
    -----------
    S : float
        Current stock price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    simulations : int
        Number of paths to generate
    steps : int
        Number of time steps
        
    Returns:
    --------
    tuple
        (time_points, price_paths) where:
        - time_points is an array of time points
        - price_paths is a 2D array of price paths
    """
    dt = T / steps
    nudt = (r - 0.5 * sigma ** 2) * dt
    sigsdt = sigma * np.sqrt(dt)
    
    # Generate time points
    time_points = np.linspace(0, T, steps + 1)
    
    # Initialize prices array
    price_paths = np.zeros((simulations, steps + 1))
    price_paths[:, 0] = S
    
    # Generate random normally distributed returns
    Z = np.random.standard_normal(size=(simulations, steps))
    
    # Simulate price paths
    for t in range(1, steps + 1):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(nudt + sigsdt * Z[:, t-1])
    
    return time_points, price_paths