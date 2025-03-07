import numpy as np

def binomial_tree_european(S, K, T, r, sigma, option_type='call', steps=100):
    """
    Calculate European option price using the binomial tree model
    
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
    steps : int
        Number of time steps in the binomial tree
        
    Returns:
    --------
    float
        Option price
    """
    # Time step
    dt = T / steps
    
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize arrays for stock price and option value
    stock_prices = np.zeros((steps + 1, steps + 1))
    option_values = np.zeros((steps + 1, steps + 1))
    
    # Fill in the stock price tree
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Determine option values at expiration
    if option_type.lower() == 'call':
        option_values[:, steps] = np.maximum(stock_prices[:, steps] - K, 0)
    elif option_type.lower() == 'put':
        option_values[:, steps] = np.maximum(K - stock_prices[:, steps], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Backward induction to find option price at t=0
    discount_factor = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # For European options, we always use the expected value (no early exercise)
            option_values[j, i] = discount_factor * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1])
    
    return option_values[0, 0]

def binomial_tree_american(S, K, T, r, sigma, option_type='call', steps=100):
    """
    Calculate American option price using the binomial tree model
    
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
    steps : int
        Number of time steps in the binomial tree
        
    Returns:
    --------
    float
        Option price
    """
    # Time step
    dt = T / steps
    
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize arrays for stock price and option value
    stock_prices = np.zeros((steps + 1, steps + 1))
    option_values = np.zeros((steps + 1, steps + 1))
    
    # Fill in the stock price tree
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Determine option values at expiration
    if option_type.lower() == 'call':
        option_values[:, steps] = np.maximum(stock_prices[:, steps] - K, 0)
    elif option_type.lower() == 'put':
        option_values[:, steps] = np.maximum(K - stock_prices[:, steps], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Backward induction to find option price at t=0
    discount_factor = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Expected value from holding
            expected_value = discount_factor * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1])
            
            # Value from immediate exercise
            if option_type.lower() == 'call':
                exercise_value = np.maximum(stock_prices[j, i] - K, 0)
            else:  # put
                exercise_value = np.maximum(K - stock_prices[j, i], 0)
            
            # For American options, we take the maximum of expected value and exercise value
            option_values[j, i] = np.maximum(expected_value, exercise_value)
    
    return option_values[0, 0]

def get_binomial_tree_data(S, K, T, r, sigma, option_type='call', steps=20):
    """
    Generate binomial tree data for visualization
    
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
    steps : int
        Number of time steps in the binomial tree
        
    Returns:
    --------
    dict
        Dictionary containing tree structure for visualization
    """
    # Time step
    dt = T / steps
    
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize arrays for stock price and option value
    stock_prices = np.zeros((steps + 1, steps + 1))
    option_values = np.zeros((steps + 1, steps + 1))
    
    # Fill in the stock price tree
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Determine option values at expiration
    if option_type.lower() == 'call':
        option_values[:, steps] = np.maximum(stock_prices[:, steps] - K, 0)
    elif option_type.lower() == 'put':
        option_values[:, steps] = np.maximum(K - stock_prices[:, steps], 0)
    
    # Backward induction to find option price at t=0
    discount_factor = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = discount_factor * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1])
    
    # Create tree structure for visualization
    tree_data = {
        'time_points': np.linspace(0, T, steps + 1),
        'stock_prices': stock_prices,
        'option_values': option_values,
        'up_factor': u,
        'down_factor': d,
        'probability': p
    }
    
    return tree_data