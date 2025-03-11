import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from numba import jit

class HestonModel:
    """
    Implementation of the Heston stochastic volatility model for option pricing.
    
    The Heston model assumes that the volatility of the underlying asset follows its own
    stochastic process. This addresses a key limitation of the Black-Scholes model,
    which assumes constant volatility.
    
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
    v0 : float
        Initial variance
    kappa : float
        Rate of mean reversion for variance
    theta : float
        Long-term mean of variance
    sigma : float
        Volatility of volatility (volvol)
    rho : float
        Correlation between stock returns and variance
    """
    
    def __init__(self, S, K, T, r, v0, kappa, theta, sigma, rho):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
    
    def _characteristic_function(self, phi, j):
        """
        Calculate the characteristic function for the Heston model.
        
        Parameters:
        -----------
        phi : float
            Integration variable
        j : int
            Indicator for the characteristic function (j=1 or j=2)
            
        Returns:
        --------
        complex
            Value of the characteristic function
        """
        # Adjust u for j=1 or j=2
        u = phi * 1j
        if j == 1:
            b = self.kappa - self.rho * self.sigma
            a = u * 1j
        else:  # j == 2
            b = self.kappa
            a = 0
            
        # Common calculations
        d = np.sqrt((self.rho * self.sigma * u * 1j - b)**2 - (self.sigma**2) * (u * 1j - u**2))
        g = (b - self.rho * self.sigma * u * 1j + d) / (b - self.rho * self.sigma * u * 1j - d)
        
        # Characteristic function formula
        exp1 = np.exp(u * 1j * np.log(self.S) + u * 1j * self.r * self.T)
        exp2 = np.exp((self.v0 / (self.sigma**2)) * 
                     (b - self.rho * self.sigma * u * 1j + d) * 
                     ((1 - np.exp(-d * self.T)) / (1 - g * np.exp(-d * self.T))))
        exp3 = np.exp((self.kappa * self.theta / (self.sigma**2)) * 
                     ((b - self.rho * self.sigma * u * 1j + d) * self.T - 
                      2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))))
        
        return exp1 * exp2 * exp3
    
    def _integrand(self, phi, j):
        """
        Integrand for the option pricing formula.
        
        Parameters:
        -----------
        phi : float
            Integration variable
        j : int
            Indicator for P1 or P2
            
        Returns:
        --------
        float
            Real part of the integrand
        """
        numerator = np.exp(-1j * phi * np.log(self.K)) * self._characteristic_function(phi, j)
        denominator = 1j * phi
        return np.real(numerator / denominator)
    
    def price(self, option_type='call', integration_limit=100, integration_points=1000):
        """
        Calculate the option price using the Heston model.
        
        Parameters:
        -----------
        option_type : str
            Type of option - 'call' or 'put'
        integration_limit : float
            Upper limit for the numerical integration
        integration_points : int
            Number of points for the numerical integration
            
        Returns:
        --------
        float
            Option price
        """
        # Calculate probabilities P1 and P2
        p1, _ = quad(lambda x: self._integrand(x, 1), 0, integration_limit, limit=integration_points)
        p2, _ = quad(lambda x: self._integrand(x, 2), 0, integration_limit, limit=integration_points)
        
        p1 = 0.5 + (1/np.pi) * p1
        p2 = 0.5 + (1/np.pi) * p2
        
        # Calculate option price
        if option_type.lower() == 'call':
            return self.S * p1 - self.K * np.exp(-self.r * self.T) * p2
        elif option_type.lower() == 'put':
            return self.K * np.exp(-self.r * self.T) * (1 - p2) - self.S * (1 - p1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def _simulate_path(self, steps=252, simulations=1000, seed=None):
        """
        Simulate price paths using the Heston model.
        
        Parameters:
        -----------
        steps : int
            Number of time steps per path
        simulations : int
            Number of paths to simulate
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (time_points, price_paths, volatility_paths)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = self.T / steps
        sqrt_dt = np.sqrt(dt)
        
        # Precompute constants for efficiency
        kappa_dt = self.kappa * dt
        theta_kappa_dt = self.theta * kappa_dt
        sigma_sqrt_dt = self.sigma * sqrt_dt
        
        # Initialize arrays
        prices = np.zeros((simulations, steps + 1))
        variances = np.zeros((simulations, steps + 1))
        
        # Set initial values
        prices[:, 0] = self.S
        variances[:, 0] = self.v0
        
        # Generate correlated random numbers
        for i in range(1, steps + 1):
            z1 = np.random.standard_normal(simulations)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(simulations)
            
            # Update variance (ensuring non-negative values)
            variances[:, i] = np.maximum(
                variances[:, i-1] + kappa_dt * (self.theta - variances[:, i-1]) + 
                sigma_sqrt_dt * np.sqrt(np.maximum(variances[:, i-1], 0)) * z1,
                0
            )
            
            # Update prices
            prices[:, i] = prices[:, i-1] * np.exp(
                (self.r - 0.5 * variances[:, i-1]) * dt + 
                np.sqrt(np.maximum(variances[:, i-1], 0)) * sqrt_dt * z2
            )
        
        time_points = np.linspace(0, self.T, steps + 1)
        volatility_paths = np.sqrt(variances)
        
        return time_points, prices, volatility_paths
    
    def simulate_and_price(self, option_type='call', simulations=10000, steps=252):
        """
        Price options using Monte Carlo simulation with the Heston model.
        
        Parameters:
        -----------
        option_type : str
            Type of option - 'call' or 'put'
        simulations : int
            Number of simulation paths
        steps : int
            Number of time steps per path
            
        Returns:
        --------
        dict
            Dictionary containing pricing results and statistics
        """
        time_points, prices, _ = self._simulate_path(steps, simulations)
        
        # Terminal stock prices
        S_T = prices[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Calculate option price and statistics
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(simulations)
        std_error_pv = np.exp(-self.r * self.T) * std_error
        confidence_interval = (
            option_price - 1.96 * std_error_pv, 
            option_price + 1.96 * std_error_pv
        )
        
        return {
            'price': option_price,
            'std_error': std_error_pv,
            'confidence_interval': confidence_interval,
            'accuracy': 1.96 * std_error_pv / option_price * 100 if option_price > 0 else float('inf')
        }
    
    def plot_simulation(self, simulations=10, steps=252, plot_volatility=True, figsize=(12, 10)):
        """
        Plot simulated price and volatility paths.
        
        Parameters:
        -----------
        simulations : int
            Number of paths to plot
        steps : int
            Number of time steps per path
        plot_volatility : bool
            Whether to also plot volatility paths
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        time_points, prices, volatilities = self._simulate_path(steps, simulations)
        
        if plot_volatility:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # Plot price paths
        for i in range(simulations):
            ax1.plot(time_points, prices[i], alpha=0.7, linewidth=1)
        
        ax1.set_title(f'Heston Model: {simulations} Simulated Price Paths')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Stock Price')
        ax1.axhline(y=self.K, color='r', linestyle='--', label=f'Strike Price (K={self.K})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot volatility paths if requested
        if plot_volatility:
            for i in range(simulations):
                ax2.plot(time_points, volatilities[i] * 100, alpha=0.7, linewidth=1)  # Convert to percentage
            
            ax2.set_title('Corresponding Volatility Paths')
            ax2.set_xlabel('Time (years)')
            ax2.set_ylabel('Volatility (%)')
            ax2.axhline(y=np.sqrt(self.theta) * 100, color='g', linestyle='--', 
                       label=f'Long-term Volatility (θ={np.sqrt(self.theta)*100:.1f}%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def implied_volatility_surface(self, strikes_range=(0.7, 1.3), tenors_range=(0.1, 2.0), 
                                 num_strikes=20, num_tenors=10, option_type='call'):
        """
        Generate implied volatility surface data.
        
        Parameters:
        -----------
        strikes_range : tuple
            Range of strikes as a fraction of current price (min, max)
        tenors_range : tuple
            Range of tenors in years (min, max)
        num_strikes : int
            Number of strike points
        num_tenors : int
            Number of tenor points
        option_type : str
            Type of option - 'call' or 'put'
            
        Returns:
        --------
        tuple
            (strikes, tenors, volatility_surface)
        """
        # Generate strike and tenor grids
        strikes = np.linspace(self.S * strikes_range[0], self.S * strikes_range[1], num_strikes)
        tenors = np.linspace(tenors_range[0], tenors_range[1], num_tenors)
        
        # Initialize volatility surface
        implied_vol_surface = np.zeros((num_strikes, num_tenors))
        
        # Calculate option prices for each strike and tenor
        for i, strike in enumerate(strikes):
            for j, tenor in enumerate(tenors):
                # Create a new model instance with the current strike and tenor
                model = HestonModel(
                    S=self.S, 
                    K=strike, 
                    T=tenor, 
                    r=self.r, 
                    v0=self.v0, 
                    kappa=self.kappa, 
                    theta=self.theta, 
                    sigma=self.sigma, 
                    rho=self.rho
                )
                
                # Calculate option price using the Heston model
                heston_price = model.price(option_type=option_type)
                
                # Convert strike to moneyness for output
                moneyness = strike / self.S
                
                # For IV surface, we directly compute the implied volatility from the model
                # For simplicity, we'll use the square root of the initial variance as the IV
                # In practice, you'd typically solve for the Black-Scholes IV that matches the Heston price
                vol = np.sqrt(self.v0 * (1 + self.kappa * tenor * (self.theta / self.v0 - 1) + 
                                         self.rho * self.sigma * tenor / (2 * self.v0)))
                
                implied_vol_surface[i, j] = vol
        
        return strikes / self.S, tenors, implied_vol_surface  # Return moneyness instead of absolute strikes


# Example usage:
if __name__ == "__main__":
    # Typical Heston parameters for an equity index
    params = {
        'S': 100.0,        # Current stock price
        'K': 100.0,        # Strike price
        'T': 1.0,          # 1 year to maturity
        'r': 0.02,         # 2% risk-free rate
        'v0': 0.04,        # Initial variance (0.04 = 20% volatility squared)
        'kappa': 2.0,      # Mean reversion speed
        'theta': 0.04,     # Long-term variance
        'sigma': 0.3,      # Volatility of volatility
        'rho': -0.7        # Correlation (typically negative for equity indexes)
    }
    
    # Create model instance
    heston = HestonModel(**params)
    
    # Price a call option
    call_price = heston.price(option_type='call')
    print(f"Heston Call Price: ${call_price:.4f}")
    
    # Price a put option
    put_price = heston.price(option_type='put')
    print(f"Heston Put Price: ${put_price:.4f}")
    
    # Simulate and plot
    heston.plot_simulation(simulations=5, steps=252)
    plt