import torch
import numpy as np

class PPO:
    """
    Percentage Price Oscillator (PPO) indicator with GPU acceleration.
    
    The PPO is a momentum oscillator that measures the difference between two moving averages
    as a percentage. It consists of three components:
    1. PPO Line: ((Fast EMA - Slow EMA) / Slow EMA) * 100
    2. Signal Line: EMA of the PPO Line
    3. Histogram: PPO Line - Signal Line
    
    This implementation uses PyTorch for GPU acceleration when available.
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, use_gpu=True):
        """
        Initialize the PPO indicator.
        
        Args:
            fast_period (int): Period for the fast EMA
            slow_period (int): Period for the slow EMA
            signal_period (int): Period for the signal line EMA
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
    def _exponential_moving_average(self, data, period):
        """
        Calculate the Exponential Moving Average (EMA) using PyTorch.
        
        Args:
            data (torch.Tensor): Price data
            period (int): EMA period
            
        Returns:
            torch.Tensor: EMA values
        """
        alpha = 2.0 / (period + 1)
        alpha_tensor = torch.tensor(alpha, device=self.device)
        
        ema = torch.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha_tensor * data[i] + (1 - alpha_tensor) * ema[i-1]
            
        return ema
    
    def calculate(self, prices):
        """
        Calculate the PPO, signal line, and histogram.
        
        Args:
            prices (numpy.ndarray or pandas.DataFrame): Price data
            
        Returns:
            tuple: (ppo, signal, histogram) as numpy arrays
        """
        if hasattr(prices, 'values'):
            prices_np = prices.values
        elif hasattr(prices, 'to_numpy'):
            prices_np = prices.to_numpy()
        else:
            prices_np = prices
            
        if hasattr(prices_np, 'flatten'):
            prices_np = prices_np.flatten()
            
        prices_tensor = torch.tensor(prices_np, dtype=torch.float32, device=self.device)
        
        fast_ema = self._exponential_moving_average(prices_tensor, self.fast_period)
        slow_ema = self._exponential_moving_average(prices_tensor, self.slow_period)
        
        ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100.0
        
        signal_line = self._exponential_moving_average(ppo_line, self.signal_period)
        
        histogram = ppo_line - signal_line
        
        return (
            ppo_line.cpu().numpy().flatten(),
            signal_line.cpu().numpy().flatten(),
            histogram.cpu().numpy().flatten()
        )
    
    def generate_signals(self, prices):
        """
        Generate buy/sell signals based on PPO crossovers and histogram confirmation.
        
        Args:
            prices (numpy.ndarray): Array of price data
            
        Returns:
            tuple: (ppo, signal, histogram, buy_signals, sell_signals) as numpy arrays
        """
        ppo, signal, histogram = self.calculate(prices)
        
        buy_signals = np.zeros_like(ppo)
        sell_signals = np.zeros_like(ppo)
        
        for i in range(1, len(ppo)):
            if ppo[i-1] < signal[i-1] and ppo[i] > signal[i] and histogram[i] > 0:
                buy_signals[i] = 1
            elif (ppo[i-1] > signal[i-1] and ppo[i] < signal[i]) or (histogram[i-1] > 0 and histogram[i] < 0):
                sell_signals[i] = 1
                
        return ppo, signal, histogram, buy_signals, sell_signals
