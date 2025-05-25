import numpy as np
import pandas as pd
import torch
from src.indicators.ppo import PPO

class PPOStrategy:
    """
    Trading strategy based on the Percentage Price Oscillator (PPO).
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, use_gpu=True):
        """
        Initialize the PPO strategy.
        
        Args:
            fast_period (int): Period for the fast EMA
            slow_period (int): Period for the slow EMA
            signal_period (int): Period for the signal line EMA
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.ppo = PPO(fast_period, slow_period, signal_period, use_gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
    def generate_signals(self, prices):
        """
        Generate trading signals based on PPO.
        
        Args:
            prices (numpy.ndarray): Array of price data
            
        Returns:
            tuple: (ppo, signal, histogram, positions) as numpy arrays
        """
        ppo, signal, histogram, buy_signals, sell_signals = self.ppo.generate_signals(prices)
        
        positions = np.zeros_like(ppo)
        position = 0  # Start with no position
        
        for i in range(len(ppo)):
            if buy_signals[i] == 1:
                position = 1  # Buy/Long
            elif sell_signals[i] == 1:
                position = 0  # Sell/Cash (or -1 for short)
            
            positions[i] = position
        
        return ppo, signal, histogram, positions
    
    def backtest(self, prices, dates):
        """
        Backtest the PPO strategy.
        
        Args:
            prices (numpy.ndarray): Array of price data
            dates (numpy.ndarray): Array of dates
            
        Returns:
            pandas.DataFrame: Backtest results
        """
        if hasattr(prices, 'flatten'):
            prices_1d = prices.flatten()
        else:
            prices_1d = prices
            
        ppo, signal, histogram, positions = self.generate_signals(prices_1d)
        
        daily_returns = np.zeros_like(prices_1d)
        daily_returns[1:] = (prices_1d[1:] / prices_1d[:-1]) - 1
        
        shifted_positions = np.zeros_like(positions)
        shifted_positions[2:] = positions[:-2]
        
        strategy_returns = daily_returns * shifted_positions
        
        cumulative_returns = np.zeros_like(strategy_returns)
        cumulative_value = 1.0
        for i in range(len(strategy_returns)):
            cumulative_value *= (1 + strategy_returns[i])
            cumulative_returns[i] = cumulative_value - 1
        
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.to_numpy()
        
        results = pd.DataFrame({
            'Date': dates,
            'Price': prices_1d,
            'PPO': ppo,
            'Signal': signal,
            'Histogram': histogram,
            'Position': positions,
            'Daily_Return': daily_returns,
            'Strategy_Return': strategy_returns,
            'Cumulative_Return': cumulative_returns
        })
        
        return results
