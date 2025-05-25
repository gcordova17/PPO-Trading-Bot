import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class PerformanceMetrics:
    """
    Class for calculating and visualizing trading strategy performance metrics.
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the PerformanceMetrics calculator.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    
    def calculate_returns(self, prices, positions):
        """
        Calculate returns based on positions.
        
        Args:
            prices (numpy.ndarray): Array of price data
            positions (numpy.ndarray): Array of positions (1 for long, 0 for cash, -1 for short)
            
        Returns:
            numpy.ndarray: Strategy returns
        """
        price_returns = np.zeros_like(prices)
        price_returns[1:] = (prices[1:] / prices[:-1]) - 1
        
        shifted_positions = np.zeros_like(positions)
        shifted_positions[2:] = positions[:-2]
        
        strategy_returns = price_returns * shifted_positions
        
        return strategy_returns
    
    def calculate_cumulative_returns(self, returns):
        """
        Calculate cumulative returns.
        
        Args:
            returns (numpy.ndarray): Array of returns
            
        Returns:
            numpy.ndarray: Cumulative returns
        """
        return np.cumprod(1 + returns) - 1
    
    def calculate_drawdowns(self, cumulative_returns):
        """
        Calculate drawdowns.
        
        Args:
            cumulative_returns (numpy.ndarray): Array of cumulative returns
            
        Returns:
            numpy.ndarray: Drawdowns
        """
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        
        drawdowns = (cumulative_returns + 1) / running_max - 1
        
        return drawdowns
    
    def calculate_ulcer_index(self, cumulative_returns, window=14):
        """
        Calculate the Ulcer Index, which measures downside risk.
        
        The Ulcer Index is the square root of the mean of the squared percentage drawdowns.
        
        Args:
            cumulative_returns (numpy.ndarray): Array of cumulative returns
            window (int): Window size for calculation
            
        Returns:
            float: Ulcer Index
        """
        drawdowns = self.calculate_drawdowns(cumulative_returns)
        
        squared_drawdowns = drawdowns ** 2
        
        mean_squared_drawdown = np.mean(squared_drawdowns)
        
        ulcer_index = np.sqrt(mean_squared_drawdown)
        
        return ulcer_index
    
    def calculate_ulcer_performance_index(self, returns, risk_free_rate=0.0):
        """
        Calculate the Ulcer Performance Index (UPI).
        
        UPI = (Annual Return - Risk-Free Rate) / Ulcer Index
        
        Args:
            returns (numpy.ndarray): Array of returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Ulcer Performance Index
        """
        cumulative_returns = self.calculate_cumulative_returns(returns)
        
        total_return = cumulative_returns[-1]
        num_years = len(returns) / 252  # Assuming 252 trading days per year
        annual_return = (1 + total_return) ** (1 / num_years) - 1
        
        ulcer_index = self.calculate_ulcer_index(cumulative_returns)
        
        if ulcer_index == 0:
            return float('inf')  # Avoid division by zero
        
        upi = (annual_return - risk_free_rate) / ulcer_index
        
        return upi
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate the Sharpe Ratio.
        
        Args:
            returns (numpy.ndarray): Array of returns
            risk_free_rate (float): Annual risk-free rate
            annualization_factor (int): Number of periods in a year
            
        Returns:
            float: Sharpe Ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)
        
        return sharpe_ratio
    
    def calculate_max_drawdown(self, cumulative_returns):
        """
        Calculate the maximum drawdown.
        
        Args:
            cumulative_returns (numpy.ndarray): Array of cumulative returns
            
        Returns:
            float: Maximum drawdown
        """
        drawdowns = self.calculate_drawdowns(cumulative_returns)
        return np.min(drawdowns)
    
    def calculate_performance_metrics(self, prices, positions, risk_free_rate=0.0):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            prices (numpy.ndarray): Array of price data
            positions (numpy.ndarray): Array of positions
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            dict: Dictionary of performance metrics
        """
        returns = self.calculate_returns(prices, positions)
        
        cumulative_returns = self.calculate_cumulative_returns(returns)
        
        total_return = cumulative_returns[-1]
        num_years = len(returns) / 252  # Assuming 252 trading days per year
        annual_return = (1 + total_return) ** (1 / num_years) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        ulcer_index = self.calculate_ulcer_index(cumulative_returns)
        upi = self.calculate_ulcer_performance_index(returns, risk_free_rate)
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Ulcer Index': ulcer_index,
            'Ulcer Performance Index': upi
        }
        
        return metrics
    
    def plot_equity_curve(self, dates, strategy_returns, benchmark_returns=None, title='Equity Curve'):
        """
        Plot the equity curve of the strategy vs. benchmark.
        
        Args:
            dates (numpy.ndarray): Array of dates
            strategy_returns (numpy.ndarray): Array of strategy returns
            benchmark_returns (numpy.ndarray, optional): Array of benchmark returns
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        strategy_cum_returns = self.calculate_cumulative_returns(strategy_returns)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, (1 + strategy_cum_returns) * 100, label='Strategy', linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_cum_returns = self.calculate_cumulative_returns(benchmark_returns)
            ax.plot(dates, (1 + benchmark_cum_returns) * 100, label='Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdowns(self, dates, cumulative_returns, title='Drawdowns'):
        """
        Plot the drawdowns of the strategy.
        
        Args:
            dates (numpy.ndarray): Array of dates
            cumulative_returns (numpy.ndarray): Array of cumulative returns
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        drawdowns = self.calculate_drawdowns(cumulative_returns)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(dates, 0, drawdowns * 100, color='red', alpha=0.3)
        ax.plot(dates, drawdowns * 100, color='red', linewidth=1)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True)
        
        return fig
