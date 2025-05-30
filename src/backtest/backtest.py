import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from src.utils.data_loader import DataLoader
from src.strategies.ppo_strategy import PPOStrategy
from src.utils.performance import PerformanceMetrics

class Backtest:
    """
    Backtesting framework for trading strategies.
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the backtesting framework.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.data_loader = DataLoader(use_gpu)
        self.performance = PerformanceMetrics(use_gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
    def run(self, ticker, start_date, end_date, strategy, benchmark_ticker='SPY'):
        """
        Run a backtest for a given strategy.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            strategy: Strategy object with a backtest method
            benchmark_ticker (str): Ticker symbol for benchmark
            
        Returns:
            tuple: (results, benchmark_results, performance_metrics)
        """
        data = self.data_loader.load_ticker_data(ticker, start_date, end_date)
        if data is None or data.empty:
            print(f"Error: No data available for {ticker}")
            return None, None, None
        
        processed_data = self.data_loader.preprocess_data(data)
        if processed_data is None or processed_data.empty:
            print(f"Error: Failed to preprocess data for {ticker}")
            return None, None, None
        
        benchmark_data = None
        if benchmark_ticker:
            benchmark_data = self.data_loader.load_ticker_data(benchmark_ticker, start_date, end_date)
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_data = self.data_loader.preprocess_data(benchmark_data)
        
        prices = processed_data['Close'].values
        dates = processed_data.index.values
        
        results = strategy.backtest(prices, dates)
        if results is None:
            print(f"Error: Strategy backtest failed")
            return None, None, None
        
        benchmark_results = None
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_prices = benchmark_data['Close'].values
            
            # Ensure proper conversion of DatetimeIndex to numpy array
            if isinstance(benchmark_data.index, pd.DatetimeIndex):
                benchmark_dates = benchmark_data.index.to_numpy()
            else:
                benchmark_dates = benchmark_data.index.values
                
            if hasattr(benchmark_dates, 'flatten'):
                benchmark_dates = benchmark_dates.flatten()
            
            if hasattr(benchmark_prices, 'shape') and len(benchmark_prices.shape) > 1:
                benchmark_prices = benchmark_prices.flatten()
            
            # Calculate benchmark returns
            benchmark_returns = np.zeros_like(benchmark_prices)
            benchmark_returns[1:] = (benchmark_prices[1:] / benchmark_prices[:-1]) - 1
            
            # Calculate benchmark cumulative returns manually to avoid broadcasting issues
            benchmark_cumulative_returns = np.zeros_like(benchmark_returns)
            cumulative_value = 1.0
            for i in range(len(benchmark_returns)):
                cumulative_value *= (1 + benchmark_returns[i])
                benchmark_cumulative_returns[i] = cumulative_value - 1
            
            benchmark_results = pd.DataFrame({
                'Date': benchmark_dates,
                'Price': benchmark_prices,
                'Daily_Return': benchmark_returns,
                'Cumulative_Return': benchmark_cumulative_returns
            })
        
        performance_metrics = self._calculate_performance_metrics(results, benchmark_results)
        
        return results, benchmark_results, performance_metrics
    
    def _calculate_performance_metrics(self, results, benchmark_results):
        """
        Calculate performance metrics for the strategy and benchmark.
        
        Args:
            results (pandas.DataFrame): Strategy results
            benchmark_results (pandas.DataFrame): Benchmark results
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Ensure proper conversion of values to 1D arrays
        price_values = results['Price'].values
        if hasattr(price_values, 'flatten'):
            price_values = price_values.flatten()
            
        position_values = results['Position'].values
        if hasattr(position_values, 'flatten'):
            position_values = position_values.flatten()
            
        strategy_metrics = self.performance.calculate_performance_metrics(
            price_values,
            position_values
        )
        
        benchmark_metrics = None
        if benchmark_results is not None:
            benchmark_positions = np.ones_like(benchmark_results['Price'].values)
            
            # Ensure proper conversion of benchmark values to 1D arrays
            benchmark_price_values = benchmark_results['Price'].values
            if hasattr(benchmark_price_values, 'flatten'):
                benchmark_price_values = benchmark_price_values.flatten()
                
            if hasattr(benchmark_positions, 'flatten'):
                benchmark_positions = benchmark_positions.flatten()
                
            benchmark_metrics = self.performance.calculate_performance_metrics(
                benchmark_price_values,
                benchmark_positions
            )
        
        metrics = {
            'Strategy': strategy_metrics,
            'Benchmark': benchmark_metrics
        }
        
        return metrics
    
    def plot_results(self, results, benchmark_results=None, save_path=None):
        """
        Plot backtest results.
        
        Args:
            results (pandas.DataFrame): Strategy results
            benchmark_results (pandas.DataFrame, optional): Benchmark results
            save_path (str, optional): Path to save the plots
            
        Returns:
            list: List of figure objects
        """
        figures = []
        
        if benchmark_results is not None:
            common_dates = np.intersect1d(results['Date'], benchmark_results['Date'])
            strategy_data = results[results['Date'].isin(common_dates)]
            benchmark_data = benchmark_results[benchmark_results['Date'].isin(common_dates)]
            
            fig = self.performance.plot_equity_curve(
                common_dates,
                strategy_data['Strategy_Return'].values,
                benchmark_data['Daily_Return'].values,
                title=f'Equity Curve: Strategy vs Benchmark'
            )
        else:
            fig = self.performance.plot_equity_curve(
                results['Date'].values,
                results['Strategy_Return'].values,
                title='Equity Curve: Strategy'
            )
        
        figures.append(fig)
        
        fig = self.performance.plot_drawdowns(
            results['Date'].values,
            results['Cumulative_Return'].values,
            title='Strategy Drawdowns'
        )
        figures.append(fig)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results['Date'], results['PPO'], label='PPO', color='blue')
        ax.plot(results['Date'], results['Signal'], label='Signal', color='red')
        ax.bar(results['Date'], results['Histogram'], label='Histogram', color='green', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('PPO Indicator')
        ax.legend()
        ax.grid(True)
        figures.append(fig)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results['Date'], results['Price'], label='Price', color='blue')
        
        buy_signals = results[results['Position'].diff() > 0]
        ax.scatter(buy_signals['Date'], buy_signals['Price'], color='green', label='Buy', marker='^', s=100)
        
        sell_signals = results[results['Position'].diff() < 0]
        ax.scatter(sell_signals['Date'], sell_signals['Price'], color='red', label='Sell', marker='v', s=100)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Buy/Sell Signals')
        ax.legend()
        ax.grid(True)
        figures.append(fig)
        
        if save_path:
            for i, fig in enumerate(figures):
                fig.savefig(f"{save_path}/figure_{i}.png")
        
        return figures
