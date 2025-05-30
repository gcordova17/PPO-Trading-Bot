#!/usr/bin/env python3
"""
Script to fix the backtesting error related to pandas Series conversion to numeric values.
"""
import os
import sys
import logging
import traceback
import shutil
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    main_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'backend', 'main.py')
    
    backup_path = os.path.join(os.path.dirname(main_file_path), 'main_backup_backtest.py')
    shutil.copy2(main_file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    with open(main_file_path, 'r') as f:
        content = f.read()
    
    train_error_pattern = r"tasks\[task_id\]\[\"message\"\] = f\"Error: \{str\(e\) \| \{dt\}\}\""
    if train_error_pattern in content:
        content = content.replace(
            'tasks[task_id]["message"] = f"Error: {str(e) | {dt}}"',
            'tasks[task_id]["message"] = f"Error: {str(e)}"'
        )
        logger.info("Fixed string formatting error in _train_model_task")
    
    content = re.sub(
        r'f"Error: \{str\(e\) \| \{.*?\}\}"',
        r'f"Error: {str(e)}"',
        content
    )
    
    with open(main_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {main_file_path}")
    
    backtest_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'backtest', 'backtest.py')
    
    backup_path = os.path.join(os.path.dirname(backtest_file_path), 'backtest_backup.py')
    shutil.copy2(backtest_file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    with open(backtest_file_path, 'r') as f:
        content = f.read()
    
    ticker_pattern = """            if isinstance(benchmark_data.index, pd.DatetimeIndex):
                benchmark_dates = benchmark_data.index.to_numpy()
            else:
                benchmark_dates = benchmark_data.index.values"""
    
    ticker_replacement = """            # Ensure proper conversion of DatetimeIndex to numpy array
            if isinstance(benchmark_data.index, pd.DatetimeIndex):
                benchmark_dates = benchmark_data.index.to_numpy()
            else:
                benchmark_dates = benchmark_data.index.values
                
            if hasattr(benchmark_dates, 'flatten'):
                benchmark_dates = benchmark_dates.flatten()"""
    
    content = content.replace(ticker_pattern, ticker_replacement)
    
    metrics_pattern = """        strategy_metrics = self.performance.calculate_performance_metrics(
            results['Price'].values,
            results['Position'].values
        )"""
    
    metrics_replacement = """        # Ensure proper conversion of values to 1D arrays
        price_values = results['Price'].values
        if hasattr(price_values, 'flatten'):
            price_values = price_values.flatten()
            
        position_values = results['Position'].values
        if hasattr(position_values, 'flatten'):
            position_values = position_values.flatten()
            
        strategy_metrics = self.performance.calculate_performance_metrics(
            price_values,
            position_values
        )"""
    
    content = content.replace(metrics_pattern, metrics_replacement)
    
    benchmark_pattern = """            benchmark_metrics = self.performance.calculate_performance_metrics(
                benchmark_results['Price'].values,
                benchmark_positions
            )"""
    
    benchmark_replacement = """            # Ensure proper conversion of benchmark values to 1D arrays
            benchmark_price_values = benchmark_results['Price'].values
            if hasattr(benchmark_price_values, 'flatten'):
                benchmark_price_values = benchmark_price_values.flatten()
                
            if hasattr(benchmark_positions, 'flatten'):
                benchmark_positions = benchmark_positions.flatten()
                
            benchmark_metrics = self.performance.calculate_performance_metrics(
                benchmark_price_values,
                benchmark_positions
            )"""
    
    content = content.replace(benchmark_pattern, benchmark_replacement)
    
    with open(backtest_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {backtest_file_path}")
    
    env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'models', 'rl', 'trading_env.py')
    
    backup_path = os.path.join(os.path.dirname(env_file_path), 'trading_env_backup_backtest.py')
    shutil.copy2(env_file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    with open(env_file_path, 'r') as f:
        content = f.read()
    
    metrics_pattern = """        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        market_series = pd.Series(self.market_values, index=self.dates)"""
    
    metrics_replacement = """        # Ensure portfolio_values and market_values are properly converted to 1D arrays
        portfolio_values = [float(val) if hasattr(val, 'item') else float(val) for val in self.portfolio_values]
        market_values = [float(val) if hasattr(val, 'item') else float(val) for val in self.market_values]
        
        portfolio_series = pd.Series(portfolio_values, index=self.dates)
        market_series = pd.Series(market_values, index=self.dates)"""
    
    content = content.replace(metrics_pattern, metrics_replacement)
    
    series_pattern = """        total_return = (values.iloc[-1] / values.iloc[0]) - 1"""
    
    series_replacement = """        # Ensure proper numeric conversion for calculations
        first_value = float(values.iloc[0])
        last_value = float(values.iloc[-1])
        total_return = (last_value / first_value) - 1"""
    
    content = content.replace(series_pattern, series_replacement)
    
    with open(env_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {env_file_path}")
    
    logger.info("All fixes have been applied successfully")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
    logger.error(traceback.format_exc())
