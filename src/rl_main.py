#!/usr/bin/env python3
"""
Main script for the PPO RL Trading System.

This script provides a command-line interface for training and testing
a PPO (Proximal Policy Optimization) reinforcement learning agent for
stock trading. It integrates with the existing PPO (Percentage Price Oscillator)
indicator system to provide a comprehensive trading solution.
"""

import os
import sys
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.rl.ppo_rl_strategy import PPORLStrategy
from src.strategies.ppo_strategy import PPOStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPO RL Trading System')
    
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol to trade')
    parser.add_argument('--start-date', type=str, default='2018-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for data (YYYY-MM-DD)')
    
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance for trading')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost as a percentage')
    parser.add_argument('--window-size', type=int, default=30, help='Window size for feature calculation')
    parser.add_argument('--reward-scaling', type=float, default=1.0, help='Scaling factor for rewards')
    
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for PPO')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for PPO')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs for PPO')
    
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--model-name', type=str, default='ppo_trading', help='Name of the model')
    parser.add_argument('--tensorboard-log', type=str, default='./tensorboard/', help='Directory for TensorBoard logs')
    
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'compare'], default='train',
                        help='Mode: train, test, or compare with PPO indicator')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for testing')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training if available')
    
    return parser.parse_args()

def train_model(args):
    """Train a PPO RL model."""
    logger.info(f"Training PPO RL model for {args.ticker} from {args.start_date} to {args.end_date}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    strategy = PPORLStrategy(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        transaction_cost_pct=args.transaction_cost,
        window_size=args.window_size,
        reward_scaling=args.reward_scaling,
        model_name=args.model_name,
        tensorboard_log=args.tensorboard_log,
        use_gpu=args.use_gpu
    )
    
    model_path = os.path.join(args.output_dir, f"{args.model_name}.zip")
    strategy.train(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_path=model_path
    )
    
    results = strategy.run_backtest()
    
    metrics_path = os.path.join(args.output_dir, f"{args.model_name}_metrics.csv")
    metrics_df = pd.DataFrame({
        'Metric': list(results['metrics']['portfolio'].keys()),
        'Portfolio': list(results['metrics']['portfolio'].values()),
        'Market': list(results['metrics']['market'].values())
    })
    metrics_df.to_csv(metrics_path, index=False)
    
    plot_path = os.path.join(args.output_dir, f"{args.model_name}_performance.png")
    results['performance_plot'].savefig(plot_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Performance plot saved to {plot_path}")
    
    logger.info("Portfolio Metrics:")
    for key, value in results['metrics']['portfolio'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nMarket Metrics:")
    for key, value in results['metrics']['market'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nComparison:")
    for key, value in results['metrics']['comparison'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    return strategy, results

def test_model(args):
    """Test a trained PPO RL model."""
    logger.info(f"Testing PPO RL model for {args.ticker} from {args.start_date} to {args.end_date}")
    
    if args.model_path is None:
        logger.error("Model path must be provided for testing")
        return None, None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    strategy = PPORLStrategy(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        transaction_cost_pct=args.transaction_cost,
        window_size=args.window_size,
        reward_scaling=args.reward_scaling,
        model_name=args.model_name,
        tensorboard_log=args.tensorboard_log,
        use_gpu=args.use_gpu
    )
    
    strategy.load(args.model_path)
    
    results = strategy.run_backtest()
    
    metrics_path = os.path.join(args.output_dir, f"{args.model_name}_test_metrics.csv")
    metrics_df = pd.DataFrame({
        'Metric': list(results['metrics']['portfolio'].keys()),
        'Portfolio': list(results['metrics']['portfolio'].values()),
        'Market': list(results['metrics']['market'].values())
    })
    metrics_df.to_csv(metrics_path, index=False)
    
    plot_path = os.path.join(args.output_dir, f"{args.model_name}_test_performance.png")
    results['performance_plot'].savefig(plot_path)
    
    logger.info(f"Test metrics saved to {metrics_path}")
    logger.info(f"Test performance plot saved to {plot_path}")
    
    logger.info("Portfolio Metrics:")
    for key, value in results['metrics']['portfolio'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nMarket Metrics:")
    for key, value in results['metrics']['market'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nComparison:")
    for key, value in results['metrics']['comparison'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    return strategy, results

def compare_strategies(args):
    """Compare PPO RL strategy with PPO indicator strategy."""
    logger.info(f"Comparing strategies for {args.ticker} from {args.start_date} to {args.end_date}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    rl_strategy = PPORLStrategy(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        transaction_cost_pct=args.transaction_cost,
        window_size=args.window_size,
        reward_scaling=args.reward_scaling,
        model_name=args.model_name,
        tensorboard_log=args.tensorboard_log,
        use_gpu=args.use_gpu
    )
    
    if args.model_path:
        rl_strategy.load(args.model_path)
    else:
        model_path = os.path.join(args.output_dir, f"{args.model_name}.zip")
        rl_strategy.train(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            save_path=model_path
        )
    
    rl_results = rl_strategy.run_backtest()
    
    ppo_strategy = PPOStrategy(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        use_gpu=args.use_gpu
    )
    
    import yfinance as yf
    data = yf.download(args.ticker, start=args.start_date, end=args.end_date)
    
    ppo_results = ppo_strategy.backtest(data['Close'], data.index)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rl_portfolio_values = rl_results['portfolio_values']
    rl_dates = rl_results['dates']
    
    ppo_cumulative_returns = (1 + ppo_results['Strategy_Return']).cumprod()
    
    market_values = rl_results['market_values']
    
    rl_portfolio_values = [val / rl_portfolio_values[0] for val in rl_portfolio_values]
    market_values = [val / market_values[0] for val in market_values]
    
    ax.plot(rl_dates, rl_portfolio_values, label='PPO RL Strategy')
    ax.plot(ppo_results.index, ppo_cumulative_returns, label='PPO Indicator Strategy')
    ax.plot(rl_dates, market_values, label='Market (Buy & Hold)')
    
    ax.set_title('Strategy Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    ax.grid(True)
    
    plot_path = os.path.join(args.output_dir, "strategy_comparison.png")
    plt.savefig(plot_path)
    
    logger.info(f"Comparison plot saved to {plot_path}")
    
    logger.info("PPO RL Strategy Metrics:")
    for key, value in rl_results['metrics']['portfolio'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    ppo_total_return = ppo_cumulative_returns.iloc[-1] - 1
    ppo_annual_return = (1 + ppo_total_return) ** (252 / len(ppo_results)) - 1
    ppo_volatility = ppo_results['Strategy_Return'].std() * np.sqrt(252)
    ppo_sharpe = ppo_annual_return / ppo_volatility if ppo_volatility > 0 else 0
    
    logger.info("\nPPO Indicator Strategy Metrics:")
    logger.info(f"  Total Return: {ppo_total_return:.4f}")
    logger.info(f"  Annual Return: {ppo_annual_return:.4f}")
    logger.info(f"  Volatility: {ppo_volatility:.4f}")
    logger.info(f"  Sharpe Ratio: {ppo_sharpe:.4f}")
    
    logger.info("\nMarket Metrics:")
    for key, value in rl_results['metrics']['market'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    return rl_strategy, ppo_strategy

def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'compare':
        compare_strategies(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
