#!/usr/bin/env python3
"""
Test script for the PPO Trading System
This script tests the core components of the system
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.trading_env import TradingEnv
from app.models.ppo_agent import PPOTradingAgent
from app.models.backtest import BacktestAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trading_env():
    """Test the trading environment"""
    logger.info("Testing trading environment...")
    
    env = TradingEnv(
        ticker="SPY",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_balance=10000.0,
        transaction_cost_pct=0.001,
        window_size=30,
        reward_scaling=1.0
    )
    
    obs, info = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    logger.info(f"Initial info: {info}")
    
    actions = [0, 1, 2]  # Sell, Hold, Buy
    for action in actions:
        logger.info(f"Testing action: {action}")
        obs, reward, done, _, info = env.step(action)
        logger.info(f"Reward: {reward}")
        logger.info(f"Info: {info}")
    
    metrics = env.calculate_metrics()
    logger.info("Metrics calculated successfully")
    
    plt.figure(figsize=(10, 6))
    env.plot_performance()
    plt.savefig("test_env_performance.png")
    plt.close()
    logger.info("Performance plot created successfully")
    
    return env

def test_ppo_agent(env, train=True):
    """Test the PPO agent"""
    logger.info("Testing PPO agent...")
    
    agent = PPOTradingAgent(
        env=env,
        model_name="test_ppo",
        tensorboard_log="./tensorboard/",
        device="auto"
    )
    
    os.makedirs("app/data/models", exist_ok=True)
    
    if train:
        logger.info("Training model (small number of steps for testing)...")
        agent.train(
            total_timesteps=1000,  # Small number for testing
            save_path="app/data/models/test_model.zip"
        )
    else:
        if os.path.exists("app/data/models/test_model.zip"):
            logger.info("Loading model...")
            agent.load("app/data/models/test_model.zip")
        else:
            logger.info("No model found, training...")
            agent.train(
                total_timesteps=1000,  # Small number for testing
                save_path="app/data/models/test_model.zip"
            )
    
    logger.info("Running backtest...")
    results = agent.backtest(deterministic=True)
    
    metrics = results['metrics']
    logger.info("Portfolio metrics:")
    for key, value in metrics['portfolio'].items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("\nMarket metrics:")
    for key, value in metrics['market'].items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("\nComparison:")
    for key, value in metrics['comparison'].items():
        logger.info(f"{key}: {value:.4f}")
    
    os.makedirs("app/data/plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plot = results['performance_plot']
    plot.savefig("app/data/plots/test_performance.png")
    logger.info("Plot saved to app/data/plots/test_performance.png")
    
    return results

def test_backtest_analyzer():
    """Test the backtest analyzer"""
    logger.info("Testing backtest analyzer...")
    
    from datetime import timedelta
    dates = [datetime(2022, 1, 1) + i * timedelta(days=1) for i in range(100)]
    portfolio_values = [10000 * (1 + 0.001 * i + 0.0005 * np.sin(i/10)) for i in range(100)]
    market_values = [10000 * (1 + 0.0008 * i) for i in range(100)]
    
    analyzer = BacktestAnalyzer(portfolio_values, market_values, dates)
    
    metrics = analyzer.calculate_metrics()
    logger.info("Metrics calculated successfully")
    
    report = analyzer.generate_report()
    logger.info("Report generated successfully")
    
    plt.figure(figsize=(12, 6))
    analyzer.plot_performance()
    plt.savefig("test_analyzer_performance.png")
    plt.close()
    logger.info("Performance plot created successfully")
    
    return analyzer

def main():
    """Main test function"""
    logger.info("Starting system tests...")
    
    env = test_trading_env()
    
    test_ppo_agent(env)
    
    test_backtest_analyzer()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()
