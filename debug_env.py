#!/usr/bin/env python3
"""
Debug script for the TradingEnv class
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_trading_env():
    """Debug the trading environment observation space"""
    env = TradingEnv(
        ticker="SPY",
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_balance=10000.0,
        transaction_cost_pct=0.001,
        window_size=30,
        reward_scaling=1.0
    )
    
    current_step = env.window_size
    features = env.features.iloc[current_step]
    
    print(f"Features type: {type(features)}")
    print(f"Features shape: {features.shape}")
    print(f"Features values: {features.values}")
    print(f"Features values type: {type(features.values)}")
    print(f"Features values shape: {features.values.shape}")
    
    for col in env.features.columns:
        val = features[col]
        print(f"Feature {col}: {val} (type: {type(val)})")
    
    current_price = env.data['Close'].iloc[current_step]
    portfolio_value = env.balance + env.shares_held * current_price
    
    normalized_balance = env.balance / env.initial_balance - 1
    normalized_shares = env.shares_held * current_price / env.initial_balance
    normalized_portfolio_value = portfolio_value / env.initial_balance - 1
    
    print(f"normalized_balance: {normalized_balance} (type: {type(normalized_balance)})")
    print(f"normalized_shares: {normalized_shares} (type: {type(normalized_shares)})")
    print(f"normalized_portfolio_value: {normalized_portfolio_value} (type: {type(normalized_portfolio_value)})")
    
    account_info = np.array([float(normalized_balance), float(normalized_shares), float(normalized_portfolio_value)], dtype=np.float32)
    print(f"account_info: {account_info} (type: {type(account_info)}, shape: {account_info.shape})")
    
    try:
        features_array = features.values.astype(np.float32)
        print(f"features_array shape: {features_array.shape}")
        
        observation = np.concatenate([features_array, account_info])
        print(f"observation: {observation} (type: {type(observation)}, shape: {observation.shape})")
    except Exception as e:
        print(f"Error concatenating: {e}")
        
        try:
            features_flat = features.values.flatten().astype(np.float32)
            print(f"features_flat shape: {features_flat.shape}")
            
            observation = np.concatenate([features_flat, account_info])
            print(f"observation with flattened features: {observation} (type: {type(observation)}, shape: {observation.shape})")
        except Exception as e2:
            print(f"Error concatenating with flattened features: {e2}")

if __name__ == "__main__":
    debug_trading_env()
