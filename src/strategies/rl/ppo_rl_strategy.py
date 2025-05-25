import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import matplotlib.pyplot as plt

from src.models.rl.trading_env import TradingEnv
from src.models.rl.ppo_agent import PPOTradingAgent

class PPORLStrategy:
    """
    Trading strategy based on Proximal Policy Optimization (PPO) reinforcement learning.
    
    This strategy uses a PPO agent from StableBaselines3 to learn a trading policy
    that maximizes returns while minimizing drawdowns, with a focus on optimizing
    the Ulcer Performance Index (UPI).
    """
    
    def __init__(
        self,
        ticker: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1.0,
        model_name: str = "ppo_trading",
        tensorboard_log: str = "./tensorboard/",
        use_gpu: bool = True
    ):
        """
        Initialize the PPO RL strategy.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            initial_balance: Starting cash balance
            transaction_cost_pct: Transaction cost as a percentage
            window_size: Number of days to use for feature calculation
            reward_scaling: Scaling factor for rewards
            model_name: Name of the model
            tensorboard_log: Directory for TensorBoard logs
            use_gpu: Whether to use GPU acceleration if available
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.model_name = model_name
        self.tensorboard_log = tensorboard_log
        self.use_gpu = use_gpu
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        self.env = TradingEnv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost_pct,
            window_size=window_size,
            reward_scaling=reward_scaling,
            use_gpu=use_gpu
        )
        
        self.agent = PPOTradingAgent(
            env=self.env,
            model_name=model_name,
            tensorboard_log=tensorboard_log,
            device="auto" if use_gpu else "cpu"
        )
        
        self.trained = False
    
    def train(
        self,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        verbose: int = 1
    ):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total timesteps for training
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            clip_range_vf: Clipping parameter for value function
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum norm for the gradient clipping
            policy_kwargs: Additional arguments to be passed to the policy on creation
            save_path: Path to save the model
            verbose: Verbosity level
            
        Returns:
            Trained PPO agent
        """
        os.makedirs(self.tensorboard_log, exist_ok=True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.agent.train(
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            save_path=save_path,
            verbose=verbose
        )
        
        self.trained = True
        
        return self.agent
    
    def load(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.agent.load(path)
        self.trained = True
        return self.agent
    
    def generate_signals(self, prices):
        """
        Generate trading signals based on the trained PPO agent.
        
        Args:
            prices (numpy.ndarray): Array of price data
            
        Returns:
            tuple: (positions) as numpy array
        """
        if not self.trained:
            raise ValueError("Model not trained or loaded")
        
        observation, _ = self.env.reset()
        
        positions = np.zeros(len(prices))
        
        for i in range(len(prices)):
            action, _ = self.agent.predict(observation, deterministic=True)
            
            if action == 0:
                positions[i] = 0  # Sell/Cash
            elif action == 2:
                positions[i] = 1  # Buy/Long
            else:
                positions[i] = positions[i-1] if i > 0 else 0  # Hold previous position
            
            observation, _, done, _, _ = self.env.step(action)
            
            if done:
                break
        
        return positions
    
    def backtest(self, prices, dates):
        """
        Backtest the PPO strategy.
        
        Args:
            prices (numpy.ndarray): Array of price data
            dates (numpy.ndarray): Array of dates
            
        Returns:
            pandas.DataFrame: Backtest results
        """
        if not self.trained:
            raise ValueError("Model not trained or loaded")
        
        if hasattr(prices, 'flatten'):
            prices_1d = prices.flatten()
        else:
            prices_1d = prices
        
        positions = self.generate_signals(prices_1d)
        
        daily_returns = np.zeros_like(prices_1d)
        daily_returns[1:] = (prices_1d[1:] / prices_1d[:-1]) - 1
        
        shifted_positions = np.zeros_like(positions)
        shifted_positions[1:] = positions[:-1]
        
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
            'Position': positions,
            'Daily_Return': daily_returns,
            'Strategy_Return': strategy_returns,
            'Cumulative_Return': cumulative_returns
        })
        
        return results
    
    def run_backtest(self, deterministic=True):
        """
        Run a full backtest using the agent's backtest method.
        
        Args:
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with backtest results
        """
        if not self.trained:
            raise ValueError("Model not trained or loaded")
        
        return self.agent.backtest(deterministic=deterministic)
    
    def plot_results(self, results, benchmark_results=None, save_path=None):
        """
        Plot the backtest results.
        
        Args:
            results: DataFrame with backtest results
            benchmark_results: DataFrame with benchmark results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        (1 + results['Cumulative_Return']).plot(ax=ax, label='PPO RL Strategy')
        
        if benchmark_results is not None:
            (1 + benchmark_results['Cumulative_Return']).plot(ax=ax, label='Benchmark')
        
        ax.set_title('PPO RL Strategy Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
