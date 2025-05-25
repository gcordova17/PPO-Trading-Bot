import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training.
    
    This callback logs portfolio performance metrics to TensorBoard,
    including returns, Sharpe ratio, and Ulcer Performance Index.
    """
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.market_values = []
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        info = self.locals['infos'][0]
        
        if 'portfolio_value' in info and 'market_value' in info:
            self.portfolio_values.append(info['portfolio_value'])
            self.market_values.append(info['market_value'])
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(sum(self.locals['rewards']))
            self.episode_lengths.append(self.num_timesteps - sum(self.episode_lengths))
            
            if len(self.portfolio_values) > 10:
                portfolio_series = pd.Series(self.portfolio_values)
                market_series = pd.Series(self.market_values)
                
                portfolio_returns = portfolio_series.pct_change().dropna()
                market_returns = market_series.pct_change().dropna()
                
                if len(portfolio_returns) > 1:
                    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
                    market_return = (market_series.iloc[-1] / market_series.iloc[0]) - 1
                    excess_return = total_return - market_return
                    
                    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
                    
                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    running_max = cumulative_returns.cummax()
                    drawdowns = (cumulative_returns / running_max) - 1
                    max_drawdown = drawdowns.min()
                    
                    squared_drawdowns = drawdowns ** 2
                    ulcer_index = np.sqrt(squared_drawdowns.mean())
                    
                    risk_free_rate = 0.0
                    upi = (portfolio_returns.mean() - risk_free_rate) / ulcer_index if ulcer_index > 0 else 0
                    
                    self.logger.record('metrics/total_return', total_return)
                    self.logger.record('metrics/excess_return', excess_return)
                    self.logger.record('metrics/sharpe_ratio', sharpe_ratio)
                    self.logger.record('metrics/max_drawdown', max_drawdown)
                    self.logger.record('metrics/ulcer_index', ulcer_index)
                    self.logger.record('metrics/ulcer_performance_index', upi)
            
            self.portfolio_values = []
            self.market_values = []
        
        return True

class PPOTradingAgent:
    """
    PPO Trading Agent using StableBaselines3.
    
    This agent uses Proximal Policy Optimization (PPO) to learn a trading policy
    that maximizes returns while minimizing drawdowns.
    
    Attributes:
        env: Trading environment
        model_name: Name of the model
        tensorboard_log: Directory for TensorBoard logs
        device: Device to use for training (CPU or GPU)
        model: Trained PPO model
    """
    
    def __init__(
        self,
        env,
        model_name: str = "ppo_trading",
        tensorboard_log: str = "./tensorboard/",
        device: str = "auto"
    ):
        """
        Initialize the PPO Trading Agent.
        
        Args:
            env: Trading environment (gym.Env)
            model_name: Name of the model
            tensorboard_log: Directory for TensorBoard logs
            device: Device to use for training (auto, cpu, cuda)
        """
        self.env = env
        self.model_name = model_name
        self.tensorboard_log = tensorboard_log
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.vec_env = DummyVecEnv([lambda: env])
        
        self.model = None
    
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
        Train the PPO model.
        
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
            Trained PPO model
        """
        os.makedirs(self.tensorboard_log, exist_ok=True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": torch.nn.Tanh
            }
        
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                self.vec_env,
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
                tensorboard_log=self.tensorboard_log,
                verbose=verbose,
                device=self.device
            )
        
        callback = TensorboardCallback()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=self.model_name
        )
        
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return self.model
    
    def load(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model = PPO.load(path, env=self.vec_env, device=self.device)
        return self.model
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action based on observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def backtest(self, deterministic: bool = True):
        """
        Backtest the model on the environment.
        
        Args:
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        observation, info = self.env.reset()
        
        done = False
        portfolio_values = []
        market_values = []
        dates = []
        actions = []
        
        while not done:
            action, _ = self.predict(observation, deterministic=deterministic)
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            market_values.append(info.get('market_value', 0))
            dates.append(info['date'])
            actions.append(action)
        
        metrics = self.env.calculate_metrics()
        
        performance_plot = self.env.plot_performance()
        
        return {
            'portfolio_values': portfolio_values,
            'market_values': market_values,
            'dates': dates,
            'actions': actions,
            'metrics': metrics,
            'performance_plot': performance_plot
        }
