import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'portfolio_value' in info and 'initial_portfolio_value' in info:
            portfolio_return = info['portfolio_value'] / info['initial_portfolio_value'] - 1
            self.logger.record('metrics/portfolio_return', portfolio_return)
        
        return True

class PPOTradingAgent:
    """
    PPO agent for trading using StableBaselines3
    """
    def __init__(
        self,
        env,
        model_name: str = "ppo_trading",
        tensorboard_log: str = "./tensorboard/",
        device: str = "auto",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        """
        Initialize the PPO trading agent
        
        Args:
            env: Trading environment
            model_name: Name of the model
            tensorboard_log: Path to tensorboard logs
            device: Device to run the model on ('auto', 'cuda', or 'cpu')
            policy_kwargs: Additional arguments to be passed to the policy
            verbose: Verbosity level
        """
        self.env = env
        self.model_name = model_name
        self.tensorboard_log = tensorboard_log
        self.device = device
        self.verbose = verbose
        
        os.makedirs(tensorboard_log, exist_ok=True)
        
        if policy_kwargs is None:
            self.policy_kwargs = {
                "net_arch": [dict(pi=[128, 64, 32], vf=[128, 64, 32])]
            }
        else:
            self.policy_kwargs = policy_kwargs
            
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
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total timesteps to train for
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            clip_range: Clipping parameter for PPO
            clip_range_vf: Clipping parameter for the value function
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum norm for the gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every n steps
            target_kl: Target KL divergence threshold
            save_path: Path to save the model
        """
        env = Monitor(self.env)
        env = DummyVecEnv([lambda: env])
        
        callback = TensorboardCallback()
        
        if self.model is None:
            logger.info("Creating new PPO model")
            self.model = PPO(
                "MlpPolicy",
                env,
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
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                target_kl=target_kl,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                verbose=self.verbose
            )
        
        logger.info(f"Training model for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            
        return self.model
    
    def load(self, path: str):
        """
        Load a trained model
        
        Args:
            path: Path to the trained model
        """
        logger.info(f"Loading model from {path}")
        self.model = PPO.load(path, env=self.env)
        return self.model
    
    def predict(self, observation, state=None, deterministic=True):
        """
        Predict action based on observation
        
        Args:
            observation: Current observation
            state: Current state (for recurrent policies)
            deterministic: Whether to use deterministic actions
            
        Returns:
            action, state
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load() first.")
            
        return self.model.predict(observation, state, deterministic=deterministic)
    
    def backtest(self, env=None, deterministic=True, render=False):
        """
        Backtest the model on the environment
        
        Args:
            env: Environment to backtest on (uses self.env if None)
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment
            
        Returns:
            Dictionary of backtest results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() or load() first.")
            
        backtest_env = env if env is not None else self.env
        
        obs, info = backtest_env.reset()
        done = False
        
        while not done:
            action, _ = self.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = backtest_env.step(action)
            
            if render:
                backtest_env.render()
        
        metrics = backtest_env.calculate_metrics()
        
        performance_plot = backtest_env.plot_performance()
        
        return {
            'metrics': metrics,
            'performance_plot': performance_plot
        }
