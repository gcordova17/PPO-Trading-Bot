#!/usr/bin/env python3
"""
FastAPI backend for the PPO Trading System.

This module provides API endpoints for training and testing
PPO trading models, as well as configuring training parameters.
"""

import os
import sys
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.rl.ppo_rl_strategy import PPORLStrategy
from src.strategies.ppo_strategy import PPOStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs("app/data/models", exist_ok=True)
os.makedirs("app/data/plots", exist_ok=True)

app = FastAPI(
    title="PPO Trading System API",
    description="API for training and testing PPO trading models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")
app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

tasks = {}

models = {}

class TrainingParameters(BaseModel):
    """Training parameters for PPO model."""
    ticker: str = Field(default="SPY", description="Ticker symbol to trade")
    start_date: str = Field(default="2018-01-01", description="Start date for data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date for data (YYYY-MM-DD)")
    initial_balance: float = Field(default=10000.0, description="Initial balance for trading")
    transaction_cost: float = Field(default=0.001, description="Transaction cost as a percentage")
    window_size: int = Field(default=30, description="Window size for feature calculation")
    reward_scaling: float = Field(default=1.0, description="Scaling factor for rewards")
    timesteps: int = Field(default=100000, description="Total timesteps for training")
    learning_rate: float = Field(default=3e-4, description="Learning rate for PPO")
    batch_size: int = Field(default=64, description="Batch size for PPO")
    n_epochs: int = Field(default=10, description="Number of epochs for PPO")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    clip_range: float = Field(default=0.2, description="PPO clip range")
    ent_coef: float = Field(default=0.0, description="Entropy coefficient")
    vf_coef: float = Field(default=0.5, description="Value function coefficient")
    max_grad_norm: float = Field(default=0.5, description="Maximum gradient norm")
    use_gpu: bool = Field(default=True, description="Use GPU for training if available")

class BacktestParameters(BaseModel):
    """Backtest parameters for PPO model."""
    model_id: str = Field(..., description="ID of the model to backtest")
    start_date: Optional[str] = Field(default=None, description="Start date for backtest (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date for backtest (YYYY-MM-DD)")
    initial_balance: float = Field(default=10000.0, description="Initial balance for trading")
    transaction_cost: float = Field(default=0.001, description="Transaction cost as a percentage")

class TaskStatus(BaseModel):
    """Task status."""
    task_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    ticker: str
    start_date: str
    end_date: Optional[str]
    created_at: str
    metrics: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PPO Trading System API"}

@app.post("/train", response_model=TaskStatus)
async def train_model(params: TrainingParameters, background_tasks: BackgroundTasks):
    """Train a PPO model."""
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Task created",
        "result": None
    }
    
    background_tasks.add_task(
        _train_model_task,
        task_id=task_id,
        params=params
    )
    
    return TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Training started"
    )

async def _train_model_task(task_id: str, params: TrainingParameters):
    """Background task for training a PPO model."""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "Initializing training"
        
        model_id = f"{params.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_dir = f"app/data/models/{model_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        strategy = PPORLStrategy(
            ticker=params.ticker,
            start_date=params.start_date,
            end_date=params.end_date,
            initial_balance=params.initial_balance,
            transaction_cost_pct=params.transaction_cost,
            window_size=params.window_size,
            reward_scaling=params.reward_scaling,
            model_name=model_id,
            tensorboard_log=f"{model_dir}/tensorboard",
            use_gpu=params.use_gpu
        )
        
        tasks[task_id]["progress"] = 0.1
        tasks[task_id]["message"] = "Training model"
        
        model_path = f"{model_dir}/{model_id}.zip"
        strategy.train(
            total_timesteps=params.timesteps,
            learning_rate=params.learning_rate,
            batch_size=params.batch_size,
            n_epochs=params.n_epochs,
            gamma=params.gamma,
            gae_lambda=params.gae_lambda,
            clip_range=params.clip_range,
            ent_coef=params.ent_coef,
            vf_coef=params.vf_coef,
            max_grad_norm=params.max_grad_norm,
            save_path=model_path
        )
        
        tasks[task_id]["progress"] = 0.8
        tasks[task_id]["message"] = "Running backtest"
        
        results = strategy.run_backtest()
        
        metrics_path = f"{model_dir}/metrics.json"
        metrics = {
            "portfolio": results["metrics"]["portfolio"],
            "market": results["metrics"]["market"],
            "comparison": results["metrics"]["comparison"]
        }
        
        import json
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        plot_path = f"{model_dir}/performance.png"
        results["performance_plot"].savefig(plot_path)
        
        models[model_id] = {
            "model_id": model_id,
            "ticker": params.ticker,
            "start_date": params.start_date,
            "end_date": params.end_date,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["message"] = "Training completed"
        tasks[task_id]["result"] = {
            "model_id": model_id,
            "metrics": metrics,
            "plot_url": f"/plots/{model_id}/performance.png"
        }
        
    except Exception as e:
        logger.exception(f"Error in training task: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        result=task["result"]
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all models."""
    return [
        ModelInfo(
            model_id=model_id,
            ticker=info["ticker"],
            start_date=info["start_date"],
            end_date=info["end_date"],
            created_at=info["created_at"],
            metrics=info["metrics"]
        )
        for model_id, info in models.items()
    ]

@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    info = models[model_id]
    
    return ModelInfo(
        model_id=model_id,
        ticker=info["ticker"],
        start_date=info["start_date"],
        end_date=info["end_date"],
        created_at=info["created_at"],
        metrics=info["metrics"]
    )

@app.post("/backtest", response_model=TaskStatus)
async def backtest_model(params: BacktestParameters, background_tasks: BackgroundTasks):
    """Backtest a PPO model."""
    if params.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Task created",
        "result": None
    }
    
    background_tasks.add_task(
        _backtest_model_task,
        task_id=task_id,
        params=params
    )
    
    return TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Backtest started"
    )

async def _backtest_model_task(task_id: str, params: BacktestParameters):
    """Background task for backtesting a PPO model."""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "Initializing backtest"
        
        model_info = models[params.model_id]
        
        strategy = PPORLStrategy(
            ticker=model_info["ticker"],
            start_date=params.start_date if params.start_date else model_info["start_date"],
            end_date=params.end_date if params.end_date else model_info["end_date"],
            initial_balance=params.initial_balance,
            transaction_cost_pct=params.transaction_cost,
            window_size=30,  # Default
            reward_scaling=1.0,  # Default
            model_name=params.model_id,
            tensorboard_log=f"app/data/models/{params.model_id}/tensorboard",
            use_gpu=True
        )
        
        tasks[task_id]["progress"] = 0.3
        tasks[task_id]["message"] = "Loading model"
        
        model_path = f"app/data/models/{params.model_id}/{params.model_id}.zip"
        strategy.load(model_path)
        
        tasks[task_id]["progress"] = 0.5
        tasks[task_id]["message"] = "Running backtest"
        
        results = strategy.run_backtest()
        
        metrics = {
            "portfolio": results["metrics"]["portfolio"],
            "market": results["metrics"]["market"],
            "comparison": results["metrics"]["comparison"]
        }
        
        plot_path = f"app/data/plots/{params.model_id}_backtest.png"
        results["performance_plot"].savefig(plot_path)
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 1.0
        tasks[task_id]["message"] = "Backtest completed"
        tasks[task_id]["result"] = {
            "model_id": params.model_id,
            "metrics": metrics,
            "plot_url": f"/plots/{params.model_id}_backtest.png"
        }
        
    except Exception as e:
        logger.exception(f"Error in backtest task: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"

@app.get("/plots/{model_id}/{plot_name}")
async def get_plot(model_id: str, plot_name: str):
    """Get plot for a model."""
    if plot_name == "performance.png":
        plot_path = f"app/data/models/{model_id}/performance.png"
    elif plot_name == "backtest.png":
        plot_path = f"app/data/plots/{model_id}_backtest.png"
    else:
        raise HTTPException(status_code=404, detail="Plot not found")
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path)

@app.get("/compare")
async def compare_strategies(
    ticker: str = Query("SPY", description="Ticker symbol to trade"),
    start_date: str = Query("2018-01-01", description="Start date for data (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for data (YYYY-MM-DD)"),
    model_id: Optional[str] = Query(None, description="ID of the RL model to compare")
):
    """Compare PPO RL strategy with PPO indicator strategy."""
    try:
        ppo_strategy = PPOStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            use_gpu=True
        )
        
        data = yf.download(ticker, start=start_date, end=end_date)
        
        ppo_results = ppo_strategy.backtest(data['Close'], data.index)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ppo_cumulative_returns = (1 + ppo_results['Strategy_Return']).cumprod()
        
        ax.plot(ppo_results.index, ppo_cumulative_returns, label='PPO Indicator Strategy')
        
        market_returns = data['Close'] / data['Close'].iloc[0]
        ax.plot(data.index, market_returns, label='Market (Buy & Hold)')
        
        rl_metrics = None
        if model_id:
            if model_id not in models:
                raise HTTPException(status_code=404, detail="Model not found")
            
            rl_strategy = PPORLStrategy(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000.0,
                transaction_cost_pct=0.001,
                window_size=30,
                reward_scaling=1.0,
                model_name=model_id,
                tensorboard_log=f"app/data/models/{model_id}/tensorboard",
                use_gpu=True
            )
            
            model_path = f"app/data/models/{model_id}/{model_id}.zip"
            rl_strategy.load(model_path)
            
            rl_results = rl_strategy.run_backtest()
            
            rl_portfolio_values = rl_results['portfolio_values']
            rl_dates = rl_results['dates']
            
            rl_portfolio_values = [val / rl_portfolio_values[0] for val in rl_portfolio_values]
            
            ax.plot(rl_dates, rl_portfolio_values, label='PPO RL Strategy')
            
            rl_metrics = rl_results['metrics']
        
        ax.set_title('Strategy Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True)
        
        plot_path = f"app/data/plots/comparison_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        
        ppo_total_return = ppo_cumulative_returns.iloc[-1] - 1
        ppo_annual_return = (1 + ppo_total_return) ** (252 / len(ppo_results)) - 1
        ppo_volatility = ppo_results['Strategy_Return'].std() * np.sqrt(252)
        ppo_sharpe = ppo_annual_return / ppo_volatility if ppo_volatility > 0 else 0
        
        market_total_return = market_returns.iloc[-1] - 1
        market_annual_return = (1 + market_total_return) ** (252 / len(market_returns)) - 1
        market_returns_series = market_returns.pct_change().dropna()
        market_volatility = market_returns_series.std() * np.sqrt(252)
        market_sharpe = market_annual_return / market_volatility if market_volatility > 0 else 0
        
        return {
            "plot_url": f"/plots/{os.path.basename(plot_path)}",
            "ppo_indicator_metrics": {
                "total_return": float(ppo_total_return),
                "annual_return": float(ppo_annual_return),
                "volatility": float(ppo_volatility),
                "sharpe_ratio": float(ppo_sharpe)
            },
            "market_metrics": {
                "total_return": float(market_total_return),
                "annual_return": float(market_annual_return),
                "volatility": float(market_volatility),
                "sharpe_ratio": float(market_sharpe)
            },
            "rl_metrics": rl_metrics
        }
        
    except Exception as e:
        logger.exception(f"Error in compare_strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/plots/{plot_name}")
async def get_comparison_plot(plot_name: str):
    """Get comparison plot."""
    plot_path = f"app/data/plots/{plot_name}"
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path)

@app.get("/documentation")
async def get_documentation():
    """Get API documentation."""
    return {
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Root endpoint"
            },
            {
                "path": "/train",
                "method": "POST",
                "description": "Train a PPO model",
                "parameters": TrainingParameters.schema()
            },
            {
                "path": "/tasks/{task_id}",
                "method": "GET",
                "description": "Get task status"
            },
            {
                "path": "/models",
                "method": "GET",
                "description": "List all models"
            },
            {
                "path": "/models/{model_id}",
                "method": "GET",
                "description": "Get model information"
            },
            {
                "path": "/backtest",
                "method": "POST",
                "description": "Backtest a PPO model",
                "parameters": BacktestParameters.schema()
            },
            {
                "path": "/plots/{model_id}/{plot_name}",
                "method": "GET",
                "description": "Get plot for a model"
            },
            {
                "path": "/compare",
                "method": "GET",
                "description": "Compare PPO RL strategy with PPO indicator strategy"
            },
            {
                "path": "/plots/{plot_name}",
                "method": "GET",
                "description": "Get comparison plot"
            },
            {
                "path": "/documentation",
                "method": "GET",
                "description": "Get API documentation"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
