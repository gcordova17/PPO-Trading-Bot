from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
import sys
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid
import asyncio
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.trading_env import TradingEnv
from models.ppo_agent import PPOTradingAgent
from models.backtest import BacktestAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPO Trading System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

os.makedirs("../data/models", exist_ok=True)
os.makedirs("../data/plots", exist_ok=True)
os.makedirs("../data/custom_data", exist_ok=True)
os.makedirs("../data/tensorboard", exist_ok=True)

training_tasks = {}

class TrainingParameters(BaseModel):
    ticker: str = Field(default="SPY", description="Stock ticker symbol")
    start_date: str = Field(default="2018-01-01", description="Start date for training data")
    end_date: str = Field(default="2023-01-01", description="End date for training data")
    test_start_date: str = Field(default="2023-01-01", description="Start date for test data")
    test_end_date: str = Field(default="2023-12-31", description="End date for test data")
    initial_balance: float = Field(default=10000.0, description="Initial account balance")
    transaction_cost_pct: float = Field(default=0.001, description="Transaction cost percentage")
    window_size: int = Field(default=30, description="Size of observation window")
    reward_scaling: float = Field(default=1.0, description="Scaling factor for rewards")
    total_timesteps: int = Field(default=100000, description="Total timesteps to train for")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    n_steps: int = Field(default=2048, description="Number of steps to run for each environment per update")
    batch_size: int = Field(default=64, description="Minibatch size")
    n_epochs: int = Field(default=10, description="Number of epoch when optimizing the surrogate loss")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="Factor for trade-off of bias vs variance for GAE")
    clip_range: float = Field(default=0.2, description="Clipping parameter for PPO")
    ent_coef: float = Field(default=0.01, description="Entropy coefficient for the loss calculation")
    vf_coef: float = Field(default=0.5, description="Value function coefficient for the loss calculation")
    max_grad_norm: float = Field(default=0.5, description="Maximum norm for the gradient clipping")
    use_sde: bool = Field(default=False, description="Whether to use generalized State Dependent Exploration")
    device: str = Field(default="auto", description="Device to run the model on ('auto', 'cuda', or 'cpu')")

class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    metrics: Optional[Dict[str, Any]] = None
    plots: Optional[List[str]] = None

class BacktestRequest(BaseModel):
    model_id: str
    ticker: str = Field(default="SPY", description="Stock ticker symbol")
    start_date: str = Field(default="2023-01-01", description="Start date for backtest data")
    end_date: str = Field(default="2023-12-31", description="End date for backtest data")
    initial_balance: float = Field(default=10000.0, description="Initial account balance")
    transaction_cost_pct: float = Field(default=0.001, description="Transaction cost percentage")

# @app.get("/")
# async def root():
#     return {"message": "PPO Trading System API"}

@app.post("/train", response_model=TrainingResponse)
async def train_model(params: TrainingParameters, background_tasks: BackgroundTasks):
    """
    Start training a PPO model with the given parameters
    """
    task_id = str(uuid.uuid4())
    
    training_tasks[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Task queued",
        "params": params.dict(),
        "metrics": None,
        "plots": []
    }
    
    background_tasks.add_task(run_training, task_id, params)
    
    return TrainingResponse(
        task_id=task_id,
        status="queued",
        message="Training task has been queued"
    )

async def run_training(task_id: str, params: TrainingParameters):
    """
    Run the training process in the background
    """
    try:
        training_tasks[task_id]["status"] = "running"
        training_tasks[task_id]["message"] = "Initializing training environment"
        
        env = TradingEnv(
            ticker=params.ticker,
            start_date=params.start_date,
            end_date=params.end_date,
            initial_balance=params.initial_balance,
            transaction_cost_pct=params.transaction_cost_pct,
            window_size=params.window_size,
            reward_scaling=params.reward_scaling
        )
        
        agent = PPOTradingAgent(
            env=env,
            model_name=f"ppo_{params.ticker}_{task_id}",
            tensorboard_log=f"../data/tensorboard/{task_id}/",
            device=params.device
        )
        
        training_tasks[task_id]["message"] = "Training model"
        
        total_timesteps = params.total_timesteps
        progress_callback = lambda progress: update_training_progress(task_id, progress)
        
        model_path = f"../data/models/{params.ticker}_{task_id}.zip"
        
        agent.train(
            total_timesteps=total_timesteps,
            learning_rate=params.learning_rate,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            n_epochs=params.n_epochs,
            gamma=params.gamma,
            gae_lambda=params.gae_lambda,
            clip_range=params.clip_range,
            ent_coef=params.ent_coef,
            vf_coef=params.vf_coef,
            max_grad_norm=params.max_grad_norm,
            use_sde=params.use_sde,
            save_path=model_path
        )
        
        training_tasks[task_id]["message"] = "Running backtest"
        
        test_env = TradingEnv(
            ticker=params.ticker,
            start_date=params.test_start_date,
            end_date=params.test_end_date,
            initial_balance=params.initial_balance,
            transaction_cost_pct=params.transaction_cost_pct,
            window_size=params.window_size,
            reward_scaling=params.reward_scaling
        )
        
        backtest_results = agent.backtest(env=test_env, deterministic=True)
        
        plot_path = f"../data/plots/{params.ticker}_{task_id}.png"
        backtest_results['performance_plot'].savefig(plot_path)
        plt.close(backtest_results['performance_plot'])
        
        training_tasks[task_id]["status"] = "completed"
        training_tasks[task_id]["progress"] = 1.0
        training_tasks[task_id]["message"] = "Training and backtest completed"
        training_tasks[task_id]["metrics"] = backtest_results['metrics']
        training_tasks[task_id]["plots"] = [plot_path]
        training_tasks[task_id]["model_path"] = model_path
        
    except Exception as e:
        logger.error(f"Error in training task {task_id}: {str(e)}")
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["message"] = f"Error: {str(e)}"

def update_training_progress(task_id: str, progress: float):
    """Update the progress of a training task"""
    if task_id in training_tasks:
        training_tasks[task_id]["progress"] = progress

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status of a training task
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        metrics=task.get("metrics"),
        plots=task.get("plots", [])
    )

@app.get("/tasks")
async def list_tasks():
    """
    List all training tasks
    """
    return {
        task_id: {
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "ticker": task["params"]["ticker"],
            "created_at": task.get("created_at", "Unknown")
        }
        for task_id, task in training_tasks.items()
    }

@app.get("/models")
async def list_models():
    """
    List all trained models
    """
    models = []
    model_dir = "../../app/data/models"
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith(".zip"):
                model_path = os.path.join(model_dir, filename)
                model_stats = os.stat(model_path)
                
                parts = filename.replace(".zip", "").split("_")
                ticker = parts[0] if len(parts) > 0 else "unknown"
                task_id = parts[1] if len(parts) > 1 else "unknown"
                
                models.append({
                    "id": filename.replace(".zip", ""),
                    "ticker": ticker,
                    "task_id": task_id,
                    "file_size": model_stats.st_size,
                    "created_at": datetime.fromtimestamp(model_stats.st_ctime).isoformat(),
                    "path": model_path
                })
    
    return models

@app.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest on a trained model
    """
    model_path = f"../data/models/{request.model_id}.zip"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    task_id = str(uuid.uuid4())
    
    training_tasks[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Backtest queued",
        "params": request.dict(),
        "metrics": None,
        "plots": []
    }
    
    background_tasks.add_task(run_backtest_task, task_id, request, model_path)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Backtest has been queued"
    }

async def run_backtest_task(task_id: str, request: BacktestRequest, model_path: str):
    """
    Run the backtest process in the background
    """
    try:
        training_tasks[task_id]["status"] = "running"
        training_tasks[task_id]["message"] = "Initializing backtest environment"
        
        env = TradingEnv(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            transaction_cost_pct=request.transaction_cost_pct
        )
        
        agent = PPOTradingAgent(env=env)
        agent.load(model_path)
        
        training_tasks[task_id]["message"] = "Running backtest"
        
        backtest_results = agent.backtest(deterministic=True)
        
        plot_path = f"../data/plots/{request.ticker}_{task_id}_backtest.png"
        backtest_results['performance_plot'].savefig(plot_path)
        plt.close(backtest_results['performance_plot'])
        
        training_tasks[task_id]["status"] = "completed"
        training_tasks[task_id]["progress"] = 1.0
        training_tasks[task_id]["message"] = "Backtest completed"
        training_tasks[task_id]["metrics"] = backtest_results['metrics']
        training_tasks[task_id]["plots"] = [plot_path]
        
    except Exception as e:
        logger.error(f"Error in backtest task {task_id}: {str(e)}")
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["message"] = f"Error: {str(e)}"

@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """
    Get a plot by filename
    """
    plot_path = f"../data/plots/{filename}"
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path)

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload custom data file (CSV)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_path = f"../data/custom_data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "path": file_path}

@app.get("/custom_data")
async def list_custom_data():
    """
    List all custom data files
    """
    data_files = []
    data_dir = "../data/custom_data"
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(data_dir, filename)
                file_stats = os.stat(file_path)
                
                data_files.append({
                    "filename": filename,
                    "file_size": file_stats.st_size,
                    "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "path": file_path
                })
    
    return data_files

app.mount("/", StaticFiles(directory="/home/ubuntu/ppo_trading_system/app/frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
