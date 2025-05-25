# PPO Trading System

A reinforcement learning-based trading system that uses Proximal Policy Optimization (PPO) to recommend when to buy and sell specific stocks. The system optimizes for market returns and Ulcer Performance Index (UPI) to protect equity from drawdowns.

## Features

- Uses StableBaselines3 PPO implementation for robust training
- Optimizes for market returns and Ulcer Performance Index (UPI)
- Backtests against market performance to validate strategy
- Supports GPU acceleration for faster training
- Web interface for parameter configuration
- Docker support with CUDA for cloud deployment

## System Architecture

The system consists of three main components:

1. **Trading Environment**: A custom Gymnasium environment that simulates stock trading with technical indicators
2. **PPO Agent**: A reinforcement learning agent that learns optimal trading strategies
3. **Web Interface**: A user-friendly interface for configuring training parameters and viewing results

## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional but recommended)
- NVIDIA Container Toolkit (for GPU support)

### Using Docker

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ppo-trading-system
   ```

2. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

3. Access the web interface at http://localhost:8000

### Manual Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ppo-trading-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:
   ```bash
   cd app
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

5. Access the web interface at http://localhost:8000

## Usage

### Training a Model

1. Navigate to the Training page in the web interface
2. Configure the training parameters:
   - **Ticker Symbol**: The stock to train on (e.g., SPY for S&P 500 ETF)
   - **Date Range**: Historical data period for training and testing
   - **Initial Balance**: Starting capital for the trading agent
   - **Transaction Cost**: Cost per trade as a percentage
   - **PPO Parameters**: Learning rate, batch size, etc.
3. Click "Start Training" to begin the training process
4. Monitor the training progress and view results when complete

### Backtesting a Model

1. Navigate to the Models page to view trained models
2. Click "Backtest" on a model or go to the Backtest page
3. Configure the backtest parameters:
   - **Model**: Select a trained model
   - **Ticker**: Stock to backtest on
   - **Date Range**: Period for backtesting
   - **Initial Balance**: Starting capital
   - **Transaction Cost**: Cost per trade
4. Click "Run Backtest" to start the backtest
5. View the performance metrics and charts

## Performance Metrics

The system calculates various performance metrics to evaluate trading strategies:

- **Total Return**: Overall percentage return
- **Annualized Return**: Return normalized to yearly rate
- **Sharpe Ratio**: Return per unit of risk
- **Sortino Ratio**: Return per unit of downside risk
- **Max Drawdown**: Largest percentage drop from peak
- **Ulcer Index (UI)**: Measure of downside risk
- **Ulcer Performance Index (UPI)**: Return per unit of downside risk

## API Documentation

The system provides a RESTful API for programmatic access:

- `POST /train` - Start a new training task
- `GET /tasks/{task_id}` - Get status of a training task
- `GET /models` - List all trained models
- `POST /backtest` - Run a backtest on a trained model

For full API documentation, visit http://localhost:8000/docs when the server is running.

## Technical Details

### Trading Environment

The trading environment is implemented as a custom Gymnasium environment with the following features:

- **Actions**: Buy, Hold, Sell
- **Observations**: Technical indicators and account information
- **Reward**: Excess return over market benchmark
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Momentum

### PPO Agent

The PPO agent uses StableBaselines3 with the following components:

- **Policy Network**: Multi-layer perceptron (MLP)
- **Value Network**: MLP for value function estimation
- **Optimization**: PPO algorithm with clipped objective
- **Exploration**: Entropy-based exploration

### Ulcer Performance Index (UPI)

The Ulcer Performance Index is calculated as:

```
UPI = (Annual Return - Risk-Free Rate) / Ulcer Index
```

where the Ulcer Index is:

```
UI = sqrt(sum(squared drawdowns) / n)
```

This metric provides a measure of return per unit of downside risk, with higher values indicating better risk-adjusted performance.

## License

[MIT License](LICENSE)

## Acknowledgements

- [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) for the PPO implementation
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the reinforcement learning environment framework
- [yfinance](https://github.com/ranaroussi/yfinance) for stock data retrieval
