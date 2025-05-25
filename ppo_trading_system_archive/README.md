# PPO Trading System with GPU Acceleration

A Python-based trading system that uses the Percentage Price Oscillator (PPO) indicator to generate buy/sell signals. The system includes GPU acceleration via PyTorch, comprehensive backtesting capabilities, and performance metrics with a focus on the Ulcer Performance Index for risk assessment.

## Features

- **PPO Indicator**: Implementation of the Percentage Price Oscillator with customizable parameters
- **GPU Acceleration**: PyTorch-based calculations that utilize CUDA when available
- **Backtesting Framework**: Comprehensive backtesting with performance metrics and visualizations
- **Risk Assessment**: Ulcer Performance Index calculation to measure drawdown risk
- **Docker Support**: Containerization with NVIDIA CUDA support for deployment

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Docker with NVIDIA Container Toolkit (for containerized deployment)

### Local Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ppo_trading_system
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Docker Installation

1. Build the Docker image:
   ```
   docker build -t ppo_trading_system .
   ```

2. Run the container with GPU support:
   ```
   docker run --gpus all ppo_trading_system
   ```

## Usage

### Command Line Interface

The system can be run from the command line with various parameters:

```
python -m src.main --ticker SPY --start-date 2018-01-01 --end-date 2023-01-01
```

#### Available Parameters

- `--ticker`: Ticker symbol to analyze (default: SPY)
- `--start-date`: Start date for backtesting (default: 2018-01-01)
- `--end-date`: End date for backtesting (default: 2023-01-01)
- `--fast-period`: Fast EMA period for PPO (default: 12)
- `--slow-period`: Slow EMA period for PPO (default: 26)
- `--signal-period`: Signal line period for PPO (default: 9)
- `--output-dir`: Directory to save results (default: ./results)
- `--use-gpu`: Enable GPU acceleration (default: True)

### Docker Usage

Run the container with custom parameters:

```
docker run --gpus all ppo_trading_system --ticker AAPL --start-date 2020-01-01
```

## Performance Metrics

The system calculates various performance metrics, with a focus on the Ulcer Performance Index:

- **Total Return**: Overall return of the strategy
- **Annual Return**: Annualized return
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return using standard deviation
- **Max Drawdown**: Maximum peak-to-trough decline
- **Ulcer Index**: Measure of downside risk that considers both depth and duration of drawdowns
- **Ulcer Performance Index (UPI)**: Risk-adjusted return using the Ulcer Index

A UPI greater than 1.0 indicates good risk-adjusted returns, with higher values being better.

## PPO Indicator

The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the difference between two moving averages as a percentage:

1. **PPO Line**: ((Fast EMA - Slow EMA) / Slow EMA) * 100
2. **Signal Line**: EMA of the PPO Line
3. **Histogram**: PPO Line - Signal Line

Buy signals are generated when the PPO line crosses above the signal line, and sell signals when it crosses below.

## Output

The system generates the following outputs:

1. **Performance Metrics**: CSV file with comprehensive metrics
2. **Equity Curve**: Plot showing strategy performance vs benchmark
3. **Drawdowns**: Plot showing drawdowns over time
4. **PPO Indicator**: Plot showing PPO line, signal line, and histogram
5. **Buy/Sell Signals**: Plot showing price with buy/sell signals

## Testing

Run the test suite to verify functionality:

```
python -m pytest tests/
```

## License

[MIT License](LICENSE)
