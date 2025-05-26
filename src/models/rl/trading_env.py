import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for stock trading using reinforcement learning.
    
    This environment allows an agent to learn when to buy, sell, or hold a specific
    ticker based on technical indicators and price data. It calculates rewards based
    on portfolio performance relative to a buy-and-hold strategy.
    
    Attributes:
        ticker (str): The stock ticker symbol
        start_date (str): Start date for data in YYYY-MM-DD format
        end_date (str): End date for data in YYYY-MM-DD format
        initial_balance (float): Starting cash balance
        transaction_cost_pct (float): Transaction cost as a percentage
        window_size (int): Number of days to use for feature calculation
        reward_scaling (float): Scaling factor for rewards
        device (torch.device): Device to use for calculations
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        ticker: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1.0,
        use_gpu: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            initial_balance: Starting cash balance
            transaction_cost_pct: Transaction cost as a percentage
            window_size: Number of days to use for feature calculation
            reward_scaling: Scaling factor for rewards
            use_gpu: Whether to use GPU acceleration if available
        """
        super(TradingEnv, self).__init__()
        
        self.ticker = ticker
        
        if start_date is None:
            end_date_dt = datetime.now()
            start_date_dt = end_date_dt - timedelta(days=5*365)  # 5 years of data
            self.start_date = start_date_dt.strftime('%Y-%m-%d')
            self.end_date = end_date_dt.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        self.data = self._download_data()
        self.features = self._calculate_features()
        
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        
        n_features = len(self.features.columns)
        account_info_size = 3  # normalized_balance, normalized_shares, normalized_portfolio_value
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features + account_info_size,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _download_data(self) -> pd.DataFrame:
        """
        Download historical data for the ticker.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            
            if data.empty or len(data) < self.window_size + 10:
                raise ValueError(f"Not enough data for ticker {self.ticker} between {self.start_date} and {self.end_date}")
            
            self.benchmark_data = data.copy()
            
            return data
        except Exception as e:
            raise ValueError(f"Error downloading data: {e}")
    
    def _calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators as features.
        
        Returns:
            DataFrame with calculated features
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("Empty data in _calculate_features, returning minimal DataFrame")
            return pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
            
        df = pd.DataFrame(index=self.data.index)
        
        df['Close'] = self.data['Close']
        df['Volume'] = self.data['Volume']
        df['High'] = self.data['High']
        df['Low'] = self.data['Low']
        
        df['Daily_Return'] = self.data['Close'].pct_change()
        
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            df[f'SMA_Volume_{window}'] = self.data['Volume'].rolling(window=window).mean()
        
        for window in [5, 10, 20, 50, 200]:
            # Calculate Price_to_SMA ratio safely
            close_series = self.data['Close']
            sma_series = df[f'SMA_{window}']
            ratio_series = pd.Series(index=close_series.index)
            
            for i in range(len(close_series)):
                if i < window - 1 or pd.isna(sma_series.iloc[i]) or sma_series.iloc[i] == 0:
                    ratio_series.iloc[i] = np.nan
                else:
                    ratio_series.iloc[i] = close_series.iloc[i] / sma_series.iloc[i]
            
            df[f'Price_to_SMA_{window}'] = ratio_series
        
        for window in [20]:
            df[f'BB_Middle_{window}'] = self.data['Close'].rolling(window=window).mean()
            df[f'BB_Std_{window}'] = self.data['Close'].rolling(window=window).std()
            df[f'BB_Upper_{window}'] = df[f'BB_Middle_{window}'] + 2 * df[f'BB_Std_{window}']
            df[f'BB_Lower_{window}'] = df[f'BB_Middle_{window}'] - 2 * df[f'BB_Std_{window}']
            # Calculate BB_Width safely using element-wise operations
            upper_values = df[f'BB_Upper_{window}'].values
            lower_values = df[f'BB_Lower_{window}'].values
            middle_values = df[f'BB_Middle_{window}'].values
            width_values = np.zeros(len(middle_values))
            
            for i in range(len(middle_values)):
                if np.isnan(middle_values[i]) or middle_values[i] == 0:
                    width_values[i] = np.nan
                else:
                    width_values[i] = (upper_values[i] - lower_values[i]) / middle_values[i]
            
            df[f'BB_Width_{window}'] = width_values
            # Calculate BB_Position safely using element-wise operations
            close_values = self.data['Close'].values
            lower_values = df[f'BB_Lower_{window}'].values
            upper_values = df[f'BB_Upper_{window}'].values
            position_values = np.zeros(len(close_values))
            
            for i in range(len(close_values)):
                if np.isnan(lower_values[i]) or np.isnan(upper_values[i]) or (upper_values[i] - lower_values[i]) == 0:
                    position_values[i] = np.nan
                else:
                    position_values[i] = (close_values[i] - lower_values[i]) / (upper_values[i] - lower_values[i])
            
            df[f'BB_Position_{window}'] = position_values
        
        # Calculate RSI using a different approach to avoid type errors
        delta = self.data['Close'].pct_change()
        up = delta.clip(lower=0).fillna(0)
        down = -delta.clip(upper=0).fillna(0)
        
        for window in [14]:
            avg_gain = up.rolling(window=window).mean().fillna(0)
            avg_loss = down.rolling(window=window).mean().fillna(0)
            
            # Use vectorized operations to avoid Series comparison issues
            avg_gain_values = avg_gain.values.flatten()  # Ensure 1D array
            avg_loss_values = avg_loss.values.flatten()  # Ensure 1D array
            
            rs_values = np.where(
                avg_loss_values > 0,
                avg_gain_values / avg_loss_values,
                1.0  # Default to 1.0 for zero losses
            )
            
            rs = pd.Series(rs_values, index=avg_gain.index)
            
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        for window in [5, 10, 20]:
            # Calculate Momentum safely using element-wise operations
            close_values = self.data['Close'].values
            shifted_values = self.data['Close'].shift(window).values
            momentum_values = np.zeros(len(close_values))
            
            for i in range(len(close_values)):
                if i < window or np.isnan(shifted_values[i]) or shifted_values[i] == 0:
                    momentum_values[i] = np.nan
                else:
                    momentum_values[i] = close_values[i] / shifted_values[i] - 1
            
            df[f'Momentum_{window}'] = momentum_values
        
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = self.data['Close'].pct_change().rolling(window=window).std()
        
        df = df.fillna(0)
        
        for col in df.columns:
            if col not in ['Close', 'High', 'Low', 'Volume']:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Ensure current_step is within valid range for features
        self.current_step = min(self.window_size, len(self.features) - 1)
        if self.current_step < 0:
            self.current_step = 0

        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.total_transaction_costs = 0
        self.portfolio_values = []
        self.market_values = []
        self.dates = []
        
        initial_price = self.data['Close'].iloc[self.current_step]
        self.initial_market_shares = self.initial_balance / initial_price
        
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        market_value = self.initial_market_shares * current_price
        
        self.portfolio_values.append(portfolio_value)
        self.market_values.append(market_value)
        self.dates.append(self.data.index[self.current_step])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Sell, 1: Hold, 2: Buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        current_price = self.data['Close'].iloc[self.current_step]
        
        # Convert to scalar values to avoid Series comparison issues
        shares_held = float(self.shares_held.iloc[0]) if hasattr(self.shares_held, 'iloc') else float(self.shares_held)
        balance = float(self.balance.iloc[0]) if hasattr(self.balance, 'iloc') else float(self.balance)
        
        if action == 0:  # Sell
            if shares_held > 0:
                transaction_cost = shares_held * current_price * self.transaction_cost_pct
                self.total_transaction_costs += transaction_cost
                
                self.balance = balance + shares_held * current_price - transaction_cost
                self.shares_held = 0.0
        
        elif action == 2:  # Buy
            if balance > 0:
                max_shares = balance / (current_price * (1 + self.transaction_cost_pct))
                
                transaction_cost = max_shares * current_price * self.transaction_cost_pct
                self.total_transaction_costs += transaction_cost
                
                self.shares_held = shares_held + max_shares
                self.balance = balance - (max_shares * current_price + transaction_cost)
        
        # Increment current_step safely
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.current_step = len(self.data) - 1
            done = True
        else:
            done = False

        
        next_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * next_price
        market_value = self.initial_market_shares * next_price
        
        self.portfolio_values.append(portfolio_value)
        self.market_values.append(market_value)
        self.dates.append(self.data.index[self.current_step])
        
        # Calculate reward safely
        if len(self.portfolio_values) >= 2:
            portfolio_return = (portfolio_value / self.portfolio_values[-2]) - 1
            market_return = (market_value / self.market_values[-2]) - 1
            reward = (portfolio_return - market_return) * self.reward_scaling
        else:
            reward = 0.0
        
                
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Numpy array with features and account info
        """
        # Ensure current_step is within valid range
        if self.current_step >= len(self.features):
            self.current_step = len(self.features) - 1
        features = self.features.iloc[self.current_step]

        
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        # Convert to scalar values to avoid Series issues
        balance = float(self.balance.iloc[0]) if hasattr(self.balance, 'iloc') else float(self.balance)
        shares_held = float(self.shares_held.iloc[0]) if hasattr(self.shares_held, 'iloc') else float(self.shares_held)
        
        normalized_balance = balance / self.initial_balance - 1
        normalized_shares = shares_held * current_price / self.initial_balance
        normalized_portfolio_value = portfolio_value / self.initial_balance - 1
        
        account_info = np.array([
            float(normalized_balance) if not isinstance(normalized_balance, float) else normalized_balance if not isinstance(normalized_balance, float) else normalized_balance,
            float(normalized_shares.iloc[0]) if isinstance(normalized_shares, pd.Series) else float(normalized_shares) if not isinstance(normalized_shares, float) else normalized_shares if not isinstance(normalized_shares, float) else normalized_shares,
            float(normalized_portfolio_value.iloc[0]) if isinstance(normalized_portfolio_value, pd.Series) else float(normalized_portfolio_value) if not isinstance(normalized_portfolio_value, float) else normalized_portfolio_value if not isinstance(normalized_portfolio_value, float) else normalized_portfolio_value
        ], dtype=np.float32)
        
        observation = np.concatenate([features.values.astype(np.float32), account_info])
        
        return observation
    
    def _get_info(self) -> Dict:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with additional info
        """
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        return {
            'step': self.current_step,
            'date': self.data.index[self.current_step].strftime('%Y-%m-%d'),
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': portfolio_value,
            'transaction_costs': self.total_transaction_costs
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Date: {self.data.index[self.current_step].strftime('%Y-%m-%d')}")
            print(f"Price: ${self.data['Close'].iloc[self.current_step]:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held:.6f}")
            print(f"Portfolio value: ${self.balance + self.shares_held * self.data['Close'].iloc[self.current_step]:.2f}")
            print(f"Transaction costs: ${self.total_transaction_costs:.2f}")
            print("-" * 50)
    
    def plot_performance(self, save_path=None):
        """
        Plot the performance of the portfolio vs the market.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        market_series = pd.Series(self.market_values, index=self.dates)
        
        # Normalize to initial value using numeric values to avoid type errors
        if len(portfolio_series) > 0 and len(market_series) > 0:
            initial_portfolio = portfolio_series.iloc[0]
            initial_market = market_series.iloc[0]
            
            portfolio_values = [float(val) for val in portfolio_series.values]
            market_values = [float(val) for val in market_series.values]
            
            # Create new normalized series without type errors
            normalized_portfolio = [val / portfolio_values[0] for val in portfolio_values]
            normalized_market = [val / market_values[0] for val in market_values]
            
            portfolio_series = pd.Series(normalized_portfolio, index=portfolio_series.index)
            market_series = pd.Series(normalized_market, index=market_series.index)
        
        portfolio_series.plot(ax=ax, label='Portfolio')
        market_series.plot(ax=ax, label='Market (Buy & Hold)')
        
        ax.set_title('Portfolio Performance vs Market')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def calculate_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.portfolio_values) < 2:
            return None
        
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        market_series = pd.Series(self.market_values, index=self.dates)
        
        portfolio_returns = portfolio_series.pct_change().dropna()
        market_returns = market_series.pct_change().dropna()
        
        metrics = {}
        
        metrics['portfolio'] = self._calculate_series_metrics(portfolio_series, portfolio_returns)
        
        metrics['market'] = self._calculate_series_metrics(market_series, market_returns)
        
        metrics['comparison'] = {
            'excess_return': metrics['portfolio']['total_return'] - metrics['market']['total_return'],
            'information_ratio': (metrics['portfolio']['annual_return'] - metrics['market']['annual_return']) / 
                                (portfolio_returns - market_returns).std() * np.sqrt(252),
            'tracking_error': (portfolio_returns - market_returns).std() * np.sqrt(252)
        }
        
        return metrics
    
    def _calculate_series_metrics(self, values, returns):
        """
        Calculate metrics for a value series.
        
        Args:
            values: Series of values
            returns: Series of returns
            
        Returns:
            Dictionary with metrics
        """
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        
        days = (values.index[-1] - values.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        volatility = returns.std() * np.sqrt(252)
        
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_risk if downside_risk > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        risk_free_rate = 0.0  # Assuming 0 for simplicity
        upi = (annual_return - risk_free_rate) / ulcer_index if ulcer_index > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'ulcer_index': ulcer_index,
            'ulcer_performance_index': upi
        }
