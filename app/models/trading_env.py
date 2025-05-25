import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    A trading environment for OpenAI gym
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                ticker: str = "SPY", 
                start_date: str = "2018-01-01",
                end_date: str = "2023-01-01",
                initial_balance: float = 10000.0,
                transaction_cost_pct: float = 0.001,
                window_size: int = 30,
                reward_scaling: float = 1.0):
        """
        Initialize the trading environment
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            initial_balance: Initial account balance
            transaction_cost_pct: Transaction cost percentage
            window_size: Size of observation window
            reward_scaling: Scaling factor for rewards
        """
        super(TradingEnv, self).__init__()
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        self.data = self._download_data()
        self.returns = self.data['Close'].pct_change().fillna(0)
        
        self._calculate_features()
        
        self.action_space = spaces.Discrete(3)
        
        num_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features + 3,),  # +3 for balance, shares, portfolio value
            dtype=np.float32
        )
        
        self.reset()
        
    def _download_data(self) -> pd.DataFrame:
        """Download stock data using yfinance"""
        logger.info(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data
    
    def _calculate_features(self):
        """Calculate technical indicators as features"""
        df = self.data.copy()
        
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        df['Momentum'] = df['Close'].pct_change(periods=10)
        
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        feature_columns = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                          'BB_Width', 'Momentum', 'Volatility']
        
        df = df.fillna(method='bfill')
        
        for col in feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
        
        self.features = df[feature_columns]
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_sales_value = 0
        self.total_buy_value = 0
        self.total_transaction_costs = 0
        
        self.portfolio_values = [self.initial_balance]
        self.market_values = [self.initial_balance / self.data['Close'].iloc[self.current_step] * self.data['Close'].iloc[self.current_step]]
        
        self.positions = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self):
        """Return the current observation"""
        features = self.features.iloc[self.current_step].values.flatten().astype(np.float32)
        
        current_price = self.data['Close'].iloc[self.current_step]
        
        portfolio_value = self.balance + self.shares_held * current_price
        
        normalized_balance = float(self.balance / self.initial_balance - 1)
        normalized_shares = float(self.shares_held * current_price / self.initial_balance)
        normalized_portfolio_value = float(portfolio_value / self.initial_balance - 1)
        
        account_info = np.array([normalized_balance, normalized_shares, normalized_portfolio_value], dtype=np.float32)
        
        observation = np.concatenate([features, account_info])
        
        return observation
    
    def _get_info(self):
        """Return current info"""
        current_price = self.data['Close'].iloc[self.current_step]
        
        if isinstance(self.balance, (pd.Series, pd.DataFrame)):
            balance = float(self.balance.iloc[0])
        else:
            balance = float(self.balance)
            
        if isinstance(self.shares_held, (pd.Series, pd.DataFrame)):
            shares = float(self.shares_held.iloc[0])
        else:
            shares = float(self.shares_held)
            
        if isinstance(current_price, (pd.Series, pd.DataFrame)):
            price = float(current_price.iloc[0])
        else:
            price = float(current_price)
        
        portfolio_value = balance + shares * price
        
        return {
            'step': self.current_step,
            'balance': balance,
            'shares_held': shares,
            'current_price': price,
            'portfolio_value': portfolio_value,
            'initial_portfolio_value': float(self.initial_balance),
            'return': (portfolio_value / float(self.initial_balance)) - 1
        }
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: 0 (Sell), 1 (Hold), 2 (Buy)
        """
        current_price = self.data['Close'].iloc[self.current_step]
        
        if isinstance(self.shares_held, (pd.Series, pd.DataFrame)):
            shares = float(self.shares_held.iloc[0])
        else:
            shares = float(self.shares_held)
            
        if isinstance(self.balance, (pd.Series, pd.DataFrame)):
            balance = float(self.balance.iloc[0])
        else:
            balance = float(self.balance)
            
        if isinstance(current_price, (pd.Series, pd.DataFrame)):
            price = float(current_price.iloc[0])
        else:
            price = float(current_price)
        
        reward = 0
        done = False
        
        if action == 0:  # Sell
            if shares > 0:
                transaction_cost = shares * price * self.transaction_cost_pct
                self.total_transaction_costs += transaction_cost
                
                sale_value = shares * price - transaction_cost
                self.balance += sale_value
                self.total_sales_value += sale_value
                
                self.positions.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': price,
                    'shares': shares,
                    'value': sale_value,
                    'transaction_cost': transaction_cost
                })
                
                self.shares_held = 0
                
        elif action == 2:  # Buy
            if balance > 0:
                max_shares = balance / (price * (1 + self.transaction_cost_pct))
                shares_to_buy = max_shares  # Buy all available
                
                transaction_cost = shares_to_buy * price * self.transaction_cost_pct
                self.total_transaction_costs += transaction_cost
                
                buy_value = shares_to_buy * price + transaction_cost
                self.balance -= buy_value
                self.shares_held += shares_to_buy
                self.total_buy_value += buy_value
                
                self.positions.append({
                    'type': 'buy',
                    'step': self.current_step,
                    'price': price,
                    'shares': shares_to_buy,
                    'value': buy_value,
                    'transaction_cost': transaction_cost
                })
        
        self.current_step += 1
        
        if isinstance(self.balance, (pd.Series, pd.DataFrame)):
            balance = float(self.balance.iloc[0])
        else:
            balance = float(self.balance)
            
        if isinstance(self.shares_held, (pd.Series, pd.DataFrame)):
            shares = float(self.shares_held.iloc[0])
        else:
            shares = float(self.shares_held)
            
        if isinstance(current_price, (pd.Series, pd.DataFrame)):
            price = float(current_price.iloc[0])
        else:
            price = float(current_price)
        
        portfolio_value = balance + shares * price
        self.portfolio_values.append(portfolio_value)
        
        initial_shares = float(self.initial_balance) / float(self.data['Close'].iloc[self.window_size])
        market_value = initial_shares * price
        self.market_values.append(market_value)
        
        if len(self.portfolio_values) >= 2:
            daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
            
            market_return = (self.market_values[-1] / self.market_values[-2]) - 1
            
            reward = (daily_return - market_return) * self.reward_scaling
        
        if self.current_step >= len(self.data) - 1:
            done = True
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            current_price = self.data['Close'].iloc[self.current_step]
            portfolio_value = self.balance + self.shares_held * current_price
            
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held:.6f}")
            print(f"Portfolio value: ${portfolio_value:.2f}")
            print(f"Return: {(portfolio_value / self.initial_balance - 1) * 100:.2f}%")
            
    def plot_performance(self):
        """Plot the performance of the agent vs market"""
        plt.figure(figsize=(12, 6))
        
        portfolio_values = np.array([float(val) if isinstance(val, (pd.Series, pd.DataFrame)) 
                                    else float(val) for val in self.portfolio_values])
        market_values = np.array([float(val) if isinstance(val, (pd.Series, pd.DataFrame)) 
                                 else float(val) for val in self.market_values])
        
        portfolio_values = portfolio_values / portfolio_values[0]
        market_values = market_values / market_values[0]
        
        dates = self.data.index[self.window_size:self.window_size + len(portfolio_values)]
        
        plt.plot(dates, portfolio_values, label='Agent Portfolio')
        plt.plot(dates, market_values, label='Market (Buy & Hold)')
        
        for position in self.positions:
            idx = position['step'] - self.window_size
            if idx >= 0 and idx < len(dates):
                if position['type'] == 'buy':
                    plt.scatter(dates[idx], portfolio_values[idx], color='green', marker='^', s=100)
                else:  # sell
                    plt.scatter(dates[idx], portfolio_values[idx], color='red', marker='v', s=100)
        
        plt.title('Agent Performance vs Market')
        plt.xlabel('Date')
        plt.ylabel('Normalized Return')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def calculate_metrics(self):
        """Calculate performance metrics including UPI"""
        portfolio_values = np.array([float(val) if isinstance(val, (pd.Series, pd.DataFrame)) 
                                    else float(val) for val in self.portfolio_values])
        market_values = np.array([float(val) if isinstance(val, (pd.Series, pd.DataFrame)) 
                                 else float(val) for val in self.market_values])
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        market_returns = np.diff(market_values) / market_values[:-1]
        
        portfolio_cum_returns = portfolio_values / portfolio_values[0] - 1
        market_cum_returns = market_values / market_values[0] - 1
        
        portfolio_drawdowns = self._calculate_drawdowns(portfolio_values)
        market_drawdowns = self._calculate_drawdowns(market_values)
        
        portfolio_ui = self._calculate_ulcer_index(portfolio_drawdowns)
        market_ui = self._calculate_ulcer_index(market_drawdowns)
        
        days = len(portfolio_values)
        years = days / 252
        
        portfolio_annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years)) - 1
        market_annual_return = ((market_values[-1] / market_values[0]) ** (1 / years)) - 1
        
        portfolio_upi = portfolio_annual_return / portfolio_ui if portfolio_ui > 0 else 0
        market_upi = market_annual_return / market_ui if market_ui > 0 else 0
        
        portfolio_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        market_sharpe = np.mean(market_returns) / np.std(market_returns) * np.sqrt(252)
        
        portfolio_max_dd = np.min(portfolio_drawdowns)
        market_max_dd = np.min(market_drawdowns)
        
        metrics = {
            'portfolio': {
                'total_return': portfolio_values[-1] / portfolio_values[0] - 1,
                'annualized_return': portfolio_annual_return,
                'sharpe_ratio': portfolio_sharpe,
                'max_drawdown': portfolio_max_dd,
                'ulcer_index': portfolio_ui,
                'ulcer_performance_index': portfolio_upi
            },
            'market': {
                'total_return': market_values[-1] / market_values[0] - 1,
                'annualized_return': market_annual_return,
                'sharpe_ratio': market_sharpe,
                'max_drawdown': market_max_dd,
                'ulcer_index': market_ui,
                'ulcer_performance_index': market_upi
            },
            'comparison': {
                'excess_return': (portfolio_values[-1] / portfolio_values[0]) - (market_values[-1] / market_values[0]),
                'upi_improvement': portfolio_upi - market_upi
            }
        }
        
        return metrics
    
    def _calculate_drawdowns(self, values):
        """Calculate drawdowns for a series of values"""
        running_max = np.maximum.accumulate(values)
        
        drawdowns = values / running_max - 1
        
        return drawdowns
    
    def _calculate_ulcer_index(self, drawdowns):
        """Calculate Ulcer Index from drawdowns"""
        squared_drawdowns = drawdowns ** 2
        
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
