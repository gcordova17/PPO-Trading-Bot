import yfinance as yf
import pandas as pd
import numpy as np
import torch

class DataLoader:
    """
    Utility class for loading and preprocessing financial data.
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the DataLoader.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
    def load_ticker_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Load historical data for a given ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1wk', '1mo', etc.)
            
        Returns:
            pandas.DataFrame: Historical data
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        Preprocess the data for analysis.
        
        Args:
            data (pandas.DataFrame): Raw data
            
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        if data is None or data.empty:
            return None
        
        df = data.copy()
        
        df = df.ffill()
        df = df.bfill()
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        df.dropna(inplace=True)
        
        return df
    
    def prepare_tensor_data(self, data, column='Close'):
        """
        Convert data to PyTorch tensors for GPU acceleration.
        
        Args:
            data (pandas.DataFrame): Preprocessed data
            column (str): Column to convert to tensor
            
        Returns:
            torch.Tensor: Data as PyTorch tensor
        """
        if data is None or data.empty:
            return None
        
        numpy_data = data[column].values
        
        tensor_data = torch.tensor(numpy_data, dtype=torch.float32, device=self.device)
        
        return tensor_data
