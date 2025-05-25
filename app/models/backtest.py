import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """
    Analyze backtest results and calculate performance metrics
    """
    def __init__(self, portfolio_values: List[float], market_values: List[float], dates=None):
        """
        Initialize the backtest analyzer
        
        Args:
            portfolio_values: List of portfolio values over time
            market_values: List of market values over time (buy and hold)
            dates: Optional list of dates corresponding to values
        """
        self.portfolio_values = np.array(portfolio_values)
        self.market_values = np.array(market_values)
        self.dates = dates
        
        self.portfolio_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        self.market_returns = np.diff(self.market_values) / self.market_values[:-1]
        
    def calculate_metrics(self, risk_free_rate: float = 0.0, trading_days: int = 252) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
            
        Returns:
            Dictionary of performance metrics
        """
        portfolio_total_return = self.portfolio_values[-1] / self.portfolio_values[0] - 1
        market_total_return = self.market_values[-1] / self.market_values[0] - 1
        
        days = len(self.portfolio_values)
        years = days / trading_days
        
        portfolio_annual_return = ((1 + portfolio_total_return) ** (1 / years)) - 1
        market_annual_return = ((1 + market_total_return) ** (1 / years)) - 1
        
        portfolio_volatility = np.std(self.portfolio_returns) * np.sqrt(trading_days)
        market_volatility = np.std(self.market_returns) * np.sqrt(trading_days)
        
        daily_risk_free = ((1 + risk_free_rate) ** (1 / trading_days)) - 1
        portfolio_sharpe = (np.mean(self.portfolio_returns) - daily_risk_free) / np.std(self.portfolio_returns) * np.sqrt(trading_days)
        market_sharpe = (np.mean(self.market_returns) - daily_risk_free) / np.std(self.market_returns) * np.sqrt(trading_days)
        
        portfolio_drawdowns = self._calculate_drawdowns(self.portfolio_values)
        market_drawdowns = self._calculate_drawdowns(self.market_values)
        
        portfolio_max_dd = np.min(portfolio_drawdowns)
        market_max_dd = np.min(market_drawdowns)
        
        portfolio_ui = self._calculate_ulcer_index(portfolio_drawdowns)
        market_ui = self._calculate_ulcer_index(market_drawdowns)
        
        portfolio_upi = (portfolio_annual_return - risk_free_rate) / portfolio_ui if portfolio_ui > 0 else 0
        market_upi = (market_annual_return - risk_free_rate) / market_ui if market_ui > 0 else 0
        
        target_return = daily_risk_free
        portfolio_downside_returns = self.portfolio_returns[self.portfolio_returns < target_return]
        market_downside_returns = self.market_returns[self.market_returns < target_return]
        
        portfolio_downside_deviation = np.sqrt(np.mean(np.square(portfolio_downside_returns - target_return))) * np.sqrt(trading_days)
        market_downside_deviation = np.sqrt(np.mean(np.square(market_downside_returns - target_return))) * np.sqrt(trading_days)
        
        portfolio_sortino = (portfolio_annual_return - risk_free_rate) / portfolio_downside_deviation if portfolio_downside_deviation > 0 else 0
        market_sortino = (market_annual_return - risk_free_rate) / market_downside_deviation if market_downside_deviation > 0 else 0
        
        portfolio_calmar = portfolio_annual_return / abs(portfolio_max_dd) if portfolio_max_dd < 0 else 0
        market_calmar = market_annual_return / abs(market_max_dd) if market_max_dd < 0 else 0
        
        metrics = {
            'portfolio': {
                'total_return': portfolio_total_return,
                'annualized_return': portfolio_annual_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'sortino_ratio': portfolio_sortino,
                'calmar_ratio': portfolio_calmar,
                'max_drawdown': portfolio_max_dd,
                'ulcer_index': portfolio_ui,
                'ulcer_performance_index': portfolio_upi
            },
            'market': {
                'total_return': market_total_return,
                'annualized_return': market_annual_return,
                'volatility': market_volatility,
                'sharpe_ratio': market_sharpe,
                'sortino_ratio': market_sortino,
                'calmar_ratio': market_calmar,
                'max_drawdown': market_max_dd,
                'ulcer_index': market_ui,
                'ulcer_performance_index': market_upi
            },
            'comparison': {
                'excess_return': portfolio_total_return - market_total_return,
                'excess_annual_return': portfolio_annual_return - market_annual_return,
                'relative_volatility': portfolio_volatility / market_volatility if market_volatility > 0 else 0,
                'information_ratio': (portfolio_annual_return - market_annual_return) / (np.std(self.portfolio_returns - self.market_returns) * np.sqrt(trading_days)) if np.std(self.portfolio_returns - self.market_returns) > 0 else 0,
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
    
    def plot_performance(self, title: str = "Performance Comparison", figsize: Tuple[int, int] = (12, 8)):
        """
        Plot performance comparison between portfolio and market
        
        Args:
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        norm_portfolio = self.portfolio_values / self.portfolio_values[0]
        norm_market = self.market_values / self.market_values[0]
        
        if self.dates is not None:
            axes[0].plot(self.dates, norm_portfolio, label='Portfolio', linewidth=2)
            axes[0].plot(self.dates, norm_market, label='Market (Buy & Hold)', linewidth=2)
        else:
            axes[0].plot(norm_portfolio, label='Portfolio', linewidth=2)
            axes[0].plot(norm_market, label='Market (Buy & Hold)', linewidth=2)
            
        axes[0].set_title(title)
        axes[0].set_ylabel('Normalized Value')
        axes[0].legend()
        axes[0].grid(True)
        
        portfolio_drawdowns = self._calculate_drawdowns(self.portfolio_values)
        market_drawdowns = self._calculate_drawdowns(self.market_values)
        
        if self.dates is not None:
            axes[1].fill_between(self.dates, 0, portfolio_drawdowns, color='red', alpha=0.3, label='Portfolio Drawdown')
            axes[1].fill_between(self.dates, 0, market_drawdowns, color='blue', alpha=0.3, label='Market Drawdown')
        else:
            axes[1].fill_between(range(len(portfolio_drawdowns)), 0, portfolio_drawdowns, color='red', alpha=0.3, label='Portfolio Drawdown')
            axes[1].fill_between(range(len(market_drawdowns)), 0, market_drawdowns, color='blue', alpha=0.3, label='Market Drawdown')
            
        axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None):
        """
        Generate a comprehensive performance report
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as a string
        """
        metrics = self.calculate_metrics()
        
        report = []
        report.append("# Performance Report\n")
        
        report.append("## Portfolio Performance\n")
        for key, value in metrics['portfolio'].items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
        
        report.append("\n## Market Performance (Buy & Hold)\n")
        for key, value in metrics['market'].items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
        
        report.append("\n## Comparison\n")
        for key, value in metrics['comparison'].items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
        
        report.append("\n## Ulcer Performance Index (UPI) Analysis\n")
        report.append(f"- **Portfolio UPI**: {metrics['portfolio']['ulcer_performance_index']:.4f}\n")
        report.append(f"- **Market UPI**: {metrics['market']['ulcer_performance_index']:.4f}\n")
        report.append(f"- **UPI Improvement**: {metrics['comparison']['upi_improvement']:.4f}\n")
        
        if metrics['comparison']['upi_improvement'] > 0:
            report.append("\nThe trading strategy shows improved drawdown protection compared to the market.\n")
        else:
            report.append("\nThe trading strategy does not show improved drawdown protection compared to the market.\n")
        
        report_str = "".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {output_path}")
        
        return report_str
