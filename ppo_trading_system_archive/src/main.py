import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch

from src.utils.data_loader import DataLoader
from src.indicators.ppo import PPO
from src.strategies.ppo_strategy import PPOStrategy
from src.backtest.backtest import Backtest
from src.utils.performance import PerformanceMetrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPO Trading System')
    
    parser.add_argument('--ticker', type=str, default='SPY',
                        help='Ticker symbol to analyze')
    parser.add_argument('--benchmark', type=str, default='SPY',
                        help='Benchmark ticker symbol')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--fast-period', type=int, default=12,
                        help='Fast EMA period for PPO')
    parser.add_argument('--slow-period', type=int, default=26,
                        help='Slow EMA period for PPO')
    parser.add_argument('--signal-period', type=int, default=9,
                        help='Signal line period for PPO')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    
    return parser.parse_args()

def main():
    """Main function to run the PPO trading system."""
    args = parse_args()
    
    if args.start_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years of data
        args.start_date = start_date.strftime('%Y-%m-%d')
        args.end_date = end_date.strftime('%Y-%m-%d')
    elif args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    data_loader = DataLoader(use_gpu=use_gpu)
    ppo_strategy = PPOStrategy(
        fast_period=args.fast_period,
        slow_period=args.slow_period,
        signal_period=args.signal_period,
        use_gpu=use_gpu
    )
    backtest = Backtest(use_gpu=use_gpu)
    
    print(f"Running backtest for {args.ticker} from {args.start_date} to {args.end_date}")
    results, benchmark_results, performance_metrics = backtest.run(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy=ppo_strategy,
        benchmark_ticker=args.benchmark
    )
    
    if results is None:
        print("Backtest failed. Please check your inputs.")
        return
    
    print("\nPerformance Metrics:")
    print("===================")
    
    if performance_metrics is not None and 'Strategy' in performance_metrics:
        print("\nStrategy Metrics:")
        for metric, value in performance_metrics['Strategy'].items():
            print(f"{metric}: {value:.4f}")
        
        if 'Ulcer Performance Index' in performance_metrics['Strategy']:
            upi = performance_metrics['Strategy']['Ulcer Performance Index']
            print(f"\nUlcer Performance Index: {upi:.4f}")
            if upi > 1.0:
                print("The strategy has a good Ulcer Performance Index (> 1.0), indicating good risk-adjusted returns.")
            else:
                print("The strategy has a low Ulcer Performance Index (<= 1.0), indicating poor risk-adjusted returns.")
    else:
        print("\nNo strategy metrics available.")
    
    if performance_metrics is not None and 'Benchmark' in performance_metrics and performance_metrics['Benchmark'] is not None:
        print("\nBenchmark Metrics:")
        for metric, value in performance_metrics['Benchmark'].items():
            print(f"{metric}: {value:.4f}")
        
        if ('Strategy' in performance_metrics and 
            'Annual Return' in performance_metrics['Strategy'] and 
            'Annual Return' in performance_metrics['Benchmark'] and
            performance_metrics['Strategy']['Annual Return'] > performance_metrics['Benchmark']['Annual Return']):
            print("\nThe strategy outperforms the benchmark in terms of annual return.")
        else:
            print("\nThe strategy does not outperform the benchmark in terms of annual return.")
    else:
        print("\nNo benchmark metrics available.")
    
    figures = backtest.plot_results(results, benchmark_results, save_path=args.output_dir)
    
    if results is not None:
        results.to_csv(f"{args.output_dir}/results.csv", index=False)
    
    if benchmark_results is not None:
        benchmark_results.to_csv(f"{args.output_dir}/benchmark_results.csv", index=False)
    
    if performance_metrics is not None and 'Strategy' in performance_metrics:
        metrics_df = pd.DataFrame({
            'Metric': list(performance_metrics['Strategy'].keys()),
            'Strategy': list(performance_metrics['Strategy'].values()),
            'Benchmark': (list(performance_metrics['Benchmark'].values()) 
                         if performance_metrics.get('Benchmark') is not None 
                         else [None] * len(performance_metrics['Strategy']))
        })
        metrics_df.to_csv(f"{args.output_dir}/performance_metrics.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
