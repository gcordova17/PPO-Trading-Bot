import unittest
import numpy as np
import pandas as pd
from src.strategies.ppo_strategy import PPOStrategy

class TestPPOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PPOStrategy(fast_period=12, slow_period=26, signal_period=9, use_gpu=False)
        
        self.prices = np.array([
            100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0, 109.0, 
            108.0, 107.0, 106.0, 105.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
            114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0,
            106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0
        ])
        
        self.dates = pd.date_range(start='2020-01-01', periods=len(self.prices))
    
    def test_generate_signals(self):
        ppo, signal, histogram, positions = self.strategy.generate_signals(self.prices)
        
        self.assertEqual(len(positions), len(self.prices))
        
        self.assertTrue(np.all((positions == 0) | (positions == 1)))
        
        position_changes = np.diff(positions)
        self.assertTrue(np.any(position_changes != 0))
    
    def test_backtest(self):
        results = self.strategy.backtest(self.prices, self.dates)
        
        expected_columns = [
            'Date', 'Price', 'PPO', 'Signal', 'Histogram', 
            'Position', 'Daily_Return', 'Strategy_Return', 'Cumulative_Return'
        ]
        for col in expected_columns:
            self.assertIn(col, results.columns)
        
        self.assertEqual(len(results), len(self.prices))
        
        for i in range(2, len(results)):
            expected_return = results.iloc[i-2]['Position'] * (
                results.iloc[i]['Price'] / results.iloc[i-1]['Price'] - 1
            )
            self.assertAlmostEqual(results.iloc[i]['Strategy_Return'], expected_return, places=6)
        
        strategy_returns = results['Strategy_Return'].to_numpy()
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        
        actual_cum_returns = results['Cumulative_Return'].to_numpy()
        np.testing.assert_almost_equal(
            actual_cum_returns,
            cumulative_returns,
            decimal=5
        )

if __name__ == '__main__':
    unittest.main()
