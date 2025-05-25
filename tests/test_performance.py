import unittest
import numpy as np
import pandas as pd
from src.utils.performance import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.performance = PerformanceMetrics(use_gpu=False)
        
        self.prices = np.array([
            100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0, 109.0, 
            108.0, 107.0, 106.0, 105.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
            114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0,
            106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0
        ])
        
        self.positions = np.array([
            0, 0, 1, 1, 1, 1, 1, 1, 
            1, 1, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1
        ])
    
    def test_calculate_returns(self):
        returns = self.performance.calculate_returns(self.prices, self.positions)
        
        self.assertEqual(len(returns), len(self.prices))
        
        for i in range(2, len(self.prices)):
            expected_return = self.positions[i-2] * (self.prices[i] / self.prices[i-1] - 1)
            self.assertAlmostEqual(returns[i], expected_return, places=6)
    
    def test_calculate_cumulative_returns(self):
        returns = self.performance.calculate_returns(self.prices, self.positions)
        
        cumulative_returns = self.performance.calculate_cumulative_returns(returns)
        
        self.assertEqual(len(cumulative_returns), len(returns))
        
        expected_cumulative_returns = np.cumprod(1 + returns) - 1
        np.testing.assert_allclose(cumulative_returns, expected_cumulative_returns)
    
    def test_calculate_drawdowns(self):
        returns = self.performance.calculate_returns(self.prices, self.positions)
        cumulative_returns = self.performance.calculate_cumulative_returns(returns)
        
        drawdowns = self.performance.calculate_drawdowns(cumulative_returns)
        
        self.assertEqual(len(drawdowns), len(cumulative_returns))
        
        self.assertTrue(np.all(drawdowns <= 0))
        
        max_drawdown = self.performance.calculate_max_drawdown(cumulative_returns)
        self.assertEqual(max_drawdown, np.min(drawdowns))
    
    def test_calculate_ulcer_index(self):
        returns = self.performance.calculate_returns(self.prices, self.positions)
        cumulative_returns = self.performance.calculate_cumulative_returns(returns)
        
        ulcer_index = self.performance.calculate_ulcer_index(cumulative_returns)
        
        self.assertTrue(ulcer_index >= 0)
        
        drawdowns = self.performance.calculate_drawdowns(cumulative_returns)
        expected_ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        self.assertAlmostEqual(ulcer_index, expected_ulcer_index, places=6)
    
    def test_calculate_ulcer_performance_index(self):
        returns = self.performance.calculate_returns(self.prices, self.positions)
        
        upi = self.performance.calculate_ulcer_performance_index(returns, risk_free_rate=0.0)
        
        self.assertIsNotNone(upi)
        
        upi_with_rf = self.performance.calculate_ulcer_performance_index(returns, risk_free_rate=0.02)
        self.assertIsNotNone(upi_with_rf)
        self.assertNotEqual(upi, upi_with_rf)
    
    def test_calculate_performance_metrics(self):
        metrics = self.performance.calculate_performance_metrics(self.prices, self.positions)
        
        expected_metrics = [
            'Total Return', 'Annual Return', 'Volatility', 
            'Sharpe Ratio', 'Max Drawdown', 'Ulcer Index', 
            'Ulcer Performance Index'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        self.assertTrue(metrics['Volatility'] >= 0)
        self.assertTrue(metrics['Ulcer Index'] >= 0)
        self.assertTrue(metrics['Max Drawdown'] <= 0)

if __name__ == '__main__':
    unittest.main()
