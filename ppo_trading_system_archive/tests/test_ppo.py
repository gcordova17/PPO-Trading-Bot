import unittest
import numpy as np
import torch
from src.indicators.ppo import PPO

class TestPPO(unittest.TestCase):
    def setUp(self):
        self.ppo = PPO(fast_period=12, slow_period=26, signal_period=9, use_gpu=False)
        
        self.prices = np.array([
            100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0, 109.0, 
            108.0, 107.0, 106.0, 105.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
            114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0,
            106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0
        ])
    
    def test_calculate(self):
        ppo, signal, histogram = self.ppo.calculate(self.prices)
        
        self.assertEqual(len(ppo), len(self.prices))
        self.assertEqual(len(signal), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))
        
        np.testing.assert_allclose(histogram, ppo - signal)
        
        self.assertTrue(np.max(ppo) < 10.0)  # PPO should be small for this data
        self.assertTrue(np.min(ppo) > -10.0)
    
    def test_generate_signals(self):
        ppo, signal, histogram, buy_signals, sell_signals = self.ppo.generate_signals(self.prices)
        
        self.assertEqual(len(buy_signals), len(self.prices))
        self.assertEqual(len(sell_signals), len(self.prices))
        
        self.assertTrue(np.all((buy_signals == 0) | (buy_signals == 1)))
        self.assertTrue(np.all((sell_signals == 0) | (sell_signals == 1)))
        
        self.assertTrue(np.all((buy_signals * sell_signals) == 0))
        
        for i in range(1, len(ppo)):
            if buy_signals[i] == 1:
                self.assertTrue(ppo[i-1] < signal[i-1] and ppo[i] > signal[i])
            if sell_signals[i] == 1:
                self.assertTrue(ppo[i-1] > signal[i-1] and ppo[i] < signal[i])
    
    def test_gpu_support(self):
        if torch.cuda.is_available():
            ppo_gpu = PPO(fast_period=12, slow_period=26, signal_period=9, use_gpu=True)
            ppo_cpu = PPO(fast_period=12, slow_period=26, signal_period=9, use_gpu=False)
            
            ppo_gpu_result, _, _ = ppo_gpu.calculate(self.prices)
            ppo_cpu_result, _, _ = ppo_cpu.calculate(self.prices)
            
            np.testing.assert_allclose(ppo_gpu_result, ppo_cpu_result, rtol=1e-5)
            
            self.assertEqual(ppo_gpu.device.type, 'cuda')
        else:
            print("CUDA not available, skipping GPU test")

if __name__ == '__main__':
    unittest.main()
