"""
test_model.py

This file contains the implementation of the unit tests and integration tests for the EZ Diffusion Model.
- The test codes were generated with the assistance of ChatGPT o3-mini-high and ChatGPT Diffusion Models for Python Code.

Author: Aiden Hai
Date: 03/20/2025
"""

import unittest
import numpy as np
import math
from src.ez_diffusion import forward_stats, simulate_data, inverse_stats
from src.simulate_and_recover import simulate_and_recover_iteration, run_simulations

class TestEZDiffusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set a fixed random seed for reproducibility."""
        np.random.seed(6657)

    # ========== Testing forward_stats ==========
    def test_forward_stats_values(self):
        """Test if forward_stats outputs valid values within expected ranges."""
        a, v, t = 1.0, 1.0, 0.3
        r_pred, m_pred, v_pred = forward_stats(a, v, t)

        self.assertGreaterEqual(r_pred, 0)
        self.assertLessEqual(r_pred, 1)
        self.assertGreater(m_pred, t)  # Mean RT should be greater than non-decision time
        self.assertGreaterEqual(v_pred, 0)  # Variance should be non-negative

    def test_forward_stats_computation(self):
        """Test forward_stats against manually computed expected values."""
        a, v, t = 1.0, 1.0, 0.3
        r_pred, m_pred, v_pred = forward_stats(a, v, t)

        # Expected values
        y = math.exp(-1)
        expected_r_pred = 1 / (1 + y)
        expected_m_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        expected_v_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))

        self.assertAlmostEqual(r_pred, expected_r_pred, places=5)
        self.assertAlmostEqual(m_pred, expected_m_pred, places=5)
        self.assertAlmostEqual(v_pred, expected_v_pred, places=4)

    # ========== Testing simulate_data ==========
    def test_simulate_data_validity(self):
        """Test simulate_data returns valid values within expected ranges."""
        r_pred, m_pred, v_pred = 0.7, 0.5, 0.04
        N = 100
        r_obs, m_obs, v_obs = simulate_data(r_pred, m_pred, v_pred, N)

        self.assertGreaterEqual(r_obs, 0)
        self.assertLessEqual(r_obs, 1)
        self.assertTrue(np.isfinite(m_obs))
        self.assertGreater(v_obs, 0)

    def test_simulate_data_small_N(self):
        """Test simulate_data behavior when N is very small."""
        r_pred, m_pred, v_pred = 0.7, 0.5, 0.04
        N = 2  # Very small sample size
        r_obs, m_obs, v_obs = simulate_data(r_pred, m_pred, v_pred, N)

        self.assertGreaterEqual(r_obs, 0)
        self.assertLessEqual(r_obs, 1)
        self.assertTrue(np.isfinite(m_obs))
        self.assertGreater(v_obs, 0)

    def test_simulate_data_large_N(self):
        """Test if simulate_data approximates theoretical predictions as N increases."""
        r_pred, m_pred, v_pred = 0.7, 0.5, 0.04
        N = 10000  # Large sample size
        r_obs, m_obs, v_obs = simulate_data(r_pred, m_pred, v_pred, N)

        self.assertAlmostEqual(r_obs, r_pred, delta=0.05)
        self.assertAlmostEqual(m_obs, m_pred, delta=0.05)
        self.assertAlmostEqual(v_obs, v_pred, delta=0.05)

    # ========== Testing inverse_stats ==========
    def test_inverse_stats_recover(self):
        """Test if inverse_stats correctly recovers parameters from simulated data."""
        a_true, v_true, t_true = 1.0, 1.0, 0.3
        r_pred, m_pred, v_pred = forward_stats(a_true, v_true, t_true)
        r_obs, m_obs, v_obs = simulate_data(r_pred, m_pred, v_pred, 1000)
        v_est, a_est, t_est = inverse_stats(r_obs, m_obs, v_obs)

        self.assertAlmostEqual(v_est, v_true, delta=0.2)
        self.assertAlmostEqual(a_est, a_true, delta=0.2)
        self.assertAlmostEqual(t_est, t_true, delta=0.05)

    def test_inverse_stats_extreme_r_obs(self):
        """Test inverse_stats with extreme values of r_obs."""
        v_est, a_est, t_est = inverse_stats(0.99, 0.5, 0.02)  # r_obs near 1
        self.assertTrue(np.isfinite([v_est, a_est, t_est]).all())
        self.assertGreater(a_est, 0)

        v_est, a_est, t_est = inverse_stats(0.01, 0.5, 0.02)  # r_obs near 0
        self.assertTrue(np.isfinite([v_est, a_est, t_est]).all())

    # ========== Testing simulate_and_recover ==========
    def test_simulate_and_recover_iteration(self):
        """Test if simulate_and_recover_iteration produces reasonable bias values."""
        bias_v, bias_a, bias_t = simulate_and_recover_iteration(100)

        self.assertLess(abs(bias_v), 1.0)
        self.assertLess(abs(bias_a), 1.0)
        self.assertLess(abs(bias_t), 0.2)

    def test_run_simulations(self):
        """Test if mean bias is close to 0 and MSE decreases with increasing N."""
        mean_bias_10, mse_10 = run_simulations(10, 500)
        mean_bias_40, mse_40 = run_simulations(40, 500)
        mean_bias_4000, mse_4000 = run_simulations(4000, 500)

        # Mean bias should be close to 0
        self.assertAlmostEqual(mean_bias_4000[0], 0, delta=0.1)
        self.assertAlmostEqual(mean_bias_4000[1], 0, delta=0.1)
        self.assertAlmostEqual(mean_bias_4000[2], 0, delta=0.05)

        # MSE should decrease as N increases
        self.assertLess(mse_40[0], mse_10[0])
        self.assertLess(mse_40[1], mse_10[1])
        self.assertLess(mse_40[2], mse_10[2])

        self.assertLess(mse_4000[0], mse_40[0])
        self.assertLess(mse_4000[1], mse_40[1])
        self.assertLess(mse_4000[2], mse_40[2])


    # ========== Integration tests ==========
    def test_full_simulation_pipeline(self):
        """Integration Test: Ensure the full simulation pipeline works correctly."""
        N = 100  # Moderate sample size to balance accuracy and runtime
        iterations = 1000
        
        mean_bias, mse = run_simulations(N, iterations)

        # Ensure outputs are finite numbers
        self.assertTrue(np.isfinite(mean_bias).all(), "Mean Bias contains non-finite values.")
        self.assertTrue(np.isfinite(mse).all(), "MSE contains non-finite values.")

        # Ensure MSE is non-negative
        self.assertTrue((mse >= 0).all(), "MSE values should be non-negative.")

    def test_increasing_sample_size_improves_accuracy(self):
        """Integration Test: Verify that increasing N improves parameter recovery."""
        mean_bias_10, mse_10 = run_simulations(10, 100)
        mean_bias_40, mse_40 = run_simulations(40, 100)
        mean_bias_4000, mse_4000 = run_simulations(4000, 100)

        # Verify that the absolute bias decreases as N increases
        self.assertLessEqual(abs(mean_bias_40[0]), abs(mean_bias_10[0]), "Drift Rate bias should decrease")
        self.assertLessEqual(abs(mean_bias_40[1]), abs(mean_bias_10[1]), "Boundary Separation bias should decrease")
        self.assertLessEqual(abs(mean_bias_40[2]), abs(mean_bias_10[2]), "Non-decision Time bias should decrease")

        self.assertLessEqual(abs(mean_bias_4000[0]), abs(mean_bias_40[0]), "Drift Rate bias should further decrease")
        self.assertLessEqual(abs(mean_bias_4000[1]), abs(mean_bias_40[1]), "Boundary Separation bias should further decrease")
        self.assertLessEqual(abs(mean_bias_4000[2]), abs(mean_bias_40[2]), "Non-decision Time bias should further decrease")

        # Verify that MSE decreases as N increases
        self.assertLessEqual(mse_40[0], mse_10[0], "Drift Rate MSE should decrease")
        self.assertLessEqual(mse_40[1], mse_10[1], "Boundary Separation MSE should decrease")
        self.assertLessEqual(mse_40[2], mse_10[2], "Non-decision Time MSE should decrease")

        self.assertLessEqual(mse_4000[0], mse_40[0], "Drift Rate MSE should further decrease")
        self.assertLessEqual(mse_4000[1], mse_40[1], "Boundary Separation MSE should further decrease")
        self.assertLessEqual(mse_4000[2], mse_40[2], "Non-decision Time MSE should further decrease")


if __name__ == "__main__":
    unittest.main()