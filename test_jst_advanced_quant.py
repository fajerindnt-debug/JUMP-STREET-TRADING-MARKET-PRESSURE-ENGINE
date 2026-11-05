"""
JST APEX QUANT — TEST RUNNER
File 2: test_jst_advanced_quant.py

Letakkan file ini satu folder dengan jst_advanced_quant.py.
Jalankan:
    python test_jst_advanced_quant.py

File ini hanya menguji modul riset. Tidak mengubah strategi V8.5.
"""

import unittest
import numpy as np
import pandas as pd

import jst_advanced_quant as jq


class TestAdvancedQuant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.returns = pd.Series(rng.normal(0.0002, 0.01, 300))
        cls.prices = 2300 * np.exp(cls.returns.cumsum())

    def test_distribution_report(self):
        result = jq.distribution_report(self.returns)
        self.assertEqual(result["n"], 300)
        self.assertIn("skewness", result)
        self.assertIn("excess_kurtosis", result)

    def test_volatility_report(self):
        result = jq.volatility_report(self.returns)
        self.assertGreaterEqual(result["ewma_volatility_per_bar"], 0)

    def test_bayesian_probability(self):
        result = jq.bayesian_win_probability(wins=20, losses=10)
        self.assertGreater(result["posterior_mean"], 0.5)
        self.assertLess(result["posterior_mean"], 1.0)

    def test_probability_validation(self):
        result = jq.probability_validation(
            probabilities=[0.8, 0.7, 0.2, 0.1],
            outcomes=[1, 1, 0, 0]
        )
        self.assertEqual(result["n"], 4)
        self.assertGreaterEqual(result["brier_score"], 0)

    def test_markov_transition(self):
        result = jq.markov_transition_matrix(
            ["UP", "UP", "DOWN", "DOWN", "UP", "DOWN"]
        )
        self.assertIn("UP", result["states"])
        self.assertIn("DOWN", result["states"])

    def test_monte_carlo(self):
        result = jq.monte_carlo_gbm(
            self.returns, horizon=5, paths=200, seed=7
        )
        self.assertEqual(result["paths"], 200)
        self.assertGreater(result["mean_terminal"], 0)

    def test_risk_report(self):
        result = jq.risk_report(self.returns)
        self.assertIn("historical_var_loss_per_bar", result)
        self.assertIn("max_drawdown", result)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            jq.probability_validation([1.2, 0.5], [1, 0])

    def test_short_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            jq.distribution_report([0.01, -0.01, 0.02])


if __name__ == "__main__":
    unittest.main(verbosity=2)
