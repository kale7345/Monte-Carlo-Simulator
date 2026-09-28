import unittest

import numpy as np

from mc import (
    black_scholes_call,
    calculate_var_es,
    get_market_config,
    mc_call_price,
    mc_call_price_antithetic,
    simulate_gbm,
    simulate_portfolio_values,
)


class MonteCarloTests(unittest.TestCase):
    def test_market_configurations_use_local_tickers_and_currency(self):
        us = get_market_config("US")
        australia = get_market_config("au")
        self.assertEqual(us.currency, "US$")
        self.assertEqual(australia.currency, "A$")
        self.assertTrue(all(ticker.endswith(".AX") for ticker in australia.tickers))
        self.assertAlmostEqual(sum(australia.weights), 1.0)

    def test_gbm_shape_starting_price_and_reproducibility(self):
        first = simulate_gbm(100, 0.08, 0.25, 1, 20, 30, seed=7)
        second = simulate_gbm(100, 0.08, 0.25, 1, 20, 30, seed=7)
        self.assertEqual(first.shape, (21, 30))
        np.testing.assert_array_equal(first[0], np.full(30, 100.0))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0))

    def test_monte_carlo_call_matches_black_scholes(self):
        expected = black_scholes_call(100, 105, 1, 0.05, 0.25)
        estimated = mc_call_price(100, 105, 1, 0.05, 0.25, 250_000, seed=12)
        self.assertAlmostEqual(estimated, expected, delta=0.12)

    def test_antithetic_call_matches_black_scholes(self):
        expected = black_scholes_call(100, 105, 1, 0.05, 0.25)
        estimated = mc_call_price_antithetic(100, 105, 1, 0.05, 0.25, 125_000, seed=12)
        self.assertAlmostEqual(estimated, expected, delta=0.12)

    def test_var_and_expected_shortfall_ordering(self):
        pnl = np.array([-10.0, -8.0, -6.0, -2.0, 1.0, 3.0])
        var, expected_shortfall = calculate_var_es(pnl, confidence=0.80)
        self.assertGreaterEqual(expected_shortfall, var)

    def test_correlated_portfolio_simulation_shape(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0002, 0.01, size=(100, 2))
        terminal, paths, _ = simulate_portfolio_values(
            returns, np.array([0.6, 0.4]), 10_000, 5, 12, seed=9
        )
        self.assertEqual(terminal.shape, (12,))
        self.assertEqual(paths.shape, (5, 12))
        self.assertTrue(np.all(terminal > 0))


if __name__ == "__main__":
    unittest.main()