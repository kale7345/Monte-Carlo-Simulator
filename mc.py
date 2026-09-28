"""Monte Carlo demonstrations for option pricing and portfolio risk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Option-pricing example
S0 = 100.0
STRIKE = 105.0
MATURITY = 1.0
RISK_FREE_RATE = 0.05
VOLATILITY = 0.25
OPTION_PATHS = 500_000
OPTION_SEED = 42

# Historical portfolio-risk examples. Australian symbols use Yahoo Finance's
# .AX suffix for securities listed on the Australian Securities Exchange.
@dataclass(frozen=True)
class MarketConfig:
    name: str
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    currency: str


MARKETS = {
    "us": MarketConfig("United States", ("AAPL", "MSFT", "JPM", "XOM"), (0.30, 0.30, 0.20, 0.20), "US$"),
    "au": MarketConfig("Australia (ASX)", ("BHP.AX", "CBA.AX", "CSL.AX", "WES.AX"), (0.30, 0.30, 0.20, 0.20), "A$"),
}
PORTFOLIO_VALUE = 100_000.0
HORIZON_DAYS = 10
RISK_PATHS = 50_000
RISK_START_DATE = "2020-01-01"
RISK_SEED = 42

OUTPUT_DIR = Path("output")


def get_market_config(market: str) -> MarketConfig:
    """Return the configured portfolio for a supported market code."""
    try:
        return MARKETS[market.lower()]
    except KeyError as error:
        supported = ", ".join(MARKETS)
        raise ValueError(f"Unknown market '{market}'. Choose from: {supported}") from error


def simulate_gbm(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Return GBM paths with shape (steps + 1, n_paths), including time zero."""
    if initial_price <= 0 or volatility < 0 or years < 0:
        raise ValueError("initial_price must be positive; volatility and years cannot be negative")
    if steps < 1 or n_paths < 1:
        raise ValueError("steps and n_paths must be positive")

    rng = np.random.default_rng(seed)
    dt = years / steps
    shocks = rng.standard_normal((steps, n_paths))
    increments = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_paths = np.vstack((np.zeros((1, n_paths)), np.cumsum(increments, axis=0)))
    return initial_price * np.exp(log_paths)


def black_scholes_call(
    initial_price: float,
    strike: float,
    years: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """Calculate the Black-Scholes value of a European call option."""
    if initial_price <= 0 or strike <= 0 or years < 0 or volatility < 0:
        raise ValueError("prices must be positive; years and volatility cannot be negative")
    if years == 0 or volatility == 0:
        terminal_price = initial_price * np.exp(risk_free_rate * years)
        return float(np.exp(-risk_free_rate * years) * max(terminal_price - strike, 0.0))

    root_time = np.sqrt(years)
    d1 = (
        np.log(initial_price / strike)
        + (risk_free_rate + 0.5 * volatility**2) * years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    return float(
        initial_price * norm.cdf(d1)
        - strike * np.exp(-risk_free_rate * years) * norm.cdf(d2)
    )


def mc_call_price(
    initial_price: float,
    strike: float,
    years: float,
    risk_free_rate: float,
    volatility: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """Price a call from exact GBM terminal draws without storing full paths."""
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)
    terminal_prices = initial_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * years
        + volatility * np.sqrt(years) * shocks
    )
    payoffs = np.maximum(terminal_prices - strike, 0.0)
    return float(np.exp(-risk_free_rate * years) * payoffs.mean())


def mc_call_price_antithetic(
    initial_price: float,
    strike: float,
    years: float,
    risk_free_rate: float,
    volatility: float,
    n_pairs: int = 50_000,
    seed: int = 42,
) -> float:
    """Price a call using antithetic shock pairs; n_pairs counts pairs."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_pairs)
    log_drift = (risk_free_rate - 0.5 * volatility**2) * years
    log_shock = volatility * np.sqrt(years) * shocks
    positive_payoffs = np.maximum(initial_price * np.exp(log_drift + log_shock) - strike, 0.0)
    negative_payoffs = np.maximum(initial_price * np.exp(log_drift - log_shock) - strike, 0.0)
    pair_average = 0.5 * (positive_payoffs + negative_payoffs)
    return float(np.exp(-risk_free_rate * years) * pair_average.mean())


def calculate_var_es(pnl: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Return positive loss VaR and expected shortfall at the given confidence."""
    if pnl.size == 0 or not 0 < confidence < 1:
        raise ValueError("pnl must be non-empty and confidence must be between 0 and 1")
    pnl_quantile = float(np.percentile(pnl, 100.0 * (1.0 - confidence)))
    tail = pnl[pnl <= pnl_quantile]
    return -pnl_quantile, float(-tail.mean())


def plot_gbm_paths() -> None:
    paths = simulate_gbm(S0, 0.08, VOLATILITY, MATURITY, 252, 10_000, seed=OPTION_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(paths[:, :200], alpha=0.15, linewidth=0.5)
    plt.plot(paths.mean(axis=1), color="black", linewidth=2, label="Simulated mean")
    plt.axhline(S0, color="grey", linestyle="--", linewidth=0.8)
    plt.title("GBM paths (200 shown) | S0=100, drift=8%, volatility=25%")
    plt.xlabel("Trading day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "paths.png", dpi=150)
    plt.close()


def run_option_demo() -> None:
    benchmark = black_scholes_call(S0, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY)
    mc_price = mc_call_price(
        S0, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY, OPTION_PATHS, OPTION_SEED
    )
    anti_price = mc_call_price_antithetic(
        S0, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY, OPTION_PATHS // 2, OPTION_SEED
    )
    print(f"Monte Carlo price : ${mc_price:.3f}")
    print(f"Antithetic price  : ${anti_price:.3f}")
    print(f"Black-Scholes     : ${benchmark:.3f}")
    print(f"MC difference     : ${abs(mc_price - benchmark):.3f}")

    sizes = np.array([1_000, 5_000, 10_000, 50_000, 100_000, OPTION_PATHS])
    rng = np.random.default_rng(OPTION_SEED + 1)
    shocks = rng.standard_normal(int(sizes[-1]))
    estimates = []
    for size in sizes:
        terminal = S0 * np.exp(
            (RISK_FREE_RATE - 0.5 * VOLATILITY**2) * MATURITY
            + VOLATILITY * np.sqrt(MATURITY) * shocks[:size]
        )
        estimates.append(
            float(np.exp(-RISK_FREE_RATE * MATURITY) * np.maximum(terminal - STRIKE, 0).mean())
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(sizes))
    plt.axhline(benchmark, color="red", linestyle="--", label="Black-Scholes")
    plt.plot(indices, estimates, "o-", label="Monte Carlo estimate")
    plt.fill_between(indices, benchmark - 0.5, benchmark + 0.5, alpha=0.15, color="red")
    plt.xticks(indices, [f"{size // 1000}k" for size in sizes])
    plt.xlabel("Number of paths")
    plt.ylabel("Call price ($)")
    plt.title("Monte Carlo convergence to Black-Scholes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.close()
    plot_gbm_paths()
    print(f"Option charts saved in {OUTPUT_DIR.resolve()}")


def download_log_returns(tickers: list[str]) -> np.ndarray:
    """Download adjusted prices and return aligned daily log returns."""
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError("Install project dependencies before running --risk") from error

    prices = yf.download(
        tickers,
        start=RISK_START_DATE,
        auto_adjust=True,
        progress=False,
    )["Close"]
    if hasattr(prices, "to_frame"):
        prices = prices.to_frame()
    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty or len(prices) < 2 or prices.isna().any().any():
        raise RuntimeError("Could not download complete price history for all requested tickers")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.to_numpy(dtype=float)


def simulate_portfolio_values(
    log_returns: np.ndarray,
    weights: np.ndarray,
    initial_value: float,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate correlated asset paths and return terminal and daily portfolio values."""
    daily_mean = log_returns.mean(axis=0)
    covariance = np.cov(log_returns, rowvar=False)
    covariance = np.atleast_2d(covariance)
    factor = np.linalg.cholesky(covariance + np.eye(len(weights)) * 1e-12)
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((horizon_days, len(weights), n_paths))
    correlated = np.einsum("ij,djn->din", factor, independent)
    daily_log_returns = daily_mean[None, :, None] + correlated
    relative_prices = np.exp(np.cumsum(daily_log_returns, axis=0))
    paths = initial_value * np.einsum("i,dis->ds", weights, relative_prices)
    return paths[-1], paths, daily_mean


def plot_risk_dashboard(
    pnl: np.ndarray,
    paths: np.ndarray,
    var_95: float,
    es_95: float,
    currency: str,
    output_name: str,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(pnl, bins=100, color="steelblue", alpha=0.8)
    axes[0].axvline(-var_95, color="orange", linewidth=2, label=f"VaR 95% ({currency}{var_95:,.0f})")
    axes[0].axvline(-es_95, color="red", linewidth=2, label=f"ES 95% ({currency}{es_95:,.0f})")
    axes[0].set_title(f"10-day P&L ({currency})")
    axes[0].legend()

    axes[1].plot(paths[:, :200], alpha=0.15, linewidth=0.5)
    axes[1].axhline(PORTFOLIO_VALUE, color="grey", linestyle="--")
    axes[1].set_title("Correlated portfolio paths (200 shown)")
    axes[1].set_xlabel("Trading day")

    levels = np.arange(90.0, 100.0, 0.5)
    var_curve = [calculate_var_es(pnl, level / 100.0)[0] for level in levels]
    axes[2].plot(levels, var_curve, linewidth=2, color="darkred")
    axes[2].set_title("VaR vs confidence level")
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_ylabel(f"VaR ({currency})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=150)
    plt.close(fig)


def run_risk_demo(market: str) -> None:
    config = get_market_config(market)
    tickers = list(config.tickers)
    weights = np.array(config.weights)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("WEIGHTS must sum to 1")
    log_returns = download_log_returns(tickers)
    daily_mean = log_returns.mean(axis=0)

    terminal_values, paths, _ = simulate_portfolio_values(
        log_returns, weights, PORTFOLIO_VALUE, HORIZON_DAYS, RISK_PATHS, RISK_SEED
    )
    correlated_pnl = terminal_values - PORTFOLIO_VALUE
    correlated_var, correlated_es = calculate_var_es(correlated_pnl)

    portfolio_returns = np.expm1(log_returns) @ weights
    portfolio_mean = float(portfolio_returns.mean())
    portfolio_volatility = float(portfolio_returns.std(ddof=1))
    rng = np.random.default_rng(RISK_SEED)
    shocks = rng.standard_normal((HORIZON_DAYS, RISK_PATHS))
    increments = (
        portfolio_mean - 0.5 * portfolio_volatility**2
    ) + portfolio_volatility * shocks
    portfolio_paths = PORTFOLIO_VALUE * np.exp(np.cumsum(increments, axis=0))
    aggregate_pnl = portfolio_paths[-1] - PORTFOLIO_VALUE
    aggregate_var, aggregate_es = calculate_var_es(aggregate_pnl)

    print(f"Market: {config.name} ({', '.join(tickers)})")
    print(f"Historical log returns: {len(log_returns)} days")
    print(f"Horizon: {HORIZON_DAYS} trading days; portfolio: {config.currency}{PORTFOLIO_VALUE:,.0f}")
    print(f"Portfolio daily return: {portfolio_mean:.5f} +/- {portfolio_volatility:.5f}")
    print(f"Portfolio-level 95% VaR: {config.currency}{aggregate_var:,.0f}; ES: {config.currency}{aggregate_es:,.0f}")
    print(f"Correlated-assets 95% VaR: {config.currency}{correlated_var:,.0f}; ES: {config.currency}{correlated_es:,.0f}")
    output_name = f"risk_dashboard_{market.lower()}.png"
    plot_risk_dashboard(correlated_pnl, paths, correlated_var, correlated_es, config.currency, output_name)
    print(f"Risk chart saved to {OUTPUT_DIR / output_name}")
    print(f"Weights: {np.array2string(weights, precision=2)}; daily means: {np.array2string(daily_mean, precision=5)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk",
        action="store_true",
        help="download adjusted prices and run the portfolio VaR/ES demonstration",
    )
    parser.add_argument(
        "--market",
        choices=MARKETS,
        default="us",
        help="market portfolio for --risk: us or au (default: us)",
    )
    args = parser.parse_args()
    if args.risk:
        run_risk_demo(args.market)
    else:
        run_option_demo()


if __name__ == "__main__":
    main()