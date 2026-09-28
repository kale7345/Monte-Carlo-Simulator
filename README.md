# Monte Carlo Finance Simulator

A small Python project that uses geometric Brownian motion (GBM) to price a European call option and estimate portfolio market risk. The option demo compares simulation results with the Black-Scholes formula; the risk demo estimates 10-day Value-at-Risk (VaR) and expected shortfall (ES) from adjusted historical stock prices.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or newer
- Internet access only for the optional Yahoo Finance risk demo

## Setup

From this project directory, create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

If the `py` launcher is unavailable, use the full path to your Python executable in place of `py -3`. In VS Code, select `.venv` as the workspace interpreter (`Python: Select Interpreter`).

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Run

Run the option pricing example:

```powershell
python mc.py
```

This prints the standard Monte Carlo, antithetic Monte Carlo, and Black-Scholes prices. It writes `paths.png` and `convergence.png` to the `output/` directory. Monte Carlo uses the risk-free rate as its drift, discounts the terminal payoff, and samples the exact GBM terminal distribution. The separate path visualization uses time-stepped GBM paths.

Run the historical portfolio risk example:

```powershell
python mc.py --risk
python mc.py --risk --market au
```

The default `us` market uses AAPL, MSFT, JPM, and XOM and reports results in US dollars. Use `--market au` for the Australian Securities Exchange portfolio (BHP.AX, CBA.AX, CSL.AX, and WES.AX), reported in Australian dollars. Both portfolios' tickers and weights are configured near the top of `mc.py`. Each command downloads adjusted closes, estimates daily log-return statistics, prints 95% 10-day VaR and ES for both a portfolio-level GBM and a correlated multi-asset simulation, and saves a market-specific chart (`output/risk_dashboard_us.png` or `output/risk_dashboard_au.png`). The risk command needs an internet connection and may be affected by Yahoo Finance availability or data changes.

The market selector runs one local-currency portfolio at a time. Combining U.S. and Australian holdings would require modeling the USD/AUD exchange rate, which this project does not currently do.

Run the tests:

```powershell
python -m unittest -v
```

## Model notes

GBM evolves prices in log space, so simulated prices remain positive. For a call option, the risk-neutral drift is the risk-free rate; the stock's historical return is not the option-pricing drift. VaR is reported as a positive loss at the selected percentile. ES is the mean loss in the tail beyond that percentile.

The portfolio-level model is a compact approximation that fits one return distribution to the weighted portfolio. The correlated-assets model simulates each holding using its own historical mean and the full covariance matrix, then values the weighted holdings together. Both assume recent return behavior is informative and that the fitted distribution is stable. GBM does not capture volatility clustering, jumps, or fat tails; results are estimates, not forecasts or investment advice.

See [PROJECT_NOTES.txt](PROJECT_NOTES.txt) for the project record, formulas, assumptions, and extension ideas.