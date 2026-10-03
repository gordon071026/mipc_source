#!/usr/bin/env python3
"""Summarize a user-specified portfolio without optimization.

The first price column after the date column is treated as cash_eur, the
risk-free benchmark. The script asks for the percentage allocation to each
remaining asset, validates that allocations total 100%, and writes the same
main output filenames as efficient_frontier.py. No efficient-frontier
optimization is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRADING_PERIODS_DEFAULT = 12


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=None, engine="python")
    if raw.empty or raw.shape[1] < 3:
        raise ValueError("CSV must contain a date column and at least two asset columns.")
    date_candidates = [c for c in raw.columns if str(c).strip().lower() in {"date", "dates", "time"}]
    date_col = date_candidates[0] if date_candidates else raw.columns[0]
    dates = pd.to_datetime(raw[date_col], format="mixed", errors="coerce")
    prices = raw.drop(columns=[date_col]).copy()
    for col in prices.columns:
        if prices[col].dtype == "object":
            prices[col] = prices[col].astype(str).str.replace(",", "", regex=False).str.strip()
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices.index = dates
    prices = prices.loc[:, prices.notna().any(axis=0)]
    prices = prices.dropna(how="all")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if len(prices) < 3 or prices.shape[1] < 3:
        raise ValueError("At least three observations and cash_eur plus two risky assets are required.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive; check the CSV columns and formatting.")
    return prices


def ask_allocations(asset_names: list[str]) -> pd.Series:
    print("\nEnter the portfolio allocation for each risky asset.")
    print("Use percentages. Press Enter for 0%. cash_eur is used as the risk-free benchmark and is not prompted.\n")
    values: dict[str, float] = {}
    for asset in asset_names:
        while True:
            answer = input(f"Allocation for {asset} (%): ").strip()
            if answer == "":
                values[asset] = 0.0
                break
            try:
                value = float(answer.replace("%", ""))
            except ValueError:
                print("Please enter a numeric percentage, such as 10, or press Enter for 0%.")
                continue
            if not 0 <= value <= 100:
                print("Each allocation must be between 0% and 100%.")
                continue
            values[asset] = value / 100.0
            break
    weights = pd.Series(values, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Allocations must total 100%; your total is {weights.sum():.2%}.")
    return weights


def calculate_stats(weights: np.ndarray, returns: pd.DataFrame, cash_returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    portfolio_returns = returns @ weights
    annualized_return = float(portfolio_returns.mean() * periods_per_year)
    annualized_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year))
    aligned = pd.concat([portfolio_returns.rename("portfolio"), cash_returns.rename("cash")], axis=1).dropna()
    excess = aligned["portfolio"] - aligned["cash"]
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) > 0 else float("nan")
    historical_cagr = float((1.0 + portfolio_returns).prod() ** (periods_per_year / len(portfolio_returns)) - 1.0)
    equity = (1.0 + portfolio_returns).cumprod()
    maximum_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "historical_cagr": historical_cagr,
        "maximum_drawdown": maximum_drawdown,
        "sharpe_ratio": sharpe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV containing a date column followed by price columns")
    parser.add_argument("--output-dir", default="portfolio_summary_output", help="Directory for the output files")
    parser.add_argument("--periods-per-year", type=int, default=TRADING_PERIODS_DEFAULT, help="12 for monthly data, 252 for daily data")
    args = parser.parse_args()
    if args.periods_per_year <= 0:
        parser.error("--periods-per-year must be positive")

    prices = load_prices(args.csv)
    all_returns = prices.pct_change().dropna(how="any")
    cash_column = prices.columns[0]
    cash_returns = all_returns.iloc[:, 0].rename("cash_eur")
    returns = all_returns.iloc[:, 1:]
    asset_names = list(returns.columns)

    try:
        weights_series = ask_allocations(asset_names)
    except ValueError as error:
        parser.error(str(error))
    weights = weights_series.to_numpy()
    portfolio_returns = returns @ weights
    stats = calculate_stats(weights, returns, cash_returns, args.periods_per_year)
    equity_curves = pd.DataFrame(
        {
            "user_portfolio": (1.0 + portfolio_returns).cumprod(),
            "cash_eur": (1.0 + cash_returns).cumprod(),
        }
    )
    equity_curves.index.name = "date"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_series.rename("user_portfolio").to_csv(output_dir / "optimal_portfolio_weights.csv", header=True)
    pd.DataFrame([{"return": stats["annualized_return"], "volatility": stats["annualized_volatility"], "sharpe": stats["sharpe_ratio"]}]).to_csv(output_dir / "efficient_frontier_points.csv", index=False)
    returns_with_cash = returns.copy()
    returns_with_cash.insert(0, "cash_eur_risk_free_return", cash_returns)
    returns_with_cash.insert(1, "user_portfolio_return", portfolio_returns)
    returns_with_cash.to_csv(output_dir / "periodic_returns.csv")
    equity_curves.to_csv(output_dir / "portfolio_equity_curves.csv", date_format="%Y-%m-%d")

    plt.style.use("seaborn-v0_8-whitegrid")
    equity_fig, equity_ax = plt.subplots(figsize=(11, 7), dpi=160)
    equity_ax.plot(equity_curves.index, equity_curves["user_portfolio"], linewidth=2.2, label=f"User portfolio (MDD {stats['maximum_drawdown']:.2%})")
    cash_mdd = float((equity_curves["cash_eur"] / equity_curves["cash_eur"].cummax() - 1.0).min())
    equity_ax.plot(equity_curves.index, equity_curves["cash_eur"], linewidth=1.8, linestyle="--", label=f"Cash / cash_eur (MDD {cash_mdd:.2%})")
    equity_ax.set_title("User Portfolio Equity Curve")
    equity_ax.set_xlabel("Date")
    equity_ax.set_ylabel("Growth of 1.0")
    equity_ax.legend(loc="best")
    equity_fig.tight_layout()
    equity_fig.savefig(output_dir / "portfolio_equity_curves.png", bbox_inches="tight")
    plt.close(equity_fig)

    frontier_fig, frontier_ax = plt.subplots(figsize=(11, 7), dpi=160)
    frontier_ax.scatter(stats["annualized_volatility"], stats["annualized_return"], s=120, color="#1b9e77", label="User portfolio")
    risk_free_annual = float(cash_returns.mean() * args.periods_per_year)
    frontier_ax.scatter(0.0, risk_free_annual, s=80, color="#7570b3", label="Cash / risk-free rate")
    if stats["annualized_volatility"] > 0:
        cal_volatility = np.linspace(0.0, stats["annualized_volatility"] * 1.25, 100)
        cal_slope = (stats["annualized_return"] - risk_free_annual) / stats["annualized_volatility"]
        frontier_ax.plot(cal_volatility, risk_free_annual + cal_slope * cal_volatility, "--", color="#7570b3", label="CAL through user portfolio")
    frontier_ax.set_title("User Portfolio Risk/Return Summary")
    frontier_ax.set_xlabel("Annualized volatility")
    frontier_ax.set_ylabel("Annualized return")
    frontier_ax.xaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
    frontier_ax.yaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
    frontier_ax.legend(loc="best")
    frontier_fig.tight_layout()
    frontier_fig.savefig(output_dir / "efficient_frontier.png", bbox_inches="tight")
    plt.close(frontier_fig)

    summary = pd.DataFrame([{"portfolio": "user_portfolio", **stats}])
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    print(f"Loaded {len(prices):,} price observations for {len(asset_names)} risky assets: {', '.join(asset_names)}")
    print(f"Risk-free series: {cash_column} (annualized mean return: {risk_free_annual:.2%})")
    print("\nUser allocation:")
    print(weights_series.to_string(float_format=lambda value: f"{value:.2%}"))
    print("\nPortfolio summary:")
    print(summary.to_string(index=False, formatters={
        "annualized_return": lambda value: f"{value:.2%}",
        "annualized_volatility": lambda value: f"{value:.2%}",
        "historical_cagr": lambda value: f"{value:.2%}",
        "maximum_drawdown": lambda value: f"{value:.2%}",
        "sharpe_ratio": lambda value: f"{value:.3f}",
    }))
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
