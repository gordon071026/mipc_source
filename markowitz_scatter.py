#!/usr/bin/env python3
"""Plot a Monte Carlo portfolio risk/return scatter plot.

The first price column after the date column is treated as cash_eur, the
risk-free series. Every remaining column is a risky asset. The script prompts
for asset bounds, samples feasible fully invested portfolios, colors points by
Sharpe ratio, and marks the minimum-volatility and maximum-Sharpe portfolios.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.optimize import minimize


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


def ask_bounds(asset_names: list[str], allow_short: bool, max_absolute_weight: float) -> list[tuple[float, float]]:
    default_lower = -max_absolute_weight if allow_short else 0.0
    default_upper = max_absolute_weight
    bounds: list[tuple[float, float]] = []
    print("\nEnter lower and upper allocation bounds for each risky asset.")
    print("Use percentages; press Enter to accept the displayed default.\n")
    for asset in asset_names:
        while True:
            lo_text = input(f"Lower bound for {asset} (%) [{default_lower * 100:.1f}]: ").strip()
            hi_text = input(f"Upper bound for {asset} (%) [{default_upper * 100:.1f}]: ").strip()
            try:
                lo = default_lower if lo_text == "" else float(lo_text.replace("%", ""))
                hi = default_upper if hi_text == "" else float(hi_text.replace("%", ""))
            except ValueError:
                print("Please enter numeric percentages or press Enter for the defaults.")
                continue
            lo /= 100.0
            hi /= 100.0
            if lo > hi or (not allow_short and lo < 0) or lo < -max_absolute_weight or hi > max_absolute_weight:
                print("Invalid bounds for the selected short-selling settings.")
                continue
            bounds.append((lo, hi))
            break
    if sum(lo for lo, _ in bounds) > 1.0 or sum(hi for _, hi in bounds) < 1.0:
        raise ValueError("The bounds cannot produce a fully invested portfolio totaling 100%.")
    return bounds


def feasible_weights(bounds: list[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    n = len(bounds)
    target = rng.dirichlet(np.ones(n))
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    result = minimize(
        lambda w: float(np.sum((w - target) ** 2)),
        np.clip(np.repeat(1.0 / n, n), lower, upper),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError("Could not generate a feasible portfolio under the supplied bounds.")
    return result.x


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, covariance: np.ndarray, risk_free: float) -> tuple[float, float, float]:
    annual_return = float(weights @ mean_returns)
    volatility = float(np.sqrt(weights @ covariance @ weights))
    sharpe = (annual_return - risk_free) / volatility if volatility > 0 else np.nan
    return annual_return, volatility, sharpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV containing date and price columns")
    parser.add_argument("--output-dir", default="scatter_output")
    parser.add_argument("--periods-per-year", type=int, default=12)
    parser.add_argument("--portfolios", type=int, default=5000, help="Number of random feasible portfolios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--max-absolute-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.periods_per_year <= 0 or args.portfolios < 100:
        parser.error("periods-per-year must be positive and portfolios must be at least 100")
    if args.max_absolute_weight <= 0:
        parser.error("max-absolute-weight must be positive")

    prices = load_prices(args.csv)
    all_returns = prices.pct_change().dropna(how="any")
    cash_column = prices.columns[0]
    risk_free_returns = all_returns.iloc[:, 0]
    returns = all_returns.iloc[:, 1:]
    asset_names = list(returns.columns)
    bounds = ask_bounds(asset_names, args.allow_short, args.max_absolute_weight)

    mean_returns = returns.mean().to_numpy() * args.periods_per_year
    covariance = returns.cov().to_numpy() * args.periods_per_year
    risk_free_annual = float(risk_free_returns.mean() * args.periods_per_year)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float]] = []
    for _ in range(args.portfolios):
        weights = feasible_weights(bounds, rng)
        annual_return, volatility, sharpe = portfolio_stats(weights, mean_returns, covariance, risk_free_annual)
        rows.append({"return": annual_return, "volatility": volatility, "sharpe": sharpe, **dict(zip(asset_names, weights))})
    portfolios = pd.DataFrame(rows)

    minvol_row = portfolios.loc[portfolios["volatility"].idxmin()]
    maxsharpe_row = portfolios.loc[portfolios["sharpe"].idxmax()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolios.to_csv(output_dir / "random_portfolios.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    # Reserve balanced space for the axes and colorbar so the portfolio cloud
    # is visually centered rather than being pushed to one side.
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160, constrained_layout=True)
    scatter = ax.scatter(
        portfolios["volatility"],
        portfolios["return"],
        c=portfolios["sharpe"],
        cmap="Blues",
        s=10,
        alpha=0.45,
        edgecolors="none",
        norm=Normalize(vmin=portfolios["sharpe"].min(), vmax=portfolios["sharpe"].max()),
    )
    ax.scatter(minvol_row["volatility"], minvol_row["return"], color="#d95f02", s=100, label="Minimum volatility", zorder=5)
    ax.scatter(maxsharpe_row["volatility"], maxsharpe_row["return"], color="#e31a1c", s=100, label="Maximum sampled Sharpe", zorder=5)
    ax.scatter(0.0, risk_free_annual, color="#7570b3", s=80, label="Cash / risk-free rate", zorder=5)
    ax.annotate("Minimum volatility", (minvol_row["volatility"], minvol_row["return"]), xytext=(8, -16), textcoords="offset points")
    ax.annotate("Maximum Sharpe", (maxsharpe_row["volatility"], maxsharpe_row["return"]), xytext=(8, 8), textcoords="offset points")
    ax.set_title("Portfolio Optimization & Efficient Frontier")
    ax.set_xlabel("Annualized risk")
    ax.set_ylabel("Annualized return")
    # Set limits from all plotted markers with symmetric proportional padding.
    # This avoids excessive whitespace and keeps the displayed results centered.
    plotted_x = np.array([
        portfolios["volatility"].min(),
        portfolios["volatility"].max(),
        minvol_row["volatility"],
        maxsharpe_row["volatility"],
        0.0,
    ])
    plotted_y = np.array([
        portfolios["return"].min(),
        portfolios["return"].max(),
        minvol_row["return"],
        maxsharpe_row["return"],
        risk_free_annual,
    ])
    x_span = max(float(plotted_x.max() - plotted_x.min()), 1e-6)
    y_span = max(float(plotted_y.max() - plotted_y.min()), 1e-6)
    x_pad = 0.10 * x_span
    y_pad = 0.12 * y_span
    ax.set_xlim(float(plotted_x.min() - x_pad), float(plotted_x.max() + x_pad))
    ax.set_ylim(float(plotted_y.min() - y_pad), float(plotted_y.max() + y_pad))
    ax.xaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
    ax.legend(loc="upper left", frameon=True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Sharpe ratio")
    figure_path = output_dir / "portfolio_scatter.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(prices):,} observations for {len(asset_names)} risky assets.")
    print(f"Risk-free series: {cash_column}; annualized mean return: {risk_free_annual:.2%}")
    print(f"Generated {len(portfolios):,} feasible portfolios using seed {args.seed}.")
    print(f"Minimum sampled volatility: {minvol_row['volatility']:.2%} return {minvol_row['return']:.2%}")
    print(f"Maximum sampled Sharpe: {maxsharpe_row['sharpe']:.3f} return {maxsharpe_row['return']:.2%} risk {maxsharpe_row['volatility']:.2%}")
    print(f"Scatter plot: {figure_path.resolve()}")
    print(f"Portfolio sample: {(output_dir / 'random_portfolios.csv').resolve()}")


if __name__ == "__main__":
    main()
