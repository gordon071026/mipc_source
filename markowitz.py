#!/usr/bin/env python3
"""Plot an efficient frontier and calculate optimal portfolios from price data.

Input CSV format:
    date,asset_1,asset_2,...
    2021-08-31,121.1174,207.2813,...

The script uses percentage changes between observations, annualizes the
estimated mean returns and covariance matrix, and solves long-only portfolio
optimization problems by default. The second CSV column (the first price
column after the date column, expected to be ``cash_eur``) is treated as the
risk-free cash series and is excluded from the risky-asset frontier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_PERIODS_DEFAULT = 12  # Monthly observations in the supplied template.


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean a date-indexed price CSV.

    The first column is treated as the date column unless a column named
    'date' exists. All other columns must contain numeric prices.
    """
    # Excel exports may use commas or tabs depending on regional settings.
    # Let pandas infer the separator so both formats are accepted.
    raw = pd.read_csv(csv_path, sep=None, engine="python")
    if raw.empty or raw.shape[1] < 3:
        raise ValueError("CSV must contain a date column and at least two asset columns.")

    date_candidates = [c for c in raw.columns if str(c).strip().lower() in {"date", "dates", "time"}]
    date_col = date_candidates[0] if date_candidates else raw.columns[0]
    dates = pd.to_datetime(raw[date_col], format="mixed", errors="coerce")
    if dates.isna().all():
        raise ValueError(f"Could not parse '{date_col}' as dates.")

    prices = raw.drop(columns=[date_col]).copy()
    # Handle numbers exported with thousands separators and/or whitespace.
    for col in prices.columns:
        if prices[col].dtype == "object":
            prices[col] = prices[col].astype(str).str.replace(",", "", regex=False).str.strip()
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    # Attach dates before removing empty rows so the date index stays aligned
    # with the price rows, including files containing blank lines.
    prices.index = dates
    prices = prices.loc[:, prices.notna().any(axis=0)]
    if prices.shape[1] < 2:
        raise ValueError("At least two numeric asset price columns are required.")
    if prices.isna().all(axis=1).any():
        prices = prices.dropna(how="all")

    prices = prices[~prices.index.isna()].sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(axis=1, how="all").ffill().dropna()

    if len(prices) < 3:
        raise ValueError("At least three dated price observations are required.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive; check the CSV columns and formatting.")
    return prices


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple[float, float]:
    """Return annualized expected return and volatility."""
    expected_return = float(weights @ mean_returns)
    volatility = float(np.sqrt(weights @ cov_matrix @ weights))
    return expected_return, volatility


def ask_asset_bounds(asset_names: list[str], allow_short: bool, max_absolute_weight: float) -> list[tuple[float, float]]:
    """Prompt for individual lower and upper portfolio-weight bounds."""
    default_lower = -max_absolute_weight if allow_short else 0.0
    default_upper = max_absolute_weight
    print("\nEnter lower and upper portfolio-weight bounds for each risky asset.")
    print("Use percentages; press Enter to accept the displayed default.\n")

    bounds: list[tuple[float, float]] = []
    for asset in asset_names:
        while True:
            lower_text = input(f"Lower bound for {asset} (%) [{default_lower * 100:.1f}]: ").strip()
            upper_text = input(f"Upper bound for {asset} (%) [{default_upper * 100:.1f}]: ").strip()
            try:
                lower = default_lower if lower_text == "" else float(lower_text.replace("%", ""))
                upper = default_upper if upper_text == "" else float(upper_text.replace("%", ""))
            except ValueError:
                print("Please enter numeric percentages, or press Enter for the defaults.")
                continue
            lower /= 100.0
            upper /= 100.0
            if lower > upper:
                print("The lower bound cannot exceed the upper bound.")
                continue
            if not allow_short and lower < 0:
                print("Negative lower bounds require the --allow-short option.")
                continue
            if lower < -max_absolute_weight or upper > max_absolute_weight:
                print(f"Bounds must be between {-max_absolute_weight:.2%} and {max_absolute_weight:.2%}.")
                continue
            bounds.append((lower, upper))
            break
    return bounds


def find_feasible_initial(bounds: list[tuple[float, float]]) -> np.ndarray:
    """Find fully invested starting weights satisfying individual bounds."""
    n = len(bounds)
    equal = np.repeat(1.0 / n, n)
    result = minimize(
        lambda w: float(np.sum((w - equal) ** 2)),
        np.clip(equal, [b[0] for b in bounds], [b[1] for b in bounds]),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError("The asset bounds cannot produce a fully invested portfolio totaling 100%.")
    return result.x


def solve_min_variance(
    target_return: float,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    initial_weights: np.ndarray,
) -> np.ndarray:
    """Find the minimum-volatility portfolio at a target return."""
    n = len(mean_returns)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_matrix @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mean_returns - target_return},
    ]
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Frontier optimization failed at return {target_return:.6f}: {result.message}")
    return result.x


def solve_global_min_variance(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    initial_weights: np.ndarray,
) -> np.ndarray:
    n = len(mean_returns)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_matrix @ w)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return result.x


def solve_max_sharpe(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    bounds: list[tuple[float | None, float | None]],
    initial_weights: np.ndarray,
) -> np.ndarray:
    n = len(mean_returns)

    def negative_sharpe(w: np.ndarray) -> float:
        ret, vol = portfolio_stats(w, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol if vol > 0 else 1e9

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {result.message}")
    return result.x


def calculate_time_varying_sharpe(
    portfolio_returns: pd.Series,
    risk_free_returns: pd.Series,
    periods_per_year: int,
) -> float:
    """Annualize Sharpe using period-by-period excess returns over cash_eur."""
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), risk_free_returns.rename("cash_eur")],
        axis=1,
    ).dropna()
    excess = aligned["portfolio"] - aligned["cash_eur"]
    volatility = excess.std(ddof=1)
    if volatility <= 0:
        return float("nan")
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))


def calculate_cagr(portfolio_returns: pd.Series, periods_per_year: int) -> float:
    """Calculate compounded annual growth from periodic portfolio returns."""
    returns = portfolio_returns.dropna()
    if returns.empty or (1.0 + returns <= 0).any():
        return float("nan")
    return float((1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0)


def build_equity_curves(
    minvol_returns: pd.Series,
    max_sharpe_returns: pd.Series,
    risk_free_returns: pd.Series,
) -> pd.DataFrame:
    """Build normalized cumulative wealth indices from periodic returns."""
    curves = pd.DataFrame(
        {
            "minimum_volatility": minvol_returns,
            "maximum_sharpe": max_sharpe_returns,
            "cash_eur": risk_free_returns,
        }
    ).dropna()
    equity = (1.0 + curves).cumprod()
    equity.index.name = "date"
    return equity


def build_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    points: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    initial = find_feasible_initial(bounds)
    gmv = solve_global_min_variance(mean_returns, cov_matrix, bounds, initial)
    gmv_return, _ = portfolio_stats(gmv, mean_returns, cov_matrix)

    # The feasible target-return range is bounded by individual asset returns
    # for long-only portfolios. With shorting enabled, use a wider practical
    # range to avoid an unbounded frontier.
    if all(bound[0] >= 0 for bound in bounds):
        low, high = float(np.min(mean_returns)), float(np.max(mean_returns))
    else:
        low = min(float(np.min(mean_returns)), gmv_return) - 0.25
        high = max(float(np.max(mean_returns)), gmv_return) + 0.25
    target_returns = np.linspace(low, high, points)

    rows: list[dict[str, float]] = []
    valid_weights: list[np.ndarray] = []
    for target in target_returns:
        try:
            w = solve_min_variance(target, mean_returns, cov_matrix, bounds, gmv)
            ret, vol = portfolio_stats(w, mean_returns, cov_matrix)
            rows.append({"return": ret, "volatility": vol})
            valid_weights.append(w)
        except RuntimeError:
            # Some targets can be infeasible because of solver tolerances.
            continue
    if not rows:
        raise RuntimeError("No feasible frontier points were found.")
    return pd.DataFrame(rows), gmv, np.vstack(valid_weights)


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV containing a date column followed by price columns")
    parser.add_argument("--output-dir", default="frontier_output", help="Directory for PNG and CSV outputs")
    parser.add_argument("--periods-per-year", type=int, default=TRADING_PERIODS_DEFAULT, help="12 for monthly data, 252 for daily data")
    parser.add_argument("--frontier-points", type=int, default=100, help="Number of target-return points")
    parser.add_argument("--allow-short", action="store_true", help="Allow short positions; default is long-only")
    parser.add_argument("--max-absolute-weight", type=float, default=1.0, help="Maximum absolute bound allowed in the interactive prompts")
    args = parser.parse_args()

    if args.periods_per_year <= 0 or args.frontier_points < 5:
        parser.error("--periods-per-year must be positive and --frontier-points must be at least 5")
    if args.max_absolute_weight <= 0:
        parser.error("--max-absolute-weight must be positive")
    prices = load_prices(args.csv)
    if prices.shape[1] < 3:
        raise ValueError("CSV must contain cash_eur plus at least two risky-asset price columns.")
    cash_column = prices.columns[0]
    if str(cash_column).strip().lower() != "cash_eur":
        print(f"Warning: treating the first price column '{cash_column}' as the cash_eur risk-free series.")

    all_returns = prices.pct_change().dropna(how="any")
    risk_free_returns = all_returns.iloc[:, 0].rename("cash_eur")
    returns = all_returns.iloc[:, 1:]
    if len(returns) < 2:
        raise ValueError("Not enough return observations after calculating percentage changes.")

    mean_returns = returns.mean().to_numpy() * args.periods_per_year
    cov_matrix = returns.cov().to_numpy() * args.periods_per_year
    risk_free_annual = float(risk_free_returns.mean() * args.periods_per_year)
    asset_names = list(returns.columns)
    n = len(asset_names)
    bounds = ask_asset_bounds(asset_names, args.allow_short, args.max_absolute_weight)
    initial = find_feasible_initial(bounds)

    frontier, gmv_weights, _ = build_frontier(mean_returns, cov_matrix, bounds, args.frontier_points)
    minvol_return, minvol_vol = portfolio_stats(gmv_weights, mean_returns, cov_matrix)
    max_sharpe_weights = solve_max_sharpe(mean_returns, cov_matrix, risk_free_annual, bounds, initial)
    max_sharpe_return, max_sharpe_vol = portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix)
    minvol_returns = returns @ gmv_weights
    max_sharpe_returns = returns @ max_sharpe_weights
    equity_curves = build_equity_curves(minvol_returns, max_sharpe_returns, risk_free_returns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = pd.DataFrame(
        {"minimum_volatility": gmv_weights, "maximum_sharpe": max_sharpe_weights},
        index=asset_names,
    )
    weights.index.name = "asset"
    weights.to_csv(output_dir / "optimal_portfolio_weights.csv")

    frontier.to_csv(output_dir / "efficient_frontier_points.csv", index=False)
    returns_with_cash = returns.copy()
    returns_with_cash.insert(0, "cash_eur_risk_free_return", risk_free_returns)
    returns_with_cash.to_csv(output_dir / "periodic_returns.csv")
    equity_curves.to_csv(output_dir / "portfolio_equity_curves.csv", date_format="%Y-%m-%d")

    plt.style.use("seaborn-v0_8-whitegrid")
    equity_fig, equity_ax = plt.subplots(figsize=(11, 7), dpi=160)
    equity_ax.plot(equity_curves.index, equity_curves["minimum_volatility"], linewidth=2.0, label="Minimum volatility")
    equity_ax.plot(equity_curves.index, equity_curves["maximum_sharpe"], linewidth=2.0, label="Maximum Sharpe")
    equity_ax.plot(equity_curves.index, equity_curves["cash_eur"], linewidth=1.8, linestyle="--", label="Cash / cash_eur")
    equity_ax.set_title("Optimized Portfolio Equity Curves")
    equity_ax.set_xlabel("Date")
    equity_ax.set_ylabel("Growth of 1.0")
    equity_ax.legend(loc="best")
    equity_fig.tight_layout()
    equity_fig.savefig(output_dir / "portfolio_equity_curves.png", bbox_inches="tight")
    plt.close(equity_fig)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    ax.plot(frontier["volatility"], frontier["return"], color="#1769aa", linewidth=2.2, label="Efficient frontier")
    ax.scatter(minvol_vol, minvol_return, s=90, color="#d95f02", zorder=4, label="Minimum-volatility portfolio")
    ax.scatter(max_sharpe_vol, max_sharpe_return, s=90, color="#1b9e77", zorder=4, label="Maximum-Sharpe portfolio")
    max_plot_vol = max(float(frontier["volatility"].max()), max_sharpe_vol) * 1.10
    cal_volatility = np.linspace(0.0, max_plot_vol, 100)
    cal_slope = (max_sharpe_return - risk_free_annual) / max_sharpe_vol if max_sharpe_vol > 0 else 0.0
    cal_returns = risk_free_annual + cal_slope * cal_volatility
    ax.plot(
        cal_volatility,
        cal_returns,
        color="#7570b3",
        linestyle="--",
        linewidth=1.8,
        label="Capital Allocation Line (CAL)",
    )
    ax.scatter(0.0, risk_free_annual, s=60, color="#7570b3", zorder=4, label="Cash / risk-free rate")
    ax.annotate("Minimum volatility", (minvol_vol, minvol_return), xytext=(8, -18), textcoords="offset points")
    ax.annotate("Maximum Sharpe", (max_sharpe_vol, max_sharpe_return), xytext=(8, 8), textcoords="offset points")
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Annualized volatility (standard deviation)")
    ax.set_ylabel("Annualized expected return")
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_formatter(lambda x, pos: format_pct(x))
    ax.yaxis.set_major_formatter(lambda x, pos: format_pct(x))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "efficient_frontier.png", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "portfolio": ["minimum_volatility", "maximum_sharpe"],
            "annualized_return": [minvol_return, max_sharpe_return],
            "annualized_volatility": [minvol_vol, max_sharpe_vol],
            "historical_cagr": [
                calculate_cagr(minvol_returns, args.periods_per_year),
                calculate_cagr(max_sharpe_returns, args.periods_per_year),
            ],
            "sharpe_ratio": [
                calculate_time_varying_sharpe(
                    minvol_returns,
                    risk_free_returns,
                    args.periods_per_year,
                ),
                calculate_time_varying_sharpe(
                    max_sharpe_returns,
                    risk_free_returns,
                    args.periods_per_year,
                ),
            ],
        }
    )
    summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    print(f"Loaded {len(prices):,} price observations for {n} risky assets: {', '.join(asset_names)}")
    print(f"Risk-free series: {cash_column} (annualized mean period return: {format_pct(risk_free_annual)})")
    print("Individual risky-asset bounds:")
    for name, (lower, upper) in zip(asset_names, bounds):
        print(f"  {name}: {lower:.2%} to {upper:.2%}")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(f"Equity curves: {(output_dir / 'portfolio_equity_curves.png').resolve()}")
    print("\nOptimal portfolios:")
    print(summary.to_string(index=False, formatters={
        "annualized_return": format_pct,
        "annualized_volatility": format_pct,
        "historical_cagr": format_pct,
        "sharpe_ratio": lambda x: f"{x:.3f}",
    }))
    print("\nWeights:")
    print(weights.to_string(float_format=lambda x: f"{x:.2%}"))


if __name__ == "__main__":
    main()


__all__: Iterable[str] = ["load_prices", "portfolio_stats", "build_frontier", "main"]
