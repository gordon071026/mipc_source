#!/usr/bin/env python3
"""Download monthly USD/EUR short-term rates for the EUR-hedge approximation.

Sources (FRED public CSV endpoints):
  * FEDFUNDS: monthly effective federal funds rate, USD.
  * EONIARATE: daily EONIA, EUR, used through December 2021.
  * ECBESTRVOLWGTTRMDMNRT: daily euro short-term rate (€STR), EUR,
    used from January 2022 onward.

Output columns are compatible with usd_index_to_eur_hedged.py:
    date,usd_rate_annual,eur_rate_annual

Rates are exported as annualized percentage rates in decimal form, e.g. 0.0525
means 5.25%. Daily EUR observations are averaged within each calendar month;
the monthly USD FEDFUNDS observations are used as published.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USD_SERIES = "FEDFUNDS"
EUR_EONIA_SERIES = "EONIARATE"
EUR_ESTR_SERIES = "ECBESTRVOLWGTTRMDMNRT"


def fred_csv(series_id: str, start: str, end: str) -> pd.DataFrame:
    params = urlencode({"id": series_id, "cosd": start, "coed": end})
    url = f"{FRED_CSV_URL}?{params}"
    with urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    data = pd.read_csv(StringIO(text))
    if data.shape[1] < 2:
        raise RuntimeError(f"FRED returned no usable columns for {series_id}")
    data = data.rename(columns={data.columns[0]: "date", data.columns[1]: series_id})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    data = data.dropna(subset=["date", series_id]).sort_values("date")
    if data.empty:
        raise RuntimeError(f"FRED returned no usable observations for {series_id}")
    return data


def to_monthly_percent(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Convert daily or monthly percentage rates to calendar-month averages."""
    out = data.copy()
    out["month"] = out["date"].dt.to_period("M")
    monthly = out.groupby("month", as_index=False)[value_col].mean()
    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    return monthly[["date", value_col]]


def build_rates(start: str, end: str) -> pd.DataFrame:
    usd = to_monthly_percent(fred_csv(USD_SERIES, start, end), USD_SERIES)
    eonia = to_monthly_percent(fred_csv(EUR_EONIA_SERIES, start, end), EUR_EONIA_SERIES)
    estr = to_monthly_percent(fred_csv(EUR_ESTR_SERIES, start, end), EUR_ESTR_SERIES)

    usd = usd.rename(columns={USD_SERIES: "usd_rate_annual_pct"})
    eonia = eonia.rename(columns={EUR_EONIA_SERIES: "eur_eonia_pct"})
    estr = estr.rename(columns={EUR_ESTR_SERIES: "eur_estr_pct"})

    months = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="ME")})
    out = months.merge(usd, on="date", how="left")
    out = out.merge(eonia, on="date", how="left")
    out = out.merge(estr, on="date", how="left")

    # EONIA was discontinued after early January 2022; use EONIA through
    # 2021-12 and €STR starting 2022-01. Do not backfill either series.
    out["eur_rate_annual_pct"] = out["eur_eonia_pct"]
    switch_date = pd.Timestamp("2022-01-31")
    out.loc[out["date"] >= switch_date, "eur_rate_annual_pct"] = out.loc[
        out["date"] >= switch_date, "eur_estr_pct"
    ]

    result = pd.DataFrame(
        {
            "date": out["date"],
            "usd_rate_annual": out["usd_rate_annual_pct"] / 100.0,
            "eur_rate_annual": out["eur_rate_annual_pct"] / 100.0,
            "usd_source_series": USD_SERIES,
            "eur_source_series": out["date"].map(
                lambda d: EUR_EONIA_SERIES if d < switch_date else EUR_ESTR_SERIES
            ),
        }
    )
    result["rates_available"] = result[["usd_rate_annual", "eur_rate_annual"]].notna().all(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2006-08-31", help="First month-end, YYYY-MM-DD")
    parser.add_argument("--end", default="2026-08-31", help="Last month-end, YYYY-MM-DD")
    parser.add_argument("--output", default="usd_eur_rates.csv")
    args = parser.parse_args()

    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error("--start must not be after --end")

    result = build_rates(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, date_format="%Y-%m-%d")

    missing = int((~result["rates_available"]).sum())
    print(f"Wrote {len(result):,} month-end rows to {output.resolve()}")
    print(f"Rows with both USD and EUR rates: {len(result) - missing:,}")
    print(f"Rows with a missing rate: {missing:,}")
    print("USD source: FRED FEDFUNDS; EUR source: FRED EONIARATE through 2021-12, then ECBESTRVOLWGTTRMDMNRT (€STR)")


if __name__ == "__main__":
    main()
