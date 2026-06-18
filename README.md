# Cross-Asset-Market-Regime-Detection-and-Forecasting

## What This Project Does

This project builds an end-to-end data science pipeline that identifies 
hidden structural patterns in complex time series data using unsupervised 
machine learning, then tests whether those patterns improve predictive 
accuracy in a supervised forecasting model.

The application is financial markets, but the core methodology, finding 
hidden regime shifts in data without labelled examples, is directly 
transferable to any domain where an organisation faces structural changes 
in demand, behaviour, or resource needs.

---

## Research Question

Can a Gaussian Hidden Markov Model identify latent market regimes from 
cross-asset financial data without any labelled training data, and does 
conditioning an XGBoost return forecast on the current regime improve 
directional accuracy versus a baseline model?

---

## Data

Five daily time series from Yahoo Finance via yfinance, January 2010 
to present:

- SPY — US equity market (S&P 500 ETF)
- TLT — Long-duration US Treasury bonds
- GLD — Gold (safe haven proxy)
- VIX — CBOE Volatility Index (market fear gauge)
- HYG — High-yield corporate bonds (credit risk proxy)

---

## Pipeline

**Step 1 — Feature engineering**
Four features computed from raw price data: 20-day rolling volatility, 
20-day momentum, VIX level, and a credit risk proxy from high-yield 
bond returns.

**Step 2 — Unsupervised regime detection (Gaussian HMM)**
A Hidden Markov Model is fitted to the normalised feature matrix. 
No labels are provided. The model discovers that market days fall 
into two distinct structural states, which correspond closely to 
calm and stressed market environments when inspected against 
historical events. Validated with a two-sample t-test confirming 
the two regimes have statistically distinct return distributions.

**Step 3 — Supervised forecasting (XGBoost, walk-forward)**
An XGBoost classifier predicts next-week market return direction. 
Run twice: once with raw features only, once with the HMM regime 
label added. Walk-forward accuracy and F1 score compared across 
both to isolate the contribution of regime information.

---

## Key Finding

[Update after running tonight]

Adding the HMM regime label as a feature improved walk-forward 
directional accuracy from X% to Y% versus the baseline model.

---

## Why This Methodology Matters Beyond Finance

The most important insight from this project is not financial. It is 
methodological.

Hidden Markov Models find structure in data that human labellers 
cannot easily see. They identify when a system has shifted into a 
different behavioural state, without being told what those states 
are in advance.

This has direct applications for organisations facing structural 
shifts in demand or need:

- A food bank experiencing high-demand and low-demand regimes 
  driven by economic conditions and benefit payment cycles could 
  use an HMM on public economic indicators to anticipate regime 
  shifts and plan procurement and staffing accordingly.

- A social enterprise tracking engagement or usage data could 
  identify when their user base has structurally shifted behaviour, 
  enabling earlier and more targeted interventions.

- An NGO allocating resources across multiple programmes could 
  detect which programmes are entering a high-need phase before 
  that need becomes visible in aggregate statistics.

The same pipeline built here, feature engineering from available 
data, unsupervised regime identification, supervised decision 
support, applies directly to these problems.

---

## Tools

Python · hmmlearn · yfinance · XGBoost · scikit-learn · pandas · 
numpy · matplotlib · scipy · statsmodels

---

## Status

In progress · June 2026

---

## Next Steps

- Extend to a particle filter for real-time online regime updating
- Test on non-financial time series including social sector datasets
- Build a simple Streamlit dashboard for interactive regime 
  visualisation
