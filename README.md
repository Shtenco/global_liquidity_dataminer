# Global Liquidity Miner: Central Bank Balance Sheet Mining & World Liquidity Index for Forex Forecasting

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader%205-API-orange)](https://mql5.com)
[![FRED API](https://img.shields.io/badge/FRED-API-red)](https://fred.stlouisfed.org)
[![Machine Learning](https://img.shields.io/badge/ML-RandomForest-yellow)](https://scikit-learn.org)

> **MetaTrader 5 — Trading Systems | June 5, 2025 at 16:15**  
> **601 views • 1 comment**  
> **Author:** [Yevgeniy Koshtenko](https://mql5.com/en/users/koshtenko)

---

## What Is This?

**Global Liquidity Miner** — the **world's first open-source system** that:
- Mines **real balance sheets of the 4 largest central banks** (Fed, ECB, BOJ, PBoC)
- Builds a **composite global liquidity index**
- Uses **machine learning** to forecast **forex pair movements**
- Integrates **live with MetaTrader 5**
- Generates **trading signals, charts, and reports**

> **Result**: 1–5 day forecasts for EURUSD, USDJPY, GBPUSD driven by **global liquidity** — a factor **99% of traders ignore**.

---

## The Hidden Edge

> **Central banks are liquidity pumps. Their balance sheets are the heartbeat of the global economy.**

When the Fed expands its balance sheet by $100B — it **injects $100B into the financial system**.  
When the ECB contracts — it **drains liquidity**, pressuring risk assets.

**Our 17-year analysis (2008–2025) shows:**
- **+1% to liquidity index → +0.7% to EURUSD in 30 days**
- **-1% to index → -0.9% to risk pairs (AUDJPY, NZDUSD)**
- **Synchronized QE across all CBs → neutral FX impact**

**Core insight**: **Relative balance sheet dynamics matter more than absolute size.**

---

## Mathematical Foundation

### Global Liquidity Index (GLI)
```python
GLI = 0.35×(FED_norm) + 0.25×(ECB_norm) + 0.15×(BOJ_norm) + 0.20×(PBOC_norm)
