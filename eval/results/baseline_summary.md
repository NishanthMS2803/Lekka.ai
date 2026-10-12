# Evaluation Benchmark Summary: Baseline Results (Run: `baseline_20261012_100648`)

- **Date**: 2026-10-12 10:20:59 UTC
- **Model**: `gemini-3.6-flash`
- **Dataset**: `facts_v0` (20 examples)
- **Evaluated Variants**: direct, single

## 1. Executive Summary Table

| Variant | Examples | Success Rate | Factual Accuracy | Avg Tool Calls | Avg Tokens | Total Tokens | Avg Latency (s) | Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **direct** | 20 | 100% | 75% | 0.0 | 589 | 11773 | 5.91s | $0.0050 |
| **single** | 20 | 100% | 5% | 5.5 | 59454 | 1189086 | 36.60s | $0.1842 |

## 2. Per-Question Results Breakdown

| ID | Ticker | Metric | Expected Ground Truth | Variant | Factual Match | Tokens | Tools | Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `fact_01` | **AAPL** | `total_revenue` | `391035000000.0` | `direct` | ✅ PASS | 459 | 0 | 6.92s |
| `fact_01` | **AAPL** | `total_revenue` | `391035000000.0` | `single` | ❌ FAIL | 22782 | 3 | 8.26s |
| `fact_02` | **AAPL** | `net_income` | `93736000000.0` | `direct` | ✅ PASS | 533 | 0 | 2.30s |
| `fact_02` | **AAPL** | `net_income` | `93736000000.0` | `single` | ❌ FAIL | 86759 | 6 | 13.22s |
| `fact_03` | **AAPL** | `diluted_eps` | `6.08` | `direct` | ✅ PASS | 435 | 0 | 2.20s |
| `fact_03` | **AAPL** | `diluted_eps` | `6.08` | `single` | ❌ FAIL | 85796 | 7 | 16.23s |
| `fact_04` | **MSFT** | `total_revenue` | `245122000000.0` | `direct` | ❌ FAIL | 1048 | 0 | 2.86s |
| `fact_04` | **MSFT** | `total_revenue` | `245122000000.0` | `single` | ❌ FAIL | 100882 | 6 | 10.42s |
| `fact_05` | **MSFT** | `net_income` | `88136000000.0` | `direct` | ❌ FAIL | 462 | 0 | 2.34s |
| `fact_05` | **MSFT** | `net_income` | `88136000000.0` | `single` | ❌ FAIL | 31430 | 3 | 6.34s |
| `fact_06` | **MSFT** | `diluted_eps` | `11.8` | `direct` | ❌ FAIL | 548 | 0 | 2.49s |
| `fact_06` | **MSFT** | `diluted_eps` | `11.8` | `single` | ❌ FAIL | 29984 | 3 | 35.78s |
| `fact_07` | **NVDA** | `total_revenue` | `60922000000.0` | `direct` | ✅ PASS | 555 | 0 | 2.77s |
| `fact_07` | **NVDA** | `total_revenue` | `60922000000.0` | `single` | ❌ FAIL | 52532 | 5 | 85.15s |
| `fact_08` | **NVDA** | `gross_margin_pct` | `72.7` | `direct` | ✅ PASS | 641 | 0 | 2.33s |
| `fact_08` | **NVDA** | `gross_margin_pct` | `72.7` | `single` | ❌ FAIL | 47475 | 5 | 72.01s |
| `fact_09` | **NVDA** | `net_income` | `29760000000.0` | `direct` | ✅ PASS | 648 | 0 | 2.79s |
| `fact_09` | **NVDA** | `net_income` | `29760000000.0` | `single` | ❌ FAIL | 111902 | 7 | 67.35s |
| `fact_10` | **GOOGL** | `total_revenue` | `307394000000.0` | `direct` | ✅ PASS | 461 | 0 | 2.19s |
| `fact_10` | **GOOGL** | `total_revenue` | `307394000000.0` | `single` | ❌ FAIL | 30888 | 4 | 18.87s |
| `fact_11` | **GOOGL** | `operating_income` | `84293000000.0` | `direct` | ✅ PASS | 407 | 0 | 2.30s |
| `fact_11` | **GOOGL** | `operating_income` | `84293000000.0` | `single` | ✅ PASS | 65210 | 5 | 23.64s |
| `fact_12` | **AMZN** | `net_sales` | `574785000000.0` | `direct` | ✅ PASS | 613 | 0 | 2.62s |
| `fact_12` | **AMZN** | `net_sales` | `574785000000.0` | `single` | ❌ FAIL | 90782 | 7 | 49.52s |
| `fact_13` | **AMZN** | `aws_net_sales` | `90757000000.0` | `direct` | ✅ PASS | 599 | 0 | 2.40s |
| `fact_13` | **AMZN** | `aws_net_sales` | `90757000000.0` | `single` | ❌ FAIL | 55887 | 7 | 68.91s |
| `fact_14` | **META** | `total_revenue` | `134902000000.0` | `direct` | ✅ PASS | 559 | 0 | 14.73s |
| `fact_14` | **META** | `total_revenue` | `134902000000.0` | `single` | ❌ FAIL | 72042 | 7 | 37.26s |
| `fact_15` | **META** | `net_income` | `39098000000.0` | `direct` | ✅ PASS | 1219 | 0 | 20.29s |
| `fact_15` | **META** | `net_income` | `39098000000.0` | `single` | ❌ FAIL | 71969 | 7 | 30.39s |
| `fact_16` | **TSLA** | `total_revenue` | `96773000000.0` | `direct` | ✅ PASS | 448 | 0 | 9.26s |
| `fact_16` | **TSLA** | `total_revenue` | `96773000000.0` | `single` | ❌ FAIL | 23278 | 3 | 22.01s |
| `fact_17` | **TSLA** | `regulatory_credits` | `1790000000.0` | `direct` | ✅ PASS | 774 | 0 | 30.04s |
| `fact_17` | **TSLA** | `regulatory_credits` | `1790000000.0` | `single` | ❌ FAIL | 60552 | 7 | 52.64s |
| `fact_18` | **JPM** | `total_net_revenue` | `162398000000.0` | `direct` | ❌ FAIL | 483 | 0 | 3.29s |
| `fact_18` | **JPM** | `total_net_revenue` | `162398000000.0` | `single` | ❌ FAIL | 48070 | 7 | 46.83s |
| `fact_19` | **WMT** | `total_revenue` | `648125000000.0` | `direct` | ❌ FAIL | 474 | 0 | 1.98s |
| `fact_19` | **WMT** | `total_revenue` | `648125000000.0` | `single` | ❌ FAIL | 32725 | 4 | 29.48s |
| `fact_20` | **LLY** | `total_revenue` | `34124100000.0` | `direct` | ✅ PASS | 407 | 0 | 2.10s |
| `fact_20` | **LLY** | `total_revenue` | `34124100000.0` | `single` | ❌ FAIL | 68141 | 7 | 37.64s |

## 3. Observations & Notes

- **V1 (Direct LLM Baseline)**: Direct prompts without tools rely solely on model internal weights and memory.
- **V2 (Single Agent Analyst Baseline)**: Uses autonomous tool calls against SEC filing search, XBRL facts, market data, and financial statements.
- **Status**: Baseline benchmark calibration complete. Zero hyperparameter tuning applied.

