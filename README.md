# AutoSales Intelligence Dashboard

> Sales Volume Planning, Stock Balancing & AI-Powered Insights
> Portfolio Project · Ambika Sugganahalli Mahadevappa · 2026

---

## Project Overview

The **AutoSales Intelligence Dashboard** is a full-stack data intelligence tool for automotive sales volume planning. It simulates how planners balance **production capacity**, **stock levels**, **sales forecasts**, and **market KPIs** — and demonstrates skills in data visualisation, anomaly detection, and dashboard development directly applicable to quality monitoring roles in the automotive industry.

| Skill Demonstrated | Implementation |
| --- | --- |
| Interactive KPI dashboards | Live browser-based dashboard, zero install required |
| Anomaly detection & defect flagging | Rule-based AI engine auto-detects 33 planning deviations |
| Scenario planning & forecasting | 5-slider real-time recalculation with 3-band scenario chart |
| Testing (smoke / regression / E2E / UAT) | Full `test_suite.py` covering all four phases |
| Python data pipeline | `data_engine.py` — data generation, analysis, CSV/JSON export |
| Data visualisation | Chart.js (browser) + Matplotlib (Python reports) |

---

## Dashboard Features

### Overview Tab
- 4 live KPIs: units sold, revenue, plan accuracy, stock balance index
- Monthly Volume vs Plan chart (actual vs planned with deviation shading)
- Regional market share, top models by volume, powertrain mix (BEV/PHEV/ICE)
- Market filter: Global, Europe, Germany, USA, China

### Scenario Planning Tab
- 5 real-time sliders: demand uplift, production capacity, BEV mix, price, market weight
- Instant recalculation of volume, revenue, BEV share, plan accuracy, capacity utilisation
- Optimistic / Base / Pessimistic / Planned scenario chart

### Stock & Capacity Tab
- Days of Supply per model — colour-coded (red = critical, amber = warning, green = ok)
- Plant capacity utilisation across 5 facilities
- Live stock tracker: 8 models × key metrics with status indicators

### AI Insights Tab
- 6 auto-generated insight cards: anomaly, recommendation, forecast deviation, optimisation, risk alert, data integrity
- LSTM-style forecast confidence band chart
- Defect/issue tracker with severity, owner, and age

---

## How to Run

**Dashboard** (no install needed)
