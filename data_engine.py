"""
AutoSales Intelligence Dashboard — Data Generator & Analysis Engine
====================================================================
Author  : Ambika Sugganahalli Mahadevappa
Project : BMW IT Intern Portfolio Project — Sales Volume Planning & AI Support
Tools   : Python 3.x · pandas · numpy · matplotlib
Purpose : Generates simulated automotive sales data, performs volume planning
          analysis, detects anomalies, and exports KPI reports.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

MODELS   = ['3 Series', '5 Series', 'X5', 'iX', 'i4', '7 Series', 'X3', 'M3', 'Z4', 'iX1']
REGIONS  = ['Europe', 'Germany', 'USA', 'China', 'RoW']
PLANTS   = ['Munich', 'Leipzig', 'Dingolfing', 'Spartanburg', 'Shenyang']
MONTHS   = pd.date_range('2025-01-01', periods=12, freq='MS')

COLORS   = ['#3b82f6','#06b6d4','#f59e0b','#10b981','#ef4444','#8b5cf6','#f97316','#ec4899','#14b8a6','#84cc16']

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  AutoSales Intelligence Dashboard — Data Engine")
print("  BMW IT Intern Portfolio Project")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
#  1. GENERATE SALES VOLUME DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Generating sales volume dataset...")

def seasonality_factor(month_idx):
    """Automotive sales follow a strong seasonal pattern."""
    factors = [0.88, 0.82, 1.05, 1.08, 1.12, 1.04,
               0.95, 1.00, 1.10, 1.08, 0.92, 1.08]
    return factors[month_idx]

def generate_volume(base, noise_pct=0.04, trend_pct=0.005):
    """Generate monthly volume with seasonality, trend, and noise."""
    volumes = []
    for i, month in enumerate(MONTHS):
        seasonal = seasonality_factor(month.month - 1)
        trend    = 1 + trend_pct * i
        noise    = 1 + np.random.normal(0, noise_pct)
        volumes.append(int(base * seasonal * trend * noise))
    return volumes

# Generate actual & planned volumes per model
records = []
model_bases = {
    '3 Series':35000, '5 Series':26500, 'X5':24600, 'iX':15600,
    'i4':20400, '7 Series':12100, 'X3':17800, 'M3':8200, 'Z4':3500, 'iX1':13700
}

for model in MODELS:
    base = model_bases[model]
    actual_vol  = generate_volume(base, noise_pct=0.045)
    planned_vol = generate_volume(base * 0.98, noise_pct=0.015)
    for i, month in enumerate(MONTHS):
        records.append({
            'Month':     month,
            'Model':     model,
            'Actual':    actual_vol[i],
            'Planned':   planned_vol[i],
            'Variance':  round((actual_vol[i] - planned_vol[i]) / planned_vol[i] * 100, 2),
        })

df_volume = pd.DataFrame(records)

# Regional split
region_weights = {'Europe':0.38, 'Germany':0.13, 'USA':0.20, 'China':0.25, 'RoW':0.04}
region_records = []
for _, row in df_volume.iterrows():
    for region, w in region_weights.items():
        region_records.append({
            'Month':   row['Month'],
            'Model':   row['Model'],
            'Region':  region,
            'Actual':  int(row['Actual'] * w * (1 + np.random.normal(0, 0.03))),
            'Planned': int(row['Planned'] * w),
        })

df_regional = pd.DataFrame(region_records)

print(f"   ✓ Generated {len(df_volume):,} volume records ({len(df_regional):,} regional records)")

# ─────────────────────────────────────────────────────────────
#  2. STOCK & CAPACITY DATA
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Building stock & capacity tables...")

daily_sales = {m: model_bases[m] / 30 for m in MODELS}

stock_data = []
for model in MODELS:
    for region in REGIONS:
        stock_units = int(daily_sales[model] * region_weights[region] *
                          np.random.uniform(25, 90))
        days_supply = int(stock_units / (daily_sales[model] * region_weights[region]))
        cap_util    = round(np.random.uniform(62, 99), 1)
        plan_gap    = round(np.random.normal(0, 7), 1)
        status = 'critical' if days_supply < 35 or days_supply > 80 or cap_util > 97 else \
                 'warning'  if days_supply < 45 or days_supply > 70 or cap_util > 90 else \
                 'ok'
        stock_data.append({
            'Model':         model,
            'Region':        region,
            'Stock_Units':   stock_units,
            'Days_Supply':   days_supply,
            'Target_Min_DOS':45,
            'Target_Max_DOS':60,
            'Capacity_Util': cap_util,
            'Plan_Gap_Pct':  plan_gap,
            'Status':        status,
        })

df_stock = pd.DataFrame(stock_data)

capacity_data = []
for plant in PLANTS:
    planned_u = np.random.randint(18000, 45000)
    capacity_data.append({
        'Plant':        plant,
        'Capacity_Util': round(np.random.uniform(72, 98), 1),
        'Planned_Units': planned_u,
        'Actual_Units':  int(planned_u * np.random.uniform(0.88, 1.04)),
        'Downtime_Days': np.random.randint(0, 6),
    })

df_capacity = pd.DataFrame(capacity_data)

critical_count = len(df_stock[df_stock['Status'] == 'critical'])
warning_count  = len(df_stock[df_stock['Status'] == 'warning'])
print(f"   ✓ Stock records: {len(df_stock)}  |  Critical: {critical_count}  |  Warning: {warning_count}")

# ─────────────────────────────────────────────────────────────
#  3. ANOMALY DETECTION (Rule-based AI Engine)
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Running anomaly detection engine...")

anomalies = []

# Rule 1: Large variance between plan and actual
for _, row in df_volume.iterrows():
    if abs(row['Variance']) > 10:
        anomalies.append({
            'Type':        'Plan Deviation',
            'Severity':    'Critical' if abs(row['Variance']) > 15 else 'Warning',
            'Model':       row['Model'],
            'Month':       row['Month'].strftime('%b %Y'),
            'Description': f"Variance of {row['Variance']:+.1f}% detected (threshold: ±10%)",
            'Recommended': 'Recalibrate Q+1 plan' if row['Variance'] < 0 else 'Review overstock risk',
        })

# Rule 2: Stock imbalance
for _, row in df_stock.iterrows():
    if row['Days_Supply'] < 35:
        anomalies.append({
            'Type': 'Stock Alert', 'Severity': 'Critical',
            'Model': row['Model'], 'Month': 'Current',
            'Description': f"{row['Region']}: Only {row['Days_Supply']}d supply (min target: 45d)",
            'Recommended': 'Emergency replenishment order',
        })
    elif row['Days_Supply'] > 75:
        anomalies.append({
            'Type': 'Overstock Risk', 'Severity': 'Warning',
            'Model': row['Model'], 'Month': 'Current',
            'Description': f"{row['Region']}: {row['Days_Supply']}d supply (max target: 60d)",
            'Recommended': 'Pause production order / push retail incentives',
        })

# Rule 3: Capacity overload
for _, row in df_capacity.iterrows():
    if row['Capacity_Util'] > 95:
        anomalies.append({
            'Type': 'Capacity Overload', 'Severity': 'Critical',
            'Model': 'All Models', 'Month': 'Current',
            'Description': f"{row['Plant']} at {row['Capacity_Util']}% utilisation",
            'Recommended': 'Shift production to alternate plant or delay delivery',
        })

df_anomalies = pd.DataFrame(anomalies)
print(f"   ✓ Detected {len(df_anomalies)} anomalies  |  "
      f"Critical: {len(df_anomalies[df_anomalies['Severity']=='Critical'])}  |  "
      f"Warning: {len(df_anomalies[df_anomalies['Severity']=='Warning'])}")

# ─────────────────────────────────────────────────────────────
#  4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Generating visualisations...")

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0a0c10',
    'axes.facecolor':   '#10141c',
    'axes.edgecolor':   '#1e2535',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'grid.color':       '#1e2535',
    'grid.alpha':       0.8,
    'text.color':       '#e2e8f0',
    'font.family':      'monospace',
    'font.size':        9,
})

# ── Chart A: Monthly Volume vs Plan (top 4 models) ──────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Sales Volume vs Plan — Top 4 Models (2025)', fontsize=14, fontweight='bold', y=0.98)

top4 = ['3 Series', 'X5', 'iX', 'i4']
month_labels = [m.strftime('%b') for m in MONTHS]

for ax, model in zip(axes.flat, top4):
    sub = df_volume[df_volume['Model'] == model].sort_values('Month')
    ax.bar(month_labels, sub['Actual'], color='#3b82f6', alpha=0.75, label='Actual', zorder=3)
    ax.plot(month_labels, sub['Planned'], color='#f59e0b', linewidth=2, marker='o', markersize=4, label='Planned', zorder=4)
    ax.fill_between(month_labels, sub['Actual'], sub['Planned'],
                    where=sub['Actual'] >= sub['Planned'], alpha=0.15, color='#10b981', label='Above plan')
    ax.fill_between(month_labels, sub['Actual'], sub['Planned'],
                    where=sub['Actual'] < sub['Planned'], alpha=0.15, color='#ef4444', label='Below plan')
    ax.set_title(f'{model}', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_ylabel('Units', fontsize=8)
    ax.grid(True, axis='y', linewidth=0.5)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
    if model == top4[0]:
        ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart_A_volume_vs_plan.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ chart_A_volume_vs_plan.png")

# ── Chart B: KPI Dashboard ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Key Performance Indicators — Full Year 2025', fontsize=13, fontweight='bold')

# KPI 1: Plan accuracy by model
monthly_acc = df_volume.groupby('Model').apply(
    lambda g: 100 - g['Variance'].abs().mean()
).sort_values()
colors_bar = ['#ef4444' if v < 92 else '#f59e0b' if v < 95 else '#10b981' for v in monthly_acc]
axes[0].barh(monthly_acc.index, monthly_acc.values, color=colors_bar, alpha=0.85)
axes[0].axvline(95, color='#3b82f6', linestyle='--', linewidth=1.5, label='Target (95%)')
axes[0].set_xlabel('Plan Accuracy %')
axes[0].set_title('Plan Accuracy by Model', fontweight='bold')
axes[0].set_xlim(80, 101)
axes[0].legend(fontsize=8)
for i, (val, idx) in enumerate(zip(monthly_acc.values, monthly_acc.index)):
    axes[0].text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=8, color='#94a3b8')

# KPI 2: Total annual volume by region
reg_vol = df_regional.groupby('Region')['Actual'].sum().sort_values()
wedge_colors = COLORS[:len(REGIONS)]
axes[1].pie(reg_vol.values, labels=reg_vol.index, colors=wedge_colors,
            autopct='%1.1f%%', pctdistance=0.75, startangle=140,
            wedgeprops={'edgecolor':'#0a0c10', 'linewidth':2})
axes[1].set_title('Volume by Region', fontweight='bold')

# KPI 3: Powertrain mix trend
bev_models  = ['iX', 'i4', 'iX1']
phev_models = []
bev_vol  = df_volume[df_volume['Model'].isin(bev_models)].groupby('Month')['Actual'].sum()
ice_vol  = df_volume[~df_volume['Model'].isin(bev_models)].groupby('Month')['Actual'].sum()
total    = bev_vol + ice_vol
bev_pct  = (bev_vol / total * 100).values
ice_pct  = (ice_vol / total * 100).values

x = np.arange(12)
axes[2].bar(x, bev_pct, label='BEV', color='#3b82f6', alpha=0.85)
axes[2].bar(x, ice_pct, bottom=bev_pct, label='ICE/PHEV', color='#334155', alpha=0.85)
axes[2].axhline(27, color='#f59e0b', linestyle='--', linewidth=1.5, label='BEV Target 27%')
axes[2].set_xticks(x); axes[2].set_xticklabels(month_labels, rotation=45, fontsize=8)
axes[2].set_ylabel('%'); axes[2].set_title('Powertrain Mix Trend', fontweight='bold')
axes[2].legend(fontsize=8); axes[2].set_ylim(0, 105)
axes[2].grid(True, axis='y', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart_B_kpi_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ chart_B_kpi_dashboard.png")

# ── Chart C: Stock imbalance heatmap ────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
pivot = df_stock.pivot_table(index='Model', columns='Region', values='Days_Supply', aggfunc='mean')
pivot = pivot.reindex(index=MODELS, columns=REGIONS)

cmap = plt.cm.RdYlGn
im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=20, vmax=90)
ax.set_xticks(range(len(REGIONS)));  ax.set_xticklabels(REGIONS, fontsize=10)
ax.set_yticks(range(len(MODELS)));   ax.set_yticklabels(MODELS, fontsize=10)
ax.set_title('Stock Days of Supply — Heatmap (Target: 45–60 days)', fontsize=13, fontweight='bold', pad=15)

for i in range(len(MODELS)):
    for j in range(len(REGIONS)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            col = 'white' if val < 38 or val > 70 else 'black'
            ax.text(j, i, f'{int(val)}d', ha='center', va='center', fontsize=9, color=col, fontweight='bold')

cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Days of Supply', color='#94a3b8')

# Add target band markers
ax.axhline(-0.5, color='#3b82f6', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart_C_stock_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ chart_C_stock_heatmap.png")

# ─────────────────────────────────────────────────────────────
#  5. EXPORT REPORTS
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Exporting reports...")

# Monthly KPI Summary CSV
monthly_summary = df_volume.groupby('Month').agg(
    Total_Actual  = ('Actual',  'sum'),
    Total_Planned = ('Planned', 'sum'),
).reset_index()
monthly_summary['Variance_Pct']    = ((monthly_summary['Total_Actual'] - monthly_summary['Total_Planned']) / monthly_summary['Total_Planned'] * 100).round(2)
monthly_summary['Plan_Accuracy']   = (100 - monthly_summary['Variance_Pct'].abs()).round(2)
monthly_summary['Cumulative_Actual'] = monthly_summary['Total_Actual'].cumsum()
monthly_summary['Month'] = monthly_summary['Month'].dt.strftime('%Y-%m')

monthly_summary.to_csv(f'{OUTPUT_DIR}/kpi_monthly_summary.csv', index=False)
print(f"   ✓ kpi_monthly_summary.csv")

df_stock.to_csv(f'{OUTPUT_DIR}/stock_status_report.csv', index=False)
print(f"   ✓ stock_status_report.csv")

df_anomalies.to_csv(f'{OUTPUT_DIR}/anomaly_defect_log.csv', index=False)
print(f"   ✓ anomaly_defect_log.csv  ({len(df_anomalies)} issues logged)")

# JSON summary for dashboard
summary = {
    'generated_at':  datetime.now().isoformat(),
    'total_units':   int(df_volume['Actual'].sum()),
    'total_planned': int(df_volume['Planned'].sum()),
    'plan_accuracy': round(100 - df_volume['Variance'].abs().mean(), 2),
    'anomaly_count': len(df_anomalies),
    'critical_stock':len(df_stock[df_stock['Status'] == 'critical']),
    'bev_share_pct': round(df_volume[df_volume['Model'].isin(['iX','i4','iX1'])]['Actual'].sum() / df_volume['Actual'].sum() * 100, 2),
    'top_model':     df_volume.groupby('Model')['Actual'].sum().idxmax(),
}
with open(f'{OUTPUT_DIR}/dashboard_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   ✓ dashboard_summary.json")

# ─────────────────────────────────────────────────────────────
#  PRINT SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — RESULTS SUMMARY")
print("=" * 65)
print(f"  Total Units Sold:    {summary['total_units']:>12,}")
print(f"  Total Planned:       {summary['total_planned']:>12,}")
print(f"  Plan Accuracy:       {summary['plan_accuracy']:>11.2f}%")
print(f"  BEV Share:           {summary['bev_share_pct']:>11.2f}%")
print(f"  Top Model:           {summary['top_model']:>15}")
print(f"  Anomalies Detected:  {summary['anomaly_count']:>12,}")
print(f"  Critical Stock Flags:{summary['critical_stock']:>12,}")
print("\n  Output files saved to: ./output/")
print("=" * 65)
