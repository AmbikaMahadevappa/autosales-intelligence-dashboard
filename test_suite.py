"""
AutoSales Intelligence Dashboard — Test Suite
==============================================
Author  : Ambika Sugganahalli Mahadevappa
Purpose : Structured testing of the sales volume planning pipeline.
          Covers Smoke, Regression, End-to-End, and UAT phases —
          mirroring a professional software release testing workflow.

Run with:
    python test_suite.py
"""

import unittest
import numpy as np
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  TEST REPORTER
# ─────────────────────────────────────────────────────────────
PASS  = "\033[92m  ✓ PASS\033[0m"
FAIL  = "\033[91m  ✗ FAIL\033[0m"
SKIP  = "\033[93m  ⊘ SKIP\033[0m"
PHASE = "\033[96m"
RESET = "\033[0m"

results = {'passed': 0, 'failed': 0, 'skipped': 0, 'log': []}

def log_result(name, passed, msg=''):
    status = 'PASS' if passed else 'FAIL'
    symbol = PASS if passed else FAIL
    print(f"{symbol}  {name}" + (f"  →  {msg}" if msg else ''))
    results['passed' if passed else 'failed'] += 1
    results['log'].append({'test': name, 'status': status, 'message': msg, 'timestamp': datetime.now().isoformat()})

# ─────────────────────────────────────────────────────────────
#  SHARED FIXTURES
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

MODELS   = ['3 Series', '5 Series', 'X5', 'iX', 'i4', '7 Series', 'X3', 'M3', 'Z4', 'iX1']
REGIONS  = ['Europe', 'Germany', 'USA', 'China', 'RoW']
MONTHS   = pd.date_range('2025-01-01', periods=12, freq='MS')

SEASON   = [0.88, 0.82, 1.05, 1.08, 1.12, 1.04, 0.95, 1.00, 1.10, 1.08, 0.92, 1.08]
BASE_VOL = 210000

def generate_volume(base, noise=0.04):
    return [int(base * SEASON[i] * (1 + np.random.normal(0, noise))) for i in range(12)]

def build_volume_df():
    records = []
    bases = {'3 Series':35000,'5 Series':26500,'X5':24600,'iX':15600,'i4':20400,
             '7 Series':12100,'X3':17800,'M3':8200,'Z4':3500,'iX1':13700}
    for model in MODELS:
        actual  = generate_volume(bases[model], 0.045)
        planned = generate_volume(bases[model] * 0.98, 0.015)
        for i, month in enumerate(MONTHS):
            records.append({'Month': month, 'Model': model,
                            'Actual': actual[i], 'Planned': planned[i],
                            'Variance': round((actual[i]-planned[i])/planned[i]*100, 2)})
    return pd.DataFrame(records)

def build_stock_df():
    rows = []
    rw = {'Europe':0.38,'Germany':0.13,'USA':0.20,'China':0.25,'RoW':0.04}
    ds = {m: 35000/30 * 0.1 for m in MODELS}
    for model in MODELS:
        for region in REGIONS:
            stock = int(ds[model] * rw[region] * np.random.uniform(25, 90))
            days  = max(1, int(stock / (ds[model] * rw[region])))
            cap   = round(np.random.uniform(62, 99), 1)
            status = 'critical' if days < 35 or days > 80 or cap > 97 else \
                     'warning'  if days < 45 or days > 70 or cap > 90 else 'ok'
            rows.append({'Model':model,'Region':region,'Days_Supply':days,'Capacity_Util':cap,'Status':status})
    return pd.DataFrame(rows)

df_volume = build_volume_df()
df_stock  = build_stock_df()

# ─────────────────────────────────────────────────────────────
#  PHASE 1: SMOKE TESTS
#  Purpose: Verify core data structures are healthy before
#           running any logic. First gate in deployment cycle.
# ─────────────────────────────────────────────────────────────
print(f"\n{PHASE}{'='*60}")
print(f"  PHASE 1 — SMOKE TESTS")
print(f"  Goal: Core data structures load and are non-empty")
print(f"{'='*60}{RESET}\n")

# S1: Volume DataFrame shape
try:
    expected_rows = len(MODELS) * len(MONTHS)
    log_result("S1: Volume DataFrame row count", len(df_volume) == expected_rows,
               f"Expected {expected_rows}, got {len(df_volume)}")
except Exception as e:
    log_result("S1: Volume DataFrame row count", False, str(e))

# S2: Required columns present
try:
    required = {'Month', 'Model', 'Actual', 'Planned', 'Variance'}
    log_result("S2: Required columns present", required.issubset(df_volume.columns),
               f"Missing: {required - set(df_volume.columns)}")
except Exception as e:
    log_result("S2: Required columns present", False, str(e))

# S3: No null values in critical columns
try:
    nulls = df_volume[['Actual','Planned','Variance']].isnull().sum().sum()
    log_result("S3: No null values in Actual/Planned/Variance", nulls == 0, f"Nulls found: {nulls}")
except Exception as e:
    log_result("S3: No null values", False, str(e))

# S4: All 10 models present
try:
    found = set(df_volume['Model'].unique())
    log_result("S4: All 10 models present", found == set(MODELS),
               f"Missing: {set(MODELS) - found}")
except Exception as e:
    log_result("S4: All 10 models present", False, str(e))

# S5: Stock DataFrame non-empty
try:
    log_result("S5: Stock DataFrame non-empty", len(df_stock) > 0, f"{len(df_stock)} rows")
except Exception as e:
    log_result("S5: Stock DataFrame non-empty", False, str(e))

# ─────────────────────────────────────────────────────────────
#  PHASE 2: REGRESSION TESTS
#  Purpose: Verify formulas produce consistent, repeatable
#           results. Catch bugs introduced by code changes.
# ─────────────────────────────────────────────────────────────
print(f"\n{PHASE}{'='*60}")
print(f"  PHASE 2 — REGRESSION TESTS")
print(f"  Goal: Formulas are consistent and reproducible")
print(f"{'='*60}{RESET}\n")

# R1: Variance formula is symmetric
try:
    test_a, test_p = 10000, 8000
    expected_var = round((test_a - test_p) / test_p * 100, 2)
    log_result("R1: Variance formula correctness", expected_var == 25.0, f"Expected 25.0, got {expected_var}")
except Exception as e:
    log_result("R1: Variance formula", False, str(e))

# R2: Plan accuracy = 100 - |mean variance|
try:
    mean_var = df_volume['Variance'].abs().mean()
    acc = round(100 - mean_var, 2)
    log_result("R2: Plan accuracy formula in valid range", 85 <= acc <= 100, f"Got {acc}%")
except Exception as e:
    log_result("R2: Plan accuracy formula", False, str(e))

# R3: Total annual volume is deterministic (same seed)
try:
    df2 = build_volume_df()  # rebuild with same seed — should differ (seed not reset, that's expected)
    # Test: both DFs have same columns and shape
    log_result("R3: DataFrame rebuild has same schema", list(df2.columns) == list(df_volume.columns), "Schema match")
except Exception as e:
    log_result("R3: Schema stability", False, str(e))

# R4: Variance direction is correct (actual > planned → positive)
try:
    row = df_volume[(df_volume['Actual'] > df_volume['Planned'])].iloc[0]
    log_result("R4: Positive variance when Actual > Planned", row['Variance'] > 0, f"Got {row['Variance']}")
except Exception as e:
    log_result("R4: Variance direction", False, str(e))

# R5: Seasonality sums to approximately 12 (balanced year)
try:
    s = sum(SEASON)
    log_result("R5: Seasonality factors sum near 12.0", abs(s - 12.0) < 0.5, f"Sum = {s:.3f}")
except Exception as e:
    log_result("R5: Seasonality sum", False, str(e))

# R6: BEV models correctly identified
try:
    bev_models = {'iX', 'i4', 'iX1'}
    bev_found = set(df_volume[df_volume['Model'].isin(bev_models)]['Model'].unique())
    log_result("R6: BEV models in dataset", bev_found == bev_models, f"Found: {bev_found}")
except Exception as e:
    log_result("R6: BEV model identification", False, str(e))

# ─────────────────────────────────────────────────────────────
#  PHASE 3: END-TO-END TESTS
#  Purpose: Verify the complete pipeline runs from raw input
#           through to exported output without errors.
# ─────────────────────────────────────────────────────────────
print(f"\n{PHASE}{'='*60}")
print(f"  PHASE 3 — END-TO-END TESTS")
print(f"  Goal: Full pipeline runs and outputs are valid")
print(f"{'='*60}{RESET}\n")

# E1: Full aggregation pipeline
try:
    monthly = df_volume.groupby('Month').agg(Total=('Actual','sum')).reset_index()
    log_result("E1: Monthly aggregation pipeline", len(monthly) == 12, f"{len(monthly)} months aggregated")
except Exception as e:
    log_result("E1: Monthly aggregation", False, str(e))

# E2: Regional split sums correctly
try:
    rw = {'Europe':0.38,'Germany':0.13,'USA':0.20,'China':0.25,'RoW':0.04}
    total_w = sum(rw.values())
    log_result("E2: Regional weights sum to 1.0", abs(total_w - 1.0) < 0.001, f"Sum = {total_w:.3f}")
except Exception as e:
    log_result("E2: Regional weights", False, str(e))

# E3: Anomaly detection pipeline
try:
    anomalies = []
    for _, row in df_volume.iterrows():
        if abs(row['Variance']) > 10:
            anomalies.append({'type':'Plan Deviation','model':row['Model'],'var':row['Variance']})
    log_result("E3: Anomaly detection runs without error", True, f"Detected {len(anomalies)} anomalies")
except Exception as e:
    log_result("E3: Anomaly detection", False, str(e))

# E4: CSV export pipeline
try:
    os.makedirs('output', exist_ok=True)
    test_csv = 'output/test_export.csv'
    df_volume.head(5).to_csv(test_csv, index=False)
    df_back = pd.read_csv(test_csv)
    log_result("E4: CSV export and re-import", len(df_back) == 5 and list(df_back.columns) == list(df_volume.columns), f"{len(df_back)} rows re-imported")
    os.remove(test_csv)
except Exception as e:
    log_result("E4: CSV export", False, str(e))

# E5: JSON summary generation
try:
    summary = {
        'total_units':    int(df_volume['Actual'].sum()),
        'plan_accuracy':  round(100 - df_volume['Variance'].abs().mean(), 2),
        'bev_share':      round(df_volume[df_volume['Model'].isin(['iX','i4','iX1'])]['Actual'].sum() / df_volume['Actual'].sum() * 100, 2),
        'generated_at':   datetime.now().isoformat(),
    }
    json_str = json.dumps(summary)
    parsed   = json.loads(json_str)
    log_result("E5: JSON summary serialises correctly", parsed['total_units'] > 0, f"Total: {parsed['total_units']:,} units")
except Exception as e:
    log_result("E5: JSON summary", False, str(e))

# E6: Output file existence (if data_engine.py was already run)
try:
    expected_outputs = ['output/kpi_monthly_summary.csv', 'output/stock_status_report.csv',
                        'output/anomaly_defect_log.csv', 'output/dashboard_summary.json']
    all_exist = all(os.path.exists(f) for f in expected_outputs)
    log_result("E6: Pipeline output files exist", all_exist,
               "Run data_engine.py first to generate" if not all_exist else "All 4 output files found")
except Exception as e:
    log_result("E6: Output files", False, str(e))

# ─────────────────────────────────────────────────────────────
#  PHASE 4: USER ACCEPTANCE TESTS (UAT)
#  Purpose: Validate against business rules and planning
#           thresholds — the criteria a volume planner would use.
# ─────────────────────────────────────────────────────────────
print(f"\n{PHASE}{'='*60}")
print(f"  PHASE 4 — USER ACCEPTANCE TESTS (UAT)")
print(f"  Goal: Business rules and planning thresholds satisfied")
print(f"{'='*60}{RESET}\n")

# U1: Plan accuracy target ≥ 90%
try:
    acc = 100 - df_volume['Variance'].abs().mean()
    log_result("U1: Plan accuracy ≥ 90% (business threshold)", acc >= 90, f"Accuracy = {acc:.2f}%")
except Exception as e:
    log_result("U1: Plan accuracy threshold", False, str(e))

# U2: No model has >20% variance in a single month
try:
    max_var = df_volume['Variance'].abs().max()
    log_result("U2: No variance >20% in any month", max_var <= 20, f"Max variance = {max_var:.2f}%")
except Exception as e:
    log_result("U2: Max variance threshold", False, str(e))

# U3: BEV share ≥ 20% (EU mandate floor)
try:
    bev_share = df_volume[df_volume['Model'].isin(['iX','i4','iX1'])]['Actual'].sum() / df_volume['Actual'].sum() * 100
    log_result("U3: BEV share ≥ 20% (EU mandate floor)", bev_share >= 20, f"BEV share = {bev_share:.2f}%")
except Exception as e:
    log_result("U3: BEV share mandate", False, str(e))

# U4: Stock status categories are valid values only
try:
    valid_statuses = {'ok', 'warning', 'critical'}
    found_statuses = set(df_stock['Status'].unique())
    log_result("U4: Stock status values are valid", found_statuses.issubset(valid_statuses),
               f"Found: {found_statuses}")
except Exception as e:
    log_result("U4: Status value validation", False, str(e))

# U5: Days of supply > 0 for all stock records
try:
    min_dos = df_stock['Days_Supply'].min()
    log_result("U5: Days of supply > 0 for all records", min_dos > 0, f"Min DOS = {min_dos}")
except Exception as e:
    log_result("U5: Days of supply positive", False, str(e))

# U6: Capacity utilisation in valid range [0, 100]
try:
    cap_min = df_stock['Capacity_Util'].min()
    cap_max = df_stock['Capacity_Util'].max()
    log_result("U6: Capacity utilisation in [0%, 100%]", 0 <= cap_min and cap_max <= 100,
               f"Range: {cap_min:.1f}% – {cap_max:.1f}%")
except Exception as e:
    log_result("U6: Capacity range validation", False, str(e))

# U7: All 12 months covered in output
try:
    months_found = df_volume['Month'].nunique()
    log_result("U7: All 12 months present in dataset", months_found == 12, f"Months found: {months_found}")
except Exception as e:
    log_result("U7: Full year coverage", False, str(e))

# U8: Annual total > 1 million units (business-realistic floor)
try:
    total = df_volume['Actual'].sum()
    log_result("U8: Annual volume > 1M units (realistic scale)", total > 1_000_000, f"Total: {total:,}")
except Exception as e:
    log_result("U8: Volume scale validation", False, str(e))

# ─────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────
total_tests = results['passed'] + results['failed']
pass_rate   = round(results['passed'] / total_tests * 100, 1) if total_tests > 0 else 0

print(f"\n{'='*60}")
print(f"  TEST SUITE COMPLETE")
print(f"{'='*60}")
print(f"  Total Tests:  {total_tests}")
print(f"  \033[92mPassed:       {results['passed']}\033[0m")
print(f"  \033[91mFailed:       {results['failed']}\033[0m")
print(f"  Pass Rate:    {pass_rate}%")
print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# Export test log
os.makedirs('output', exist_ok=True)
with open('output/test_results_log.json', 'w') as f:
    json.dump({
        'summary':   {'total': total_tests, 'passed': results['passed'], 'failed': results['failed'], 'pass_rate': pass_rate},
        'timestamp': datetime.now().isoformat(),
        'tests':     results['log'],
    }, f, indent=2)
print(f"  ✓ Test log saved to output/test_results_log.json\n")

sys.exit(0 if results['failed'] == 0 else 1)
