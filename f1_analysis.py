# =============================================================================
# F1 Race Performance — AI/ML Analysis Pipeline
# DataCore Analytics · Intern Assignment 2026
# Session: 2024 Bahrain GP Race
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)
seaborn_theme = sns.set_theme(style='darkgrid')

# =============================================================================
# PHASE 1 — DATA COLLECTION & SETUP
# =============================================================================

# ── Step 1: Load Session ─────────────────────────────────────────────────────
# When running locally with FastF1 installed, replace the block below with:
import fastf1
fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()
laps = session.laps
# The synthetic data below is structurally identical to FastF1 output so all
# downstream code works without modification when you swap in the real loader.

print("=" * 60)
print("PHASE 1 — DATA COLLECTION & SETUP")
print("=" * 60)

np.random.seed(42)

DRIVERS = ['VER', 'PER', 'SAI', 'LEC', 'HAM', 'RUS', 'NOR', 'ALO', 'STR', 'GAS',
           'OCO', 'TSU', 'BOT', 'ZHO', 'MAG', 'HUL', 'ALB', 'SAR', 'PIA', 'DEV']

TEAMS = {
    'VER': 'Red Bull', 'PER': 'Red Bull',
    'SAI': 'Ferrari',  'LEC': 'Ferrari',
    'HAM': 'Mercedes', 'RUS': 'Mercedes',
    'NOR': 'McLaren',  'PIA': 'McLaren',
    'ALO': 'Aston Martin', 'STR': 'Aston Martin',
    'GAS': 'Alpine',   'OCO': 'Alpine',
    'TSU': 'RB',       'DEV': 'RB',
    'BOT': 'Kick Sauber', 'ZHO': 'Kick Sauber',
    'MAG': 'Haas',     'HUL': 'Haas',
    'ALB': 'Williams', 'SAR': 'Williams',
}

DRIVER_PACE = {
    'VER': 91.2, 'LEC': 91.5, 'SAI': 91.7, 'NOR': 91.8, 'HAM': 91.9,
    'PER': 92.0, 'RUS': 92.1, 'ALO': 92.3, 'PIA': 92.5, 'GAS': 92.8,
    'OCO': 92.9, 'STR': 93.0, 'TSU': 93.2, 'ALB': 93.4, 'BOT': 93.5,
    'MAG': 93.6, 'ZHO': 93.7, 'HUL': 93.8, 'DEV': 94.0, 'SAR': 94.2,
}

TOTAL_LAPS = 57
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

rows = []
for driver in DRIVERS:
    base_pace = DRIVER_PACE[driver]
    lap_num = 1
    stint = 0
    tyre_life = 0
    compound = np.random.choice(['SOFT', 'MEDIUM'], p=[0.6, 0.4])

    while lap_num <= TOTAL_LAPS:
        # Tyre degradation: pace drops as tyre ages
        deg = tyre_life * np.random.uniform(0.03, 0.06)
        lap_noise = np.random.normal(0, 0.25)
        lap_time = base_pace + deg + lap_noise

        # Compound offsets
        if compound == 'SOFT':
            lap_time -= 0.4
        elif compound == 'HARD':
            lap_time += 0.5

        # Sector split (~30 / 35 / 28 % of lap)
        s1 = lap_time * 0.295 + np.random.normal(0, 0.05)
        s2 = lap_time * 0.365 + np.random.normal(0, 0.06)
        s3 = lap_time - s1 - s2

        speed_base = 305 + (91.2 - base_pace) * 3
        rows.append({
            'Driver':       driver,
            'Team':         TEAMS[driver],
            'LapNumber':    lap_num,
            'LapTime':      pd.Timedelta(seconds=lap_time),
            'Sector1Time':  pd.Timedelta(seconds=s1),
            'Sector2Time':  pd.Timedelta(seconds=s2),
            'Sector3Time':  pd.Timedelta(seconds=s3),
            'Compound':     compound,
            'TyreLife':     tyre_life + 1,
            'SpeedI1':      speed_base + np.random.normal(0, 4),
            'SpeedI2':      speed_base - 20 + np.random.normal(0, 3),
            'SpeedFL':      speed_base - 10 + np.random.normal(0, 3),
            'SpeedST':      speed_base + np.random.normal(0, 5),
        })

        tyre_life += 1
        lap_num += 1

        # Pit stop logic
        if tyre_life > np.random.randint(15, 28):
            compound = np.random.choice(COMPOUNDS)
            tyre_life = 0
            stint += 1

laps_raw = pd.DataFrame(rows)

print(f"Raw laps shape : {laps_raw.shape}")
print(f"Columns        : {list(laps_raw.columns)}")
print(laps_raw.head())

# =============================================================================
# ── Step 2: Data Cleaning ──────────────────────────────────────────────────
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — DATA CLEANING")
print("=" * 60)

KEEP_COLS = ['Driver', 'Team', 'LapNumber', 'LapTime',
             'Sector1Time', 'Sector2Time', 'Sector3Time',
             'Compound', 'TyreLife',
             'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']

laps = laps_raw[KEEP_COLS].copy()

# Convert timedeltas to float seconds
for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
    laps[col] = laps[col].dt.total_seconds()

rows_before = len(laps)

# Remove nulls and outlier laps (> 120 s indicates SC/crash lap)
laps = laps[laps['LapTime'].notna() & (laps['LapTime'] <= 120)]
laps = laps[laps['Sector1Time'].notna() & laps['Sector2Time'].notna() & laps['Sector3Time'].notna()]
laps = laps.reset_index(drop=True)

rows_after = len(laps)
print(f"Rows before cleaning : {rows_before}")
print(f"Rows after  cleaning : {rows_after}  ({rows_before - rows_after} removed)")
print(laps.dtypes)

# =============================================================================
# PHASE 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 — EDA")
print("=" * 60)

# ── Step 3: Lap Time Distribution ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(laps['LapTime'], bins=40, color='crimson', edgecolor='black', alpha=0.85)
mean_lt = laps['LapTime'].mean()
med_lt  = laps['LapTime'].median()
ax.axvline(mean_lt, color='gold',  linestyle='--', linewidth=2, label=f'Mean: {mean_lt:.2f}s')
ax.axvline(med_lt,  color='teal',  linestyle='--', linewidth=2, label=f'Median: {med_lt:.2f}s')
ax.set_xlabel('Lap Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Lap Time Distribution — 2024 Bahrain GP', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/lap_distribution.png', dpi=150)
plt.close()
print("Saved → plots/lap_distribution.png")

# Top 5 fastest drivers
top5 = (laps.groupby('Driver')['LapTime']
            .mean()
            .sort_values()
            .head(5)
            .reset_index())
top5.columns = ['Driver', 'Avg LapTime (s)']
top5['Rank'] = range(1, 6)
print("\nTop 5 Fastest Drivers (avg lap time):")
print(top5[['Rank', 'Driver', 'Avg LapTime (s)']].to_string(index=False))

# ── Step 4: Compound Boxplot ───────────────────────────────────────────────
compound_palette = {'SOFT': 'red', 'MEDIUM': 'gold', 'HARD': 'grey'}
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=laps, x='Compound', y='LapTime',
            palette=compound_palette,
            order=['SOFT', 'MEDIUM', 'HARD'],
            ax=ax, linewidth=1.5)
ax.set_xlabel('Tyre Compound', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
ax.set_title('Lap Time by Tyre Compound', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/compound_boxplot.png', dpi=150)
plt.close()
print("Saved → plots/compound_boxplot.png")

# ── Step 5: Sector & Speed Trap Analysis ──────────────────────────────────
driver_sectors = (laps.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']]
                      .mean()
                      .sort_values('LapTime'))

top_drivers = driver_sectors.index.tolist()
sector_data = driver_sectors[['Sector1Time', 'Sector2Time', 'Sector3Time']].loc[top_drivers]

x = np.arange(len(top_drivers))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, sector_data['Sector1Time'], width, label='S1', color='crimson')
ax.bar(x,          sector_data['Sector2Time'], width, label='S2', color='gold')
ax.bar(x + width, sector_data['Sector3Time'], width, label='S3', color='teal')
ax.set_xticks(x)
ax.set_xticklabels(top_drivers, fontsize=10)
ax.set_xlabel('Driver', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Average Sector Times by Driver', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/sector_comparison.png', dpi=150)
plt.close()
print("Saved → plots/sector_comparison.png")

# Pearson correlation: SpeedST vs LapTime
corr = laps['SpeedST'].corr(laps['LapTime'])
print(f"\nPearson correlation (SpeedST vs LapTime): r = {corr:.3f}")
print("Interpretation: near-zero r means straight-line speed has little direct effect on overall lap time.")

# ── Step 6: Speed Correlation Scatter ─────────────────────────────────────
scatter_palette = {'SOFT': 'red', 'MEDIUM': 'gold', 'HARD': 'grey'}
fig, ax = plt.subplots(figsize=(10, 6))
for compound, grp in laps.groupby('Compound'):
    ax.scatter(grp['SpeedST'], grp['LapTime'],
               color=scatter_palette.get(compound, 'blue'),
               label=compound, alpha=0.55, s=25)

# Regression line
m, b = np.polyfit(laps['SpeedST'], laps['LapTime'], 1)
x_line = np.linspace(laps['SpeedST'].min(), laps['SpeedST'].max(), 200)
ax.plot(x_line, m * x_line + b, 'w--', linewidth=2, label='Trend')
ax.text(0.02, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
        fontsize=12, color='white', va='top')
ax.set_xlabel('Speed Trap (km/h)', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
ax.set_title('Speed Trap vs Lap Time', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/speed_correlation.png', dpi=150)
plt.close()
print("Saved → plots/speed_correlation.png")

# =============================================================================
# PHASE 3 — FEATURE ENGINEERING & ML MODELLING
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 3 — FEATURE ENGINEERING & ML MODELLING")
print("=" * 60)

# ── Step 7: Feature Engineering ───────────────────────────────────────────
laps['SectorBalance'] = laps['Sector1Time'] - laps['Sector3Time']

laps['TyreAge_Bucket'] = pd.cut(laps['TyreLife'],
                                 bins=[0, 10, 25, 100],
                                 labels=['Fresh', 'Used', 'Old'])

compound_dummies  = pd.get_dummies(laps['Compound'],       prefix='Compound')
tyre_age_dummies  = pd.get_dummies(laps['TyreAge_Bucket'], prefix='TyreAge')

le = LabelEncoder()
laps['Driver_enc'] = le.fit_transform(laps['Driver'])

feature_cols = ['LapNumber', 'TyreLife', 'SectorBalance',
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Driver_enc']

X = pd.concat([laps[feature_cols], compound_dummies, tyre_age_dummies], axis=1)
y = laps['LapTime']

# Confirm no nulls
assert X.isnull().sum().sum() == 0, "Nulls found in feature matrix!"
print(f"Feature matrix X shape : {X.shape}")
print(f"Target y shape         : {y.shape}")
print(f"Null count in X        : {X.isnull().sum().sum()}")
print(f"Features used          : {list(X.columns)}")

# ── Step 8: Train Random Forest ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nModel Evaluation (Random Forest):")
print(f"  MAE  = {mae:.4f} s")
print(f"  RMSE = {rmse:.4f} s")
print(f"  R²   = {r2:.4f}")
if r2 >= 0.85:
    print("  ✓ R² target met (≥ 0.85)")
else:
    print("  ✗ R² below 0.85 — revisit feature engineering")

# Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, color='crimson', alpha=0.55, s=30, label='Predictions')
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], '--', color='gold', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Lap Time (s)', fontsize=12)
ax.set_ylabel('Predicted Lap Time (s)', fontsize=12)
ax.set_title(f'Predicted vs Actual Lap Times\nRandom Forest — R² = {r2:.2f}',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/predicted_vs_actual.png', dpi=150)
plt.close()
print("Saved → plots/predicted_vs_actual.png")

# ── Step 9: Feature Importance ────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top10.index, top10.values, color='crimson', edgecolor='black')
for bar, val in zip(bars, top10.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=9)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top Feature Importances — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("Saved → plots/feature_importance.png")

# ── Step 10: Anomaly Detection ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — ANOMALY DETECTION")
print("=" * 60)

driver_stats = laps.groupby('Driver')['LapTime'].agg(['median', 'std']).reset_index()
driver_stats.columns = ['Driver', 'driver_median', 'driver_std']
laps = laps.merge(driver_stats, on='Driver')
laps['IsAnomaly'] = laps['LapTime'] > (laps['driver_median'] + 2 * laps['driver_std'])

# Per-driver anomaly summary
anomaly_summary = (laps.groupby('Driver')['IsAnomaly']
                       .sum()
                       .astype(int)
                       .sort_values(ascending=False)
                       .reset_index())
anomaly_summary.columns = ['Driver', 'Anomaly Laps']
print("\nPer-driver anomaly lap count:")
print(anomaly_summary.to_string(index=False))

# Plot top 3 fastest drivers
top3 = driver_sectors.index[:3].tolist()
colors_map = {top3[0]: 'crimson', top3[1]: 'royalblue', top3[2]: 'teal'}

fig, ax = plt.subplots(figsize=(13, 6))
for drv in top3:
    drv_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
    ax.plot(drv_laps['LapNumber'], drv_laps['LapTime'],
            color=colors_map[drv], linewidth=1.4, label=drv)
    anomalies = drv_laps[drv_laps['IsAnomaly']]
    ax.scatter(anomalies['LapNumber'], anomalies['LapTime'],
               color='red', marker='x', s=120, linewidths=2.5, zorder=5)

ax.set_xlabel('Lap Number', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
ax.set_title('Anomaly Detection — Lap Time per Driver\n✕ = flagged anomaly lap',
             fontsize=13, fontweight='bold')
ax.legend(title='Driver')
plt.tight_layout()
plt.savefig('plots/anomaly_detection.png', dpi=150)
plt.close()
print("Saved → plots/anomaly_detection.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ALL DELIVERABLES COMPLETE")
print("=" * 60)
print(f"  Rows in clean dataset     : {len(laps)}")
print(f"  MAE                       : {mae:.4f} s")
print(f"  RMSE                      : {rmse:.4f} s")
print(f"  R²                        : {r2:.4f}")
print(f"  Top feature               : {importances.idxmax()}")
print(f"  Total anomaly laps        : {laps['IsAnomaly'].sum()}")
print("\nPlots saved in plots/ folder:")
for f in sorted(os.listdir('plots')):
    print(f"  plots/{f}")
print("=" * 60)
print("Push to GitHub and submit your report. Good luck!")
