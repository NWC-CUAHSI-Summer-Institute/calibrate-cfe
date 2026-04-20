#!/usr/bin/env python3
"""
run_daily_testrun.py
Evaluates the daily Snow17+PET+CFE model on a test period using
previously calibrated parameters saved by run_daily_calibration.py.

Units: mm/day for both simulated and observed (per Suma's direction).

Usage:
    python run_daily_testrun.py \
        --params ./results/theta_opt_03463300.txt \
        --test-forcing ./gage_03463300_data/forcing_test.csv \
        --test-obs ./gage_03463300_data/obs_usgs_daily_test.csv \
        --out_dir ./results/test_run
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import pet
import snow17_pet_cfe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_nse(obs, sim):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def compute_kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Test daily CFE with best calibrated params")
    parser.add_argument('--params', type=str, required=True,
                        help="Path to calibrated params JSON (theta_opt_*.txt from calibration)")
    parser.add_argument('--test-forcing', type=str, required=True,
                        help="Path to test period forcing CSV")
    parser.add_argument('--test-obs', type=str, required=True,
                        help="Path to test period observations CSV")
    parser.add_argument('--out_dir', type=str, default='./results/test_run',
                        help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    basin = '03463300'
    area_km2 = 113.18356816373296
    lat = 35.83138889
    elev = 806.0
    albedo = 0.20
    time_step_size = 3600 * 24
    time_step_units = 'day'

    # Load calibrated parameters
    print(f"Loading calibrated params from: {args.params}")
    with open(args.params, 'r') as f:
        all_params = json.load(f)
    pet_params = all_params['pet_params']
    snow_params = all_params['snow_params']
    cfg_data = all_params['cfg_data']
    print(f"  alpha_PT={pet_params['alpha_PT']:.4f}, scf={snow_params['scf']:.4f}")

    # Load test forcing and observations
    print("Loading test data...")
    met_df = pd.read_csv(args.test_forcing)
    obs_df = pd.read_csv(args.test_obs)
    met_df['time'] = pd.to_datetime(met_df['time'])
    obs_df['time'] = pd.to_datetime(obs_df['time'])

    merged = pd.merge(met_df, obs_df, on='time')
    if 'streamflow' in merged.columns:
        merged['discharge_mm_day'] = merged['streamflow'] * 86400 * 1000 / (area_km2 * 1_000_000)
    else:
        merged['discharge_mm_day'] = 0.0

    dates = merged['time']
    qobs = merged['discharge_mm_day']
    print(f"  {len(merged)} days, obs range: {qobs.min():.3f}–{qobs.max():.3f} mm/day")

    # Compute net radiation
    Jul = dates.dt.dayofyear.values
    merged['Rn(W/m2)'] = pet.net_radiation(
        merged['srad_daily(W/m2)'].values, merged['tmin(C)'].values,
        merged['tmax(C)'].values, elev, lat, Jul, merged['vp(Pa)'].values, albedo)
    merged['G(W/m2)'] = 0.0

    # Run Snow17 + PET + CFE
    print("Running Snow17 + PET + CFE...")
    initial_snow_state = {'ait': 0.0, 'w_q': 0.0, 'w_i': 0.0, 'deficit': 0}
    initial_cfe_state = {'gw_initial_storage_m': 0.00, 'soil_initial_storage_m': 0.00}

    output_lists, _ = snow17_pet_cfe.run_snow_pet_cfe(
        merged, dates, cfg_data, snow_params, pet_params,
        initial_snow_state, initial_cfe_state, lat, elev, time_step_size, time_step_units)

    sim = np.array(output_lists['land_surface_water__runoff_depth']) * 1000.0  # m → mm/day
    obs = qobs.values

    # Align lengths
    n = min(len(sim), len(obs))
    sim, obs, dates_plot = sim[:n], obs[:n], dates.values[:n]

    # Metrics — skip if obs has no variance (e.g. very short test period)
    if np.var(obs) > 0:
        nse = compute_nse(obs, sim)
        kge = compute_kge(obs, sim)
    else:
        nse, kge = float('nan'), float('nan')

    print(f"\n  Test NSE : {nse:.4f}")
    print(f"  Test KGE : {kge:.4f}")
    print(f"  Obs  max : {obs.max():.2f} mm/day")
    print(f"  Sim  max : {sim.max():.2f} mm/day")

    # Save metrics
    metrics_path = os.path.join(args.out_dir, f'test_metrics_{basin}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Test NSE: {nse}\n')
        f.write(f'Test KGE: {kge}\n')
    print(f"  Saved: {metrics_path}")

    # Plot 1: Observed only
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_plot, obs, 'b-', linewidth=0.8, alpha=0.85, marker='o', markersize=2)
    ax.set_ylabel('Observed (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Observed Streamflow — Gage {basin} (Daily, Test Period)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'test_obs_only_{basin}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Separate panels — obs / sim / residual
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10),
                                         gridspec_kw={'height_ratios': [2, 2, 1]}, sharex=True)
    ax1.plot(dates_plot, obs, 'b-', linewidth=0.8, alpha=0.85)
    ax1.set_ylabel('Observed (mm/day)')
    ax1.set_title(f'Daily Test Run — Gage {basin}  NSE={nse:.4f}  KGE={kge:.4f}')
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates_plot, sim, 'r-', linewidth=0.8, alpha=0.85)
    ax2.set_ylabel('Simulated (mm/day)')
    ax2.grid(True, alpha=0.3)

    residuals = sim - obs
    ax3.bar(dates_plot, residuals, width=1.0, color='gray', alpha=0.4)
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_ylabel('Residual (mm/day)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'test_separate_{basin}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_plot, obs, 'b-', linewidth=0.8, alpha=0.75, label='Observed')
    ax.plot(dates_plot, sim, 'r--', linewidth=0.8, alpha=0.75, label='Simulated')
    ax.set_ylabel('Streamflow (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Daily Test Run — Gage {basin}  NSE={nse:.4f}  KGE={kge:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'test_overlay_{basin}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
