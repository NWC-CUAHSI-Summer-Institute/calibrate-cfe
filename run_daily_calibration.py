#!/usr/bin/env python3
"""
run_daily_calibration.py
Calibrates the CFE hydrologic model for DAILY time steps using the
snow17 + PET + CFE setup integrated via SPOTPY DDS.

Units: mm/day for both simulated and observed.

"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import pet
from spotpy.algorithms import dds
import SPOTPY_DDS_setup
import snow17_pet_cfe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Run daily CFE calibration")
    parser.add_argument('--forcing', type=str, required=True, help="Path to calibration forcing CSV")
    parser.add_argument('--obs', type=str, required=True, help="Path to calibration observations CSV")
    parser.add_argument('--test-forcing', type=str, required=True, help="Path to test forcing CSV")
    parser.add_argument('--test-obs', type=str, required=True, help="Path to test observations CSV")
    parser.add_argument('--runs', type=int, default=500, help="Number of DDS iterations")
    parser.add_argument('--out_dir', type=str, default='./results/CFE_results', help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    basin = '03463300'
    area_km2 = 113.18356816373296
    lat = 35.83138889
    elev = 806.0
    albedo = 0.20
    time_step_size = 3600 * 24
    time_step_units = 'day'
    warmup_days = 365

    # Model parameters range for calibration (Abhinav's setup)
    param_range = {                
        'alpha_PT':  [0.10, 1.74, 0],   
        'scf': [0.8, 1.2, 0],           
        'rvs': [1, 1, 0],           
        'uadj': [0.01, 0.40, 0],       
        'mbase': [0.8, 2.0, 0],        
        'mfmax': [0.8, 3.0, 0],        
        'mfmin': [0.01, 0.79, 0],      
        'tipm': [0.01, 1.0, 0],        
        'nmf': [0.04, 0.40, 0],        
        'plwhc': [0.01, 0.40, 0],      
        'pxtemp': [-1.0, 1.0, 0],       
        'pxtemp1': [-1.0, -1.0, 0],    
        'pxtemp2': [3.0, 3.0, 0],      
        "bb": [0.1, 21.94, 0],               
        "satdk":[0.000001, 0.00726, 0],       
        "satpsi":[0.069, 0.78, 0],          
        "slop":[0.0001, 1.0, 0],              
        "smcmax": [0.16, 0.8, 0],         
        "wltsmc": [0.001, 0.25, 0],         
        "D": [0.1, 2.0, 0],  
        "K_lf": [0.01, 1.0, 0],             
        'alpha_fc': [0.10, 0.50, 0],          
        'max_gw_storage': [0.01, 0.50, 0],   
        'Cgw': [24*1.8*10**(-6), 24*1.8*10**(-3), 0],            
        'expon': [1.0, 8.0, 0],             
        'K_nash_lateral': [0.0001, 1.0, 0],                   
        'num_nash_reservoirs_lateral': [1, 4.99, 0],    
        'K_nash_surface': [0.0001, 1.00, 0],                    
        'num_nash_reservoirs_surface': [1, 4.99, 0],     
        'refkdt': [0.1, 8.0, 0],                            
        'trigger_z_m': [0.25, 0.75, 0]                            
    }

    print("Loading calibration data...")
    met_df_cal = pd.read_csv(args.forcing)
    obs_df_cal = pd.read_csv(args.obs)
    met_df_cal['time'] = pd.to_datetime(met_df_cal['time'])
    obs_df_cal['time'] = pd.to_datetime(obs_df_cal['time'])
    
    merged_cal = pd.merge(met_df_cal, obs_df_cal, on='time')
    if 'streamflow' in merged_cal.columns:
        # Convert streamflow m³/s → mm/day
        merged_cal['discharge_mm_day'] = merged_cal['streamflow'] * 86400 * 1000 / (area_km2 * 1000000)
    else:
        merged_cal['discharge_mm_day'] = 0.0

    met_df_cal = merged_cal.copy()
    dates_cal = met_df_cal['time']
    qobs_cal = met_df_cal['discharge_mm_day']

    print(f"  Cal obs stats (mm/day): min={qobs_cal.min():.4f}, max={qobs_cal.max():.4f}, mean={qobs_cal.mean():.4f}")

    print("Loading test data...")
    met_df_test = pd.read_csv(args.test_forcing)
    obs_df_test = pd.read_csv(args.test_obs)
    met_df_test['time'] = pd.to_datetime(met_df_test['time'])
    obs_df_test['time'] = pd.to_datetime(obs_df_test['time'])
    
    merged_test = pd.merge(met_df_test, obs_df_test, on='time')
    if 'streamflow' in merged_test.columns:
        merged_test['discharge_mm_day'] = merged_test['streamflow'] * 86400 * 1000 / (area_km2 * 1000000)
    else:
        merged_test['discharge_mm_day'] = 0.0

    met_df_test = merged_test.copy()
    dates_test = met_df_test['time']
    qobs_test = met_df_test['discharge_mm_day']

    print(f"  Test obs stats (mm/day): min={qobs_test.min():.4f}, max={qobs_test.max():.4f}, mean={qobs_test.mean():.4f}")

    # Compute Net Radiation and G for Calibration
    Jul_cal = met_df_cal['time'].dt.dayofyear.values
    met_df_cal['Rn(W/m2)'] = pet.net_radiation(met_df_cal['srad_daily(W/m2)'].values,
                                               met_df_cal['tmin(C)'].values,
                                               met_df_cal['tmax(C)'].values,
                                               elev, lat, Jul_cal,
                                               met_df_cal['vp(Pa)'].values, albedo)
    met_df_cal['G(W/m2)'] = 0.0

    # Test
    Jul_test = met_df_test['time'].dt.dayofyear.values
    met_df_test['Rn(W/m2)'] = pet.net_radiation(met_df_test['srad_daily(W/m2)'].values,
                                                met_df_test['tmin(C)'].values,
                                                met_df_test['tmax(C)'].values,
                                                elev, lat, Jul_test,
                                                met_df_test['vp(Pa)'].values, albedo)
    met_df_test['G(W/m2)'] = 0.0

    initial_snow_state = {'ait': 0.0, 'w_q': 0.0, 'w_i': 0.0, 'deficit': 0}
    initial_cfe_state = {'gw_initial_storage_m': 0.00, 'soil_initial_storage_m': 0.00}

    print(f"\nStarting SPOTPY DDS Calibration for {args.runs} iterations")
    print(f"  Units: mm/day (both obs and sim)")
    spotpy_setup_instance = SPOTPY_DDS_setup.spotpy_setup(
        param_range, area_km2, warmup_days, 
        met_df_cal, dates_cal, qobs_cal.values[warmup_days:], 
        initial_snow_state, initial_cfe_state, lat, elev, time_step_size, time_step_units
    )

    sampler = dds(spotpy_setup_instance, dbname=os.path.join(args.out_dir, f'all_params_{basin}'), dbformat='csv')
    sampler.sample(args.runs, trials=1)

    # Re-run best
    results = pd.read_csv(os.path.join(args.out_dir, f'all_params_{basin}.csv'))
    best = results.loc[results['like1'].idxmax()]

    params = best
    pet_params = {'alpha_PT': params['paralpha_PT']}
    snow_params = {'scf': params['parscf'], 'rvs': params['parrvs'], 'uadj': params['paruadj'], 
                   'mbase': params['parmbase'], 'mfmax': params['parmfmax'], 'mfmin': params['parmfmin'], 
                   'tipm': params['partipm'], 'nmf': params['parnmf'], 'plwhc': params['parplwhc'], 
                   'pxtemp': params['parpxtemp'], 'pxtemp1': params['parpxtemp1'], 'pxtemp2': params['parpxtemp2']}
    cfg_data = {
        "catchment_area_km2": area_km2,
        "partition_scheme": "Schaake",
        "soil_params": {
            "bb": params['parbb'], "satdk": params['parsatdk'], "satpsi": params['parsatpsi'], 
            "slop": params['parslop'], "smcmax": params['parsmcmax'], "wltsmc": params['parwltsmc'], 
            "D": params['parD'], "K_lf": params['parK_lf'], "alpha_fc": params['paralpha_fc']
        },
        "max_gw_storage": params['parmax_gw_storage'], "Cgw": params['parCgw'], "expon": params['parexpon'],
        "K_nash_lateral": params['parK_nash_lateral'], "nash_storage_lateral": [0.0]*int(params['parnum_nash_reservoirs_lateral']), 
        "K_nash_surface": params['parK_nash_surface'], "nash_storage_surface": [0.0]*int(params['parnum_nash_reservoirs_surface']), 
        "refkdt": params['parrefkdt'], 'trigger_z_m': params['partrigger_z_m'],   
        "soil_scheme": "classic",
        "stand_alone": 0,
    }

    # -------- Calibration period --------
    output_lists, _ = snow17_pet_cfe.run_snow_pet_cfe(met_df_cal, dates_cal, cfg_data,
                                                      snow_params, pet_params, initial_snow_state, 
                                                      initial_cfe_state, lat, elev, time_step_size, time_step_units)
    sim_cal = np.array(output_lists['land_surface_water__runoff_depth']) * 1000.0  # m → mm/day
    sim_cal = sim_cal[warmup_days:]
    obs_cal = qobs_cal.values[warmup_days:]
    dates_cal_plot = dates_cal.values[warmup_days:]
    nse_cal = 1 - np.sum((obs_cal - sim_cal) ** 2) / np.sum((obs_cal - np.mean(obs_cal)) ** 2)

    # -------- Test period --------
    output_lists_test, _ = snow17_pet_cfe.run_snow_pet_cfe(met_df_test, dates_test, cfg_data,
                                                           snow_params, pet_params, initial_snow_state, 
                                                           initial_cfe_state, lat, elev, time_step_size, time_step_units)
    sim_test = np.array(output_lists_test['land_surface_water__runoff_depth']) * 1000.0  # m → mm/day
    obs_test = qobs_test.values
    dates_test_plot = dates_test.values

    # Handle test warmup: if test period is shorter than warmup_days, don't skip
    if len(sim_test) > warmup_days:
        sim_test_eval = sim_test[warmup_days:]
        obs_test_eval = obs_test[warmup_days:]
    else:
        sim_test_eval = sim_test
        obs_test_eval = obs_test

    if len(obs_test_eval) > 0 and np.var(obs_test_eval) > 0:
        nse_test = 1 - np.sum((obs_test_eval - sim_test_eval[:len(obs_test_eval)]) ** 2) / np.sum((obs_test_eval - np.mean(obs_test_eval)) ** 2)
    else:
        nse_test = float('nan')

    # Save NSE
    with open(os.path.join(args.out_dir, f'NSE_{basin}.txt'), 'w') as f:
        f.write(f'Calibration NSE: {nse_cal}\n')
        f.write(f'Test NSE: {nse_test}\n')

    all_params = {"pet_params": pet_params, "snow_params": snow_params, "cfg_data": cfg_data}
    with open(os.path.join(args.out_dir, f'theta_opt_{basin}.txt'), 'w') as f:
        json.dump(all_params, f, indent=4)

    print()
    print(f"  Calibration NSE: {nse_cal:.4f}")
    print(f"  Test NSE: {nse_test:.4f}")

    # ================================================================
    # PLOTS — per Suma's directions: separate panels, obs-only, mm/day
    # ================================================================

    # --- PLOT 1: Observations only (calibration period) ---
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(dates_cal_plot, obs_cal, 'b-', linewidth=0.7, alpha=0.8)
    ax.set_ylabel('Observed (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Observed Streamflow — Gage {basin} (Daily, Calibration Period)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    obs_cal_path = os.path.join(args.out_dir, f'daily_obs_only_cal_{basin}.png')
    plt.savefig(obs_cal_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {obs_cal_path}")

    # --- PLOT 2: Observations only (test period) ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_test_plot, obs_test, 'b-', linewidth=0.7, alpha=0.8, marker='o', markersize=2)
    ax.set_ylabel('Observed (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Observed Streamflow — Gage {basin} (Daily, Test Period: Hurricane Helene)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    obs_test_path = os.path.join(args.out_dir, f'daily_obs_only_test_{basin}.png')
    plt.savefig(obs_test_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {obs_test_path}")

    # --- PLOT 3: Calibration - Separate panels (no overlap) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10),
                                         gridspec_kw={'height_ratios': [2, 2, 1]},
                                         sharex=True)
    ax1.plot(dates_cal_plot, obs_cal, 'b-', linewidth=0.7, alpha=0.8)
    ax1.set_ylabel('Observed (mm/day)')
    ax1.set_title(f'Daily Calibration — Gage {basin} (NSE = {nse_cal:.4f})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates_cal_plot, sim_cal, 'r-', linewidth=0.7, alpha=0.8)
    ax2.set_ylabel('Simulated (mm/day)')
    ax2.grid(True, alpha=0.3)

    residuals_cal = sim_cal - obs_cal
    ax3.bar(dates_cal_plot, residuals_cal, width=1.0, color='gray', alpha=0.4)
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_ylabel('Residual (mm/day)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    cal_sep_path = os.path.join(args.out_dir, f'daily_separate_cal_{basin}.png')
    plt.savefig(cal_sep_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cal_sep_path}")

    # --- PLOT 4: Test - Separate panels (no overlap) ---
    min_test = min(len(sim_test), len(obs_test))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(dates_test_plot[:min_test], obs_test[:min_test], 'b-', linewidth=0.7, alpha=0.8, marker='o', markersize=3)
    ax1.set_ylabel('Observed (mm/day)')
    ax1.set_title(f'Daily Test (Hurricane Helene) — Gage {basin} (NSE = {nse_test:.4f})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates_test_plot[:min_test], sim_test[:min_test], 'r-', linewidth=0.7, alpha=0.8, marker='x', markersize=3)
    ax2.set_ylabel('Simulated (mm/day)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    test_sep_path = os.path.join(args.out_dir, f'daily_separate_test_{basin}.png')
    plt.savefig(test_sep_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {test_sep_path}")

    # --- PLOT 5: Overlay (for quick comparison) ---
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dates_cal_plot, obs_cal, 'b-', linewidth=0.7, alpha=0.7, label='Observed')
    ax.plot(dates_cal_plot, sim_cal, 'r--', linewidth=0.7, alpha=0.7, label='Simulated')
    ax.set_ylabel('Streamflow (mm/day)')
    ax.set_xlabel('Date')
    ax.set_title(f'Daily Calibration — Gage {basin} (NSE = {nse_cal:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    cal_overlay_path = os.path.join(args.out_dir, f'daily_overlay_cal_{basin}.png')
    plt.savefig(cal_overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cal_overlay_path}")

    # Sanity check
    print(f"\n  SANITY CHECK (mm/day):")
    print(f"    Obs cal peaks: max={obs_cal.max():.2f} mm/day")
    print(f"    Sim cal peaks: max={sim_cal.max():.2f} mm/day")
    print(f"    Obs test peaks: max={obs_test.max():.2f} mm/day")
    print(f"    (Suma expects max ~60-70 mm/day for CAMELS gauges)")

    print(f"\n  Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
