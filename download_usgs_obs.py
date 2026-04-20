#!/usr/bin/env python3
"""
download_usgs_obs.py — Download clean USGS daily & hourly observations for gage 03463300.

Pulls directly from USGS Water Services API.
Raw data is in CFS (cubic feet per second).
Converts to mm/day (daily) or mm/hr (hourly) using drainage area.

USGS gage 03463300: South Toe River Near Celo, NC
  Drainage area: 43.3 sq mi = 112.14 km²

Usage:
    python3 download_usgs_obs.py --output-dir ./gage_03463300_data
"""

import argparse
import os
import pandas as pd
import numpy as np
import requests
import time as time_module

SITE_NO = "03463300"
DRAINAGE_AREA_SQ_MI = 43.3
DRAINAGE_AREA_KM2 = DRAINAGE_AREA_SQ_MI * 2.58999  # = 112.14 km²


def cfs_to_mm_per_day(q_cfs, area_km2):
    """Convert CFS → mm/day."""
    return q_cfs * 0.028317 * 86400.0 * 1000.0 / (area_km2 * 1.0e6)


def cfs_to_mm_per_hr(q_cfs, area_km2):
    """Convert CFS → mm/hr."""
    return q_cfs * 0.028317 * 3600.0 * 1000.0 / (area_km2 * 1.0e6)


def cfs_to_cms(q_cfs):
    """Convert CFS → m³/s."""
    return q_cfs * 0.028317


def parse_rdb(text):
    """Parse USGS RDB format text into a DataFrame."""
    lines = text.strip().split('\n')
    data_lines = [l for l in lines if not l.startswith('#')]
    
    if len(data_lines) < 3:
        return pd.DataFrame()
    
    header = data_lines[0].split('\t')
    data = []
    for line in data_lines[2:]:
        if line.strip():
            data.append(line.split('\t'))
    
    return pd.DataFrame(data, columns=header)


def download_daily_values(site_no, start_date, end_date):
    """Download daily mean discharge from USGS."""
    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "rdb",
        "sites": site_no,
        "parameterCd": "00060",
        "startDT": start_date,
        "endDT": end_date,
        "siteStatus": "all",
        "statCd": "00003",
    }
    
    print(f"  Fetching daily values: {start_date} to {end_date}")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    
    df = parse_rdb(resp.text)
    if df.empty:
        print(f"  WARNING: No data returned!")
        return pd.DataFrame()
    
    discharge_col = [c for c in df.columns if '00060' in c and '_cd' not in c]
    if not discharge_col:
        print(f"  WARNING: No discharge column found!")
        return pd.DataFrame()
    
    q_col = discharge_col[-1]
    
    result = pd.DataFrame()
    result['time'] = pd.to_datetime(df['datetime'])
    result['discharge_cfs'] = pd.to_numeric(df[q_col], errors='coerce')
    result = result.dropna()
    
    print(f"  Got {len(result)} daily records")
    print(f"  CFS range: {result['discharge_cfs'].min():.1f} – {result['discharge_cfs'].max():.1f}")
    
    return result


def download_iv_values(site_no, start_date, end_date):
    """Download instantaneous (15-min) discharge from USGS IV service.
    
    USGS IV service limits requests to ~120 days at a time.
    We chunk into 90-day windows to be safe.
    """
    url = "https://waterservices.usgs.gov/nwis/iv/"
    
    all_data = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    chunk_days = 90
    
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        
        params = {
            "format": "rdb",
            "sites": site_no,
            "parameterCd": "00060",
            "startDT": current.strftime("%Y-%m-%d"),
            "endDT": chunk_end.strftime("%Y-%m-%d"),
            "siteStatus": "all",
        }
        
        print(f"  Fetching IV: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...", end="")
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            
            df = parse_rdb(resp.text)
            if not df.empty:
                discharge_col = [c for c in df.columns if '00060' in c and '_cd' not in c]
                if discharge_col:
                    q_col = discharge_col[-1]
                    chunk_df = pd.DataFrame()
                    chunk_df['time'] = pd.to_datetime(df['datetime'])
                    chunk_df['discharge_cfs'] = pd.to_numeric(df[q_col], errors='coerce')
                    chunk_df = chunk_df.dropna()
                    all_data.append(chunk_df)
                    print(f" {len(chunk_df)} records")
                else:
                    print(f" no discharge column")
            else:
                print(f" empty")
        except Exception as e:
            print(f" ERROR: {e}")
        
        current = chunk_end + pd.Timedelta(days=1)
        time_module.sleep(1)  # Be nice to USGS servers
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values('time').drop_duplicates(subset='time')
    
    print(f"  Total IV records: {len(result)}")
    print(f"  CFS range: {result['discharge_cfs'].min():.1f} – {result['discharge_cfs'].max():.1f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Download USGS observations for gage 03463300")
    parser.add_argument('--output-dir', type=str, default='./gage_03463300_data')
    parser.add_argument('--skip-hourly', action='store_true', help="Skip hourly IV download (slow)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    area_km2 = DRAINAGE_AREA_KM2
    print(f"USGS Gage: {SITE_NO} (South Toe River Near Celo, NC)")
    print(f"Drainage area: {DRAINAGE_AREA_SQ_MI} sq mi = {area_km2:.2f} km²")
    print()
    
    # =========================================
    # DAILY: CALIBRATION (2018-2022)
    # =========================================
    print("=" * 60)
    print("DAILY — CALIBRATION PERIOD (2018-2022)")
    print("=" * 60)
    
    df_cal = download_daily_values(SITE_NO, "2018-01-01", "2022-12-31")
    
    if len(df_cal) > 0:
        df_cal['streamflow_mm_day'] = cfs_to_mm_per_day(df_cal['discharge_cfs'], area_km2)
        df_cal['streamflow_cms'] = cfs_to_cms(df_cal['discharge_cfs'])
        
        print(f"  mm/day range: {df_cal['streamflow_mm_day'].min():.2f} – {df_cal['streamflow_mm_day'].max():.2f}")
        
        # Save full version
        df_cal[['time', 'streamflow_mm_day', 'discharge_cfs', 'streamflow_cms']].to_csv(
            os.path.join(args.output_dir, 'obs_usgs_daily_cal.csv'), index=False)
        # Save compatible version (streamflow in m³/s for existing scripts)
        pd.DataFrame({'time': df_cal['time'], 'streamflow': df_cal['streamflow_cms']}).to_csv(
            os.path.join(args.output_dir, 'obs_usgs_daily_cal_cms.csv'), index=False)
        print(f"  Saved: obs_usgs_daily_cal.csv, obs_usgs_daily_cal_cms.csv")
    
    # =========================================
    # DAILY: TEST (Sep-Oct 2024)
    # =========================================
    print()
    print("=" * 60)
    print("DAILY — TEST PERIOD (Sep-Oct 2024)")
    print("=" * 60)
    
    df_test = download_daily_values(SITE_NO, "2024-09-01", "2024-10-31")
    
    if len(df_test) > 0:
        df_test['streamflow_mm_day'] = cfs_to_mm_per_day(df_test['discharge_cfs'], area_km2)
        df_test['streamflow_cms'] = cfs_to_cms(df_test['discharge_cfs'])
        
        print(f"  mm/day range: {df_test['streamflow_mm_day'].min():.2f} – {df_test['streamflow_mm_day'].max():.2f}")
        
        peak = df_test.loc[df_test['discharge_cfs'].idxmax()]
        print(f"\n  HURRICANE HELENE PEAK:")
        print(f"    Date: {peak['time']}")
        print(f"    {peak['discharge_cfs']:.0f} CFS = {peak['streamflow_cms']:.1f} m³/s = {peak['streamflow_mm_day']:.1f} mm/day")
        
        df_test[['time', 'streamflow_mm_day', 'discharge_cfs', 'streamflow_cms']].to_csv(
            os.path.join(args.output_dir, 'obs_usgs_daily_test.csv'), index=False)
        pd.DataFrame({'time': df_test['time'], 'streamflow': df_test['streamflow_cms']}).to_csv(
            os.path.join(args.output_dir, 'obs_usgs_daily_test_cms.csv'), index=False)
        print(f"  Saved: obs_usgs_daily_test.csv, obs_usgs_daily_test_cms.csv")

    # =========================================
    # HOURLY: CALIBRATION (2018-2022) from IV data
    # =========================================
    if not args.skip_hourly:
        print()
        print("=" * 60)
        print("HOURLY — CALIBRATION PERIOD (2018-2022) [from 15-min IV data]")
        print("=" * 60)
        print("  (This will take a few minutes — fetching 5 years of 15-min data)")
        
        df_iv_cal = download_iv_values(SITE_NO, "2018-01-01", "2022-12-31")
        
        if len(df_iv_cal) > 0:
            # Resample 15-min → hourly (mean)
            df_iv_cal = df_iv_cal.set_index('time')
            df_hourly_cal = df_iv_cal.resample('1h').mean().dropna().reset_index()
            
            df_hourly_cal['streamflow_mm_hr'] = cfs_to_mm_per_hr(df_hourly_cal['discharge_cfs'], area_km2)
            df_hourly_cal['streamflow_cms'] = cfs_to_cms(df_hourly_cal['discharge_cfs'])
            
            print(f"  Hourly records: {len(df_hourly_cal)}")
            print(f"  mm/hr range: {df_hourly_cal['streamflow_mm_hr'].min():.4f} – {df_hourly_cal['streamflow_mm_hr'].max():.2f}")
            print(f"  m³/s range:  {df_hourly_cal['streamflow_cms'].min():.2f} – {df_hourly_cal['streamflow_cms'].max():.2f}")
            
            # Save full version
            df_hourly_cal[['time', 'streamflow_mm_hr', 'discharge_cfs', 'streamflow_cms']].to_csv(
                os.path.join(args.output_dir, 'obs_usgs_hourly_cal.csv'), index=False)
            # Compat: 'streamflow' column in m³/s (for run_cfe_calibration.py which expects m³/s and converts)
            pd.DataFrame({'time': df_hourly_cal['time'], 'streamflow': df_hourly_cal['streamflow_cms']}).to_csv(
                os.path.join(args.output_dir, 'obs_usgs_hourly_cal_cms.csv'), index=False)
            print(f"  Saved: obs_usgs_hourly_cal.csv, obs_usgs_hourly_cal_cms.csv")
        
        # =========================================
        # HOURLY: TEST (Sep-Oct 2024) from IV data
        # =========================================
        print()
        print("=" * 60)
        print("HOURLY — TEST PERIOD (Sep-Oct 2024) [from 15-min IV data]")
        print("=" * 60)
        
        df_iv_test = download_iv_values(SITE_NO, "2024-09-01", "2024-10-31")
        
        if len(df_iv_test) > 0:
            df_iv_test = df_iv_test.set_index('time')
            df_hourly_test = df_iv_test.resample('1h').mean().dropna().reset_index()
            
            df_hourly_test['streamflow_mm_hr'] = cfs_to_mm_per_hr(df_hourly_test['discharge_cfs'], area_km2)
            df_hourly_test['streamflow_cms'] = cfs_to_cms(df_hourly_test['discharge_cfs'])
            
            print(f"  Hourly records: {len(df_hourly_test)}")
            print(f"  mm/hr range: {df_hourly_test['streamflow_mm_hr'].min():.4f} – {df_hourly_test['streamflow_mm_hr'].max():.2f}")
            
            peak = df_hourly_test.loc[df_hourly_test['discharge_cfs'].idxmax()]
            print(f"\n  HOURLY HELENE PEAK:")
            print(f"    Time: {peak['time']}")
            print(f"    {peak['discharge_cfs']:.0f} CFS = {peak['streamflow_cms']:.1f} m³/s = {peak['streamflow_mm_hr']:.2f} mm/hr")
            
            df_hourly_test[['time', 'streamflow_mm_hr', 'discharge_cfs', 'streamflow_cms']].to_csv(
                os.path.join(args.output_dir, 'obs_usgs_hourly_test.csv'), index=False)
            pd.DataFrame({'time': df_hourly_test['time'], 'streamflow': df_hourly_test['streamflow_cms']}).to_csv(
                os.path.join(args.output_dir, 'obs_usgs_hourly_test_cms.csv'), index=False)
            print(f"  Saved: obs_usgs_hourly_test.csv, obs_usgs_hourly_test_cms.csv")
    
    # =========================================
    # SANITY CHECK
    # =========================================
    print()
    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    print(f"  Drainage area: {area_km2:.2f} km² ({DRAINAGE_AREA_SQ_MI} sq mi)")
    if len(df_cal) > 0:
        print(f"  Daily cal peak:  {df_cal['streamflow_mm_day'].max():.1f} mm/day")
    if len(df_test) > 0:
        print(f"  Daily test peak: {df_test['streamflow_mm_day'].max():.1f} mm/day")
    print(f"  Suma expects: ~60-70 mm/day max for typical events")
    print()
    print("Files saved to:", args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
