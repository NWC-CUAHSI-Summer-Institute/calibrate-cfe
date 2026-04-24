"""
GPU Run: Hourly CFE calibration with HRRR forcing for gauge 03463300
Identical setup to AORC run (no Snow17, PT-PET computed internally, fixed alpha=1.26).
Forcing starts 2018-07-13, so calibration spinup uses Jul-Dec 2018.

Forcing:  /mnt/disk2/suma_helen_poster/03463300_hrrr_hourly.csv
Obs:      /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv

Time splits:
  Spinup (cal):   2018-07-13 00:00:00 → 2018-12-31 23:00:00
  Calibration:    2019-01-01 00:00:00 → 2022-09-30 23:00:00
  Spinup (test):  2022-10-01 00:00:00 → 2023-09-30 23:00:00
  Test (Helene):  2023-10-01 00:00:00 → 2024-10-31 23:00:00

Data on GPU (svyas@dualearth1):
  Forcing (pre-built): /mnt/disk2/suma_helen_poster/03463300_hrrr_hourly.csv
  Obs:                 /mnt/disk2/suma_helen_poster/03463300_usgs_hourly_2018_2024.csv

  The HRRR forcing CSV was built by concatenating Suma's daily files:
    Raw HRRR source: /mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_hrrr/
    Structure: ~2300 daily subdirs (camels_YYYYMMDD/camels_03463300_agg.csv), 24 rows each
    Columns renamed: TMP→temperature (K→°C), DSWRF→shortwave_radiation,
                     APCP_1hr_acc_fcst→total_precipitation, PRES→pressure (/100),
                     SPFH→specific_humidity, UGRD→wind_u, VGRD→wind_v
    Build command:
      python3 -c "
      import pandas as pd, glob
      base = '/mnt/disk1/usgs_streamflow_allgauges/subdaily_15min/test/output_03463300_hrrr'
      files = sorted(glob.glob(f'{base}/camels_*/camels_03463300_agg.csv'))
      df = pd.concat([pd.read_csv(f) for f in files])
      df = df.rename(columns={'time':'date','TMP':'temperature','DSWRF':'shortwave_radiation',
                               'DLWRF':'longwave_radiation','APCP_1hr_acc_fcst':'total_precipitation',
                               'PRES':'pressure','SPFH':'specific_humidity','UGRD':'wind_u','VGRD':'wind_v'})
      df['temperature'] = df['temperature'] - 273.15
      df['pressure'] = df['pressure'] / 100
      df.to_csv('/mnt/disk2/suma_helen_poster/03463300_hrrr_hourly.csv', index=False)
      "

To reproduce:
  # 1. Clone cfe_py
  git clone https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py.git /mnt/disk2/suma_helen_poster/cfe_py

  # 2. Clone this repo and run
  python3 calibrate_hrrr_gpu.py \
    --base_dir   /mnt/disk2/suma_helen_poster \
    --cfe_dir    /mnt/disk2/suma_helen_poster/cfe_py \
    --config_dir /path/to/calibrate-cfe/results/gage_03463300_hrrr_helene \
    --N 1000

  # To skip calibration and run test period only (using existing best_run JSON):
  python3 calibrate_hrrr_gpu.py \
    --base_dir   /mnt/disk2/suma_helen_poster \
    --cfe_dir    /mnt/disk2/suma_helen_poster/cfe_py \
    --config_dir /path/to/calibrate-cfe/results/gage_03463300_hrrr_helene \
    --test_only

  # config_dir must contain:
  #   cat_03463300_bmi_config_cfe.json  (CFE model config)
  #   CFE_parameter_bounds.json         (DDS search bounds)
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import spotpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


GAUGE_ID        = '03463300'
ALBEDO          = 0.20
ALPHA_PT        = 1.26   # fixed Priestley-Taylor alpha (not calibrated)

TIME_SPLIT = {
    "spinup-for-calibration": {
        "start_datetime": "2018-07-13 00:00:00",
        "end_datetime":   "2018-12-31 23:00:00"
    },
    "calibration": {
        "start_datetime": "2019-01-01 00:00:00",
        "end_datetime":   "2022-09-30 23:00:00"
    },
    "spinup-for-testing": {
        "start_datetime": "2022-10-01 00:00:00",
        "end_datetime":   "2023-09-30 23:00:00"
    },
    "testing": {
        "start_datetime": "2023-10-01 00:00:00",
        "end_datetime":   "2024-10-31 23:00:00"
    }
}

# Set at runtime from CLI args
BASE_DIR        = None
CFE_DIR         = None
CONFIG_DIR      = None
RESULTS_DIR     = None
FORCING_FILE    = None
OBS_FILE        = None
CFE_CONFIG_FILE = None
PARAM_BOUNDS_FILE = None


def priestley_taylor_pet(srad_wm2, T_celsius, alpha=ALPHA_PT):
    """
    Priestley-Taylor PET from hourly shortwave radiation + temperature.
    Returns PET in mm/h.

    srad_wm2  : net downward shortwave radiation (W/m²)
    T_celsius : air temperature (°C)
    alpha     : PT coefficient (1.26 by default)
    """
    T = T_celsius
    delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2  # kPa/°C
    gamma = 0.0638   # kPa/°C  (~806m elevation)
    Rn_mj = np.maximum((1.0 - ALBEDO) * srad_wm2, 0.0) * 0.0036  # W/m² → MJ/m²/h
    G = 0.0
    lam = 2.501 - 0.002361 * T
    pet_mm_h = alpha * (delta / (delta + gamma)) * (Rn_mj - G) / lam
    return np.maximum(pet_mm_h, 0.0)


def prepare_forcing_with_pet(forcing_path):
    """Load HRRR CSV and add potential_evaporation column (mm/h)."""
    df = pd.read_csv(forcing_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['potential_evaporation'] = priestley_taylor_pet(
        df['shortwave_radiation'].values,
        df['temperature'].values
    )
    return df


class Spotpy_setup(object):

    def __init__(self, parameter_bounds, print_all_process=False):
        self.parameter_bounds  = parameter_bounds
        self.print_all_process = print_all_process

        with open(CFE_CONFIG_FILE) as f:
            data_loaded = json.load(f)

        optguess_dict = {
            'bb':            data_loaded['soil_params']['bb'],
            'smcmax':        data_loaded['soil_params']['smcmax'],
            'satdk':         data_loaded['soil_params']['satdk'],
            'slop':          data_loaded['soil_params']['slop'],
            'max_gw_storage': data_loaded['max_gw_storage'],
            'expon':         data_loaded['expon'],
            'Cgw':           data_loaded['Cgw'],
            'K_lf':          data_loaded['K_lf'],
            'K_nash':        data_loaded['K_nash'],
            'scheme':        1
        }

        self.params = [
            spotpy.parameter.Uniform(
                name,
                details['lower_bound'],
                details['upper_bound'],
                optguess=optguess_dict[name]
            )
            for name, details in self.parameter_bounds.items()
        ]

        # Load obs and merge onto HRRR calibration dates (NaN where obs missing/gapped)
        hrrr_df    = pd.read_csv(FORCING_FILE, usecols=['date'])
        hrrr_df['date'] = pd.to_datetime(hrrr_df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        cal_mask   = (hrrr_df['date'] >= TIME_SPLIT['calibration']['start_datetime']) & \
                     (hrrr_df['date'] <= TIME_SPLIT['calibration']['end_datetime'])
        cal_dates  = hrrr_df[cal_mask][['date']].reset_index(drop=True)

        obs_raw    = pd.read_csv(OBS_FILE)
        obs_raw['date'] = pd.to_datetime(obs_raw['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        merged     = cal_dates.merge(obs_raw[['date', 'QObs(mm/h)']], on='date', how='left')

        self.eval_dates = merged['date'].values
        self.obs_data   = merged['QObs(mm/h)'].values   # NaN where obs has gaps

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):

        def custom_load_forcing_file(self_cfe):
            df = prepare_forcing_with_pet(FORCING_FILE)
            self_cfe.forcing_data = df
            self_cfe.forcing_data.rename(columns={"date": "time"}, inplace=True)

        with open(CFE_CONFIG_FILE, 'r') as f:
            cfe_cfg = json.load(f)

        cfe_cfg['forcing_file']          = FORCING_FILE
        cfe_cfg['soil_params']['bb']     = vector['bb']
        cfe_cfg['soil_params']['smcmax'] = vector['smcmax']
        cfe_cfg['soil_params']['satdk']  = vector['satdk']
        cfe_cfg['slop']                  = vector['slop']
        cfe_cfg['max_gw_storage']        = vector['max_gw_storage']
        cfe_cfg['expon']                 = vector['expon']
        cfe_cfg['Cgw']                   = vector['Cgw']
        cfe_cfg['K_lf']                  = vector['K_lf']
        cfe_cfg['K_nash']                = vector['K_nash']
        cfe_cfg['partition_scheme']      = "Schaake" if vector['scheme'] <= 0.5 else "Xinanjiang"

        config_temp = str(CONFIG_DIR / f'cat_{GAUGE_ID}_bmi_config_cfe_temp.json')
        with open(config_temp, 'w') as f:
            json.dump(cfe_cfg, f)

        self.cfemodel = bmi_cfe.BMI_CFE(cfg_file=config_temp)
        self.cfemodel.load_forcing_file = custom_load_forcing_file.__get__(self.cfemodel)
        self.cfemodel.initialize()

        self.df_forcing = prepare_forcing_with_pet(FORCING_FILE)

        # Spinup
        spinup_start = np.where(self.df_forcing['date'] == TIME_SPLIT['spinup-for-calibration']['start_datetime'])[0][0]
        spinup_end   = np.where(self.df_forcing['date'] == TIME_SPLIT['spinup-for-calibration']['end_datetime'])[0][0]
        df_spinup    = self.df_forcing.iloc[spinup_start:spinup_end + 1]

        spinup_outputs      = self.cfemodel.get_output_var_names()
        spinup_output_lists = {o: [] for o in spinup_outputs}

        for precip, pet in zip(df_spinup['total_precipitation'], df_spinup['potential_evaporation']):
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',
                                    precip / 1000)
            self.cfemodel.set_value('water_potential_evaporation_flux',
                                    pet / 1000 / 3600)
            self.cfemodel.update()
            for o in spinup_outputs:
                spinup_output_lists[o].append(self.cfemodel.get_value(o))

        # Calibration period
        cal_start = np.where(self.df_forcing['date'] == TIME_SPLIT['calibration']['start_datetime'])[0][0]
        cal_end   = np.where(self.df_forcing['date'] == TIME_SPLIT['calibration']['end_datetime'])[0][0]
        self.df_forcing = self.df_forcing.iloc[cal_start:cal_end + 1]

        self.outputs      = self.cfemodel.get_output_var_names()
        self.output_lists = {o: [] for o in self.outputs}

        for precip, pet in zip(self.df_forcing['total_precipitation'], self.df_forcing['potential_evaporation']):
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',
                                    precip / 1000)
            self.cfemodel.set_value('water_potential_evaporation_flux',
                                    pet / 1000 / 3600)
            self.cfemodel.update()
            for o in self.outputs:
                self.output_lists[o].append(self.cfemodel.get_value(o))

        self.cfemodel.finalize(print_mass_balance=self.print_all_process)
        self.sim_results = np.array(self.output_lists['land_surface_water__runoff_depth']) * 1000  # m/h → mm/h
        return self.sim_results

    def evaluation(self, evaldates=False):
        if evaldates:
            return [pd.Timestamp(self.eval_dates[i]) for i in range(len(self.eval_dates))]
        return self.obs_data

    def objectivefunction(self, simulation, evaluation, params=None):
        if sum(np.isnan(evaluation)) == len(evaluation):
            return np.nan
        return spotpy.objectivefunctions.kge(
            evaluation[~np.isnan(evaluation)],
            simulation[~np.isnan(evaluation)]
        )


def run_testing_period(best_param_dict, parameter_bounds, output_dir):
    """Run CFE over test period (spinup Oct2022-Sep2023 then test Oct2023-Oct2024)."""

    def custom_load_forcing_file(self_cfe):
        df = prepare_forcing_with_pet(FORCING_FILE)
        self_cfe.forcing_data = df
        self_cfe.forcing_data.rename(columns={"date": "time"}, inplace=True)

    with open(CFE_CONFIG_FILE, 'r') as f:
        cfe_cfg = json.load(f)

    cfe_cfg['forcing_file']          = FORCING_FILE
    cfe_cfg['soil_params']['bb']     = best_param_dict['bb']
    cfe_cfg['soil_params']['smcmax'] = best_param_dict['smcmax']
    cfe_cfg['soil_params']['satdk']  = best_param_dict['satdk']
    cfe_cfg['slop']                  = best_param_dict['slop']
    cfe_cfg['max_gw_storage']        = best_param_dict['max_gw_storage']
    cfe_cfg['expon']                 = best_param_dict['expon']
    cfe_cfg['Cgw']                   = best_param_dict['Cgw']
    cfe_cfg['K_lf']                  = best_param_dict['K_lf']
    cfe_cfg['K_nash']                = best_param_dict['K_nash']
    cfe_cfg['partition_scheme']      = "Schaake" if best_param_dict['scheme'] <= 0.5 else "Xinanjiang"

    config_temp = str(CONFIG_DIR / f'cat_{GAUGE_ID}_bmi_config_cfe_temp_test.json')
    with open(config_temp, 'w') as f:
        json.dump(cfe_cfg, f)

    cfemodel = bmi_cfe.BMI_CFE(cfg_file=config_temp)
    cfemodel.load_forcing_file = custom_load_forcing_file.__get__(cfemodel)
    cfemodel.initialize()

    df_forcing = prepare_forcing_with_pet(FORCING_FILE)

    # Spinup for test
    sp_start = np.where(df_forcing['date'] == TIME_SPLIT['spinup-for-testing']['start_datetime'])[0][0]
    sp_end   = np.where(df_forcing['date'] == TIME_SPLIT['spinup-for-testing']['end_datetime'])[0][0]
    df_spinup = df_forcing.iloc[sp_start:sp_end + 1]

    for precip, pet in zip(df_spinup['total_precipitation'], df_spinup['potential_evaporation']):
        cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip / 1000)
        cfemodel.set_value('water_potential_evaporation_flux', pet / 1000 / 3600)
        cfemodel.update()

    # Test period
    t_start = np.where(df_forcing['date'] == TIME_SPLIT['testing']['start_datetime'])[0][0]
    t_end   = np.where(df_forcing['date'] == TIME_SPLIT['testing']['end_datetime'])[0][0]
    df_test = df_forcing.iloc[t_start:t_end + 1]

    outputs      = cfemodel.get_output_var_names()
    output_lists = {o: [] for o in outputs}

    for precip, pet in zip(df_test['total_precipitation'], df_test['potential_evaporation']):
        cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip / 1000)
        cfemodel.set_value('water_potential_evaporation_flux', pet / 1000 / 3600)
        cfemodel.update()
        for o in outputs:
            output_lists[o].append(cfemodel.get_value(o))

    cfemodel.finalize()

    sim_test = np.array(output_lists['land_surface_water__runoff_depth']) * 1000  # m/h → mm/h
    test_dates = pd.to_datetime(df_test['date'].values)

    # Load USGS obs, merge onto HRRR test dates (NaN for gaps)
    obs_raw = pd.read_csv(OBS_FILE)
    obs_raw['date'] = pd.to_datetime(obs_raw['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    test_dates_df = pd.DataFrame({'date': test_dates.strftime('%Y-%m-%d %H:%M:%S')})
    merged_test   = test_dates_df.merge(obs_raw[['date', 'QObs(mm/h)']], on='date', how='left')
    obs_test  = merged_test['QObs(mm/h)'].values
    obs_dates = merged_test['date'].values

    if obs_test is not None and len(obs_test) == len(sim_test):
        mask = ~np.isnan(obs_test)
        kge_test = spotpy.objectivefunctions.kge(obs_test[mask], sim_test[mask])
        nse_test = spotpy.objectivefunctions.nashsutcliffe(obs_test[mask], sim_test[mask])
        print(f"Test KGE (Oct2023-Oct2024): {kge_test:.4f}")
        print(f"Test NSE (Oct2023-Oct2024): {nse_test:.4f}")

        df_out = pd.DataFrame({
            'date':        test_dates.strftime('%Y-%m-%d %H:%M:%S'),
            'sim_mm_h':    sim_test,
            'obs_mm_h':    obs_test,
            'precip_mm_h': df_test['total_precipitation'].values
        })
        df_out.to_csv(os.path.join(output_dir, f'{GAUGE_ID}_test_results_hrrr.csv'), index=False)

        # Helene zoom: Sep 20 - Oct 5 2024
        helene_mask = (test_dates >= pd.Timestamp('2024-09-20')) & (test_dates <= pd.Timestamp('2024-10-05'))
        sim_h   = sim_test[helene_mask]
        obs_h   = obs_test[np.where(helene_mask)[0]] if len(obs_test) == len(sim_test) else sim_test[helene_mask]
        prcp_h  = df_test['total_precipitation'].values[helene_mask]
        dates_h = test_dates[helene_mask]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

        ax1.plot(test_dates, sim_test, 'tomato', lw=1.5, label='simulated')
        ax1.plot(test_dates, obs_test, 'k', lw=1, label='observed (USGS)')
        ax1.set_ylabel('Discharge (mm/h)')
        ax1.set_ylim([0, max(np.nanmax(obs_test), np.nanmax(sim_test)) * 1.2])
        ax1.legend()
        ax1.set_title(f'Test Period (Oct 2023 – Oct 2024) | KGE={kge_test:.4f} | NSE={nse_test:.4f} | Gauge {GAUGE_ID} | HRRR')
        ax2_twin = ax1.twinx()
        ax2_twin.plot(test_dates, df_test['total_precipitation'].values, 'tab:blue', lw=0.8, alpha=0.5, label='precip')
        ax2_twin.set_ylim([100, 0])
        ax2_twin.set_ylabel('Precip (mm/h)')

        ax2.plot(dates_h, sim_h, 'tomato', lw=2, label='simulated')
        ax2.plot(dates_h, obs_h, 'k', lw=1.5, label='observed')
        ax2.set_ylabel('Discharge (mm/h)')
        ax2_r = ax2.twinx()
        ax2_r.bar(dates_h, prcp_h, color='tab:blue', alpha=0.4, label='precip', width=0.04)
        ax2_r.set_ylim([50, 0])
        ax2_r.set_ylabel('Precip (mm/h)')
        ax2.set_title('Helene Zoom: Sep 20 – Oct 5, 2024')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{GAUGE_ID}_test_helene_hrrr.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Plots saved to {output_dir}")

    return sim_test


def main():
    global BASE_DIR, CFE_DIR, CONFIG_DIR, RESULTS_DIR
    global FORCING_FILE, OBS_FILE, CFE_CONFIG_FILE, PARAM_BOUNDS_FILE
    global bmi_cfe, cfe

    parser = argparse.ArgumentParser(description='CFE calibration with HRRR forcing, gauge 03463300')
    parser.add_argument('--base_dir',   required=True, help='Base directory with forcing and obs CSVs')
    parser.add_argument('--cfe_dir',    required=True, help='Path to cfe_py directory (bmi_cfe.py, cfe.py)')
    parser.add_argument('--config_dir', required=True, help='Path to directory with CFE config JSON and parameter bounds')
    parser.add_argument('--N',          type=int, default=1000, help='Number of DDS iterations')
    parser.add_argument('--test_only',  action='store_true', help='Skip calibration, run test period only')
    args = parser.parse_args()

    BASE_DIR   = Path(args.base_dir)
    CFE_DIR    = Path(args.cfe_dir)
    CONFIG_DIR = Path(args.config_dir)
    RESULTS_DIR = CONFIG_DIR / 'results'

    FORCING_FILE      = str(BASE_DIR / f'{GAUGE_ID}_hrrr_hourly.csv')
    OBS_FILE          = str(BASE_DIR / f'{GAUGE_ID}_usgs_hourly_2018_2024.csv')
    CFE_CONFIG_FILE   = str(CONFIG_DIR / f'cat_{GAUGE_ID}_bmi_config_cfe.json')
    PARAM_BOUNDS_FILE = str(CONFIG_DIR / 'CFE_parameter_bounds.json')

    sys.path.insert(0, str(CFE_DIR))
    import bmi_cfe as _bmi_cfe
    import cfe as _cfe
    bmi_cfe = _bmi_cfe
    cfe     = _cfe

    for d in [RESULTS_DIR, RESULTS_DIR / 'raw', RESULTS_DIR / 'images', RESULTS_DIR / 'best_runs']:
        os.makedirs(d, exist_ok=True)

    with open(PARAM_BOUNDS_FILE) as f:
        parameter_bounds = json.load(f)

    best_run_file = RESULTS_DIR / 'best_runs' / f'{GAUGE_ID}_best_run_hrrr.json'

    if not args.test_only:
        print(f"Starting calibration: N={args.N}, gauge={GAUGE_ID}, forcing=HRRR")
        print(f"Calibration period: {TIME_SPLIT['calibration']['start_datetime']} → {TIME_SPLIT['calibration']['end_datetime']}")

        calibration_instance = Spotpy_setup(parameter_bounds=parameter_bounds, print_all_process=False)

        np.random.seed(0)
        sampler = spotpy.algorithms.dds(calibration_instance, dbname='raw_result_hrrr', dbformat='ram')
        sampler.sample(args.N)
        results = sampler.getdata()

        np.save(str(RESULTS_DIR / 'raw' / f'{GAUGE_ID}_all_results_dds_hrrr.npy'), results)

        best_params     = spotpy.analyser.get_best_parameterset(results)
        best_param_dict = {name: value for name, value in zip(parameter_bounds.keys(), best_params[0])}
        obj_values      = results['like1']
        best_obj        = np.nanmax(obj_values)
        best_idx        = np.where(obj_values == best_obj)[0][0]
        best_sim        = np.array([v for v in spotpy.analyser.get_modelruns(results[best_idx])])

        print(f"\nCalibration complete!")
        print(f"Best KGE: {best_obj:.4f}")
        print(f"Best params: {best_param_dict}")

        best_run = {
            "best_parameters": best_param_dict,
            "best_kge": float(best_obj),
            "best_sim": list(best_sim),
        }
        with open(best_run_file, 'w') as f:
            json.dump(best_run, f, indent=4)

        spotpy.analyser.plot_parametertrace(
            results,
            fig_name=str(RESULTS_DIR / 'images' / f'{GAUGE_ID}_param_trace_hrrr.png'))

        dates = calibration_instance.evaluation(evaldates=True)

        fig, ax1 = plt.subplots(figsize=(20, 8))
        ax1.plot(dates[:8760], best_sim[:8760], 'tomato', lw=2, label='simulated')
        ax1.plot(dates[:8760], calibration_instance.obs_data[:8760], 'k', lw=1, label='observed')
        ax1.set_ylabel('Discharge (mm/h)')
        ax1.set_ylim([0, 2])
        ax2 = ax1.twinx()
        ax2.plot(dates[:8760], calibration_instance.df_forcing['total_precipitation'][:8760],
                 'tab:blue', alpha=0.6, label='precip')
        ax2.set_ylim([50, 0])
        ax2.set_ylabel('Precip (mm/h)')
        ax1.legend(loc='upper right')
        plt.title(f'Calibration (first year 2019) | KGE={best_obj:.4f} | {GAUGE_ID} | HRRR')
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / 'images' / f'{GAUGE_ID}_comparison_hrrr.png'), bbox_inches='tight', dpi=150)
        plt.close()

    if best_run_file.exists():
        with open(best_run_file) as f:
            best_run = json.load(f)
        print("\nRunning test period (Helene)...")
        run_testing_period(
            best_run['best_parameters'],
            parameter_bounds,
            str(RESULTS_DIR / 'images')
        )
    else:
        print("No best_run file found. Run calibration first (without --test_only).")


if __name__ == '__main__':
    main()
