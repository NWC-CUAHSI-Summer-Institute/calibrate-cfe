############################################
# Run 4: Ryoko's hourly CFE setup + Snow17 + Priestley-Taylor PET
# Based on 3-CFE_Calibration_Loop_1.py
# Changes from Run 3:
#   1. Snow17 (daily) processes NLDAS precip+temp → effective precip → disaggregated to hourly
#   2. PT-PET computed hourly from NLDAS shortwave radiation + temperature
#      (replaces pre-computed NLDAS potential_evaporation column)
#   3. Added snow params + alpha_PT to calibration (20 params total)
############################################

from omegaconf import DictConfig, OmegaConf
import hydra
import spotpy
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# CFE model path
from pathlib import Path
cfe_py_path = Path(__file__).resolve().parent.parent / 'cfe_py'
sys.path.append(str(cfe_py_path))
import bmi_cfe
import cfe

# Snow17 + PET
tonic_path = Path(__file__).resolve().parent.parent / 'tonic'
sys.path.insert(0, str(tonic_path))
from tonic.models.snow17 import snow17 as snow17_module

pet_path = Path(__file__).resolve().parent
sys.path.insert(0, str(pet_path))
import pet as pet_module

# Basin constants for gage 03463300
LAT    = 35.83138889
ELEV   = 806.0
ALBEDO = 0.20

# ── Snow17 parameter bounds (same as Abhinav's daily setup) ──────────────────
SNOW_PARAM_BOUNDS = {
    'alpha_PT': {'lower_bound': 0.10,  'upper_bound': 1.74,  'optguess': 1.26},
    'scf':      {'lower_bound': 0.80,  'upper_bound': 1.20,  'optguess': 1.0},
    'uadj':     {'lower_bound': 0.01,  'upper_bound': 0.40,  'optguess': 0.04},
    'mbase':    {'lower_bound': 0.80,  'upper_bound': 2.00,  'optguess': 1.0},
    'mfmax':    {'lower_bound': 0.80,  'upper_bound': 3.00,  'optguess': 1.05},
    'mfmin':    {'lower_bound': 0.01,  'upper_bound': 0.79,  'optguess': 0.60},
    'tipm':     {'lower_bound': 0.01,  'upper_bound': 1.00,  'optguess': 0.1},
    'nmf':      {'lower_bound': 0.04,  'upper_bound': 0.40,  'optguess': 0.15},
    'plwhc':    {'lower_bound': 0.01,  'upper_bound': 0.40,  'optguess': 0.04},
    'pxtemp':   {'lower_bound': -1.00, 'upper_bound': 1.00,  'optguess': 1.0},
}


def compute_hourly_pet(df_forcing, alpha_PT):
    """Priestley-Taylor PET from hourly NLDAS shortwave radiation + temperature.
    Returns PET in mm/h."""
    T    = df_forcing['temperature'].values          # °C
    srad = df_forcing['shortwave_radiation'].values  # W/m²
    # Simple net radiation: daytime only (srad>0), no longwave correction for hourly
    Rn   = np.maximum((1.0 - ALBEDO) * srad, 0.0)  # W/m²
    G    = np.zeros(len(T))

    # Priestley-Taylor (returns mm/day) → convert to mm/h
    pet_mm_day = pet_module.priestley_taylor_fixed_alpha(Rn, G, T, alpha=alpha_PT)
    pet_mm_h   = pet_mm_day / 24.0
    return pet_mm_h


def run_snow17_disaggregate(df_forcing, snow_params):
    """
    Aggregate NLDAS hourly → daily, run Snow17, disaggregate back to hourly.
    Returns hourly effective precip (mm/h) after snowmelt processing.
    """
    df = df_forcing.copy()
    df['date_only'] = pd.to_datetime(df['date']).dt.date

    # Aggregate to daily
    daily = df.groupby('date_only').agg(
        prcp_mm_day=('total_precipitation', 'sum'),   # mm/day
        tavg_C=('temperature', 'mean'),               # °C
        date_dt=('date', 'first')
    ).reset_index()

    dates_daily = pd.to_datetime(daily['date_dt']).dt.to_pydatetime()

    # Run Snow17 at daily timestep (dt=24)
    swe, outflow = snow17_module.snow17(
        time      = dates_daily,
        prec      = daily['prcp_mm_day'].values,   # mm/day
        tair      = daily['tavg_C'].values,         # °C
        lat       = LAT,
        elevation = ELEV,
        dt        = 24,
        scf       = snow_params['scf'],
        rvs       = 1,
        uadj      = snow_params['uadj'],
        mbase     = snow_params['mbase'],
        mfmax     = snow_params['mfmax'],
        mfmin     = snow_params['mfmin'],
        tipm      = snow_params['tipm'],
        nmf       = snow_params['nmf'],
        plwhc     = snow_params['plwhc'],
        pxtemp    = snow_params['pxtemp'],
        pxtemp1   = -1.0,
        pxtemp2   = 3.0,
    )

    # Map daily outflow back to hourly (divide evenly across 24 hours)
    daily['outflow_mm_day'] = outflow
    daily['date_only'] = daily['date_only']
    df['date_only2'] = df['date_only']
    merged = df.merge(daily[['date_only', 'outflow_mm_day']], left_on='date_only2', right_on='date_only', how='left')
    hourly_snow_outflow_mm_h = merged['outflow_mm_day'].values / 24.0  # mm/h

    return hourly_snow_outflow_mm_h


# ─────────────────────────────────────────────────────────────────────────────
class Spotpy_setup(object):

    def __init__(self, config_dir, obs_file_path, gauge_id, time_split, parameter_bounds, print_all_process):

        self.config_dir        = config_dir
        self.time_split        = time_split
        self.obs_file_path     = obs_file_path
        self.gauge_id          = gauge_id
        self.parameter_bounds  = parameter_bounds
        self.print_all_process = print_all_process

        # Load Luciana's initial CFE params as optguess
        config_filename = f'cat_{self.gauge_id}_bmi_config_cfe.json'
        with open(os.path.join(self.config_dir, config_filename)) as f:
            data_loaded = json.load(f)

        cfe_optguess = {
            'bb':            data_loaded['soil_params']['bb'],
            'smcmax':        data_loaded['soil_params']['smcmax'],
            'satdk':         data_loaded['soil_params']['satdk'],
            'slop':          data_loaded['soil_params']['slop'],
            'max_gw_storage':data_loaded['max_gw_storage'],
            'expon':         data_loaded['expon'],
            'Cgw':           data_loaded['Cgw'],
            'K_lf':          data_loaded['K_lf'],
            'K_nash':        data_loaded['K_nash'],
            'scheme':        1,
        }

        # Merge CFE + snow optguess
        all_optguess = {**cfe_optguess,
                        **{k: v['optguess'] for k, v in SNOW_PARAM_BOUNDS.items()}}

        # Build SPOTPY parameter list (CFE bounds from file + snow bounds hardcoded)
        self.params = []
        for name, details in self.parameter_bounds.items():
            self.params.append(spotpy.parameter.Uniform(
                name, details['lower_bound'], details['upper_bound'],
                optguess=all_optguess[name]))
        for name, details in SNOW_PARAM_BOUNDS.items():
            self.params.append(spotpy.parameter.Uniform(
                name, details['lower_bound'], details['upper_bound'],
                optguess=all_optguess[name]))

        # Load USGS obs, slice to calibration period
        obs_data0        = pd.read_csv(self.obs_file_path)
        self.obs_data    = obs_data0['QObs(mm/h)'].values
        self.eval_dates  = obs_data0['date'].values
        cal_start        = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])[0][0]
        cal_end          = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])[0][0]
        self.eval_dates  = self.eval_dates[cal_start:cal_end + 1]
        self.obs_data    = self.obs_data[cal_start:cal_end + 1]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):

        def custom_load_forcing_file(self):
            self.forcing_data = pd.read_csv(self.forcing_file)
            self.forcing_data.rename(columns={"date": "time"}, inplace=True)

        # ── Extract CFE params ────────────────────────────────────────────────
        config_filename = f'cat_{self.gauge_id}_bmi_config_cfe.json'
        with open(os.path.join(self.config_dir, config_filename), 'r') as f:
            self.cfe_cfg = json.load(f)

        self.cfe_cfg["soil_params"]['bb']     = vector['bb']
        self.cfe_cfg["soil_params"]['smcmax'] = vector['smcmax']
        self.cfe_cfg["soil_params"]['satdk']  = vector['satdk']
        self.cfe_cfg['slop']                  = vector['slop']
        self.cfe_cfg['max_gw_storage']        = vector['max_gw_storage']
        self.cfe_cfg['expon']                 = vector['expon']
        self.cfe_cfg['Cgw']                   = vector['Cgw']
        self.cfe_cfg['K_lf']                  = vector['K_lf']
        self.cfe_cfg['K_nash']                = vector['K_nash']
        self.cfe_cfg['partition_scheme']      = "Schaake" if vector['scheme'] <= 0.5 else "Xinanjiang"

        # ── Extract snow + PET params ─────────────────────────────────────────
        snow_params = {k: vector[k] for k in SNOW_PARAM_BOUNDS if k != 'alpha_PT'}
        alpha_PT    = vector['alpha_PT']

        config_temp = f'cat_{self.gauge_id}_bmi_config_cfe_temp.json'
        with open(os.path.join(self.config_dir, config_temp), 'w') as f:
            json.dump(self.cfe_cfg, f)

        # ── Initialize CFE model ──────────────────────────────────────────────
        self.cfemodel = bmi_cfe.BMI_CFE(cfg_file=os.path.join(self.config_dir, config_temp))
        self.cfemodel.load_forcing_file = custom_load_forcing_file.__get__(self.cfemodel)
        self.cfemodel.initialize()

        with open(self.cfemodel.forcing_file, 'r') as f:
            self.df_forcing = pd.read_csv(f)

        # ── Pre-compute Snow17 outflow + PT-PET for full forcing period ───────
        hourly_snow_mm_h = run_snow17_disaggregate(self.df_forcing, snow_params)
        hourly_pet_mm_h  = compute_hourly_pet(self.df_forcing, alpha_PT)
        self.df_forcing['snow_outflow_mm_h'] = hourly_snow_mm_h
        self.df_forcing['pet_mm_h']          = hourly_pet_mm_h

        # ── Spinup period ─────────────────────────────────────────────────────
        spinup_start = np.where(self.df_forcing['date'] == self.time_split['spinup-for-calibration']['start_datetime'])[0][0]
        spinup_end   = np.where(self.df_forcing['date'] == self.time_split['spinup-for-calibration']['end_datetime'])[0][0]
        df_spinup    = self.df_forcing.iloc[spinup_start:spinup_end + 1]

        spinup_outputs      = self.cfemodel.get_output_var_names()
        spinup_output_lists = {o: [] for o in spinup_outputs}

        for snow_precip, pet in zip(df_spinup['snow_outflow_mm_h'], df_spinup['pet_mm_h']):
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',
                                    snow_precip / 1000)       # mm/h → m/h
            self.cfemodel.set_value('water_potential_evaporation_flux',
                                    pet / 1000 / 3600)        # mm/h → m/s
            self.cfemodel.update()
            for o in spinup_outputs:
                spinup_output_lists[o].append(self.cfemodel.get_value(o))

        # ── Calibration period ────────────────────────────────────────────────
        cal_start = np.where(self.df_forcing['date'] == self.time_split['calibration']['start_datetime'])[0][0]
        cal_end   = np.where(self.df_forcing['date'].values == self.time_split['calibration']['end_datetime'])[0][0]
        self.df_forcing = self.df_forcing.iloc[cal_start:cal_end + 1]

        self.outputs      = self.cfemodel.get_output_var_names()
        self.output_lists = {o: [] for o in self.outputs}

        for snow_precip, pet in zip(self.df_forcing['snow_outflow_mm_h'], self.df_forcing['pet_mm_h']):
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux',
                                    snow_precip / 1000)
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
            simulation[~np.isnan(evaluation)])


# ─────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    N                    = cfg.calib_variables.N
    max_nbasin_per_loop  = cfg.calib_variables.max_nbasin_per_loop
    print_all_process    = cfg.calib_variables.print_all_process
    config_dir           = cfg.io_dir.config_dir
    obs_dir              = cfg.io_dir.usgs_streamflow_dir
    results_path         = cfg.io_dir.results_dir
    parameter_bound_file = cfg.model_settings.parameter_bound_file
    time_split_file      = cfg.model_settings.time_split_file
    basin_filename       = cfg.model_settings.basin_file
    missing_data_filename= cfg.model_settings.missing_data_file

    with open(basin_filename) as f:
        basin_list_str = [l.strip() for l in f.readlines()]
    with open(missing_data_filename) as f:
        missing_data_list = [l.strip() for l in f.readlines()]
    with open(time_split_file) as f:
        time_split = json.load(f)
    with open(parameter_bound_file) as f:
        parameter_bounds = json.load(f)

    # Create output dirs
    raw_results_path = os.path.join(results_path, 'raw_run4')
    png_dir          = os.path.join(results_path, 'images_run4')
    best_run_dir     = os.path.join(results_path, 'best_runs_run4')
    for d in [raw_results_path, png_dir, best_run_dir]:
        os.makedirs(d, exist_ok=True)

    for i in range(max_nbasin_per_loop):
        g_str = cfg.basin_id

        if g_str in missing_data_list:
            print(f"Skipping {g_str} — missing data.")
            continue

        print(f"Processing basin: {g_str}")
        obs_file_path = os.path.join(obs_dir, f'{g_str}-usgs-hourly.csv')

        calibration_instance = Spotpy_setup(
            config_dir        = config_dir,
            obs_file_path     = obs_file_path,
            time_split        = time_split,
            gauge_id          = g_str,
            parameter_bounds  = parameter_bounds,
            print_all_process = print_all_process,
        )

        np.random.seed(0)
        sampler = spotpy.algorithms.dds(calibration_instance, dbname='raw_result_run4', dbformat='ram')
        sampler.sample(N)
        results = sampler.getdata()

        np.save(os.path.join(raw_results_path, f'{g_str}_all_results_dds_run4.npy'), results)

        best_params   = spotpy.analyser.get_best_parameterset(results)
        all_param_names = list(parameter_bounds.keys()) + list(SNOW_PARAM_BOUNDS.keys())
        best_param_dict = {name: value for name, value in zip(all_param_names, best_params[0])}
        obj_values    = results['like1']
        best_obj      = np.nanmax(obj_values)
        best_idx      = np.where(obj_values == best_obj)[0][0]
        best_sim      = np.array([v for v in spotpy.analyser.get_modelruns(results[best_idx])])

        best_run = {
            "best parameters":         best_param_dict,
            "best objective values":    best_obj,
            "best simulation results":  list(best_sim),
        }
        with open(os.path.join(best_run_dir, f'{g_str}_best_run_run4.json'), 'w') as f:
            json.dump(best_run, f, indent=4)

        # ── Plots ─────────────────────────────────────────────────────────────
        spotpy.analyser.plot_parametertrace(
            results, fig_name=os.path.join(png_dir, f'{g_str}_param_trace_run4.png'))

        plt.figure(figsize=(18, 6))
        plt.plot(np.arange(len(obj_values)), obj_values)
        plt.ylim([-2, 1])
        plt.xlabel('Iterations'); plt.ylabel('KGE')
        plt.title(f'KGE Trace — Run 4 (Snow17 + PT-PET) [{g_str}]')
        plt.savefig(os.path.join(png_dir, f'{g_str}_obj_trace_run4.png'), bbox_inches='tight')
        plt.close()

        dates    = calibration_instance.evaluation(evaldates=True)
        fig, ax1 = plt.subplots(figsize=(18, 8))
        ax1.plot(dates[:8760], best_sim[:8760], 'tomato', lw=2, label='sim (Snow+PET)')
        ax1.plot(dates[:8760], calibration_instance.obs_data[:8760], 'k', label='obs')
        ax1.set_ylabel('Discharge (mm/h)'); ax1.set_ylim([0, 2])
        ax2 = ax1.twinx()
        ax2.plot(dates[:8760], calibration_instance.df_forcing['total_precipitation'][:8760],
                 'tab:blue', label='precip')
        ax2.set_ylim([50, 0]); ax2.set_ylabel('Precipitation (mm/h)')
        plt.legend(); plt.title(f'Run 4 — {g_str}  KGE={best_obj:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(png_dir, f'{g_str}_comparison_run4.png'), bbox_inches='tight')
        plt.close()

        print(f"\nRun 4 best KGE: {best_obj:.4f}")
        print(f"Best params: {best_param_dict}")


if __name__ == "__main__":
    main()
