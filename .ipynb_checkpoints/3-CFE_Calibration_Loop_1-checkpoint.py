
############################################
# This code runs calibration (looping through max_nbasin_per_loop basins at a time) #
############################################

# Originally written by 2022 team
# Qiyue Liu (University of Illinois at Urbana-Champaign; qiyuel3@illinois.edu) in 2022 SI
# Modified by Ryoko Araki (San Diego State University & UCSB, raraki8159@sdsu.edu) in 2023 SI 


from omegaconf import DictConfig, OmegaConf
import hydra

# import package
import spotpy

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# import the cfe model
sys.path.append(r'../cfe_py')
import bmi_cfe
import cfe


# Folder structure
# project_folder/
# ├─ data/
# ├─ cfe_py/
# ├─ calibrate_cfe/
# │  ├─ configs/
# │  ├─ results/


# ----------------------------------- Setup the Spotpy Class ----------------------------------- #
class Spotpy_setup(object): 

    def __init__(self, config_dir, obs_file_path, gauge_id, time_split, parameter_bounds, print_all_process): 
        
        self.config_dir = config_dir
        self.time_split = time_split
        self.obs_file_path = obs_file_path
        self.gauge_id = gauge_id
        self.parameter_bounds = parameter_bounds
        self.print_all_process = print_all_process
        
        # load original model and soil parameters for optguess from Luciana Cunha
        # locate config file
        config_filename = f'cat_{self.gauge_id}_bmi_config_cfe.json'

        with open(os.path.join(self.config_dir, config_filename)) as data_file:
            data_loaded = json.load(data_file)

        # TODO: add this to Yeham's code #####
        bb_optguess = data_loaded['soil_params']['bb']
        smcmax_optguess = data_loaded['soil_params']["smcmax"]
        satdk_optguess = data_loaded["soil_params"]["satdk"]
        slop_optguess = data_loaded["soil_params"]["slop"]
        maxgw_optguess = data_loaded["max_gw_storage"]
        expon_optguess = data_loaded["expon"]
        cgw_optguess = data_loaded["Cgw"]
        klf_optguess = data_loaded["K_lf"]
        knash_optguess = data_loaded["K_nash"]
        
        optguess_dict = {
            'bb': bb_optguess,
            'smcmax': smcmax_optguess,
            'satdk': satdk_optguess,
            'slop': slop_optguess,
            'max_gw_storage': maxgw_optguess,
            'expon': expon_optguess,
            'Cgw': cgw_optguess,
            'K_lf': klf_optguess,
            'K_nash': knash_optguess,
            'scheme': 1
        }

        # setup calibration parameters
        self.params = [
            spotpy.parameter.Uniform(name, details['lower_bound'], details['upper_bound'], optguess=optguess_dict[name])
            for name, details in self.parameter_bounds.items()
        ]
        
        ########################################
    
        # Load test comparison data (streamflow) from usgs data
        obs_data0 = pd.read_csv(self.obs_file_path)
        # self.obs_data = obs_data0['QObs_CAMELS(mm/h)'].values # This was daily data
        self.obs_data = obs_data0['QObs(mm/h)'].values # Use hourly data instead
        self.eval_dates = obs_data0['date'].values

        # define calibration period for usgs streamflow obs.
        cal_start_idx_usgs = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])
        cal_end_idx_usgs = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])
        self.eval_dates = self.eval_dates[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        self.obs_data = self.obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        
        if print_all_process: 
            print('###--------- usgs start date: ' + self.eval_dates[0] + '.---------')
            print('###--------- usgs end date: ' + self.eval_dates[-1] + '.---------')
            print('###---------- obs data length: ' +  str(len(self.obs_data)) + '.---------')

    def parameters(self):
        return spotpy.parameter.generate(self.params)
        
    def simulation(self, vector):
        
        # Setup custom method in BMI-CFE
        def custom_load_forcing_file(self):
            self.forcing_data = pd.read_csv(self.forcing_file)
            # Column name change to accomodate NLDAS forcing by https://zenodo.org/record/4072701
            self.forcing_data.rename(columns={"date": "time"}, inplace=True)
            pass
        
        if self.print_all_process: 
            print("### ------------ A NEW ITERATION OF CALIBRATION ------------ ###")

        # --------------------------------------- Parameter preparation --------------------------------------- # 
        # Get randomly generated parameter in Spotpy format
        self.generated_param = vector
        
        # Read the template config file 
        config_filename = f'cat_{self.gauge_id}_bmi_config_cfe.json'
        with open(os.path.join(self.config_dir, config_filename), 'r') as file:
            self.cfe_cfg = json.load(file)

        self.cfe_cfg["soil_params"]['bb'] = vector['bb']
        self.cfe_cfg["soil_params"]['smcmax'] = vector['smcmax']
        self.cfe_cfg["soil_params"]['satdk'] = vector['satdk']
        self.cfe_cfg['slop'] = vector['slop']
        self.cfe_cfg['max_gw_storage'] = vector['max_gw_storage']
        self.cfe_cfg['expon'] = vector['expon']
        self.cfe_cfg['Cgw'] = vector['Cgw']
        self.cfe_cfg['K_lf'] = vector['K_lf']
        self.cfe_cfg['K_nash'] = vector['K_nash']
        if vector['scheme'] <= 0.5:
            self.cfe_cfg['partition_scheme'] = "Schaake"
        else:
            self.cfe_cfg['partition_scheme'] = "Xinanjiang"
            
        # Dump optguess parameter into temporary config file
        config_temp_filename = f'cat_{self.gauge_id}_bmi_config_cfe_temp.json'
        with open(os.path.join(self.config_dir, config_temp_filename), 'w') as out_file:
            json.dump(self.cfe_cfg, out_file)

        if self.print_all_process: 
            print(f"###----------- parameters generated: {self.generated_param}.--------###")

        # --------------------------------------- Set-up CFE model --------------------------------------- # 
        # Set up CFE model
        self.cfemodel = bmi_cfe.BMI_CFE(cfg_file=os.path.join(self.config_dir, config_temp_filename))
        self.cfemodel.load_forcing_file = custom_load_forcing_file.__get__(self.cfemodel)
        
        if self.print_all_process: 
            print('###--------model succesfully setup----------###')
        
        self.cfemodel.initialize()
        
        if self.print_all_process: 
            print('###--------model succesfully initialized----------###')

        with open(self.cfemodel.forcing_file, 'r') as f:
            self.df_forcing = pd.read_csv(f)
            
        if self.print_all_process:
            print(f"###----- forcing_file loaded:{self.cfemodel.forcing_file}. -----###")

        # --------------------------------------- Run Spin-up Period --------------------------------------- # 
        
        # define the spin up period
        spinup_start_idx_nldas = np.where(self.df_forcing['date'] == self.time_split['spinup-for-calibration']['start_datetime'])
        spinup_end_idx_nldas = np.where(self.df_forcing['date'] == self.time_split['spinup-for-calibration']['end_datetime'])
        self.df_forcing_spinup = self.df_forcing.iloc[spinup_start_idx_nldas[0][0]:spinup_end_idx_nldas[0][0]+1,:]

        if self.print_all_process: 
            print('###-------- Model Spinning Up... ----------###')
            print('###------spinup start date: ' + self.df_forcing_spinup['date'].values[0]+ "-----###")
            print('###------spinup end date: ' + self.df_forcing_spinup['date'].values[-1]+"-----###")
            
        # Initialize
        self.spinup_outputs=self.cfemodel.get_output_var_names()
        self.spinup_output_lists = {output:[] for output in self.spinup_outputs}

        # run the model for the spin-up period
        for precip, pet in zip(self.df_forcing_spinup['total_precipitation'],self.df_forcing_spinup['potential_evaporation']):
            #print(f"###----------loaded precip, pet: {precip},{pet}.------------###")
            #sys.exit(1)

            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip/1000)   # kg/m2/h = mm/h -> m/h
            self.cfemodel.set_value('water_potential_evaporation_flux', pet/1000/3600) # kg/m2/h = mm/h -> m/s
            self.cfemodel.update()
        
            for spinup_output in self.spinup_outputs:
                self.spinup_output_lists[spinup_output].append(self.cfemodel.get_value(spinup_output))

        # --------------------------------------- Run Calibration Period --------------------------------------- # 
        # define the calibration period for nldas forcing and usgs streamflow obs.
        cal_start_idx_nldas = np.where(self.df_forcing['date'] ==  self.time_split['calibration']['start_datetime'])
        cal_end_idx_nldas = np.where(self.df_forcing['date'].values ==  self.time_split['calibration']['end_datetime'])
        self.df_forcing = self.df_forcing.iloc[cal_start_idx_nldas[0][0]:cal_end_idx_nldas[0][0]+1,:]
        
        if self.print_all_process:
            print('###-------- Model Running for the Calibration Period ----------###')
            print('###------nldas start date: ' + self.df_forcing['date'].values[0]+ "-----###")
            print('###------nldas end date: ' + self.df_forcing['date'].values[-1]+"-----###")
            print('###--------- usgs start date: ' + self.eval_dates[0] + '.---------')
            print('###--------- usgs end date: ' + self.eval_dates[-1] + '.---------')

            print('###----- nldas forcing data length: ' +  str(len(self.df_forcing['date'].values))+"------###")
            print('###---------- obs data length: ' +  str(len(self.obs_data)) + '.---------')

        # Initialize calibration period 
        self.outputs=self.cfemodel.get_output_var_names()
        self.output_lists = {output:[] for output in self.outputs}

        # Model run
        for precip, pet in zip(self.df_forcing['total_precipitation'],self.df_forcing['potential_evaporation']):
            #print(f"###----------loaded precip, pet: {precip},{pet}.------------###")
            #sys.exit(1)
            
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip/1000)   # kg/m2/h = mm/h -> m/h
            self.cfemodel.set_value('water_potential_evaporation_flux', pet/1000/3600) # kg/m2/h = mm/h -> m/s
            self.cfemodel.update()
        
            for output in self.outputs:
                self.output_lists[output].append(self.cfemodel.get_value(output))

        self.cfemodel.finalize(print_mass_balance=self.print_all_process)

        if self.print_all_process: 
            print(f'###----------output length: {len(self.output_lists["land_surface_water__runoff_depth"])}.---------###')

        self.sim_results = np.array(self.output_lists['land_surface_water__runoff_depth']) * 1000      # m/h -> mm/h
        return self.sim_results 

    def evaluation(self, evaldates=False):
        if evaldates:
            self.eval_dates_output = [pd.Timestamp(self.eval_dates[i]) for i in range(len(self.eval_dates))]
            return self.eval_dates_output
        else:
            print(f"###--------- double check - length of obs_data: {len(self.obs_data)}. -----------###")
            return self.obs_data

    def objectivefunction(self, simulation, evaluation, params=None):
        self.obj_function = spotpy.objectivefunctions.kge(evaluation[~np.isnan(evaluation)], simulation[~np.isnan(evaluation)])
        return self.obj_function


# ----------------------------------- Loop for Calibration ----------------------------------- #
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg): 
    # Read config via hydra
    print(OmegaConf.to_yaml(cfg))
    
    N = cfg.calib_variables.N

    # Number of basin to run for a loop
    max_nbasin_per_loop = cfg.calib_variables.max_nbasin_per_loop

    # If you want to print3 everything
    print_all_process = cfg.calib_variables.print_all_process

    # define config dir
    config_dir = cfg.io_dir.config_dir

    # define basin list dir
    basin_dir = cfg.io_dir.gauch_2020_dir
    basin_filename = cfg.model_settings.basin_file
    missing_data_filename = cfg.model_settings.missing_data_file

    # define observation file dir
    obs_dir = cfg.io_dir.usgs_streamflow_dir

    # define the spinup-calib-val period
    time_split_file = cfg.model_settings.time_split_file
    
    results_path = cfg.io_dir.results_dir
    
    parameter_bound_file = cfg.model_settings.parameter_bound_file
        

    # --------------------------------- Load settings  ----------------------------------- #
    # Load basin list
    basin_file = os.path.join(basin_filename)    
    with open(basin_file, 'r') as file:
        lines = file.readlines()
        # Remove leading/trailing whitespaces and newline characters
        lines = [line.strip() for line in lines]
    basin_list_str = lines

    # Basin list with missing data -> skipping these
    basin_with_nan_file = os.path.join(missing_data_filename) 
    with open(basin_with_nan_file, 'r') as file:
        lines = file.readlines()
        # Remove leading/trailing whitespaces and newline characters
        lines = [line.strip() for line in lines]
    missing_data_list = lines

    # Load time split file 
    with open(time_split_file, 'r') as file:
        time_split = json.load(file)
    print(time_split)

    # --------------------------------- Setup directories ----------------------------------- #

    # define result dir
    if os.path.exists(results_path)==False: 
        os.makedirs(results_path)

    # define raw result dir
    raw_results_path = os.path.join(results_path,'raw')
    if os.path.exists(raw_results_path)==False: 
        os.makedirs(raw_results_path)

    # define dir to save img
    png_dir = os.path.join(results_path,'images')
    if os.path.exists(png_dir)==False: 
        os.makedirs(png_dir)

    # define dir to save best run results and parameters
    best_run_dir = os.path.join(results_path,'best_runs')
    if os.path.exists(best_run_dir)==False: 
        os.makedirs(best_run_dir)

    with open(parameter_bound_file) as f:
            parameter_bounds = json.load(f)
            print(parameter_bounds)
            
    ########################################
    
    # Loop through subsets of basins
    for i in range(0, max_nbasin_per_loop):

        # ------------------ Preparation ----------------- ##
        # g_str= basin_list_str[i]
        g_str = cfg.basin_id
        
        if g_str in missing_data_list: 
            print(f"None or missing usgs streamflow data for basin {g_str}, skipping this basin.") 
            continue
        else:
            print(f"Processing basin:{g_str}.")

        # locate usgs observation file
        obs_filename = f'{g_str}-usgs-hourly.csv'
        obs_file_path = os.path.join(obs_dir, obs_filename)

        # set up spotpy class
        calibration_instance = Spotpy_setup(
            config_dir=config_dir, 
            obs_file_path=obs_file_path, 
            time_split=time_split,
            gauge_id=g_str,
            parameter_bounds=parameter_bounds,
            print_all_process=print_all_process
            )

        # ------------------ Calibration ----------------- ##
        # define algorithm and export raw result file name
        np.random.seed(0)
        sampler = spotpy.algorithms.dds(calibration_instance, dbname='raw_result_file', dbformat='ram')

        # start calibration
        sampler.sample(N)

        # export final results
        results = sampler.getdata()

        scheme = calibration_instance.cfemodel.surface_partitioning_scheme

        all_result_filename = f'{g_str}_all_results_dds_{scheme}_{str(N)}.npy'
        all_result_file = os.path.join(raw_results_path,all_result_filename)
        np.save(all_result_file,results)

        #------------------------------------------ Best parameter and Plots ------------------------------------------#

        # get best parameters and sim
        best_params = spotpy.analyser.get_best_parameterset(results)
        
        # TODO: add this to Yeham's code #####
        best_param_dict = {name: value for name, value in zip(parameter_bounds.keys(), best_params[0])}
        ########################################

        obj_values = results['like1']
        best_obj_value = np.nanmax(obj_values)
        best_idx = np.where(obj_values==best_obj_value)

        best_sim = spotpy.analyser.get_modelruns(results[best_idx[0][0]])
        best_sim = np.array([best_sim[i] for i in range(len(best_sim))])

        best_run = {"best parameters": best_param_dict,
                    "best objective values": best_obj_value, 
                    "best simulation results": list(best_sim)}

        best_run_filename = f'{g_str}_best_run_{scheme}_{str(N)}.json'
        best_run_file = os.path.join(best_run_dir,best_run_filename)

        with open(best_run_file, 'w', encoding='utf-8') as f:
            json.dump(best_run, f, ensure_ascii=False, indent=4)

        ### plot parameter trace png ###
        param_trace_imgname = f'{g_str}_param_trace_{scheme}_{str(N)}.png'
        param_trace_dir = os.path.join(png_dir,"ParamTrace")
        if os.path.exists(param_trace_dir)==False: 
            os.makedirs(param_trace_dir)
        param_trace_imgfile = os.path.join(param_trace_dir,param_trace_imgname)

        spotpy.analyser.plot_parametertrace(results, fig_name = param_trace_imgfile)

        ### plot objective value trace ###
        plt.figure(figsize = (18,12))
        plt.plot(np.arange(0,len(obj_values)), obj_values)
        #plt.vlines(x=best_idx[0],ymin=-10,ymax=0,colors = 'k',linestyles='dashed')
        plt.ylim([-2,1])
        plt.tick_params(axis='x', labelsize= 24)
        plt.tick_params(axis='y', labelsize= 24)
        plt.xlabel('Iterations',fontsize = 24)
        plt.ylabel('KGE Objective Values', fontsize = 24)
        plt.title("Trace of the Objective Values [KGE]", fontsize = 26)
        
        objvalues_imgname = f'{g_str}_obj_values_{scheme}_{str(N)}.png'
        objvalues_dir = os.path.join(png_dir,"Obj_Trace")
        if os.path.exists(objvalues_dir)==False: 
            os.makedirs(objvalues_dir)
        objvalues_imgfile = os.path.join(objvalues_dir,objvalues_imgname)
        plt.savefig(objvalues_imgfile,bbox_inches='tight')

        ### plot timeseries of simulation vs. observation ##
        
        dates = calibration_instance.evaluation(evaldates=True)
        fig, ax1 = plt.subplots(figsize = (18,12)) 
        
        # Plot obs & sim flow 
        p1, = ax1.plot(dates[0:8760], best_sim[0:8760],'tomato', linewidth = 2, label = "sim")
        p2, = ax1.plot(dates[0:8760], calibration_instance.obs_data[0:8760], 'k', label = "obs")
        ax1.set_ylabel('Discharge (mm/h)',fontsize = 24)
        ax1.set_ylim([0,2])
        ax1.margins(x=0)
        ax1.xaxis.set_ticks_position('both')
        ax1.xaxis.set_label_position('bottom')
        ax1.tick_params(axis="x",direction="in")
        
        # Plot precip
        ax2 = ax1.twinx()
        p3, = ax2.plot(dates[0:8760],calibration_instance.df_forcing['total_precipitation'][0:8760],'tab:blue', label = "precip")
        ax2.set_ylim([50,0])
        ax2.margins(x=0)
        #ax2.invert_yaxis()
        ax2.set_ylabel('Precipitation (mm/h)',fontsize = 24)
        ax2.set_xlabel('Date', fontsize = 24)
        ax2.tick_params(axis='x', labelsize= 24)
        ax2.tick_params(axis='y', labelsize= 24)
        ax1.tick_params(axis='y', labelsize= 24)
        
        plt.legend(handles = [p1,p2,p3],fontsize = 24, loc='right', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
        plt.title(f"Simulated Streamflow against Observation after {N} Iterations of Calibration [ID: {g_str}]", fontsize = 26)
        plt.tight_layout()

        comparison_imgname = f'{g_str}_comparison_{scheme}_{str(N)}.png'
        comparison_imgdir = os.path.join(png_dir,"Comparison")
        if os.path.exists(comparison_imgdir)==False: 
            os.makedirs(comparison_imgdir)
        comparison_imgfile = os.path.join(comparison_imgdir,comparison_imgname)
        plt.savefig(comparison_imgfile,bbox_inches='tight')

        # Finalize 
        print(f"The best KGE value: {best_obj_value}.")


# ----------------------------------- End of Calibration ----------------------------------- #


if __name__ == "__main__":
    main()