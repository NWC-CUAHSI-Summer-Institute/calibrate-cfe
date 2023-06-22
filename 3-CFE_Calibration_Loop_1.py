# import package
import spotpy

import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# import the cfe model
import bmi_cfe
import cfe

# TODO: Try on a CFE-ODE
# TODO: modify the py_cfe/CFE-ODE to take the xinanjiang or schakee as numbers
# TODO: change "basin list"
# TODO: check if anywhere obs is made daily 
# TODO: make it more flexible (date as json input, missing data basin as txt input)
# TODO: make a github repo for this calibration after we decide on the direction
# TODO: ? Check DDS parameters

############################################
# This code runs calibration (looping through 50 basins at a time) #
############################################

# Modified by Ryoko Araki (San Diego State University & UCSB, raraki8159@sdsu.edu) in 2023 SI 

# Originally written by 2022 team
# Lauren A. Bolotin 1, Francisco Haces-Garcia 2, Mochi Liao 3, Qiyue Liu 4
# 1 San Diego State University; lbolotin3468@sdsu.edu
# 2 University of Houston; fhacesgarcia@uh.edu
# 3 Duke University; mochi.liao@duke.edu
# 4 University of Illinois at Urbana-Champaign; qiyuel3@illinois.edu

# define working dir
working_dir = r'G:\Shared drives\SI_NextGen_Aridity\calibrate_cfe'

# ----------------------------------- Change here ----------------------------------- #
# ----------------------------------- Data Loading Dir ----------------------------------- #

# define iteration number
# N = 100
N = 3

# define basin list dir
basin_dir = r'G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020'
basin_filename = 'basin_list_561.txt'
basin_file = os.path.join(basin_dir,basin_filename)

# define config dir
config_dir = os.path.join(working_dir,'configs')

# define observation file dir
#obs_dir = os.path.join(working_dir,'usgs-streamflow')
obs_dir = r'G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020\usgs_streamflow'

# list of basins with missing data -> skipping these
missing_data_list = ['1552000','1552500','1567500','3338780','7301410','7315700','7346045','8050800','8101000','8104900','8109700','8158810','1187300','1510000','2178400','2196000','2202600','3592718','4197170','6919500','8176900']

# ----------------------------------- Saving Results Dir ----------------------------------- #
# ----------------------------------- Change end ----------------------------------- #

# define result dir
results_path = os.path.join(working_dir,'results')
if os.path.exists(results_path)==False: os.mkdir(results_path)

# define raw result dir
raw_results_path = os.path.join(results_path,'raw')
if os.path.exists(raw_results_path)==False: os.mkdir(raw_results_path)

# define dir to save img
png_dir = os.path.join(results_path,'images')
if os.path.exists(png_dir)==False: os.mkdir(png_dir)

# define dir to save best run results and parameters
best_run_dir = os.path.join(results_path,'best_runs')
if os.path.exists(best_run_dir)==False: os.mkdir(best_run_dir)



# ----------------------------------- Setup the Spotpy Class ----------------------------------- #
class Spotpy_setup(object): 

    def __init__(self,config_file,obs_file): 
        # define config file
        self.config_file = config_file
        
        # load original model and soil parameters for optguess
        with open(self.config_file) as data_file:
            data_loaded = json.load(data_file)

        bb_optguess = data_loaded['soil_params']['bb']
        smcmax_optguess = data_loaded['soil_params']["smcmax"]
        satdk_optguess = data_loaded["soil_params"]["satdk"]
        slop_optguess = data_loaded["soil_params"]["slop"]
        maxgw_optguess = data_loaded["max_gw_storage"]
        expon_optguess = data_loaded["expon"]
        cgw_optguess = data_loaded["Cgw"]
        klf_optguess = data_loaded["K_lf"]
        knash_optguess = data_loaded["K_nash"]

        # setup calibration parameters
        self.params = [spotpy.parameter.Uniform('bb',0,21.94,optguess=bb_optguess),
                       spotpy.parameter.Uniform('smcmax',0.20554,1,optguess = smcmax_optguess), #maybe max = 0.5
                       spotpy.parameter.Uniform('satdk',0,0.000726,optguess=satdk_optguess),
                       spotpy.parameter.Uniform('slop',0,1,optguess=slop_optguess),
                       spotpy.parameter.Uniform('max_gw_storage',0.01,0.25,optguess=maxgw_optguess),
                       spotpy.parameter.Uniform('expon',1,8,optguess=expon_optguess),
                       spotpy.parameter.Uniform('Cgw',1.8e-6,1.8e-3,optguess=cgw_optguess),
                       spotpy.parameter.Uniform('K_lf',0,1,optguess=klf_optguess),
                       spotpy.parameter.Uniform('K_nash',0,1,optguess=knash_optguess), 
                       spotpy.parameter.Uniform('scheme',0.01,0.99),
                       #spotpy.parameter.Uniform('mult',10,10000,optguess=1000),
                       ]
    
        # Load test comparison data (streamflow) from usgs data
        self.obs_file = obs_file
        data = pd.read_csv(self.obs_file)
        self.obs_data = data['QObs_CAMELS(mm/h)'].values
        self.eval_dates = data['date'].values

        # define calibration period for usgs streamflow obs.
        cal_start_idx_usgs = np.where(self.eval_dates=='2007-10-01 00:00:00')
        cal_end_idx_usgs = np.where(self.eval_dates=='2013-09-30 23:00:00')
        self.eval_dates = self.eval_dates[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        self.obs_data = self.obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        
        print('###--------- usgs start date: ' + self.eval_dates[0] + '.---------')
        print('###--------- usgs end date: ' + self.eval_dates[-1] + '.---------')

        print('###---------- obs data length: ' +  str(len(self.obs_data)) + '.---------')

    def parameters(self):
        return spotpy.parameter.generate(self.params)
        
    def simulation(self,vector):
        self.cfemodel = bmi_cfe.BMI_CFE(self.config_file)
        print("### ------------ A NEW ITERATION OF CALIBRATION ------------ ###")
        print('###--------model succesfully setup----------###')

        self.generated_param = vector

        # self.scheme = "Schaake"

        print(f"###----------- parameters generated: {self.generated_param}.--------###")

        self.cfemodel.initialize(param_vec = vector)
        print('###--------model succesfully initialized----------###')

        with open(self.cfemodel.forcing_file, 'r') as f:
            self.df_forcing = pd.read_csv(f)
        print(f"###----- forcing_file loaded:{self.cfemodel.forcing_file}. -----###")

        # --------------------------------------- Run Spin-up Period --------------------------------------- # 
        # define the spin up period
        spinup_start_idx_nldas = np.where(self.df_forcing['date']=='2006-10-01 00:00:00')
        spinup_end_idx_nldas = np.where(self.df_forcing['date']=='2007-09-30 23:00:00')
        self.df_forcing_spinup = self.df_forcing.iloc[spinup_start_idx_nldas[0][0]:spinup_end_idx_nldas[0][0]+1,:]

        print('###-------- Model Spinning Up... ----------###')
        print('###------spinup start date: ' + self.df_forcing_spinup['date'].values[0]+ "-----###")
        print('###------spinup end date: ' + self.df_forcing_spinup['date'].values[-1]+"-----###")
        
        # run the model for the spin-up period
        self.spinup_outputs=self.cfemodel.get_output_var_names()
        self.spinup_output_lists = {output:[] for output in self.spinup_outputs}

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
        cal_start_idx_nldas = np.where(self.df_forcing['date']=='2007-10-01 00:00:00')
        cal_end_idx_nldas = np.where(self.df_forcing['date'].values=='2013-09-30 23:00:00')
        self.df_forcing = self.df_forcing.iloc[cal_start_idx_nldas[0][0]:cal_end_idx_nldas[0][0]+1,:]

        cal_start_idx_usgs = np.where(self.eval_dates=='2007-10-01 00:00:00')
        cal_end_idx_usgs = np.where(self.eval_dates=='2013-09-30 23:00:00')
        self.eval_dates = self.eval_dates[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        self.obs_data = self.obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
        
        print('###-------- Model Running for the Calibration Period ----------###')
        print('###------nldas start date: ' + self.df_forcing['date'].values[0]+ "-----###")
        print('###------nldas end date: ' + self.df_forcing['date'].values[-1]+"-----###")
        print('###--------- usgs start date: ' + self.eval_dates[0] + '.---------')
        print('###--------- usgs end date: ' + self.eval_dates[-1] + '.---------')

        print('###----- nldas forcing data length: ' +  str(len(self.df_forcing['date'].values))+"------###")
        print('###---------- obs data length: ' +  str(len(self.obs_data)) + '.---------')

        self.outputs=self.cfemodel.get_output_var_names()
        self.output_lists = {output:[] for output in self.outputs}

        for precip, pet in zip(self.df_forcing['total_precipitation'],self.df_forcing['potential_evaporation']):
            #print(f"###----------loaded precip, pet: {precip},{pet}.------------###")
            #sys.exit(1)
            self.cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip/1000)   # kg/m2/h = mm/h -> m/h
            self.cfemodel.set_value('water_potential_evaporation_flux', pet/1000/3600) # kg/m2/h = mm/h -> m/s
            self.cfemodel.update()
        
            for output in self.outputs:
                self.output_lists[output].append(self.cfemodel.get_value(output))

        self.cfemodel.finalize(print_mass_balance=True)

        print(f'###----------output length: {len(self.output_lists["land_surface_water__runoff_depth"])}.---------###')

        self.sim_results = np.array(self.output_lists['land_surface_water__runoff_depth']) * 1000      # m/h -> mm/h
        return self.sim_results 

    def evaluation(self,evaldates=False):
        if evaldates:
            self.eval_dates_output = [pd.Timestamp(self.eval_dates[i]) for i in range(len(self.eval_dates))]
            return self.eval_dates_output
        else:
            print(f"###--------- double check - length of obs_data: {len(self.obs_data)}. -----------###")
            return self.obs_data

    def objectivefunction(self,simulation,evaluation, params=None):
        self.obj_function = spotpy.objectivefunctions.kge(evaluation,simulation)
        return self.obj_function


# ----------------------------------- Loop for Calibration ----------------------------------- #
# load basin list
with open(basin_file, "r") as f:
    basin_list = pd.read_csv(f, header=None)
    
with open(basin_file, 'r') as file:
    lines = file.readlines()
    # Remove leading/trailing whitespaces and newline characters
    lines = [line.strip() for line in lines]
basin_list_str = lines

# Loop through subsets of basins
# for i in range(0,52): 
for i in range(0,1): 
    
    g = basin_list[0][i]

    g_str= basin_list_str[i]

    print(f"Processing basin:{g_str}.")
    
    if g in missing_data_list: 
        print(f"None or missing usgs streamflow data for basin {g_str}, skipping this basin.") 
        continue

    # locate config file
    config_filename = f'cat_{g_str}_bmi_config_cfe.json'
    config_file = os.path.join(config_dir,config_filename)

    # locate usgs observation file
    obs_filename = f'{g_str}-usgs-hourly.csv'
    obs_file = os.path.join(obs_dir,obs_filename)

    # set up spotpy class
    calibration = Spotpy_setup(config_file = config_file,obs_file = obs_file)

    # define algorithm and export raw result file name
    sampler = spotpy.algorithms.dds(calibration, dbname='raw_result_file', dbformat='ram')

    # start calibration
    sampler.sample(N)

    # export final results
    results = sampler.getdata()

    scheme = calibration.cfemodel.surface_partitioning_scheme

    all_result_filename = f'{g_str}_all_results_dds_{scheme}_{str(N)}.npy'
    all_result_file = os.path.join(raw_results_path,all_result_filename)
    np.save(all_result_file,results)

    #------------------------------------------ Evaluation and Plots ------------------------------------------#

    # get best parameters and sim
    best_params = spotpy.analyser.get_best_parameterset(results)

    obj_values=results['like1']
    best_obj_value=np.nanmax(obj_values)
    best_idx=np.where(obj_values==best_obj_value)

    best_sim = spotpy.analyser.get_modelruns(results[best_idx[0][0]])
    best_sim = np.array([best_sim[i] for i in range(len(best_sim))])

    best_run = {"best parameters":list(best_params[0]),
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
        os.mkdir(param_trace_dir)
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
        os.mkdir(objvalues_dir)
    objvalues_imgfile = os.path.join(objvalues_dir,objvalues_imgname)
    plt.savefig(objvalues_imgfile,bbox_inches='tight')

    ### plot simulation vs. observation ##
    dates = calibration.evaluation(evaldates=True)
    fig, ax1 = plt.subplots(figsize = (18,12)) 
    p1, = ax1.plot(dates[0:8760],best_sim[0:8760],'tomato', linewidth = 2,label = "sim")
    p2, = ax1.plot(dates[0:8760],calibration.obs_data[0:8760],'k',label = "obs")
    ax1.set_ylabel('Discharge (mm/h)',fontsize = 24)
    ax1.set_ylim([0,2])
    ax1.margins(x=0)
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.set_label_position('bottom')
    ax1.tick_params(axis="x",direction="in")
    ax2 = ax1.twinx()
    p3, = ax2.plot(dates[0:8760],calibration.df_forcing['total_precipitation'][0:8760],'tab:blue', label = "precip")
    ax2.set_ylim([50,0])
    ax2.margins(x=0)
    #ax2.invert_yaxis()
    ax2.set_ylabel('Precipitation (mm/h)',fontsize = 24)
    ax2.set_xlabel('Date', fontsize = 24)
    ax2.tick_params(axis='x', labelsize= 24)
    ax2.tick_params(axis='y', labelsize= 24)
    ax1.tick_params(axis='y', labelsize= 24)
    plt.legend(handles = [p1,p2,p3],fontsize = 24, loc='right', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
    plt.title(f"Simulated Streamflow against Observation after {N} Iterations of Calibration [ID: 0{g}]", fontsize = 26)
    plt.tight_layout()

    comparison_imgname = f'{g_str}_comparison_{scheme}_{str(N)}.png'
    comparison_imgdir = os.path.join(png_dir,"Comparison")
    if os.path.exists(comparison_imgdir)==False: os.mkdir(comparison_imgdir)
    comparison_imgfile = os.path.join(comparison_imgdir,comparison_imgname)
    plt.savefig(comparison_imgfile,bbox_inches='tight')

    # Finalize 
    print(f"The best KGE value: {best_obj_value}.")


# ----------------------------------- End of Calibration ----------------------------------- #