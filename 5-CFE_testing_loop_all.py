import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob
import sys

# import the cfe model
sys.path.append(r'..\cfe_py')
import bmi_cfe
import cfe

# import hydro evaluation package
import hydroeval as he

# define working dir
# ----------------------------------- Data Loading Dir ----------------------------------- #
# define config dir
config_dir = r'.\configs'

# define basin list dir
basin_dir = r'..\data\camels\gauch_etal_2020'
basin_filename = 'basin_list_516.txt'
missgin_data_filename = 'basin_list_missing_data_v2023.txt'

# define observation file dir
obs_dir = r'..\data\camels\gauch_etal_2020\usgs_streamflow'

# define the spinup-calib-val period
time_split_file = r'..\data\model_common_configs\cal-val-test-period.json'
# ----------------------------------- Change end ----------------------------------- #
# ----------------------------------------------------------------------------------- #

# --------------------------------- Load settings  ----------------------------------- #
# Load basin list
basin_file = os.path.join(basin_dir,basin_filename)    
with open(basin_file, 'r') as file:
    lines = file.readlines()
    # Remove leading/trailing whitespaces and newline characters
    lines = [line.strip() for line in lines]
basin_list_str = lines

# Basin list with missing data -> skipping these
basin_with_nan_file = os.path.join(basin_dir, missgin_data_filename) 
with open(basin_with_nan_file, 'r') as file:
    lines = file.readlines()
    # Remove leading/trailing whitespaces and newline characters
    lines = [line.strip() for line in lines]
missing_data_list = lines

# Load time split file 
with open(time_split_file, 'r') as file:
    time_split = json.load(file)
print(time_split)

results_path = r'.\results'
png_dir = os.path.join(results_path,'images')
best_run_dir = os.path.join(results_path,'best_runs')

validation_imgdir = os.path.join(png_dir,"Testing")
if os.path.exists(validation_imgdir)==False: 
    os.mkdir(validation_imgdir)
    
test_dir = os.path.join(results_path,"Testing")
if os.path.exists(test_dir)==False: 
    os.mkdir(test_dir)
    
# define iteration number

# ---------------------------------------- Loop through Validation Period ---------------------------------------- #
# ---------------------------------- Validation Period: 2002/10/01 - 2007/09/30 ---------------------------------- #

# load basin list
with open(basin_file, "r") as f:
    basin_list = pd.read_csv(f, header=None)

# initialize dict to save results
performance_values = {}
performance_values["basin_id"] = []
performance_values["kge_values"] = []
performance_values["nse_values"] = []

# Setup custom method in BMI-CFE
def custom_load_forcing_file(self):
    self.forcing_data = pd.read_csv(self.forcing_file)
    # Column name change to accomodate NLDAS forcing by https://zenodo.org/record/4072701
    self.forcing_data.rename(columns={"date": "time"}, inplace=True)
    pass
    

# Loop through each basin
# for i in range(basin_list.shape[0]): 
for i in range(0, 1): 
    
    # ------------------ Preparation ----------------- ##
    g_str= basin_list_str[i]

    
    if g_str in missing_data_list: 
        print(f"None or missing usgs streamflow data for basin {g_str}, skipping this basin.") 
        continue
    else:
        print(f"Processing basin:{g_str}.")

    # ------------------ Read the best params from previous file ----------------- ##

    # load best parameters found in calibration period
    best_run_filename = '**/*' + str(g_str) + '*.*'
    #best_run_file = os.path.join(best_run_dir,best_run_filename)
    file_list = []
    for files in glob.glob(best_run_dir + best_run_filename):
        file_list.append(files)

    with open(file_list[0]) as data_file:
        data_loaded = json.load(data_file)

    best_run_params = data_loaded["best parameters"]
    
    # locate config file
    config_filename = 'cat_' + str(g_str) + '_bmi_config_cfe.json'
    with open(os.path.join(config_dir, config_filename), 'r') as file:
        cfe_cfg = json.load(file)
    
    # Read the template config file 
    cfe_cfg["soil_params"]['bb'] = best_run_params['bb']
    cfe_cfg["soil_params"]['smcmax'] = best_run_params['smcmax']
    cfe_cfg["soil_params"]['satdk'] = best_run_params['satdk']
    cfe_cfg['slop'] = best_run_params['slop']
    cfe_cfg['max_gw_storage'] = best_run_params['max_gw_storage']
    cfe_cfg['expon'] = best_run_params['expon']
    cfe_cfg['Cgw'] = best_run_params['Cgw']
    cfe_cfg['K_lf'] = best_run_params['K_lf']
    cfe_cfg['K_nash'] = best_run_params['K_nash']
    if best_run_params['scheme'] <= 0.5:
        cfe_cfg['partition_scheme'] = "Schaake"
    else:
        cfe_cfg['partition_scheme'] = "Xinanjiang"
            
    # Dump optguess parameter into temporary config file
    config_temp_filename = f'cat_{g_str}_bmi_config_cfe_temp.json'
    with open(os.path.join(config_dir, config_temp_filename), 'w') as out_file:
        json.dump(cfe_cfg, out_file)
    
    # ----------------------------------- Run the Model ----------------------------------- #

    # Set up CFE model
    cfemodel = bmi_cfe.BMI_CFE(cfg_file=os.path.join(config_dir, config_temp_filename))
    cfemodel.load_forcing_file = custom_load_forcing_file.__get__(cfemodel)

    # initialize the model
    cfemodel.initialize()
    print('###--------model succesfully initialized----------###')

    with open(cfemodel.forcing_file, 'r') as f:
        df_forcing = pd.read_csv(f)
    print(f"###----- forcing_file loaded:{cfemodel.forcing_file}. -----###")

    # --------------------------------------- Run Spin-up Period --------------------------------------- # 
    # define the spin up period
    spinup_start_idx_nldas = np.where(df_forcing['date']==time_split["spinup-for-testing"]["start_datetime"])
    spinup_end_idx_nldas = np.where(df_forcing['date']==time_split["spinup-for-testing"]["end_datetime"])
    cfemodel.df_forcing_spinup = df_forcing.iloc[spinup_start_idx_nldas[0][0]:spinup_end_idx_nldas[0][0]+1,:]

    print('###-------- model spinning up ----------###')
    print('###------spinup start date: ' + cfemodel.df_forcing_spinup['date'].values[0]+ "-----###")
    print('###------spinup end date: ' + cfemodel.df_forcing_spinup['date'].values[-1]+"-----###")
    
    # run the model for the spin-up period
    cfemodel.spinup_outputs=cfemodel.get_output_var_names()
    cfemodel.spinup_output_lists = {output:[] for output in cfemodel.spinup_outputs}

    for precip, pet in zip(cfemodel.df_forcing_spinup['total_precipitation'],cfemodel.df_forcing_spinup['potential_evaporation']):
        #print(f"###----------loaded precip, pet: {precip},{pet}.------------###")
        #sys.exit(1)
        cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip/1000)   # kg/m2/h = mm/h -> m/h
        cfemodel.set_value('water_potential_evaporation_flux', pet/1000/3600) # kg/m2/h = mm/h -> m/s
        cfemodel.update()
    
        for spinup_output in cfemodel.spinup_outputs:
            cfemodel.spinup_output_lists[spinup_output].append(cfemodel.get_value(spinup_output))

    # --------------------------------------- Rununing for the Validation Period --------------------------------------- #
    # define the calibration period for nldas forcing and usgs streamflow obs.
    cal_start_idx_nldas = np.where(df_forcing['date']==time_split["testing"]["start_datetime"])
    cal_end_idx_nldas = np.where(df_forcing['date']==time_split["testing"]["end_datetime"])
    df_forcing = df_forcing.iloc[cal_start_idx_nldas[0][0]:cal_end_idx_nldas[0][0]+1,:]

    print('###----- nldas forcing data length: ' +  str(len(df_forcing['date'].values))+"------###")

    outputs=cfemodel.get_output_var_names()
    output_lists = {output:[] for output in outputs}

    for precip, pet in zip(df_forcing['total_precipitation'],df_forcing['potential_evaporation']):
        #print(f"###----------loaded precip, pet: {precip},{pet}.------------###")
        #sys.exit(1)

        cfemodel.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip/1000)   # kg/m2/h = mm/h -> m/h
        cfemodel.set_value('water_potential_evaporation_flux', pet/1000/3600) # kg/m2/h = mm/h -> m/s
        cfemodel.update()

        for output in outputs:
            output_lists[output].append(cfemodel.get_value(output))
        
    cfemodel.finalize(print_mass_balance=True)

    # ----------------------------------- Evaluate Results ----------------------------------- #
    # Load Observation file
    obs_filename = f'{g_str}-usgs-hourly.csv'
    obs_file_path = os.path.join(obs_dir,obs_filename)

    data = pd.read_csv(obs_file_path)
    obs_data = data['QObs_CAMELS(mm/h)'].values
    eval_dates = data['date'].values

    # define calibration period for usgs streamflow obs.
    cal_start_idx_usgs = np.where(eval_dates==time_split["testing"]["start_datetime"])
    cal_end_idx_usgs = np.where(eval_dates==time_split["testing"]["end_datetime"])
    eval_dates = eval_dates[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
    obs_data = obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]
    
    dates = [pd.Timestamp(eval_dates[i]) for i in range(len(eval_dates))]

    # Export streamflow data
    sims = np.array(output_lists['land_surface_water__runoff_depth']) * 1000 

    # calculate kge values
    simulations = sims
    evaluations = obs_data
    nse = he.evaluator(he.nse, simulations, evaluations)
    kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)

    # plot sim against obs for the 1st year
    fig, ax1 = plt.subplots(figsize = (18,12)) 
    p1, = ax1.plot(dates,sims,'tomato', linewidth = 2,label = "sim")
    p2, = ax1.plot(dates,obs_data,'k',label = "obs")
    ax1.set_ylabel('Discharge (mm/h)',fontsize = 26)
    ax1.set_ylim([0,2])
    ax1.tick_params(axis='x', labelsize= 24)
    ax1.tick_params(axis='y', labelsize= 24)
    ax1.margins(x=0)
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.set_label_position('bottom')
    ax1.tick_params(axis="x",direction="in")
    ax2 = ax1.twinx()
    p3, = ax2.plot(dates,df_forcing['total_precipitation'],'tab:blue', label = "precip")
    ax2.set_ylim([50,0])
    ax2.margins(x=0)
    #ax2.invert_yaxis()
    ax2.set_ylabel('Precipitation (mm/h)',fontsize = 26)
    ax2.set_xlabel('Date', fontsize = 18)
    #ax2.tick_params(axis='x', labelsize= 24)
    ax2.tick_params(axis='y', labelsize= 24)
    plt.legend(handles = [p1,p2,p3],fontsize = 24, loc='lower right', bbox_to_anchor=(0.5, 0.5,0.5,0.5))
    textstr = '\n'.join((f"The KGE value is : {round(kge[0],4)}.",f"The NSE value is : {round(nse[0],4)}."))
    ax1.text(0.98, 0.45, textstr, transform=ax1.transAxes, fontsize=20,verticalalignment='center',horizontalalignment='right',bbox=dict(facecolor='white', alpha=0.5))
    plt.title(f"Simulated Streamflow against Observation in the Validation Period [ID: 0{g_str}]", fontsize = 28)
    plt.tight_layout()

    validation_imgname = str(g_str) + "_validation.png"
    validation_imgfile = os.path.join(validation_imgdir,validation_imgname)
    
    plt.savefig(validation_imgfile,bbox_inches='tight')
    
    # calculate kge values
    print(f"The KGE value is : {kge}.")
    print(f"The NSE value is : {nse}.")

    performance_values["basin_id"].append(g_str)
    performance_values["kge_values"].append(kge)
    performance_values["nse_values"].append(nse)

# ---------------------------------------- End of Looping Validation Period ---------------------------------------- #

# Save performance dict
df = pd.DataFrame(performance_values)
df.to_csv(os.path.join(test_dir, "cfe_validation_performance_values.csv"))