import os
import numpy as np
import pandas as pd

############################################
# This code checks nans in the data  #
############################################

# Modified by Ryoko Araki (San Diego State University & UCSB, raraki8159@sdsu.edu) in 2023 SI 

# Originally written by 2022 team
# Lauren A. Bolotin 1, Francisco Haces-Garcia 2, Mochi Liao 3, Qiyue Liu 4
# 1 San Diego State University; lbolotin3468@sdsu.edu
# 2 University of Houston; fhacesgarcia@uh.edu
# 3 Duke University; mochi.liao@duke.edu
# 4 University of Illinois at Urbana-Champaign; qiyuel3@illinois.edu

# ----------------------------------- Change here ----------------------------------- #
# ----------------------------------- Data Loading Dir ----------------------------------- #

# define working dir
working_dir = r'G:\Shared drives\SI_NextGen_Aridity\calibrate_cfe'

# define basin list dir
basin_dir = r'G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020'
basin_filename = 'basin_list_561.txt' # It was 516 basin in 2022 code 
output_dir = r'G:\Shared drives\SI_NextGen_Aridity\data\camels'
# G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020\basin_list_561.txt

# define config dir
config_dir = os.path.join(working_dir,'configs')

# define observation file dir
#obs_dir = os.path.join(working_dir,'usgs-streamflow')
obs_dir = r'G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020\usgs_streamflow'

# define atmospheric forcing file dir
forcing_path = r'G:\Shared drives\SI_NextGen_Aridity\data\camels\gauch_etal_2020\nldas_hourly'

# ----------------------------------- Data Loading Dir ----------------------------------- #
# ----------------------------------- Change end ----------------------------------- #

# load basin list
# with open(basin_file, "r") as f:
#     basin_list = pd.read_csv(f, header=None)
basin_file = os.path.join(basin_dir, basin_filename)
with open(basin_file, 'r') as file:
    lines = file.readlines()
    # Remove leading/trailing whitespaces and newline characters
    lines = [line.strip() for line in lines]
basin_list_str = lines

# ----------------------------------- Initialize ----------------------------------- #
nan_check = {}
nan_check['basin id'] = []
nan_check['Spinup - pet'] = []
nan_check['Spinup - precip'] = []
nan_check['Cal - pet'] = []
nan_check['Cal - precip'] = []
nan_check['Cal - usgs'] = []
nan_check['Val - pet'] = []
nan_check['Val - precip'] = []
nan_check['Val - usgs'] = []

# ----------------------------------- Initialize ----------------------------------- #
# Loop through each basin
for i in range(len(basin_list_str)): 
    # for i in range(3): 

    # -------- Loading files ------- #
    # Get the gauge ID
    # g = basin_list[0][i]
    g_str= basin_list_str[i]

    print(f"Processing basin:{g_str}.")

    # Load Observation file
    obs_file = os.path.join(obs_dir, f'{g_str}-usgs-hourly.csv')
    with open(obs_file) as f: 
        data = pd.read_csv(f)

    obs_data = data['QObs_CAMELS(mm/h)'].values
    eval_dates = data['date'].values
    # print(obs_data[0:5])

    # Load Forcing file
    forcing_file = os.path.join(forcing_path, f'{g_str}_hourly_nldas.csv')
    with open(forcing_file) as f:
        forcing = pd.read_csv(forcing_file)
    # print(forcing.head())

    # -------- Spin-up ------- #
    # check spin-up forcing file
    spinup_start_idx_nldas = np.where(forcing['date']=='2001-10-01 00:00:00')
    spinup_end_idx_nldas = np.where(forcing['date']=='2002-09-30 23:00:00')

    if (spinup_start_idx_nldas[0].size == 0) or (spinup_end_idx_nldas[0].size == 0) : 
        print("none or missing forcing data for spinup period")
        continue

    forcing_spinup = forcing.iloc[spinup_start_idx_nldas[0][0]:spinup_end_idx_nldas[0][0]+1,:]

    pet_nan_spinup = np.where(np.isnan(forcing_spinup['potential_evaporation']))[0].shape
    precip_nan_spinup = np.where(np.isnan(forcing_spinup['total_precipitation']))[0].shape

    # -------- Crop data for the calibration period ------- #
    
    # check calibration period usgs obs and forcing
    cal_start_idx_usgs = np.where(data['date']=='2007-10-01 00:00:00')
    cal_end_idx_usgs = np.where(data['date']=='2013-09-30 23:00:00')
    
    if (cal_start_idx_usgs[0].size == 0) or (cal_end_idx_usgs[0].size == 0): 
        print("none or missing usgs streamflow data for the calibration period for this basin.") 
        continue
    
    obs_data_cal = obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]

    cal_start_idx_nldas = np.where(data['date']=='2007-10-01 00:00:00')
    cal_end_idx_nldas = np.where(data['date']=='2013-09-30 23:00:00')

    if (cal_start_idx_nldas[0].size == 0) or (cal_end_idx_nldas[0].size == 0): 
        print("none or missing forcing data for calibration period")
        continue

    forcing_cal = forcing[cal_start_idx_nldas[0][0]:cal_end_idx_nldas[0][0]+1]

    pet_nan_cal = np.where(np.isnan(forcing_cal['potential_evaporation']))[0].shape
    precip_nan_cal = np.where(np.isnan(forcing_cal['total_precipitation']))[0].shape
    usgs_nan_cal = np.where(np.isnan(obs_data_cal))[0].shape

    # -------- Crop data for the Valifation period ------- #
    
    # check validation period usgs obs and forcing
    val_start_idx_usgs = np.where(data['date']=='2002-10-01 00:00:00')
    val_end_idx_usgs = np.where(data['date']=='2007-09-30 23:00:00')
    
    if (val_start_idx_usgs[0].size == 0) or (val_end_idx_usgs[0].size == 0): 
        print("none or missing usgs streamflow data for the validation period for this basin.") 
        continue

    obs_data_val = obs_data[val_start_idx_usgs[0][0]:val_end_idx_usgs[0][0]+1]

    val_start_idx_nldas = np.where(data['date']=='2002-10-01 00:00:00')
    val_end_idx_nldas = np.where(data['date']=='2007-09-30 23:00:00')

    if (val_start_idx_nldas[0].size == 0) or (val_end_idx_nldas[0].size == 0) : 
        print("none or missing forcing data for validation period")
        continue

    forcing_val = forcing[val_start_idx_nldas[0][0]:val_end_idx_nldas[0][0]+1]

    pet_nan_val = np.where(np.isnan(forcing_val['potential_evaporation']))[0].shape
    precip_nan_val = np.where(np.isnan(forcing_val['total_precipitation']))[0].shape
    usgs_nan_val = np.where(np.isnan(obs_data_val))[0].shape

    # -------- Finalizing ------- #
    nan_check['basin id'].append(g_str)
    nan_check['Spinup - pet'].append(pet_nan_spinup[0])
    nan_check['Spinup - precip'].append(precip_nan_spinup[0])
    nan_check['Cal - pet'].append(pet_nan_cal[0])
    nan_check['Cal - precip'].append(precip_nan_cal[0])
    nan_check['Cal - usgs'].append(usgs_nan_cal[0])
    nan_check['Val - pet'].append(pet_nan_val[0])
    nan_check['Val - precip'].append(precip_nan_val[0])
    nan_check['Val - usgs'].append(usgs_nan_val[0])

# ----------------------------------- Saving ----------------------------------- #
df = pd.DataFrame(nan_check)
df.to_csv(os.path.join(output_dir,"check_for_nan_in_data.csv"))