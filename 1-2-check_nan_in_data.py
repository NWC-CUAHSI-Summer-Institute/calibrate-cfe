import os
import numpy as np
import pandas as pd
import json

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

# define basin list dir
basin_dir = r'..\data\camels\gauch_etal_2020'
basin_filename = 'basin_list_516.txt'
output_dir = basin_dir

# define observation file dir
#obs_dir = os.path.join(working_dir,'usgs-streamflow')
obs_dir = r'..\data\camels\gauch_etal_2020\usgs_streamflow'

# define atmospheric forcing file dir
forcing_path = r'..\data\camels\gauch_etal_2020\nldas_hourly'

# define the spinup-calib-val period
time_split_file = r'..\data\model_common_configs\cal-val-test-period.json'

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

# Load time split file 
with open(time_split_file, 'r') as file:
    time_split = json.load(file)
print(json.dumps(time_split, indent=4))

# ----------------------------------- Initialize ----------------------------------- #
nan_check = {}
nan_check['basin id'] = []
nan_check['Spinup - Cal - pet'] = []
nan_check['Spinup - Cal - precip'] = []
nan_check['Spinup - Test - pet'] = []
nan_check['Spinup - Test - precip'] = []
nan_check['Cal - pet'] = []
nan_check['Cal - precip'] = []
nan_check['Cal - usgs'] = []
nan_check['Test - pet'] = []
nan_check['Test - precip'] = []
nan_check['Test - usgs'] = []

# ----------------------------------- Initialize ----------------------------------- #
# Loop through each basin
for i in range(len(basin_list_str)): 
    # for i in range(3): 

    # -------- Loading files ------- #
    # Get the gauge ID
    g_str= basin_list_str[i]

    print(f"Processing basin: {g_str}.")

    # Load Observation file
    obs_file = os.path.join(obs_dir, f'{g_str}-usgs-hourly.csv')
    with open(obs_file) as f: 
        data = pd.read_csv(f)

    obs_data = data['QObs(mm/h)'].values
    eval_dates = data['date'].values
    # print(obs_data[0:5])

    # Load Forcing file
    forcing_file = os.path.join(forcing_path, f'{g_str}_hourly_nldas.csv')
    with open(forcing_file) as f:
        forcing = pd.read_csv(forcing_file)
    # print(forcing.head())

    # -------- Spin-up for Calibration------- #
    # check spin-up forcing file
    spinup_cal_start_idx_nldas = np.where(forcing['date']==time_split["spinup-for-calibration"]["start_datetime"])
    spinup_cal_end_idx_nldas = np.where(forcing['date']==time_split["spinup-for-calibration"]["end_datetime"])

    if (spinup_cal_start_idx_nldas[0].size == 0) or (spinup_cal_end_idx_nldas[0].size == 0) : 
        print("none or missing forcing data for spinup period")
        continue

    forcing_spinup_cal = forcing.iloc[spinup_cal_start_idx_nldas[0][0]:spinup_cal_end_idx_nldas[0][0]+1,:]

    pet_nan_spinup_cal_indices = np.where(np.isnan(forcing_spinup_cal['potential_evaporation']))[0]
    pet_nan_spinup_cal = len(pet_nan_spinup_cal_indices) / len(forcing_spinup_cal['potential_evaporation'])
    
    precip_nan_spinup_cal_indices = np.where(np.isnan(forcing_spinup_cal['total_precipitation']))[0]
    precip_nan_spinup_cal = len(precip_nan_spinup_cal_indices) / len(forcing_spinup_cal['total_precipitation'])
    
    # -------- Spin-up for Testing ------- #
    # check spin-up forcing file
    spinup_test_start_idx_nldas = np.where(forcing['date']==time_split["spinup-for-testing"]["start_datetime"])
    spinup_test_end_idx_nldas = np.where(forcing['date']==time_split["spinup-for-testing"]["end_datetime"])

    if (spinup_test_start_idx_nldas[0].size == 0) or (spinup_test_end_idx_nldas[0].size == 0) : 
        print("none or missing forcing data for spinup period")
        continue

    forcing_spinup_test = forcing.iloc[spinup_test_start_idx_nldas[0][0]:spinup_test_end_idx_nldas[0][0]+1,:]

    pet_nan_spinup_test_indices = np.where(np.isnan(forcing_spinup_test['potential_evaporation']))[0]
    pet_nan_spinup_test = len(pet_nan_spinup_test_indices)/len(forcing_spinup_test['potential_evaporation'])
    
    precip_nan_spinup_test_indices = np.where(np.isnan(forcing_spinup_test['total_precipitation']))[0]
    precip_nan_spinup_test = len(precip_nan_spinup_test_indices) / len(forcing_spinup_test['total_precipitation'])


    # -------- Crop data for the calibration period ------- #
    
    # check calibration period usgs obs and forcing
    cal_start_idx_usgs = np.where(data['date']==time_split["calibration"]["start_datetime"])
    cal_end_idx_usgs = np.where(data['date']==time_split["calibration"]["end_datetime"])
    
    if (cal_start_idx_usgs[0].size == 0) or (cal_end_idx_usgs[0].size == 0): 
        print("none or missing usgs streamflow data for the calibration period for this basin.") 
        continue
    
    obs_data_cal = obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0]+1]

    cal_start_idx_nldas = np.where(data['date']==time_split["calibration"]["start_datetime"])
    cal_end_idx_nldas = np.where(data['date']==time_split["calibration"]["end_datetime"])

    if (cal_start_idx_nldas[0].size == 0) or (cal_end_idx_nldas[0].size == 0): 
        print("none or missing forcing data for calibration period")
        continue

    forcing_cal = forcing[cal_start_idx_nldas[0][0]:cal_end_idx_nldas[0][0]+1]

    pet_nan_cal = np.where(np.isnan(forcing_cal['potential_evaporation']))[0].shape
    precip_nan_cal = np.where(np.isnan(forcing_cal['total_precipitation']))[0].shape
    usgs_nan_cal = np.where(np.isnan(obs_data_cal))[0].shape

    # -------- Crop data for the Valifation period ------- #
    
    # check validation period usgs obs and forcing
    val_start_idx_usgs = np.where(data['date']==time_split["testing"]["start_datetime"])
    val_end_idx_usgs = np.where(data['date']==time_split["testing"]["end_datetime"])
    
    if (val_start_idx_usgs[0].size == 0) or (val_end_idx_usgs[0].size == 0): 
        print("none or missing usgs streamflow data for the validation period for this basin.") 
        continue

    obs_data_val = obs_data[val_start_idx_usgs[0][0]:val_end_idx_usgs[0][0]+1]

    val_start_idx_nldas = np.where(data['date']==time_split["testing"]["start_datetime"])
    val_end_idx_nldas = np.where(data['date']==time_split["testing"]["end_datetime"])

    if (val_start_idx_nldas[0].size == 0) or (val_end_idx_nldas[0].size == 0) : 
        print("none or missing forcing data for validation period")
        continue

    forcing_val = forcing[val_start_idx_nldas[0][0]:val_end_idx_nldas[0][0]+1]

    pet_nan_val = np.where(np.isnan(forcing_val['potential_evaporation']))[0].shape
    precip_nan_val = np.where(np.isnan(forcing_val['total_precipitation']))[0].shape
    usgs_nan_val = np.where(np.isnan(obs_data_val))[0].shape

    # -------- Finalizing ------- #
    nan_check['basin id'].append(g_str)
    nan_check['Spinup - Cal - pet (%)'].append(pet_nan_spinup_cal[0])
    nan_check['Spinup - Cal - precip (%)'].append(precip_nan_spinup_cal[0])
    nan_check['Spinup - Test - pet (%)'].append(pet_nan_spinup_test[0])
    nan_check['Spinup - Test - precip (%)'].append(precip_nan_spinup_test[0])
    nan_check['Cal - pet (%)'].append(pet_nan_cal[0])
    nan_check['Cal - precip (%)'].append(precip_nan_cal[0])
    nan_check['Cal - usgs (%)'].append(usgs_nan_cal[0])
    nan_check['Test - pet (%)'].append(pet_nan_val[0])
    nan_check['Test - precip (%)'].append(precip_nan_val[0])
    nan_check['Test - usgs (%)'].append(usgs_nan_val[0])

# ----------------------------------- Saving ----------------------------------- #
df = pd.DataFrame(nan_check)
df.to_csv(os.path.join(output_dir, "check_for_nan_in_data_hourly.csv"))