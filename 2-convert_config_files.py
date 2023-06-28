import pandas as pd
import numpy as np
import os
import sys
import json
import re 

from omegaconf import DictConfig, OmegaConf
import hydra

############################################
# This code creates CFE config files from Python-version 
# from Luciana's config files for C-version  #
############################################

# Modified by Ryoko Araki (San Diego State University & UCSB, raraki8159@sdsu.edu) in 2023 SI 

# Originally written by 2022 team
# Lauren A. Bolotin 1, Francisco Haces-Garcia 2, Mochi Liao 3, Qiyue Liu 4
# 1 San Diego State University; lbolotin3468@sdsu.edu
# 2 University of Houston; fhacesgarcia@uh.edu
# 3 Duke University; mochi.liao@duke.edu
# 4 University of Illinois at Urbana-Champaign; qiyuel3@illinois.edu

# Folder structure
# project_folder/
# ├─ data/
# ├─ cfe_py/
# ├─ calibrate_cfe/
# │  ├─ configs/
# │  ├─ results/

# Run this from project_folder/calibrate_cfe

# ----------------------------------- Change here ----------------------------------- #
#---------------------------- define directories ----------------------------#

# TODO: make this wget from Google Drive ... 
# Pre-requisits, optguess configs are download to calibrate_cfe/configs/CFE_Config_lumped_Luciana

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg): 
    
    # ----------------------------------- Read config via hydra ----------------------------------- #

    print(OmegaConf.to_yaml(cfg))
        
    # define GIUH and soil params files
    GIUH_soil_dir = cfg['io_dir']['cfe_c_config_dir']

    # define basin list dir
    basin_dir = cfg['io_dir']['gauch_2020_dir']
    basin_filename = cfg['model_settings']['basin_file'] # It was 516 basin in 2022 code 

    # define camel dataset dir
    camels_attr_dir = cfg['io_dir']['ucar_dir']

    # define atmospheric forcing file dir
    forcing_path = cfg['io_dir']['nldas_forcing_dir']

    # define dir for exported json
    config_dir = cfg['io_dir']['config_dir']


    # Default
    soil_scheme = "ode"
    partition_scheme = "Schaake"

    #---------------------------- Basin Attributes ----------------------------#

    # read in basin list
    # basin_filename = 'basin_list_516.txt'
    # basin_file = os.path.join(basin_dir,basin_filename)

    with open(basin_filename, "r") as f:
        basin_list = pd.read_csv(f, header=None)

    with open(basin_filename, 'r') as file:
        lines = file.readlines()
        # Remove leading/trailing whitespaces and newline characters
        lines = [line.strip() for line in lines]
    basin_list_str = lines

    # get basin attributes for each basin
    basin_attributes = {}

    for attribute_type in ['clim', 'geol', 'hydro', 'name', 'soil', 'topo', 'vege']:
        camel_filename = f"camels_{attribute_type}.txt"
        camel_file = os.path.join(camels_attr_dir, camel_filename)
        with open(camel_file, "r") as f:
            basin_attributes[attribute_type] = pd.read_csv(f, sep=";")
        basin_attributes[attribute_type] = basin_attributes[attribute_type].set_index("gauge_id")


    #---------------------------- Generate Config Files ----------------------------#

    for i in range(len(basin_list_str)): 
    # for i in range(0, 3): 

        g = basin_list[0][i]
        
        g_str= basin_list_str[i]
        print(f'Processing: {g_str}')

        # get forcing file
        forcing_file = os.path.join(forcing_path, f'{g_str}_hourly_nldas.csv')

        # get giuh and soil param file
        giuh_soil_file = os.path.join(GIUH_soil_dir, f'{g_str}_bmi_config_cfe_pass.txt')


        with open(giuh_soil_file, "r") as f:
            giuh_soil_data_all = pd.read_fwf(f,header=None)

        giuh_soil_data = giuh_soil_data_all.iloc[2:21,:]
        giuh_soil_data = pd.concat([giuh_soil_data,giuh_soil_data_all.iloc[24,:]],ignore_index=True)

        for i in range(giuh_soil_data.shape[0]): 
            giuh_soil_data[0][i] = giuh_soil_data[0][i].split('=')
            parameter_values = [giuh_soil_data[0][j][1] for j in range(giuh_soil_data.shape[0])]
            parameter_names = [giuh_soil_data[0][j][0] for j in range(giuh_soil_data.shape[0])]
            parameter_values = [re.sub("\[.*?\]", "", parameter_values[i]) for i in range(len(parameter_values))]

        parameter_values[0:18] = np.array(parameter_values[0:18],dtype ="double")
        parameter_values[18] = np.array(parameter_values[18].split(','),dtype="double")
        parameter_values[19] = np.array(parameter_values[19].split(','),dtype="double")

        param_dict = {}
        param_dict['soil_params'] = {}
        for i in range(len(parameter_names)): 
            if i <= 8: 
                soil_param_name = parameter_names[i].split('.')
                param_dict[soil_param_name[0]][soil_param_name[1]] = parameter_values[i]
            else: param_dict[parameter_names[i]] = parameter_values[i]

        del param_dict['soil_params']['expon']
        del param_dict['soil_params']['expon_secondary']
        param_dict['soil_params']['bb'] = param_dict['soil_params'].pop("b")
        param_dict['soil_params']["D"] = 2.0
        param_dict["soil_params"]["mult"] = 1000.0

        # generate json text
        dict_json = {"forcing_file": forcing_file, 
                        "catchment_area_km2": basin_attributes['topo']['area_gages2'][g], 
                        "alpha_fc": param_dict["alpha_fc"], 
                        "soil_params": param_dict["soil_params"], 
                        "refkdt": param_dict["refkdt"],
                        "max_gw_storage": param_dict["max_gw_storage"],              # [calibrating parameter]
                        "Cgw": param_dict["Cgw"],                                    # [calibrating parameter]
                        "expon": param_dict["expon"],                                # [calibrating parameter]
                        "gw_storage": param_dict["gw_storage"],         
                        "soil_storage": param_dict["soil_storage"], 
                        "K_lf": param_dict["K_lf"],                                  # [calibrating parameter]
                        "K_nash": param_dict["K_nash"],                              # [calibrating parameter]
                        "nash_storage": param_dict["nash_storage"].tolist(), 
                        "giuh_ordinates": param_dict["giuh_ordinates"].tolist(), 
                        "stand_alone": 1, 
                        "unit_test": 0, 
                        "compare_results_file": "",
                        "partition_scheme": partition_scheme,
                        "soil_scheme": soil_scheme,
                        }

        # save and export json files
        json_filename = f'cat_{g_str}_bmi_config_cfe.json'
        json_file = os.path.join(config_dir, json_filename)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dict_json, f, ensure_ascii=False, indent=4, separators=(',', ':'))
            
if __name__ == "__main__":
    main()