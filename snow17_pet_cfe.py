"""
Define a function that runs Snow module, PET computation, and CFE

Author: Abhinav Gupta (Created: 9 Feb 2026)
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import bmi_cfe_daily as bmi_cfe
import pet
from tonic.tonic.models.snow17 import snow17 

def run_snow_pet_cfe(met_df, dates, cfg_data, snow_params, pet_params, initial_snow_state, initial_cfe_state, lat, elev, time_step_size, time_step_units):
    
    
    swe, outflow = snow17.snow17(dates.dt.to_pydatetime(), met_df['prcp(mm/day)'].values, met_df['tavg(C)'].values, lat=lat, elevation=elev, dt=24, 
                                scf=snow_params['scf'], rvs=snow_params['rvs'], 
                                uadj=snow_params['uadj'], mbase=snow_params['mbase'], mfmax=snow_params['mfmax'], mfmin=snow_params['mfmin'], 
                                tipm=snow_params['tipm'], nmf=snow_params['nmf'],
                                plwhc=snow_params['plwhc'], pxtemp=snow_params['pxtemp'], pxtemp1=snow_params['pxtemp1'], 
                                pxtemp2=snow_params['pxtemp2'])       # Precip input should be in mm/day, temperature in degree C, dt in hours, 
    EP_m_per_day = outflow / 1000.0     # Convert from mm/day to m/day
    
    final_snow_state = ''

    # Compute PET using Priestley-Taylor method
    pet_mm_per_day = pet.priestley_taylor_fixed_alpha(met_df['Rn(W/m2)'].values, met_df['G(W/m2)'].values, 
                                                                met_df['tavg(C)'].values, alpha = pet_params['alpha_PT'])  # PET in mm/day

    # Convert alpha from mm/day to m/s
    pet_m_per_s = pet_mm_per_day / 1000.0 / time_step_size

    # Instantiate and initialize CFE model
    first_date = dates.iloc[0]
    year = first_date.year
    month = first_date.month
    day = first_date.day

    cfe_instance = bmi_cfe.BMI_CFE(cfg_data=cfg_data, time_step_size=time_step_size, time_step_units = time_step_units)
    cfe_instance.initialize(current_year = year, current_month = month, current_day = day, ############################################### update these dates
                            initial_state={'gw_reservoir_initial_storage_m': initial_cfe_state['gw_initial_storage_m'], 
                                           'soil_reservoir_initial_storage_m': initial_cfe_state['soil_initial_storage_m'],})
        
    outputs = cfe_instance.get_output_var_names()
    output_lists = {output:[] for output in outputs}

    # Run CFE model for the calibration period
    for (precip_time_integrated, pet_rate) in zip(EP_m_per_day, pet_m_per_s):############################################################################# MODIFY to INTEGRATE SNOW MODEL
        
        cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_time_integrated)    # Set value of the precipitation (total during the timestep) for the current timestep
        cfe_instance.set_value('water_potential_evaporation_flux', pet_rate)           # Set value of the PET rate (in m per second) for the current timestep

        cfe_instance.update()
        
        for output in outputs:
            output_lists[output].append(cfe_instance.get_value(output))
    
    return output_lists, final_snow_state