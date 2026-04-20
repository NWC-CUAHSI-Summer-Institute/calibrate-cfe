"""
SPOTPY setup file

Author: Abhinav Gupta (Created: 9 Feb 2026)
"""

import numpy as np
from spotpy.parameter import Uniform, List
import snow17_pet_cfe
import os
import spotpy

class spotpy_setup(object):
    def __init__(self, param_range, area_km2, warmup_days, met_df_cal, dates_cal, qobs_cal, initial_snow_state, initial_cfe_state, lat, elev, time_step_size, time_step_units):
        
        self.parameternames=param_range.keys()
        self.params=[]
        for parname in self.parameternames:
            low, high, discrete_flag = param_range[parname]
            if discrete_flag == 1:
                self.params.append(List(parname, list(low)))
            else:
                self.params.append(spotpy.parameter.Uniform(parname, low, high))

        self.dim = len(self.params)

        self.area_km2 = area_km2
        self.warmup_days = warmup_days
        self.met_df_cal = met_df_cal
        self.dates_cal = dates_cal
        self.observations = qobs_cal
        self.initial_snow_state = initial_snow_state
        self.initial_cfe_state = initial_cfe_state
        self.lat = lat
        self.elev = elev
        self.time_step_size = time_step_size
        self.time_step_units = time_step_units


    def parameters(self):           
        return spotpy.parameter.generate(self.params)

    def simulation(self, params):
        pet_params = {'alpha_PT': params[0]}
        snow_params = {'scf': params[1], 'rvs': params[2], 'uadj': params[3], 'mbase': params[4], 'mfmax': params[5], 'mfmin': params[6], 
                         'tipm': params[7], 'nmf': params[8], 'plwhc': params[9], 'pxtemp': params[10], 'pxtemp1': params[11], 'pxtemp2': params[12]}
        cfg_data = {
            "catchment_area_km2": self.area_km2,
            "partition_scheme":"Schaake",
            "soil_params":{"bb":params[13], "satdk":params[14], "satpsi":params[15], "slop":params[16], "smcmax":params[17], "wltsmc":params[18], "D":params[19], "K_lf":params[20], "alpha_fc":params[21]},
            "max_gw_storage":params[22], "Cgw":params[23], "expon":params[24],
            "K_nash_lateral":params[25], "nash_storage_lateral":[0.0]*int(params[26]), "K_nash_surface":params[27], "nash_storage_surface":[0.0]*int(params[28]), "refkdt":params[29], 'trigger_z_m': params[30],
            "soil_scheme":"classic",
            "stand_alone": 0,
}        

        output_lists, _ = snow17_pet_cfe.run_snow_pet_cfe(self.met_df_cal, self.dates_cal, cfg_data,
                                                           snow_params, pet_params, self.initial_snow_state, 
                                                           self.initial_cfe_state, self.lat, self.elev, 
                                                           self.time_step_size, self.time_step_units)
        simulations = np.array(output_lists['land_surface_water__runoff_depth'])*1000.0
        return simulations[self.warmup_days:]

    def evaluation(self):
        return self.observations

    def objectivefunction(self,simulation,evaluation):
        objectivefunction = -1*np.mean(np.abs(simulation - evaluation))     # -1 because the DDS in SPOTPY maximizes the ojective function, but we want to minimize the mean absolute error between simulation and evaluation
        return objectivefunction