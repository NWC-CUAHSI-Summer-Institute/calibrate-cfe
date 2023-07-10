from omegaconf import DictConfig, OmegaConf
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
        self.obs_data = obs_data0['QObs(mm/h)'].values  # Use hourly data
        self.eval_dates = obs_data0['date'].values

        # define calibration period for usgs streamflow obs.
        cal_start_idx_usgs = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])
        cal_end_idx_usgs = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])
        self.eval_dates = self.eval_dates[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0] + 1]
        self.obs_data = self.obs_data[cal_start_idx_usgs[0][0]:cal_end_idx_usgs[0][0] + 1]

        if print_all_process:
            print('###--------- usgs start date: ' + self.eval_dates[0] + '.---------')
            print('###--------- usgs end date: ' + self.eval_dates[-1] + '.---------')
            print('###---------- obs data length: ' + str(len(self.obs_data)) + '.---------')

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

        # --------------------------------------- Run CFE model --------------------------------------- #
        self.cfemodel.update()
        self.cfemodel.finalize()

        # --------------------------------------- Get simulated data --------------------------------------- #
        sim_output = self.cfemodel.q_sim.tolist()
        sim_output = np.array(sim_output).reshape((len(sim_output),))

        # --------------------------------------- Filter the simulated data --------------------------------------- #
        # Filter the data according to the calibration period
        cal_start_idx_sim = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])
        cal_end_idx_sim = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])
        sim_output = sim_output[cal_start_idx_sim[0][0]:cal_end_idx_sim[0][0] +My apologies for the incomplete response. Here's the remaining code:


        # --------------------------------------- Filter the simulated data --------------------------------------- #
        # Filter the data according to the calibration period
        cal_start_idx_sim = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])
        cal_end_idx_sim = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])
        sim_output = sim_output[cal_start_idx_sim[0][0]:cal_end_idx_sim[0][0] + 1]

        if self.print_all_process:
            print('###--------CFE model successfully run-----------###')

        return sim_output

    def evaluation(self):
        return self.obs_data

    def objectivefunction(self, simulation, evaluation):

        # Filter the observed data according to the calibration period
        cal_start_idx_obs = np.where(self.eval_dates == self.time_split['calibration']['start_datetime'])
        cal_end_idx_obs = np.where(self.eval_dates == self.time_split['calibration']['end_datetime'])
        evaluation = evaluation[cal_start_idx_obs[0][0]:cal_end_idx_obs[0][0] + 1]

        # Compute the objective function (root mean square error)
        rmse = np.sqrt(np.mean((simulation - evaluation) ** 2))

        return rmse

    def calibrate_basin(self, basin_id):

        if self.print_all_process:
            print('###------- Running calibration for basin ' + basin_id + ' --------###')

        # Set the path for saving the results
        results_dir = os.path.join(self.config_dir, 'calibration_results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Set the unique identifier for the basin's calibration run
        identifier = 'cat_' + self.gauge_id + '_' + basin_id

        # Define the spotpy object for calibration
        spotpy_calibrator = spotpy.algorithms.sceua(self, dbname=identifier,
                                                    dbformat='csv',
                                                    save_sim=False,
                                                    save_threshold=0.1,
                                                    parallel='seq',
                                                    num_workers=1)

        # Run the calibration
        spotpy_calibrator.sample(repetitions=1500, ngs=10)

        # Save the calibration results
        results_file = os.path.join(results_dir, identifier + '.csv')
        spotpy_calibrator.save(results_file)

        if self.print_all_process:
            print('###------- Finished calibration for basin ' + basin_id + ' --------###')

    def run_calibration(self):

        if self.print_all_process:
            print('###------- Starting calibration process for all basins --------###')

        # Read the basin list
        basin_list_file = os.path.join(self.config_dir, 'basin_list.txt')
        with open(basin_list_file, 'r') as f:
            basin_list = f.read().splitlines()

        # Get the task ID from the SLURM environment variable
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

        # Get the number of CPUs in one node
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])

        # Subset the basin list based on the number of CPUs
        start_idx = (task_id - 1) * num_cpus
        end_idx = start_idx + num_cpus
        subset_basin_list = basin_list[start_idx:end_idx]

        # Run calibration for each basin in the subset list
        for basin_id in subset_basin_list:
            self.calibrate_basin(basin_id)

        if self.print_all_process:
            print('###------- Finished calibration process for all basins --------###')


# ----------------------------------- Main Script ----------------------------------- #
def main():

    # Load the configuration file
    config_file = "config.yaml"
    config = OmegaConf.load(config_file)

    # Extract the required configuration parameters
    config_dir = config.config_dir
    obs_file_path = config.obs_file_path
    gauge_id = config.gauge_id
    time_split = config.time_split
    parameter_bounds = config.parameter_bounds
    print_all_process = config.print_all_process

    # Set up the Spotpy object
    spotpy_setup = Spotpy_setup(config_dir, obs_file_path, gauge_id, time_split, parameter_bounds, print_all_process)

    # Run the calibration
    spotpy_setup.run_calibration()


if __name__ == "__main__":
    main()