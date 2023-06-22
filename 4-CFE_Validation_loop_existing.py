import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob

# import the cfe model
import bmi_cfe
import cfe

# import hydro evaluation package
!pip install hydroeval
import hydroeval as he

# define working dir
working_dir = '/home/ottersloth/cfe_calibration'

# ----------------------------------- Data Loading Dir ----------------------------------- #
# define basin list dir
basin_dir = '/home/ottersloth/data/camels_hourly'
basin_filename = 'basin_list_516.txt'
basin_file = os.path.join(basin_dir,basin_filename)

# define config dir
config_dir = os.path.join(working_dir,'configs')

# define observation file dir
#obs_dir = os.path.join(working_dir,'usgs-streamflow')
obs_dir = '/home/ottersloth/data/camels_hourly/usgs_streamflow'

# list of basins with missing data -> skipping these
missing_data_list = ['1552000','1552500','1567500','3338780','7301410','7315700','7346045','8050800','8101000','8104900','8109700','8158810','1187300','1510000','2178400','2196000','2202600','3592718','4197170','6919500','8176900']

# ----------------------------------- Saving Results Dir ----------------------------------- #
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

# define iteration number
N = 300

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

file_list = os.listdir(best_run_dir) # dir is your directory path
num_files = len(file_list)

# Loop through each basin
for i in range(num_files): 
 
    filename = file_list[i].split('_')
    g = filename[0]
    print(f"current basin: {g}.")

    # if g in missing_data_list: 
    #     print(f"none or missing usgs streamflow data for basin {g}, skipping this basin.") 
    #     continue

    # locate config file
    config_filename = 'cat_' + str(g) + '_bmi_config_cfe.json'
    config_file = os.path.join(config_dir,config_filename)

    # ----------------------------------- Run the Model ----------------------------------- #

    cfemodel = bmi_cfe.BMI_CFE(config_file)
    print('###--------model succesfully setup----------###')

    # load best parameters found in calibration period
    # best_run_filename = '**/*' + str(g) + '*.*'
    # #best_run_file = os.path.join(best_run_dir,best_run_filename)
    # file_list = []
    # for files in glob.glob(best_run_dir + best_run_filename):
    #     file_list.append(files)

    best_run_file = os.path.join(best_run_dir,file_list[i])

    with open(best_run_file) as data_file:
            data_loaded = json.load(data_file)

    best_run_params = data_loaded["best parameters"]

    # initialize the model
    cfemodel.initialize(param_vec = best_run_params)
    print('###--------model succesfully initialized----------###')

    with open(cfemodel.forcing_file, 'r') as f:
        df_forcing = pd.read_csv(f)
    print(f"###----- forcing_file loaded:{cfemodel.forcing_file}. -----###")

    # --------------------------------------- Run Spin-up Period --------------------------------------- # 
    # define the spin up period
    spinup_start_idx_nldas = np.where(df_forcing['date']=='2001-10-01 00:00:00')
    spinup_end_idx_nldas = np.where(df_forcing['date']=='2002-09-30 23:00:00')
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
    cal_start_idx_nldas = np.where(df_forcing['date']=='2002-10-01 00:00:00')
    cal_end_idx_nldas = np.where(df_forcing['date']=='2007-09-30 23:00:00')
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
    if int(g) > 10000000: obs_filename = str(g) + '-usgs-hourly.csv'
    else: obs_filename = '0' + str(g) + '-usgs-hourly.csv'
    obs_file = os.path.join(obs_dir,obs_filename)

    data = pd.read_csv(obs_file)
    obs_data = data['QObs_CAMELS(mm/h)'].values
    eval_dates = data['date'].values

    # define calibration period for usgs streamflow obs.
    cal_start_idx_usgs = np.where(eval_dates=='2002-10-01 00:00:00')
    cal_end_idx_usgs = np.where(eval_dates=='2007-09-30 23:00:00')
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
    plt.title(f"Simulated Streamflow against Observation in the Validation Period [ID: 0{g}]", fontsize = 28)
    plt.tight_layout()

    validation_imgname = str(g) + "_validation_" +str(N) + ".png"
    validation_imgdir = os.path.join(png_dir,"Validation")
    if os.path.exists(validation_imgdir)==False: os.mkdir(validation_imgdir)
    validation_imgfile = os.path.join(validation_imgdir,validation_imgname)
    
    plt.savefig(validation_imgfile,bbox_inches='tight')
    
    # calculate kge values
    print(f"The KGE value is : {kge}.")
    print(f"The NSE value is : {nse}.")

    performance_values["basin_id"].append(g)
    performance_values["kge_values"].append(kge)
    performance_values["nse_values"].append(nse)

# ---------------------------------------- End of Looping Validation Period ---------------------------------------- #

# Save performance dict
df = pd.DataFrame(performance_values)
df.to_csv("cfe_validation_performance_values.csv")