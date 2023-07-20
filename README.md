# calibrate_cfe
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![versions](https://img.shields.io/pypi/pyversions/hydra-core.svg) [![CodeStyle](https://img.shields.io/badge/code%20style-Black-black)]()

#  Summary
This codes calibrate [CFE model in Python version](https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py), with all the pipelines set up. 

# Installation 
Use conda to create your own env based on our ```environment.yml``` file
```
conda env create -f environment.yml
conda activate CFE
```

# To run this codes
0. Create your own config file using ```example_config.yaml``` and name it as ```config.yaml```
1. Run ```0-create_config_files.ipynb``` to generate model time-splitting and parameter bound files. Make any changes if you need 
2. Run ```1-1-download_pipeline.ipynb``` to download dataset you need. One point, you need to manually open browser and download zip file from Hydroshare storage. To get the Hydroshare storage permission, contact anyone of the 2023 team. 
3. Run ```1-2-check_nan_in_data.ipynb``` to create a list of files with missing data 
4. Run run.sh to calibrate parameters 
```
	./run.sh /fullpath/toyour/basin_ids.txt
```
5. Run ```4-CFE_testrun_with_best_calibrated_params.py``` to test the best parameter calibrated in step #4. 
6. Visualization code is in development
7. Notebooks with names starting from ```99-``` is from 2022 team to execute model selection using random forest. Contact Francisco Haces-Garcia for the details. 

# Folder structure
If you want to completely follow the example_config.yaml file, the following folder structure will be build while going through the pipeline. 
```
project_folder/
├─ data/
├─ cfe_py/
├─ calibrate_cfe/
│  ├─ configs/
│  ├─ results/
```

# Model outputs
The initial parameters, the best calibrated parameters, and the associated steamflow ouputs are hosted here https://www.hydroshare.org/resource/f7d6db8f8677402d808531924bbcf60c/ 

## Authors 
Modified by 2023 SI team
- Ryoko Araki (San Diego State University & University of California, Santa Barbara, @ry4git)
- Soelem Aafnan Bhuiyan (George Mason University, Fairfax, Virginia @soelemaafnan)
- Tadd Bindas (Penn State University, University Park, Pennsylvania, @taddyb)
- Jeremy Rapp (Michigan State University, East Lansing, Michigan @rappjer1)

Build upon the scripts by 2022 SI team
- Lauren A. Bolotin; San Diego State University
- Francisco Haces-Garcia; University of Houston
- Mochi Liao; Duke University
- Qiyue Liu; University of Illinois at Urbana-Champaign
