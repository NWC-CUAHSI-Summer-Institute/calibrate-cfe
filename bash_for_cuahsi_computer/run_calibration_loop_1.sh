#!/bin/bash
#!/usr/bin/env python

#SBATCH -p defq
#SBATCH -J cal_loop_1
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH -o cfe_calibration_loop_1.log
#SBATCH -e cfe_calibration_loop_1.err

# memory
ulimit -s unlimited

source /home/ottersloth/anaconda3/etc/profile.d/conda.sh

conda activate conda_environment_with_spotpy

echo "I am running a CFE calibration now"

# run code
python3 CFE_Calibration_Loop_1.py > ./results/output_txt/cfe_calibration_loop_1_output.txt

echo "I am DONE running the CFE calibration now"