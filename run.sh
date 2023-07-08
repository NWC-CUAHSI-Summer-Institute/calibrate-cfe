#!/bin/bash

# Check if the file path is provided as an argument
if [ $# -lt 1 ]; then
    echo "Please provide the path to the file containing basin_ids."
    exit 1
fi

# Read the first 48 basin_ids from the file
file_path=$1
mapfile -t basin_ids < <(head -n 516 "$file_path")

# Join the basin_ids with commas
joined_basin_ids=$(IFS=, ; echo "${basin_ids[*]}")

# Execute the Python script with the joined basin_ids
python 3-CFE_Calibration_Loop_1.py --multirun "basin_id=$joined_basin_ids"