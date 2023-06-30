#!/bin/bash

# Check if the file path is provided as an argument
if [ $# -lt 1 ]; then
    echo "Please provide the path to the file containing basin_ids."
    exit 1
fi

# Read the basin_ids from the file and join them with commas
file_path=$1
IFS=$'\n' read -d '' -r -a basin_ids < "$file_path"
joined_basin_ids=$(IFS=, ; echo "${basin_ids[*]}")

# Execute the Python script with the joined basin_ids
python3 3-CFE_Calibration_Loop_1.py --multirun "basin_id=$joined_basin_ids"