#!/bin/bash

#SBATCH --job-name=cfe-classic
#SBATCH --partition=normal
#SBATCH --array=0-47
#SBATCH --output=%a.out
#SBATCH --error=%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbhuiya2@gmu.edu

module load python

# Set the filepath to the basin_ids text file
file_path="/scratch/sbhuiya2/SI2023/data/camels/gauch_etal_2020/basin_list_516.txt"

# Calculate the number of lines from the file
num_lines=$(wc -l < "$file_path")

# Calculate the start and end line numbers for the current array task ID
subset_start=$((SLURM_ARRAY_TASK_ID * 48 + 1))
subset_end=$((subset_start + 48 - 1))

# Check if the subset end is beyond the total number of lines
if [ $subset_end -gt $num_lines ]; then
    subset_end=$num_lines
fi

# Function to execute the Python script for a subset
execute_subset() {
    local subset_start=$1
    local subset_end=$2

    # Read the current batch of basin_ids from the file
    mapfile -t basin_ids < <(sed -n "${subset_start},${subset_end}p" "$file_path")

    # Join the basin_ids with commas
    joined_basin_ids=$(IFS=, ; echo "${basin_ids[*]}")

    # Execute the Python script with the joined basin_ids
    python 3-CFE_Calibration_Loop_1.py --multirun "basin_id=${joined_basin_ids}"
}

# Execute the current subset
execute_subset "$subset_start" "$subset_end"