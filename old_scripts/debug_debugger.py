import os
import pandas as pd

print(os.getcwd())

basin_dir = r'..\data\camels\gauch_etal_2020'
basin_filename = 'basin_list_516.txt' # It was 516 basin in 2022 code

basin_file = os.path.join(basin_dir,basin_filename)
with open(basin_file, "r") as f:
    basin_list = pd.read_csv(f, header=None)
    print(basin_list.head())