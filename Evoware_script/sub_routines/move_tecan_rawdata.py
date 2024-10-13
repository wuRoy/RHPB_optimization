import pandas as pd
import time
from pathlib import Path
import shutil
import os
# Evoware need absolute path
base_dir = Path('//.../Experiments')  # change this to the absolute path of the Experiments folder
raw_data_path = Path('E:/Evoware_script/cache')  # change this to the absolute path of the cache folder for the plate reader files

# read the project_name from current_job.csv with pandas
project_name = pd.read_csv(base_dir / 'current_job.csv')['project_name'][0]
project_dir = base_dir / project_name

with open(base_dir / project_name / 'status_files' / 'round.txt', 'r') as f:
    iteration = int(f.read())

print('Moving raw data for: ',project_name)

time.sleep(10)
# copy 'project_name_iteration.asc' from E:\Guangqi\Cache\Plate_reader_temp to project_dir\data\round_iteration


raw_data_activity = raw_data_path / f'{project_name}_{iteration}_activity.asc'
raw_data_after_polymer = raw_data_path / f'{project_name}_{iteration}_after_polymer.asc'
raw_data_after_GOx = raw_data_path / f'{project_name}_{iteration}_after_GOx.asc'
raw_data_dest = project_dir / 'data' / f'round_{iteration}' 

# copy the raw data to the data folder
shutil.copy(raw_data_activity, raw_data_dest)
shutil.copy(raw_data_after_polymer, raw_data_dest)
shutil.copy(raw_data_after_GOx, raw_data_dest)

# rename the raw data as 'raw_data.asc'
os.rename(raw_data_dest / f'{project_name}_{iteration}_activity.asc', raw_data_dest / 'raw_data.asc')

