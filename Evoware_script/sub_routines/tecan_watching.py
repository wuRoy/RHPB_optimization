from utils import setup_logger
import logging
import pandas as pd
import time
from pathlib import Path
import shutil
import os
# Evoware need absolute path
base_dir = Path('//.../Experiments')  # change this to the absolute path of the Experiments folder

# read the project_name from current_job.csv with pandas
project_name = pd.read_csv(base_dir / 'current_job.csv')['project_name'][0]
project_dir = base_dir / project_name

print('Current project is: ',project_name)
# write status of tecan as running
with open(base_dir / project_name / 'status_files' / 'tecan.txt', 'w') as f:
    f.write('0')

with open(base_dir / project_name / 'status_files' / 'round.txt', 'r') as f:
    iteration = int(f.read())

setup_logger(base_dir / project_name)
logging.info(f'Tecan finished running on round {iteration}.')

logging.info('Done!')
# mark as finished
with open(base_dir / project_name / 'status_files' / 'tecan.txt', 'w') as f:
    f.write('1')

# wait for the optimization to finish to start the next round
optimization_status = True
logging.info(f'Looking for the tecan command for round {iteration+1}.')
while optimization_status:
    if (base_dir / project_name / 'data' / f'round_{iteration+1}' / 'proposed_composition.csv').exists():
        optimization_status = False
    time.sleep(5)
logging.info('Found it!')