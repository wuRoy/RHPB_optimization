# This script is the main script for running the closed loop optimization
from pathlib import Path
import shutil
import os
import time
import pandas as pd
import logging
from utils import setup_logger, sort_save_joblist, update_current_job_status, get_yaml, early_stopping_checker, update_current_job_status_development 
from command_generator import run_command_generator
from foo_experiment import run_foo_experiment
from optimization import run_optimization


project_config = get_yaml()
project_name = project_config['project_name']
total_iteration = project_config['total_iteration']
opt_method = project_config['optimization_method']
polymer_list = project_config['polymer_list']
exp_mode = project_config['mode']
early_stopping = project_config['early_stopping']

base_dir = Path.cwd()

# if there is no folder in base_dir/Experiment called project_name, create one
project_dir = base_dir / 'Experiments' / project_name

project_dir.mkdir(parents=True, exist_ok=True)
setup_logger(project_dir)
if opt_method == 'predefined':
    logging.info("Waiting for job_list to be created in the project directory")
    # wait until a job_list directory is created in the project directory
    while not (project_dir / 'job_list').is_dir():
        pass
    logging.info("Found it!")
    # set a timer to wait for 5 seconds to make sure all csv files are created
    time.sleep(5)
    # sort the job_list and save it to a txt file
    job_list_dir = project_dir / 'job_list'
    job_list = sort_save_joblist(job_list_dir)
    # count the number of files in job_list directory
    total_iteration = len(job_list)
    logging.info(f"There are {total_iteration} jobs")


# create sub_directories
(project_dir / 'status_files').mkdir(exist_ok=True)
(project_dir / 'data').mkdir(exist_ok=True)
(project_dir / 'config_files').mkdir(exist_ok=True)
(project_dir / 'temp').mkdir(exist_ok=True)
# copy the project_config.yaml to the project directory/config_files
shutil.copy('project_config.yaml', project_dir / 'config_files' / 'project_config.yaml')

# initialization: read the round number, status of molgpu and tecan
# for molgpu.txt and tecan.txt, 0 means running and 1 means finished

round_file = project_dir / 'status_files' / 'round.txt'
if not round_file.is_file():
    with round_file.open('w') as f:
        f.write("1")
    (project_dir / 'data' / 'round_1').mkdir(exist_ok=True)
    # wait until a proposed_composition.csv is created in the round_1 folder
    logging.info("Waiting for proposed_composition.csv to be created in round_1 folder")
    if opt_method == 'predefined':
        # copy the first job to round_1 folder as proposed_composition.csv
        job = job_list[0]
        os.system(f"cp {job} {project_dir}/data/round_1/proposed_composition.csv")
        
    while not (project_dir / 'data' / 'round_1' / 'proposed_composition.csv').is_file():
        pass
    logging.info("Found it!")
    
molgpu_file = project_dir / 'status_files' / 'molgpu.txt'

if not molgpu_file.is_file():
    with molgpu_file.open('w') as f:
        f.write("0")

tecan_file = project_dir / 'status_files' / 'tecan.txt'
if not tecan_file.is_file():
    with tecan_file.open('w') as f:
        f.write("0")

early_stop_file = project_dir / 'status_files' / 'early_stopping.txt'
if not early_stop_file.is_file():
    with early_stop_file.open('w') as f:
        f.write("0")

with round_file.open('r') as f:
    iteration = f.read().strip()
    iteration = int(iteration)

with molgpu_file.open('r') as f:
    molgpu = f.read().strip()
    molgpu = int(molgpu)

with tecan_file.open('r') as f:
    tecan = f.read().strip()
    tecan = int(tecan)
    
# Create a dictionary with the project name and status
current_job = {
    "project_name": project_name,
    "total_iteration": total_iteration,
    "optimization strategy": opt_method,
    "status": "running",   
    "current_iteration": iteration
}
# Convert the dictionary to a pandas DataFrame
config_df = pd.DataFrame([current_job])
# Write the DataFrame to current_job.csv
if exp_mode == 'development':
    config_df.to_csv(base_dir / 'Experiments' / 'current_job_development.csv', index=False)
elif exp_mode == 'experiment':
    config_df.to_csv(base_dir / 'Experiments' / 'current_job.csv', index=False)

# Run the optimization loop
while iteration <= total_iteration:
    logging.info(f"Running {iteration}th loop")
    # if opt_method is GA_hd, run_command_generator with num_wells_source_plate=96
    if opt_method == 'GA_hd':
        run_command_generator(project_dir, num_wells_source_plate=96, total_volume=20.0)
    else:
        run_command_generator(project_dir, num_wells_source_plate=4)

    # if exp_mode is 'development', run foo_experiment
    if exp_mode == 'development':
        run_foo_experiment(project_dir)
    run_optimization(project_dir, opt_method, exp_mode)
    logging.info(f'Finished {iteration}th loop')
    iteration += 1

    if exp_mode == 'development':
        current_job = update_current_job_status_development(project_dir, base_dir, current_job, iteration)
    elif exp_mode == 'experiment':
        current_job = update_current_job_status(project_dir, base_dir, current_job, iteration)

    if early_stopping == 1:
        early_stopping_status = early_stopping_checker(project_dir)
        # add early stopping criteria
        if early_stopping_status == 1:
            # change the status of early_stop.txt to 1
            (project_dir / 'status_files' / 'early_stopping.txt').write_text('1')
            logging.info(f'Early stop after {iteration-1} iteration')
            break

current_job["status"] = "finished"
current_job["current_iteration"] = iteration - 1
config_df = pd.DataFrame([current_job])

if exp_mode == 'development':
    config_df.to_csv(base_dir / 'Experiments' / 'current_job_development.csv', index=False)
elif exp_mode == 'experiment':  
    config_df.to_csv(base_dir / 'Experiments' / 'current_job.csv', index=False)

config_df.to_csv(project_dir / 'config_files' / 'exp_config.csv', index=False)
