import numpy as np
from pathlib import Path


def run_joblist(
    project_dir: Path,
    iteration: int
) -> np.ndarray:
    '''
    The function will take files in Experiments/project_dir/joblist one by one for evaluation with the automated workflow
    Each file in ../joblist is a csv file with 96 rows and n columns, where n is the number of RHPs to be blended.
    If you want the job run in sequence, each file should be named with xxx_01.csv, xxx_02.csv, etc.
    
    No header and no index for the csv
        project_dir: the path to the project directory
        iteration: the current round of the optimization
    '''

    job_list_dir = project_dir / 'job_list'
    # read job_list.txt as a list
    job_list = [line.rstrip('\n') for line in open(job_list_dir / 'job_list.txt')]
    # read the current job
    current_job = job_list[iteration]
    # read the composition
    composition = np.genfromtxt(current_job, delimiter=',')

    return composition.round(3)
