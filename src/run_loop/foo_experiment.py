from utils import setup_logger
from pathlib import Path
import logging
from tecan.virtual_run import virtual_run


def run_foo_experiment(project_dir: Path):
    '''
        Run an optimization of a linear function virtually to test the workflow
    '''
    # write status_files/tecan.txt as 0
    (project_dir / 'status_files' / 'tecan.txt').write_text('0')
    # read the iteration number
    with open(project_dir / 'status_files' / 'round.txt', 'r') as f:
        iteration = int(f.read())
    setup_logger(project_dir)
    logging.info(f'Tecan is currently running on round {iteration}.')
    virtual_run(project_dir, iteration)
    logging.info('Done!')
    # re-write status_files/tecan.txt as 1
    (project_dir / 'status_files' / 'tecan.txt').write_text('1')

    pass
