# This script is used to generate the command file for the Tecan robot.
# It is called by run_closed_loop.py.
from pathlib import Path
from utils import setup_logger, get_iteration
import logging
from molgpu.utils.csv_to_tecan import tecan_print


def run_command_generator(
        project_dir: Path,
        num_wells_source_plate: int = 4,
        total_volume: float = 100.0
):
    # Get the current iteration
    iteration = get_iteration(project_dir / 'status_files' / 'round.txt')
    setup_logger(project_dir)
    logging.info(f'Tecan is currently printing on round {iteration}.')
    tecan_print(
        project_dir / 'data' / f'round_{iteration}' / 'proposed_composition.csv', 
        project_dir / 'data' / f'round_{iteration}' / 'tecan_command.csv',
        num_wells_source_plate=num_wells_source_plate,
        total_volume=total_volume
    )

    logging.info('Done!')
