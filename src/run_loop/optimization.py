from utils import setup_logger, get_iteration
import logging
from datetime import datetime
from pathlib import Path
from molgpu.optimization.predefined.run_predefined import run_joblist
from molgpu.optimization.BO.GP_qNEI import GP_qNEI  
from molgpu.optimization.GA.GA_constraint import constraint_genetic_algorithm, constraint_genetic_algorithm_hd
from molgpu.utils.utils import combine_data
from molgpu.utils import raw_data_processing
import numpy as np
import time


def run_optimization(project_dir: Path, opt_method: str, exp_mode: str):
    
    '''
    project_dir: the path to the project directory
    opt_method: the optimization method, could be BO, GA, BO-GA, or predefined
    exp_mode: the mode of the experiment, could be experiment or development
    '''
    setup_logger(project_dir)
    iteration = get_iteration(f'{project_dir}/status_files/round.txt')
    logging.info(f'optimization.py is watching for the raw_data.xxx file of round_{iteration}.')

    # wait for the raw_data file to be created
    fitness_status = True
    while fitness_status:
        print(f'Watching... Last checked at {datetime.now()}\r', end='')
        if (project_dir / 'data' / f'round_{iteration}' / 'raw_data.csv').exists() or (project_dir / 'data' / f'round_{iteration}' / 'raw_data.asc').exists():
            if exp_mode == 'experiment':
                raw_data_processing.tecan_plate_reader(project_dir, iteration)
            elif exp_mode == 'development':
                raw_data_processing.molecular_device(project_dir, iteration)
            fitness_status = False
        time.sleep(5)
    logging.info('Found it!')
    
    # combine the composition and activity data of the current round
    combine_data(project_dir, iteration)
    # mark molgpu as running
    (project_dir / 'status_files' / 'molgpu.txt').write_text('0')
    logging.info(f'molgpu is currently running on round {iteration}.')
    start_time = time.time()
    # make a diagonal identity matrix for control experiments 12x12
    num_control = 12
    control = np.identity(num_control)
    
    if opt_method == 'GA':
        GA = constraint_genetic_algorithm(
            project_dir=project_dir,
            iteration=iteration,
            num_candidates=88 - num_control
        )        
        next_round = GA.propose_new_candidate()
        next_round = np.vstack((control, next_round))
        # shuffle the rows
        np.random.shuffle(next_round)
        # add 8 zeros rows to the top of the next round for control experiments
        next_round = np.vstack((np.zeros((8, next_round.shape[1])), next_round))
        
    elif opt_method == 'GA_hd':
        GA = constraint_genetic_algorithm_hd(
            project_dir=project_dir,
            iteration=iteration,
            num_candidates=88
        )        
        next_round = GA.propose_new_candidate()
        next_round = np.vstack((next_round))
        # shuffle the rows
        np.random.shuffle(next_round)
        # add 8 zeros rows to the top of the next round for control experiments
        next_round = np.vstack((np.zeros((8, next_round.shape[1])), next_round))    
        
    elif opt_method == 'BO':
        BO = GP_qNEI(
            project_dir=project_dir,
            iteration=iteration,
            num_candidates=88 - num_control
        )
        next_round = BO.train_and_propose() 
        next_round = np.vstack((control, next_round))
        # shuffle the rows
        # add 8 zeros rows to the top of the next round for control experiments
        next_round = np.vstack((np.zeros((8, next_round.shape[1])), next_round))

    elif opt_method == 'BO-GA':
        BO = GP_qNEI(
            project_dir=project_dir,
            iteration=iteration,
            num_candidates=int(44 - num_control / 2)
        )
        next_round_BO = BO.train_and_propose() 
        
        GA = constraint_genetic_algorithm(
            project_dir=project_dir,
            iteration=iteration,
            num_candidates=int(44 - num_control / 2)
        )
        next_round_GA = GA.propose_new_candidate()
        
        BO_GA = np.vstack((next_round_BO, next_round_GA))
        next_round = np.vstack((control, BO_GA))
        # add 8 zeros rows to the top of the next round for control experiments
        next_round = np.vstack((np.zeros((8, next_round_BO.shape[1])), next_round))

    elif opt_method == 'predefined':
        next_round = run_joblist(project_dir, iteration)  # to avoid changing the iteration number before Tecan is finished
        
    end_time = time.time()
    time_taken = end_time - start_time
    logging.info(f'Time taken for optimization is {time_taken} seconds')
    
    # set the sleeping time
    if exp_mode == 'experiment':
        time.sleep(60) 
    elif exp_mode == 'development':
        time.sleep(2)
    
    if next_round.shape[0] > 96:
        raise ValueError('The number of candidates in the next round is more than 96!')
    
    # create the next round directory and save the proposed composition
    (project_dir / 'data' / f'round_{iteration+1}').mkdir(parents=True, exist_ok=True)
    np.savetxt(
        project_dir / 'data' / f'round_{iteration+1}' / 'proposed_composition.csv', 
        next_round, 
        delimiter=",",
        fmt='%1.3f'
    )
    logging.info('Done!')

    # make the status as finished
    (project_dir / 'status_files' / 'molgpu.txt').write_text('1')
