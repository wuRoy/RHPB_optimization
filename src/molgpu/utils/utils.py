import numpy as np
import itertools
import quantecon as qe
from pathlib import Path    


def get_data(path: Path):
    f = open(path, 'r', encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")

    return RawData


def combine_data(project_dir: Path, iteration: int):
    '''
    combine the proposed composition and fitness value
    '''
    # read the proposed composition
    x = get_data(project_dir / 'data' / f'round_{iteration}' / 'proposed_composition.csv')
    # read the fitness value
    fitness = get_data(project_dir / 'data' / f'round_{iteration}' / 'activity.csv')
    # create a new file to save the combined data
    combined_path = project_dir / 'data' / f'round_{iteration}' / 'combined_results.csv'
    # stack x with fitness value in the column and save it
    combined_data = np.hstack((x, fitness.reshape(-1, 1)))
    np.savetxt(combined_path, combined_data, delimiter=",", fmt='%1.3f')
    
    pass


def selected_enumeration(
        num_candidates: int,
        num_dim: int,
        resolution: int = 20
) -> np.ndarray:
    
    '''
        pick num_candidates from num_dim and enumerate all the possible simplex compositions
        resolution: the resolution of the simplex grid, e.g. 20 means 100/20 = 5% resolution
    '''
    
    compositions = qe.simplex_grid(num_candidates, resolution) / resolution
    vector = np.arange(0, num_dim)
    combinations = list(itertools.combinations(vector, num_candidates))
    combinations = np.array(combinations)
    all_compositions = []
    for i in range(combinations.shape[0]):
        enumerated_compositions = np.zeros((compositions.shape[0], num_dim))
        for j in range(num_candidates):
            enumerated_compositions[:, combinations[i, j]] = compositions[:, j]
        all_compositions.append(enumerated_compositions)
    all_compositions = np.array(all_compositions)
    all_compositions = all_compositions.reshape(-1, num_dim)
    all_compositions = np.unique(all_compositions, axis=0)
    
    return all_compositions
