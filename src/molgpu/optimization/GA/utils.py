import numpy as np
from pathlib import Path
import pandas as pd


def set_seeds(seed: int = 42) -> None:
    # set random seed
    np.random.seed(seed)


def get_data(path: Path):
    f = open(path, 'r', encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    
    return RawData


def get_fitness(path: Path):
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
    fitness = get_fitness(project_dir / 'data' / f'round_{iteration}' / 'fitness.csv')
    # create a new file to save the combined data
    combined_path = project_dir / 'data' / f'round_{iteration}' / 'combined_results.csv'
    # stack x with fitness value in the column and save it
    combined_data = np.hstack((x, fitness.reshape(-1, 1)))
    np.savetxt(combined_path, combined_data, delimiter=",")
    
    pass


def matrix_relaxation(x, decimal=1):
    
    '''
    make sure the sum of each row is 1
        x: a matrix to be relaxed
        dicimal: the number of dicimal to be kept
    '''
    # if the matrix is a 1-dimention array, add a zeros as the second rows
    is_one_dimention = False
    if len(x.shape) == 1:
        zeros = np.zeros((x.shape[0]))
        x = np.vstack((x, zeros))
        is_one_dimention = True

    for i in range(x.shape[0]):
        nonzero_idx = np.nonzero(x[i])
        nonzero_value = x[i][nonzero_idx]
        combined = np.vstack((nonzero_idx, nonzero_value)).T
        if combined.shape[0] < 2:
            continue
        # sort the combined array based on the second column
        combined = combined[combined[:, 1].argsort()]
        sum = 0
        for k in range(combined.shape[0] - 1):
            combined[k, 1] = round(combined[k, 1], decimal)
            sum = sum + combined[k, 1]
        combined[-1, 1] = 1 - sum
        for j in range(combined.shape[0]):
            x[i, int(combined[j, 0])] = combined[j, 1]
    if is_one_dimention:
        x = x[0]

    return x


def normalize_data(x):
    '''
    Normalize the data, to make sure the sum of each row is 1
    '''
    x = x / x.sum(axis=1, keepdims=True)
    return x


def get_previous_data(project_dir: Path):
    '''
    Get the previous data from the project_dir
    '''
    # collect all combined_results.csv for training the model
    # if there is a project_dir/data/previous_data.csv, then use it
    csv_files = sorted(Path(project_dir / 'data').glob('round_*/combined_results.csv'))
    if len(csv_files) == 0:
        pass
    previous_data = project_dir / 'data' / 'previous_data.csv'
    # Initialize an empty list to hold the tensors
    database = []
    # Loop through all csv files and read them into tensors
    for csv_file in csv_files:
        arr = np.genfromtxt(csv_file, delimiter=',')
        arr = arr[8:, :]
        database.append(arr)

    if previous_data.exists():
        # read the data from the csv file in the form of updated_results.csv
        df = pd.read_csv(previous_data)
        # remove the rows with Activity lower than 0.1 and first 12 columns are all zeros
        df = df[(df.iloc[:, :-2] != 0).any(axis=1)]
        # transform the data into numpy array
        df = df.to_numpy()
        # remove the last column 
        df = df[:, :-1]
        database.append(df)

    # Concatenate all array into one
    combined_database = np.concatenate(database)
    x = combined_database[:, :-1]
    y = combined_database[:, -1]

    return x, y


def usage_checker(
    current_data: np.ndarray,
    previous_data: np.ndarray, 
    volume_each_well: float, 
    usage_threshold: float = 10000
):
    '''
    check the usage of the of certain materials reaches the threshold
        current_data: the current fraction to be analyzed
        previous_data: the previous fraction that have been used
        volume_each_well: the volume of each well, in ul
        usage_threshold: the threshold of the usage, in ul
    '''
    usage = previous_data.sum(axis=0) * volume_each_well
    # find the index of non-zero elements in current_data
    index = np.nonzero(current_data)
    for i in index[0]:
        if usage[i] > usage_threshold:
            return True

    return False
