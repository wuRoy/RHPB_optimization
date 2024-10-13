import numpy as np


def virtual_plate_reader(fitness: np.ndarray):
    '''
    transform the fitness to the plate reader data
    '''
    
    fitness = fitness.reshape((8, 12), order='F')
    fitness[:, 0] = np.array([0, 0, 50, 50, 50, 50, 0, 0])
    blank_data = np.ones_like(fitness) * 0.5
    blank_data = np.insert(blank_data, 0, np.arange(1, 13), axis=0)
    fitness = np.insert(fitness, 0, np.arange(1, 13), axis=0)
    blank_data = np.insert(blank_data, 0, np.zeros(9), axis=1)
    fitness = np.insert(fitness, 0, np.zeros(9), axis=1)
    data = np.hstack((fitness, blank_data))
    
    return data
