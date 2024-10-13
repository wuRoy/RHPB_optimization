import numpy as np
from pathlib import Path
from tecan.utils import virtual_plate_reader


def branin(x):
    '''
        Branin Function.
            x: input array
    '''
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    result = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    return result


def rosenbrock(x):
    '''
    Rosenbrock Function.
        x: input array
    '''
    
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0) * -50 + 100


def rastrigin(x):
    '''
    Rastrigin Function.
        x: input array
    '''
    return sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10) * -1


def ackley(x):
    '''
    Ackley Function.
        x: input array
    '''
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / x.size))
    term2 = -np.exp(sum2 / x.size)
    f = term1 + term2 + a + np.exp(1)
    
    return f * -1


def fitness_function(project_dir: Path, x: np.ndarray) -> np.ndarray:
    '''
    A toy fitness function for testing
    '''
    # parent directory of the project_dir
    base_dir = project_dir.parent.parent
    # read from base_dir/src/tecan/random_parameters.csv
    coeff = np.genfromtxt(base_dir / 'src' / 'tecan' / 'random_parameters.csv', delimiter=",")
    # take first x.shape[1] elements of coeff
    coeff = coeff[:x.shape[1]]
    fitness = np.matmul(x**2, coeff) * 100
    # add noise
    noise = np.random.normal(0, 1, fitness.shape) * 1
    fitness = fitness + noise
    
    return fitness


def virtual_run(project_dir: Path, iteration: int):
    '''
    run a virtual experiment
    '''
    input_path = project_dir / 'data' / f'round_{iteration}' / 'proposed_composition.csv'
    raw_data_path = project_dir / 'data' / f'round_{iteration}' / 'raw_data.csv'
    f = open(input_path, 'r', encoding='utf-8-sig')
    x = np.genfromtxt(f, delimiter=",")
    fitness = fitness_function(project_dir, x)
    raw_data = virtual_plate_reader(fitness)
    np.savetxt(raw_data_path, raw_data, delimiter=",")
