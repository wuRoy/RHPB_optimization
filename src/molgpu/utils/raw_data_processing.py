import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def activity_normalization(activity: np.ndarray):
    
    '''
    normalize the activity data to the control group
    '''

    # calcaulate the median of the control group
    positive_group = activity[0:8]
    positive_median = np.median(positive_group)
    # normalize the activity
    activity = (activity) / (positive_median)

    return activity


def plot_heat_map(activity: np.ndarray, save_path: Path):
    '''
    plot the heatmap of activity from molecular device plate reader
    It takes a 8*12 array as input and plot the heatmap
    '''
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a heatmap
    cax = ax.imshow(activity, cmap='Reds', interpolation='nearest')
    cbar = fig.colorbar(cax, shrink=0.5, aspect=8)
    cbar.set_label('Activity', rotation=270, labelpad=25)
    columns = [str(i) for i in range(1, 13)]
    rows = [chr(i) for i in range(ord('A'), ord('H') + 1)]
    plt.xticks(np.arange(len(columns)), columns)
    plt.yticks(np.arange(len(rows)), rows)
    # Add the numerical values on each grid
    for i in range(activity.shape[0]):
        for j in range(activity.shape[1]):
            ax.text(j, i, f'{activity[i, j]:.3f}', ha='center', va='center', color='black')

    # add title and axis labels
    plt.title('Heatmap of absolute activity')
    # Save the figure as a PNG file
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf() 
    pass


def molecular_device(project_dir: Path, iteration: int):
    '''
    process the raw data from the molecular device plate reader
    
    raw_data_path: Path
        the path to the raw_data.csv file
    '''
    # read the raw data
    raw_data_path = project_dir / 'data' / f'round_{iteration}' / 'raw_data.csv'
    heatmap_path = project_dir / 'data' / f'round_{iteration}' / 'heatmap.png'
    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    # extract the fitness which is raw_data[2:9,1:13] and blank which is raw_data[2:9,15:27]
    activity = raw_data[1:9, 1:13]
    blank = raw_data[1:9, 14:26]
    activity = activity - blank
    plot_heat_map(activity, heatmap_path)
    # tranfrom activity to 1d array column by column
    activity = activity.reshape((96, 1), order='F')
    # normalize the activity to the first column
    activity = activity_normalization(activity)
    np.savetxt(
        project_dir / 'data' / f'round_{iteration}' / 'activity.csv',
        activity,
        delimiter=",",
        fmt='%1.3f'
    )


def tecan_plate_reader(project_dir: Path, iteration: int):
    '''
    process the raw data from the tecan plate reader
    
    raw_data_path: Path
        the path to the raw_data.csv file
    '''
    # read the raw data
    raw_data_path = project_dir / 'data' / f'round_{iteration}' / 'raw_data.asc'
    heatmap_path = project_dir / 'data' / f'round_{iteration}' / 'heatmap.png'
    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    # extract the A450 signal which is raw_data[2:9, 1:13] and A540 which is raw_data[10:18, 1:13] 
    activity = raw_data[1:9, 1:13]
    blank = raw_data[10:18, 1:13]
    activity = activity - blank
    plot_heat_map(activity, heatmap_path)
    # tranfrom activity to 1d array column by column
    activity = activity.reshape((96, 1), order='F')
    # normalize the activity to the first column
    activity = activity_normalization(activity)
    np.savetxt(
        project_dir / 'data' / f'round_{iteration}' / 'activity.csv',
        activity,
        delimiter=",",
        fmt='%1.3f'
    )
