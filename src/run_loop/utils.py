import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import numpy as np


def setup_logger(project_dir: Path):
    log_format = '%(asctime)s: %(process)d-%(levelname)s-%(message)s'
    log_datefmt = '%y-%b-%d %H:%M:%S'
    logging.basicConfig(
        filename=project_dir / 'output.log',
        level=logging.INFO,
        filemode='a',
        format=log_format,
        datefmt=log_datefmt
    )


def update_current_job_status(
    project_dir: Path, 
    base_dir: Path, 
    current_job: pd.DataFrame,
    iteration: int
):
    '''
    update the current_job.csv and round.txt file in the base_dir and Experiments directory
    summarize the results and save it to updated_results.csv
    plot the results and save it to results.png
    '''
    current_job["current_iteration"] = iteration
    config_df = pd.DataFrame([current_job])
    config_df.to_csv(base_dir / 'Experiments' / 'current_job.csv', index=False)
    config_df.to_csv(project_dir / 'config_files' / 'exp_config.csv', index=False)
    with open(project_dir / 'status_files' / 'round.txt', 'w') as f:
        f.write(str(iteration))
    update_summary(project_dir)
    plot_updated_results(project_dir)
    
    return current_job


def update_current_job_status_development(
    project_dir: Path,
    base_dir: Path,
    current_job: pd.DataFrame,
    iteration: int
):
    '''
    update the current_job.csv and round.txt file in the base_dir and Experiments directory
    summarize the results and save it to updated_results.csv
    plot the results and save it to results.png
    '''
    current_job["current_iteration"] = iteration
    config_df = pd.DataFrame([current_job])
    config_df.to_csv(base_dir / 'Experiments' / 'current_job_development.csv', index=False)
    config_df.to_csv(project_dir / 'config_files' / 'exp_config.csv', index=False)
    with open(project_dir / 'status_files' / 'round.txt', 'w') as f:
        f.write(str(iteration))
    update_summary(project_dir)
    plot_updated_results(project_dir)
    
    return current_job


def sort_save_joblist(job_list_dir):
    # sort all csv files in the job_list directory
    job_list = [f for f in job_list_dir.iterdir() if f.is_file()]
    # remove non-csv files
    job_list = [f for f in job_list if f.suffix == '.csv']
    job_list.sort()
    with open(job_list_dir / 'job_list.txt', 'w') as f:
        for job in job_list:
            f.write(str(job) + '\n')
    return job_list


def get_iteration(path):
    with open(path, 'r') as f:
        iteration = int(f.read())
    return iteration


def update_summary(project_dir: Path):
    # collect all combined_results.csv and save it to a file called updated_results.csv with iteration as the last column
    csv_files = sorted(Path(project_dir / 'data').glob('round_*/combined_results.csv'))
    dfs = []
    for csv_file in csv_files:
        # Extract iteration number from the directory name
        iteration = csv_file.parent.name.split('_')[-1].zfill(2)
        # Read each csv file into a DataFrame
        df = pd.read_csv(csv_file, header=None)
        # Rename the columns
        df.columns = ['Variable_' + str(i) for i in range(1, len(df.columns))] + ['Activity']
        # Add a new column for the iteration
        df['iteration'] = iteration
        # Append the DataFrame to the list
        dfs.append(df)
        
    if (project_dir / 'data' / 'previous.csv').exists():
        df = pd.read_csv(project_dir / 'data' / 'previous.csv', header=None)
        df.columns = ['Variable_' + str(i) for i in range(1, len(df.columns))] + ['Activity']
        df['iteration'] = '00'
        dfs.append(df)
    # Concatenate all the dataframes in the list
    all_combined_results = pd.concat(dfs, ignore_index=True)

    # Write the concatenated dataframe to updated_results.csv
    all_combined_results.to_csv(project_dir / 'updated_results.csv', index=False, float_format='%.3f')


def plot_heatmap(df, file_name: Path):
    # plot the heatmap of the dataframe
    num_columns = df.shape[1] - 2
    # Create a mask where True indicates rows where any variable equals 1
    mask = (df.iloc[:, :num_columns] == 1).any(axis=1)
    # Find the indices of the rows to highlight
    rows_to_highlight = np.where(mask)[0]
    # Create a GridSpec with 1 row and 2 columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[num_columns, 2]) 
    # Create a figure
    fig = plt.figure(figsize=(num_columns / 4, num_columns))
    # Create a heatmap for the first region
    ax1 = plt.subplot(gs[0])
    sns.heatmap(df.iloc[:, :num_columns], cmap='viridis', cbar=False, ax=ax1)
    for row in rows_to_highlight:
        ax1.hlines(row, *ax1.get_xlim(), color='red', linewidth=3)
    ax1.set(xticklabels=[], yticklabels=[])
    ax1.tick_params(left=False, bottom=False)
    # Create a heatmap for the second and third region
    ax2 = plt.subplot(gs[1])
    sns.heatmap(df.iloc[:, -1:], cmap='viridis', cbar=False, ax=ax2)
    ax2.set(xticklabels=[], yticklabels=[])
    ax2.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_updated_results(project_dir: Path):
    
    df = pd.read_csv(project_dir / 'updated_results.csv')
    # remove the control groups
    df = df[(df.iloc[:, :-2] != 0).any(axis=1)]
    df['iteration'] = df['iteration'].astype(float)
    max_activity = df.groupby('iteration')['Activity'].max()
    max_activity_df = pd.DataFrame({'iteration_max': max_activity.index - 1, 'Activity': max_activity.values})
    
    # Plot the frontier of the maximum value in each iteration, connecting the points with a dashed line and red color
    # plot the results as violin plots
    sns.violinplot(x='iteration', y='Activity', data=df)
    sns.lineplot(x='iteration_max', y='Activity', data=max_activity_df, color='red', linestyle='--', marker='o')

    # Set the labels for the x and y axes
    plt.xlabel('Iteration')
    plt.ylabel('Activity')
    
    # Set the x-axis ticks to integer values
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig(project_dir / 'results.png')
    plt.close()
    
    # Select columns starting with 'Variable_' for PCA
    variable_cols = [col for col in df.columns if col.startswith('Variable_')]
    X = df[variable_cols]
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Create a scatter plot of the first two principal components
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Create a scatter plot of the first two principal components for iteration
    scatter1 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['iteration'], cmap='viridis', alpha=0.8)
    fig.colorbar(scatter1, ax=axs[0], label='Iteration')
    axs[0].set_title('PCA_Iteration')
    # Create a scatter plot of the first two principal components for activity
    scatter2 = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Activity'], cmap='plasma', alpha=0.8)
    fig.colorbar(scatter2, ax=axs[1], label='Activity')
    axs[1].set_title('PCA_Activity')
    plt.tight_layout()
    plt.savefig(project_dir / 'PCA_analysis.png')
    plt.close()
    
    # Determine the number of rows and columns for the subplots
    n = len(variable_cols)
    ncols = 4
    nrows = n // ncols + (n % ncols > 0)
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    # Flatten the axes array and remove extra subplots
    axs = axs.flatten()
    for ax in axs[n:]:
        fig.delaxes(ax)
    # Create a scatter plot for each variable
    for i, col in enumerate(variable_cols):
        axs[i].scatter(df[col], df['Activity'], alpha=0.5)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Activity')
        axs[i].set_title(f'{col} vs Activity')
    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.savefig(project_dir / 'Variable_vs_Activity.png')
    plt.close()
    
    plot_heatmap(df, project_dir / 'heatmap_iteration.png')
    plot_heatmap(df.sort_values(by='Activity', ascending=False), project_dir / 'heatmap_activity.png')


def early_stopping_checker(
        project_dir: Path, 
        stop_iterations: int = 4,
        threshold: float = 0.05
):
    '''
    if the max_value have not increased over threshold for stop_iterations, return True
        project_dir: the path to the project directory
        stop_iterations: the number of iterations to stop
        threshold: the threshold for the max_activity_diff_for_checking
    '''
    # Read the updated_results.csv file into a DataFrame
    df = pd.read_csv(project_dir / 'updated_results.csv')
    # remove the rows with variable values of 0
    df = df[(df.iloc[:, :-2] != 0).any(axis=1)]
    # Convert the 'iteration' column to integer
    df['iteration'] = df['iteration'].astype(int)
    # Calculate the maximum activity for each iteration
    max_activity = df.groupby('iteration')['Activity'].max()
    # analyze the increase in max_activity
    max_activity_diff = max_activity.diff()
    max_activity_diff_for_checking = max_activity_diff.iloc[-1 * stop_iterations:]
    if (max_activity_diff_for_checking < threshold).all():
        return 1
    else:
        return 0


def get_yaml(project_config_path: Path = Path('project_config.yaml')):

    with project_config_path.open('r') as f:
        config = yaml.safe_load(f)
        
    return config
